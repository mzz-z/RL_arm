"""Rollout buffer for PPO with GAE computation."""

import numpy as np
import torch
from typing import Generator, Optional


class RolloutBuffer:
    """
    Stores rollout data from vectorized environments.

    Handles:
    - Storage of transitions (obs, actions, rewards, etc.)
    - GAE advantage computation with proper terminal handling
    - Minibatch generation for PPO updates

    Performance optimization:
    - Converts numpy arrays to torch tensors ONCE after GAE computation
    - Minibatch generation uses tensor indexing (no repeated CPU->GPU transfers)
    """

    def __init__(
        self,
        rollout_steps: int,
        num_envs: int,
        obs_dim: int,
        action_dim: int,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        device: str = "cpu",
    ):
        """
        Initialize buffer.

        Args:
            rollout_steps: Number of steps per environment per rollout
            num_envs: Number of parallel environments
            obs_dim: Observation dimension
            action_dim: Action dimension
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            device: Device for tensors ("cpu" or "cuda")
        """
        self.rollout_steps = rollout_steps
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device

        # Storage arrays (numpy, will convert to torch when needed)
        self.obs = np.zeros((rollout_steps, num_envs, obs_dim), dtype=np.float32)
        self.actions = np.zeros((rollout_steps, num_envs, action_dim), dtype=np.float32)
        self.raw_actions = np.zeros((rollout_steps, num_envs, action_dim), dtype=np.float32)
        self.log_probs = np.zeros((rollout_steps, num_envs), dtype=np.float32)
        self.rewards = np.zeros((rollout_steps, num_envs), dtype=np.float32)
        self.dones = np.zeros((rollout_steps, num_envs), dtype=np.float32)
        self.values = np.zeros((rollout_steps, num_envs), dtype=np.float32)

        # Computed during finalize
        self.advantages = np.zeros((rollout_steps, num_envs), dtype=np.float32)
        self.returns = np.zeros((rollout_steps, num_envs), dtype=np.float32)

        # Pre-converted torch tensors (set after compute_gae via prepare_for_update)
        self._tensors_ready = False
        self._obs_tensor = None
        self._actions_tensor = None
        self._raw_actions_tensor = None
        self._log_probs_tensor = None
        self._advantages_tensor = None
        self._returns_tensor = None

        # Track position
        self.step = 0
        self.full = False

    def reset(self) -> None:
        """Reset buffer for new rollout."""
        self.step = 0
        self.full = False
        self._tensors_ready = False

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        log_prob: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        value: np.ndarray,
        raw_action: np.ndarray = None,
    ) -> None:
        """
        Add a transition to the buffer.

        Args:
            obs: Observations (num_envs, obs_dim)
            action: Actions (num_envs, action_dim)
            log_prob: Log probabilities (num_envs,)
            reward: Rewards (num_envs,)
            done: Done flags (num_envs,)
            value: Value estimates (num_envs,)
            raw_action: Pre-tanh actions for log-prob consistency (num_envs, action_dim)
        """
        if self.step >= self.rollout_steps:
            raise RuntimeError("Buffer is full. Call reset() before adding more.")

        self.obs[self.step] = obs
        self.actions[self.step] = action
        self.log_probs[self.step] = log_prob
        self.rewards[self.step] = reward
        self.dones[self.step] = done
        self.values[self.step] = value
        if raw_action is not None:
            self.raw_actions[self.step] = raw_action

        self.step += 1
        if self.step == self.rollout_steps:
            self.full = True

    def compute_gae(
        self,
        final_value: np.ndarray,
        final_done: np.ndarray,
    ) -> None:
        """
        Compute Generalized Advantage Estimation.

        CRITICAL: Handles terminated vs truncated correctly.
        - terminated (done=True, truncated=False): bootstrap with 0
        - truncated (done=True, truncated=True): bootstrap with value estimate

        For simplicity, we treat all dones the same here (bootstrap with 0).
        The training loop should handle truncation by passing the correct final_value.

        Args:
            final_value: Value estimate for final state (num_envs,)
            final_done: Done flags for final state (num_envs,)
        """
        last_gae = 0.0

        for t in reversed(range(self.rollout_steps)):
            if t == self.rollout_steps - 1:
                # Last step: use final_value for bootstrap (0 if done)
                next_value = final_value * (1 - final_done)
                next_non_terminal = 1 - final_done
            else:
                next_value = self.values[t + 1]
                next_non_terminal = 1 - self.dones[t + 1]

            # TD error
            delta = (
                self.rewards[t]
                + self.gamma * next_value * (1 - self.dones[t])
                - self.values[t]
            )

            # GAE recursive computation
            self.advantages[t] = delta + self.gamma * self.gae_lambda * (1 - self.dones[t]) * last_gae
            last_gae = self.advantages[t]

        # Returns = advantages + values
        self.returns = self.advantages + self.values

        # Pre-convert to tensors for efficient minibatch generation
        self._prepare_tensors()

    def _prepare_tensors(self) -> None:
        """
        Convert numpy arrays to torch tensors ONCE after GAE computation.

        This is a key performance optimization: instead of creating new tensors
        for every minibatch (expensive Python overhead + CPU->GPU transfers),
        we convert once and then use tensor indexing for minibatches.
        """
        total_samples = self.rollout_steps * self.num_envs

        # Flatten arrays
        obs_flat = self.obs.reshape(-1, self.obs_dim)
        actions_flat = self.actions.reshape(-1, self.action_dim)
        raw_actions_flat = self.raw_actions.reshape(-1, self.action_dim)
        log_probs_flat = self.log_probs.reshape(-1)
        advantages_flat = self.advantages.reshape(-1).copy()  # copy for normalization
        returns_flat = self.returns.reshape(-1)

        # Normalize advantages (critical for stability)
        advantages_flat = (advantages_flat - advantages_flat.mean()) / (
            advantages_flat.std() + 1e-8
        )

        # Convert to tensors and move to device ONCE
        # Using pin_memory for faster CPU->GPU transfer if on CUDA
        use_pin = self.device != "cpu"

        if use_pin:
            # Create pinned memory tensors for faster transfers
            self._obs_tensor = torch.from_numpy(obs_flat).pin_memory().to(self.device, non_blocking=True)
            self._actions_tensor = torch.from_numpy(actions_flat).pin_memory().to(self.device, non_blocking=True)
            self._raw_actions_tensor = torch.from_numpy(raw_actions_flat).pin_memory().to(self.device, non_blocking=True)
            self._log_probs_tensor = torch.from_numpy(log_probs_flat).pin_memory().to(self.device, non_blocking=True)
            self._advantages_tensor = torch.from_numpy(advantages_flat).pin_memory().to(self.device, non_blocking=True)
            self._returns_tensor = torch.from_numpy(returns_flat).pin_memory().to(self.device, non_blocking=True)
        else:
            # CPU path - direct conversion
            self._obs_tensor = torch.from_numpy(obs_flat).to(self.device)
            self._actions_tensor = torch.from_numpy(actions_flat).to(self.device)
            self._raw_actions_tensor = torch.from_numpy(raw_actions_flat).to(self.device)
            self._log_probs_tensor = torch.from_numpy(log_probs_flat).to(self.device)
            self._advantages_tensor = torch.from_numpy(advantages_flat).to(self.device)
            self._returns_tensor = torch.from_numpy(returns_flat).to(self.device)

        self._tensors_ready = True

    def get_minibatches(
        self,
        minibatch_size: int,
        shuffle: bool = True,
    ) -> Generator[dict, None, None]:
        """
        Generate minibatches for PPO update.

        Performance optimization: uses pre-converted tensors and tensor indexing
        instead of creating new tensors for each minibatch.

        Args:
            minibatch_size: Size of each minibatch
            shuffle: Whether to shuffle indices

        Yields:
            Dictionary with batch tensors
        """
        if not self._tensors_ready:
            raise RuntimeError("Tensors not ready. Call compute_gae() first.")

        total_samples = self.rollout_steps * self.num_envs

        # Generate indices (on CPU, will be used for tensor indexing)
        if shuffle:
            indices = torch.randperm(total_samples, device=self.device)
        else:
            indices = torch.arange(total_samples, device=self.device)

        # Yield minibatches using tensor indexing (no new allocations)
        for start in range(0, total_samples, minibatch_size):
            end = min(start + minibatch_size, total_samples)
            batch_idx = indices[start:end]

            yield {
                "obs": self._obs_tensor[batch_idx],
                "actions": self._actions_tensor[batch_idx],
                "raw_actions": self._raw_actions_tensor[batch_idx],
                "old_log_probs": self._log_probs_tensor[batch_idx],
                "advantages": self._advantages_tensor[batch_idx],
                "returns": self._returns_tensor[batch_idx],
            }

    def get_all_data(self) -> dict:
        """
        Get all data as tensors (for logging/debugging).

        Returns:
            Dictionary with all buffer data as tensors
        """
        return {
            "obs": torch.FloatTensor(self.obs).to(self.device),
            "actions": torch.FloatTensor(self.actions).to(self.device),
            "log_probs": torch.FloatTensor(self.log_probs).to(self.device),
            "rewards": torch.FloatTensor(self.rewards).to(self.device),
            "dones": torch.FloatTensor(self.dones).to(self.device),
            "values": torch.FloatTensor(self.values).to(self.device),
            "advantages": torch.FloatTensor(self.advantages).to(self.device),
            "returns": torch.FloatTensor(self.returns).to(self.device),
        }

    @property
    def total_samples(self) -> int:
        """Total number of samples in buffer."""
        return self.rollout_steps * self.num_envs


def create_buffer(
    config: dict,
    obs_dim: int,
    action_dim: int,
    device: str = "cpu",
) -> RolloutBuffer:
    """
    Create RolloutBuffer from config.

    Args:
        config: PPO config dict
        obs_dim: Observation dimension
        action_dim: Action dimension
        device: Device for tensors

    Returns:
        RolloutBuffer instance
    """
    return RolloutBuffer(
        rollout_steps=config.get("rollout_steps", 2048),
        num_envs=config.get("num_envs", 8),
        obs_dim=obs_dim,
        action_dim=action_dim,
        gamma=config.get("gamma", 0.99),
        gae_lambda=config.get("gae_lambda", 0.95),
        device=device,
    )
