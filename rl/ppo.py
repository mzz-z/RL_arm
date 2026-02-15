"""PPO algorithm implementation."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from rl.networks import ActorCritic
from rl.buffer import RolloutBuffer
from rl.normalizer import RunningNormalizer


class PPO:
    """
    Proximal Policy Optimization algorithm.

    Features:
    - Clipped surrogate objective
    - Value function loss
    - Entropy bonus for exploration
    - Learning rate scheduling (linear or cosine)
    - Gradient clipping
    - Observation normalization
    """

    def __init__(
        self,
        policy: ActorCritic,
        buffer: RolloutBuffer,
        obs_normalizer: Optional[RunningNormalizer] = None,
        # Learning rate
        lr: float = 3e-4,
        lr_schedule: str = "linear",
        lr_min: float = 0.0,
        total_updates: int = 1000,
        # PPO parameters
        clip_range: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        target_kl: Optional[float] = 0.02,
        # Update parameters
        epochs_per_update: int = 10,
        minibatch_size: int = 64,
        # Device
        device: str = "cpu",
    ):
        """
        Initialize PPO.

        Args:
            policy: ActorCritic network
            buffer: RolloutBuffer for storing transitions
            obs_normalizer: Optional observation normalizer
            lr: Initial learning rate
            lr_schedule: "linear", "cosine", or "constant"
            lr_min: Minimum learning rate (for cosine schedule)
            total_updates: Total number of PPO updates (for scheduling)
            clip_range: PPO clip range
            entropy_coef: Entropy bonus coefficient
            value_coef: Value loss coefficient
            max_grad_norm: Maximum gradient norm for clipping
            target_kl: If set, stop epoch early when approx KL exceeds this
            epochs_per_update: Number of epochs per PPO update
            minibatch_size: Minibatch size for updates
            device: Device for tensors
        """
        self.policy = policy
        self.buffer = buffer
        self.obs_normalizer = obs_normalizer
        self.device = device

        # Move policy to device
        self.policy.to(device)

        # PPO parameters
        self.clip_range = clip_range
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl

        # Update parameters
        self.epochs_per_update = epochs_per_update
        self.minibatch_size = minibatch_size

        # Optimizer
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

        # Learning rate scheduling
        self.initial_lr = lr
        self.lr_min = lr_min
        self.lr_schedule = lr_schedule
        self.total_updates = total_updates
        self.current_update = 0

    def get_action(
        self,
        obs: np.ndarray,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get action from policy.

        Args:
            obs: Observations (num_envs, obs_dim) or (obs_dim,)
            deterministic: If True, use mean action (for evaluation)

        Returns:
            action: Actions (num_envs, action_dim) or (action_dim,)
            log_prob: Log probabilities (num_envs,) or scalar
            value: Value estimates (num_envs,) or scalar
            raw_action: Pre-tanh actions for log-prob consistency (num_envs, action_dim)
        """
        # Normalize observation if normalizer exists
        if self.obs_normalizer is not None:
            obs_normalized = self.obs_normalizer.normalize(obs)
        else:
            obs_normalized = obs

        # Convert to tensor
        obs_tensor = torch.FloatTensor(obs_normalized).to(self.device)

        # Handle single observation
        single = obs_tensor.ndim == 1
        if single:
            obs_tensor = obs_tensor.unsqueeze(0)

        # Get action from policy
        with torch.no_grad():
            action, log_prob, value, raw_action = self.policy.act(obs_tensor, deterministic)

        # Convert back to numpy
        action = action.cpu().numpy()
        value = value.cpu().numpy()

        if log_prob is not None:
            log_prob = log_prob.cpu().numpy()
        else:
            log_prob = np.zeros(action.shape[0])

        if raw_action is not None:
            raw_action = raw_action.cpu().numpy()
        else:
            raw_action = np.zeros_like(action)

        # Remove batch dimension if single
        if single:
            action = action[0]
            log_prob = log_prob[0]
            value = value[0]
            raw_action = raw_action[0]

        return action, log_prob, value, raw_action

    def get_value(self, obs: np.ndarray) -> np.ndarray:
        """
        Get value estimate only (for bootstrapping).

        Args:
            obs: Observations

        Returns:
            Value estimates
        """
        # Normalize observation
        if self.obs_normalizer is not None:
            obs_normalized = self.obs_normalizer.normalize(obs)
        else:
            obs_normalized = obs

        obs_tensor = torch.FloatTensor(obs_normalized).to(self.device)

        single = obs_tensor.ndim == 1
        if single:
            obs_tensor = obs_tensor.unsqueeze(0)

        with torch.no_grad():
            value = self.policy.get_value(obs_tensor)

        value = value.cpu().numpy()

        if single:
            value = value[0]

        return value

    def update(self) -> Dict[str, float]:
        """
        Perform PPO update on collected rollout.

        Returns:
            Dictionary with training diagnostics
        """
        # Update learning rate
        self._update_learning_rate()
        self.current_update += 1

        # Tracking metrics
        all_policy_losses = []
        all_value_losses = []
        all_entropy_losses = []
        all_approx_kl = []
        all_clip_fracs = []

        # Multiple epochs over the rollout data
        kl_exceeded = False
        for epoch in range(self.epochs_per_update):
            if kl_exceeded:
                break

            # Iterate over minibatches
            for batch in self.buffer.get_minibatches(self.minibatch_size, shuffle=True):
                # Buffer stores pre-normalized observations (normalized during rollout
                # with the same stats used for old_log_probs), so no re-normalization needed.
                obs = batch["obs"]

                # Get current policy outputs
                new_log_probs, entropy, values = self.policy.evaluate(
                    obs,
                    batch["actions"],
                    batch["raw_actions"],
                )

                # Compute policy loss (clipped surrogate)
                ratio = torch.exp(new_log_probs - batch["old_log_probs"])
                surr1 = ratio * batch["advantages"]
                surr2 = torch.clamp(
                    ratio, 1 - self.clip_range, 1 + self.clip_range
                ) * batch["advantages"]
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(values, batch["returns"])

                # Entropy loss (negative because we want to maximize entropy)
                entropy_loss = -entropy.mean()

                # Total loss
                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    + self.entropy_coef * entropy_loss
                )

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)

                self.optimizer.step()

                # Track metrics
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - torch.log(ratio)).mean().item()
                    clip_frac = (torch.abs(ratio - 1) > self.clip_range).float().mean().item()

                all_policy_losses.append(policy_loss.item())
                all_value_losses.append(value_loss.item())
                all_entropy_losses.append(-entropy_loss.item())  # Report positive entropy
                all_approx_kl.append(approx_kl)
                all_clip_fracs.append(clip_frac)

                # Early stopping: if KL diverges too much, stop updating
                if self.target_kl is not None and approx_kl > self.target_kl:
                    kl_exceeded = True
                    break

        # Compute explained variance
        with torch.no_grad():
            returns = self.buffer.returns.flatten()
            values = self.buffer.values.flatten()
            explained_var = 1 - np.var(returns - values) / (np.var(returns) + 1e-8)

        # Get current learning rate
        current_lr = self.optimizer.param_groups[0]["lr"]

        return {
            "loss/policy": np.mean(all_policy_losses),
            "loss/value": np.mean(all_value_losses),
            "loss/entropy": np.mean(all_entropy_losses),
            "stats/approx_kl": np.mean(all_approx_kl),
            "stats/clip_frac": np.mean(all_clip_fracs),
            "stats/explained_var": explained_var,
            "stats/learning_rate": current_lr,
            "stats/std": np.exp(self.policy.log_std.detach().cpu().numpy()).mean(),
        }

    def _update_learning_rate(self) -> None:
        """Update learning rate based on schedule."""
        progress = self.current_update / max(self.total_updates, 1)

        if self.lr_schedule == "linear":
            # Linear decay
            new_lr = self.initial_lr * (1 - progress)
        elif self.lr_schedule == "cosine":
            # Cosine annealing
            new_lr = self.lr_min + 0.5 * (self.initial_lr - self.lr_min) * (
                1 + np.cos(np.pi * progress)
            )
        else:
            # Constant
            new_lr = self.initial_lr

        # Ensure LR doesn't go negative
        new_lr = max(new_lr, 0.0)

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = new_lr

    def save(self, path: str) -> None:
        """
        Save checkpoint.

        Args:
            path: Path to save checkpoint
        """
        state = {
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "current_update": self.current_update,
        }

        if self.obs_normalizer is not None:
            state["obs_normalizer"] = self.obs_normalizer.state_dict()

        torch.save(state, path)

    def load(self, path: str, weights_only: bool = False) -> int:
        """
        Load checkpoint.

        Args:
            path: Path to checkpoint
            weights_only: If True, only load policy weights (for transfer learning).
                          Keeps fresh optimizer, learning rate schedule, and update count.

        Returns:
            Current update count (0 if weights_only=True)
        """
        # weights_only=False for torch.load needed because we save numpy arrays (normalizer state)
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.policy.load_state_dict(checkpoint["policy_state_dict"])

        if weights_only:
            # Transfer learning: only load policy weights, keep fresh optimizer
            # Optionally load observation normalizer stats (useful for similar tasks)
            if self.obs_normalizer is not None and "obs_normalizer" in checkpoint:
                self.obs_normalizer.load_state_dict(checkpoint["obs_normalizer"])
            return 0

        # Full resume: load optimizer and training state
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_update = checkpoint["current_update"]

        if self.obs_normalizer is not None and "obs_normalizer" in checkpoint:
            self.obs_normalizer.load_state_dict(checkpoint["obs_normalizer"])

        return self.current_update


def create_ppo(
    policy: ActorCritic,
    buffer: RolloutBuffer,
    ppo_config: dict,
    model_config: dict,
    obs_dim: int,
    total_updates: int,
    device: str = "cpu",
) -> PPO:
    """
    Create PPO from config.

    Args:
        policy: ActorCritic network
        buffer: RolloutBuffer
        ppo_config: PPO config dict
        model_config: Model config dict (for obs normalization)
        obs_dim: Observation dimension
        total_updates: Total number of PPO updates
        device: Device for tensors

    Returns:
        PPO instance
    """
    # Create observation normalizer if enabled
    obs_norm_config = model_config.get("obs_norm", {})
    if obs_norm_config.get("enabled", True):
        from rl.normalizer import RunningNormalizer
        obs_normalizer = RunningNormalizer(
            shape=(obs_dim,),
            epsilon=obs_norm_config.get("epsilon", 1e-8),
            clip=obs_norm_config.get("clip", 10.0),
        )
    else:
        obs_normalizer = None

    return PPO(
        policy=policy,
        buffer=buffer,
        obs_normalizer=obs_normalizer,
        lr=ppo_config.get("lr", 3e-4),
        lr_schedule=ppo_config.get("lr_schedule", "linear"),
        lr_min=ppo_config.get("lr_min", 0.0),
        total_updates=total_updates,
        clip_range=ppo_config.get("clip_range", 0.2),
        entropy_coef=ppo_config.get("entropy_coef", 0.01),
        value_coef=ppo_config.get("value_coef", 0.5),
        max_grad_norm=ppo_config.get("max_grad_norm", 0.5),
        target_kl=ppo_config.get("target_kl", 0.02),
        epochs_per_update=ppo_config.get("epochs_per_update", 10),
        minibatch_size=ppo_config.get("minibatch_size", 64),
        device=device,
    )
