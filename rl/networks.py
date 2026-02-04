"""Neural network architectures for PPO."""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Optional


class TanhGaussianPolicy:
    """
    Tanh-squashed Gaussian distribution for bounded continuous actions.

    Samples from a Gaussian, then applies tanh to bound actions to [-1, 1].
    Properly corrects log-probability for the change of variables.
    """

    def __init__(self, mean: torch.Tensor, log_std: torch.Tensor):
        """
        Initialize distribution.

        Args:
            mean: Mean of the Gaussian (before tanh), shape (batch_size, action_dim)
            log_std: Log standard deviation, shape (action_dim,)
        """
        self.mean = mean
        self.log_std = log_std
        self.std = torch.exp(log_std)

        # Expand std to match mean's batch shape
        # log_std is (action_dim,), mean is (batch_size, action_dim)
        # We need std to be (batch_size, action_dim) for broadcasting
        std_expanded = self.std.expand_as(mean)
        self.normal = torch.distributions.Normal(mean, std_expanded)

    def sample(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action with reparameterization trick.

        Returns:
            action: Tanh-squashed action in [-1, 1]
            raw_action: Pre-tanh (unbounded) action
        """
        # Sample from Gaussian (unbounded)
        raw_action = self.normal.rsample()

        # Squash to [-1, 1]
        action = torch.tanh(raw_action)

        return action, raw_action

    def log_prob(
        self,
        action: torch.Tensor,
        raw_action: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute log-probability with Jacobian correction.

        The correction accounts for the change of variables:
        log pi(a) = log pi(u) - log |det(da/du)|
                  = log pi(u) - sum(log(1 - tanh^2(u)))

        where u is the pre-tanh (raw) action.

        Args:
            action: Tanh-squashed action
            raw_action: Pre-tanh action (computed if not provided)

        Returns:
            Log probability (per sample, summed over action dims)
        """
        if raw_action is None:
            # Recover raw action from squashed action via atanh
            # atanh(x) = 0.5 * log((1+x)/(1-x))
            # Add small epsilon for numerical stability
            action_clamped = torch.clamp(action, -1.0 + 1e-6, 1.0 - 1e-6)
            raw_action = 0.5 * torch.log((1 + action_clamped) / (1 - action_clamped))

        # Gaussian log-prob of raw action (sum over action dims)
        log_prob_raw = self.normal.log_prob(raw_action).sum(dim=-1)

        # Jacobian correction for tanh squashing
        # log(1 - tanh^2(x)) with numerical stability
        correction = torch.sum(
            torch.log(1 - action.pow(2) + 1e-6),
            dim=-1,
        )

        log_prob = log_prob_raw - correction
        return log_prob

    def deterministic_action(self) -> torch.Tensor:
        """For evaluation: use mean, then squash."""
        return torch.tanh(self.mean)

    def entropy(self) -> torch.Tensor:
        """
        Approximate entropy of tanh-squashed Gaussian.

        Exact entropy is intractable, so we use the Gaussian entropy
        as a proxy (works well in practice).

        Returns:
            Entropy (per sample, summed over action dims)
        """
        return self.normal.entropy().sum(dim=-1)


class ActorCritic(nn.Module):
    """
    Actor-Critic network with shared trunk for PPO.

    Architecture:
    - Shared MLP trunk
    - Actor head: outputs mean for tanh-squashed Gaussian
    - Learnable log-std (state-independent, per action dimension)
    - Critic head: outputs value estimate
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: list = [256, 256],
        activation: str = "relu",
        log_std_init: float = 0.0,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
    ):
        """
        Initialize network.

        Args:
            obs_dim: Observation dimension
            action_dim: Action dimension
            hidden_sizes: List of hidden layer sizes
            activation: Activation function ("relu", "tanh", "elu")
            log_std_init: Initial log standard deviation
            log_std_min: Minimum log-std (for clamping)
            log_std_max: Maximum log-std (for clamping)
        """
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # Select activation
        activation_fn = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "elu": nn.ELU,
        }.get(activation, nn.ReLU)

        # Build shared trunk
        layers = []
        prev_size = obs_dim
        for size in hidden_sizes:
            layers.append(nn.Linear(prev_size, size))
            layers.append(activation_fn())
            prev_size = size
        self.trunk = nn.Sequential(*layers)

        # Actor head: outputs mean for tanh-squashed Gaussian
        self.actor_mean = nn.Linear(prev_size, action_dim)

        # Learnable log-std (state-independent, per action dimension)
        self.log_std = nn.Parameter(torch.full((action_dim,), log_std_init))

        # Critic head
        self.critic = nn.Linear(prev_size, 1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using orthogonal initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)

        # Smaller init for actor output (for stable initial policy)
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.constant_(self.actor_mean.bias, 0)

        # Smaller init for critic output
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.constant_(self.critic.bias, 0)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            obs: Observation tensor (batch_size, obs_dim)

        Returns:
            mean: Action mean (batch_size, action_dim)
            log_std: Log standard deviation (action_dim,)
            value: Value estimate (batch_size, 1)
        """
        features = self.trunk(obs)
        mean = self.actor_mean(features)
        value = self.critic(features)

        # Clamp log_std for numerical stability
        log_std = torch.clamp(self.log_std, self.log_std_min, self.log_std_max)

        return mean, log_std, value

    def act(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
        """
        Get action from policy.

        Args:
            obs: Observation tensor (batch_size, obs_dim)
            deterministic: If True, use mean action (for evaluation)

        Returns:
            action: Action tensor in [-1, 1] (batch_size, action_dim)
            log_prob: Log probability (None if deterministic) (batch_size,)
            value: Value estimate (batch_size,)
            raw_action: Pre-tanh action (None if deterministic) (batch_size, action_dim)
        """
        mean, log_std, value = self.forward(obs)
        value = value.squeeze(-1)

        if deterministic:
            # Evaluation: use mean, squashed
            action = torch.tanh(mean)
            return action, None, value, None
        else:
            # Training: sample from distribution
            dist = TanhGaussianPolicy(mean, log_std)
            action, raw_action = dist.sample()
            log_prob = dist.log_prob(action, raw_action)
            return action, log_prob, value, raw_action

    def evaluate(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        raw_actions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for PPO update.

        Args:
            obs: Observation tensor (batch_size, obs_dim)
            actions: Action tensor (batch_size, action_dim)
            raw_actions: Pre-tanh actions for log-prob consistency (batch_size, action_dim)
                        If provided, uses these directly instead of reconstructing via atanh.

        Returns:
            log_prob: Log probability of actions (batch_size,)
            entropy: Entropy of distribution (batch_size,)
            value: Value estimate (batch_size,)
        """
        mean, log_std, value = self.forward(obs)
        value = value.squeeze(-1)

        dist = TanhGaussianPolicy(mean, log_std)
        # Pass raw_actions to avoid atanh reconstruction errors when actions saturate
        log_prob = dist.log_prob(actions, raw_actions)
        entropy = dist.entropy()

        return log_prob, entropy, value

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Get value estimate only (for bootstrapping).

        Args:
            obs: Observation tensor

        Returns:
            Value estimate (batch_size,)
        """
        features = self.trunk(obs)
        value = self.critic(features)
        return value.squeeze(-1)


def create_actor_critic(config: dict, obs_dim: int, action_dim: int) -> ActorCritic:
    """
    Create ActorCritic network from config.

    Args:
        config: Model config dict
        obs_dim: Observation dimension
        action_dim: Action dimension

    Returns:
        ActorCritic instance
    """
    return ActorCritic(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=config.get("hidden_sizes", [256, 256]),
        activation=config.get("activation", "relu"),
        log_std_init=config.get("log_std_init", 0.0),
        log_std_min=config.get("log_std_min", -20.0),
        log_std_max=config.get("log_std_max", 2.0),
    )
