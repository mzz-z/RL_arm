"""Running observation normalizer for PPO."""

import numpy as np
from typing import Optional


class RunningNormalizer:
    """
    Online normalizer using Welford's algorithm.

    Maintains running mean and variance of observations,
    normalizes new observations using these statistics.

    Supports:
    - Batch updates
    - Normalization with optional clipping
    - State serialization for checkpointing
    """

    def __init__(
        self,
        shape: tuple,
        epsilon: float = 1e-8,
        clip: Optional[float] = 10.0,
    ):
        """
        Initialize normalizer.

        Args:
            shape: Shape of observations (e.g., (obs_dim,))
            epsilon: Small constant for numerical stability
            clip: If provided, clip normalized obs to [-clip, clip]
        """
        self.shape = shape
        self.epsilon = epsilon
        self.clip = clip

        # Running statistics (float64 for precision)
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon  # Small init to avoid division by zero

    def update(self, batch: np.ndarray) -> None:
        """
        Update running stats with a batch of observations.

        Args:
            batch: Observations to update with, shape (batch_size, *shape)
        """
        batch = np.asarray(batch, dtype=np.float64)

        # Handle single observation
        if batch.ndim == len(self.shape):
            batch = batch[np.newaxis, ...]

        batch_mean = batch.mean(axis=0)
        batch_var = batch.var(axis=0)
        batch_count = batch.shape[0]

        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(
        self,
        batch_mean: np.ndarray,
        batch_var: np.ndarray,
        batch_count: int,
    ) -> None:
        """
        Update using batch statistics (Welford's algorithm).

        Args:
            batch_mean: Mean of batch
            batch_var: Variance of batch
            batch_count: Number of samples in batch
        """
        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        # Update mean
        self.mean = self.mean + delta * batch_count / total_count

        # Update variance using parallel algorithm
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        self.var = M2 / total_count

        self.count = total_count

    def normalize(self, obs: np.ndarray) -> np.ndarray:
        """
        Normalize observation using running statistics.

        Args:
            obs: Observation(s) to normalize

        Returns:
            Normalized observation(s)
        """
        obs = np.asarray(obs, dtype=np.float32)
        normalized = (obs - self.mean.astype(np.float32)) / np.sqrt(
            self.var.astype(np.float32) + self.epsilon
        )

        # Optional clipping
        if self.clip is not None:
            normalized = np.clip(normalized, -self.clip, self.clip)

        return normalized

    def denormalize(self, normalized_obs: np.ndarray) -> np.ndarray:
        """
        Denormalize observation back to original scale.

        Args:
            normalized_obs: Normalized observation(s)

        Returns:
            Original-scale observation(s)
        """
        normalized_obs = np.asarray(normalized_obs, dtype=np.float32)
        return normalized_obs * np.sqrt(
            self.var.astype(np.float32) + self.epsilon
        ) + self.mean.astype(np.float32)

    def state_dict(self) -> dict:
        """
        Get state for checkpointing.

        Returns:
            Dictionary with mean, var, count
        """
        return {
            "mean": self.mean.copy(),
            "var": self.var.copy(),
            "count": self.count,
        }

    def load_state_dict(self, state: dict) -> None:
        """
        Load state from checkpoint.

        Args:
            state: Dictionary with mean, var, count
        """
        self.mean = np.array(state["mean"], dtype=np.float64)
        self.var = np.array(state["var"], dtype=np.float64)
        self.count = state["count"]

    def reset(self) -> None:
        """Reset statistics to initial state."""
        self.mean = np.zeros(self.shape, dtype=np.float64)
        self.var = np.ones(self.shape, dtype=np.float64)
        self.count = self.epsilon


def create_normalizer(config: dict, obs_dim: int) -> RunningNormalizer:
    """
    Create RunningNormalizer from config.

    Args:
        config: Observation normalization config dict
        obs_dim: Observation dimension

    Returns:
        RunningNormalizer instance (or None if disabled)
    """
    if not config.get("enabled", True):
        return None

    return RunningNormalizer(
        shape=(obs_dim,),
        epsilon=config.get("epsilon", 1e-8),
        clip=config.get("clip", 10.0),
    )
