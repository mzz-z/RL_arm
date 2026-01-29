"""Action space definition for the 2-DOF arm."""

import numpy as np
from dataclasses import dataclass


@dataclass
class ActionSpace:
    """
    Defines the action space for the arm controller.

    Actions are always in [-1, 1] (tanh-squashed from policy).
    The controller scales these to joint target increments.
    """

    # Action dimensions
    action_dim: int = 2

    # Action bounds (always normalized)
    action_low: float = -1.0
    action_high: float = 1.0

    # Action scaling (radians per env step)
    action_scale: float = 0.1

    def clip_action(self, action: np.ndarray) -> np.ndarray:
        """
        Clip action to valid range.

        Args:
            action: Raw action (2,)

        Returns:
            Clipped action
        """
        return np.clip(action, self.action_low, self.action_high)

    def scale_action(self, action: np.ndarray) -> np.ndarray:
        """
        Scale normalized action to delta joint targets.

        Args:
            action: Normalized action in [-1, 1]

        Returns:
            Delta joint targets (radians)
        """
        return self.action_scale * action

    @property
    def shape(self) -> tuple:
        """Return action shape."""
        return (self.action_dim,)


def create_action_space(config: dict) -> ActionSpace:
    """
    Create ActionSpace from config.

    Args:
        config: Control config dict

    Returns:
        ActionSpace instance
    """
    return ActionSpace(
        action_dim=2,
        action_scale=config.get("action_scale", 0.1),
    )
