"""Position target controller for the 3-DOF arm."""

import numpy as np
import mujoco
from typing import Optional
from dataclasses import dataclass


@dataclass
class RateLimitConfig:
    """Rate limiting configuration."""

    enabled: bool = False
    max_delta: float = 0.15  # Max target change per step (radians)


@dataclass
class LowpassConfig:
    """Low-pass filter configuration."""

    enabled: bool = False
    alpha: float = 0.3  # Smoothing factor (0 = no change, 1 = instant)


class PositionTargetController:
    """
    Incremental position target controller for the 3-DOF arm.

    Policy outputs delta targets in [-1, 1], which are scaled
    and added to current targets. Targets are clipped to joint limits
    and applied to MuJoCo position actuators.

    This controller:
    - Maintains internal joint target state
    - Converts normalized actions to target increments
    - Enforces joint limits
    - Optionally applies rate limiting and smoothing
    """

    def __init__(
        self,
        action_scale: float,
        joint_limits: dict,
        rate_limit: Optional[dict] = None,
        lowpass: Optional[dict] = None,
    ):
        """
        Initialize controller.

        Args:
            action_scale: Radians per env step per unit action
            joint_limits: Dict with 'base', 'shoulder' and 'elbow' limit tuples
            rate_limit: Optional rate limiting config
            lowpass: Optional low-pass filter config
        """
        self.action_scale = action_scale

        # Joint limits
        self.q_min = np.array([
            joint_limits["base"][0],
            joint_limits["shoulder"][0],
            joint_limits["elbow"][0],
        ], dtype=np.float64)

        self.q_max = np.array([
            joint_limits["base"][1],
            joint_limits["shoulder"][1],
            joint_limits["elbow"][1],
        ], dtype=np.float64)

        # Optional filters
        if rate_limit is None:
            rate_limit = {"enabled": False}
        self.rate_limit = RateLimitConfig(
            enabled=rate_limit.get("enabled", False),
            max_delta=rate_limit.get("max_delta", 0.15),
        )

        if lowpass is None:
            lowpass = {"enabled": False}
        self.lowpass = LowpassConfig(
            enabled=lowpass.get("enabled", False),
            alpha=lowpass.get("alpha", 0.3),
        )

        # State
        self.q_target: Optional[np.ndarray] = None

    def reset(self, q_init: np.ndarray):
        """
        Reset controller state.

        Args:
            q_init: Initial joint angles (3,)
        """
        self.q_target = np.array(q_init, dtype=np.float64)

        # Ensure targets start within limits
        self.q_target = np.clip(self.q_target, self.q_min, self.q_max)

    def apply_action(
        self,
        action: np.ndarray,
        data: mujoco.MjData,
    ) -> np.ndarray:
        """
        Apply action to update targets and write to MuJoCo.

        Args:
            action: Normalized action in [-1, 1], shape (3,)
            data: MuJoCo data object

        Returns:
            Applied control values (joint targets)
        """
        if self.q_target is None:
            raise RuntimeError("Controller not reset. Call reset() first.")

        action = np.asarray(action, dtype=np.float64)

        # 1. Clip action to [-1, 1] for safety
        action = np.clip(action, -1.0, 1.0)

        # 2. Compute delta from action
        delta = self.action_scale * action

        # 3. Optional rate limiting
        if self.rate_limit.enabled:
            delta = np.clip(
                delta,
                -self.rate_limit.max_delta,
                self.rate_limit.max_delta,
            )

        # 4. Compute new target
        new_target = self.q_target + delta

        # 5. Optional low-pass filter
        if self.lowpass.enabled:
            alpha = self.lowpass.alpha
            new_target = (1 - alpha) * self.q_target + alpha * new_target

        # 6. Clip to joint limits
        self.q_target = np.clip(new_target, self.q_min, self.q_max)

        # 7. Write to MuJoCo controls (position actuators)
        data.ctrl[:3] = self.q_target

        return self.q_target.copy()

    def get_targets(self) -> Optional[np.ndarray]:
        """Return current targets for logging."""
        return self.q_target.copy() if self.q_target is not None else None

    @property
    def joint_limits(self) -> dict:
        """Return joint limits dict."""
        return {
            "base": (self.q_min[0], self.q_max[0]),
            "shoulder": (self.q_min[1], self.q_max[1]),
            "elbow": (self.q_min[2], self.q_max[2]),
        }

    def get_debug_info(self) -> dict:
        """
        Get controller state for debug/logging info dict.

        Returns:
            Dictionary with controller state (q_target, joint limits, config)
        """
        info = {
            "ctrl_q_target": self.q_target.copy() if self.q_target is not None else None,
            "ctrl_action_scale": self.action_scale,
        }

        # Add config info
        if self.rate_limit.enabled:
            info["ctrl_rate_limit_enabled"] = True
            info["ctrl_rate_limit_max_delta"] = self.rate_limit.max_delta
        else:
            info["ctrl_rate_limit_enabled"] = False

        if self.lowpass.enabled:
            info["ctrl_lowpass_enabled"] = True
            info["ctrl_lowpass_alpha"] = self.lowpass.alpha
        else:
            info["ctrl_lowpass_enabled"] = False

        return info


def create_controller(config: dict, joint_limits: dict) -> PositionTargetController:
    """
    Create PositionTargetController from config.

    Args:
        config: Control config dict
        joint_limits: Joint limits from model validation

    Returns:
        PositionTargetController instance
    """
    return PositionTargetController(
        action_scale=config.get("action_scale", 0.1),
        joint_limits=joint_limits,
        rate_limit=config.get("rate_limit"),
        lowpass=config.get("lowpass"),
    )
