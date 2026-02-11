"""Phase 1: Reach task for the 2-DOF arm environment."""

import numpy as np
from dataclasses import dataclass
from typing import Optional

from env.rewards import RewardComputer


@dataclass
class ReachTaskConfig:
    """Configuration for reach task."""

    reach_radius: float = 0.05  # Success distance threshold (meters)
    dwell_steps: int = 5  # Steps within radius to succeed
    ee_vel_threshold: float = 0.1  # Max ee velocity for success (m/s)
    w_delta_dist: float = 5.0  # Weight for delta distance reward (getting closer)


class ReachTask:
    """
    Phase 1 task: Move end-effector to ball position.

    Success requires:
    - End-effector within reach_radius of ball
    - End-effector velocity below threshold
    - Maintained for dwell_steps consecutive steps
    """

    def __init__(
        self,
        config: ReachTaskConfig,
        reward_computer: RewardComputer,
    ):
        """
        Initialize reach task.

        Args:
            config: ReachTaskConfig with thresholds
            reward_computer: RewardComputer for reward calculation
        """
        self.config = config
        self.reward_computer = reward_computer

        # Task state
        self.dwell_count = 0
        self.prev_dist = None  # For delta distance reward

    def reset(self):
        """Reset task state for new episode."""
        self.dwell_count = 0
        self.prev_dist = None

    def check_success(
        self,
        dist: float,
        ee_vel_magnitude: float,
    ) -> tuple[bool, bool]:
        """
        Check if reach task succeeded.

        Args:
            dist: Distance from ee to ball
            ee_vel_magnitude: End-effector speed

        Returns:
            (is_dwelling, is_success)
            - is_dwelling: Currently meeting dwell conditions
            - is_success: Dwell completed (task success)
        """
        # Check if currently meeting conditions
        is_dwelling = (
            dist < self.config.reach_radius
            and ee_vel_magnitude < self.config.ee_vel_threshold
        )

        if is_dwelling:
            self.dwell_count += 1
        else:
            self.dwell_count = 0

        # Success when dwell completed
        is_success = self.dwell_count >= self.config.dwell_steps

        return is_dwelling, is_success

    def compute_reward(
        self,
        dist: float,
        ee_vel_magnitude: float,
        action: np.ndarray,
        prev_action: Optional[np.ndarray],
        joint_vel: np.ndarray,
    ) -> tuple[float, dict, bool]:
        """
        Compute reward and check for task completion.

        Args:
            dist: Distance from ee to ball
            ee_vel_magnitude: End-effector speed
            action: Current action (2,)
            prev_action: Previous action or None
            joint_vel: Joint velocities (2,)

        Returns:
            (reward, reward_terms, is_success)
        """
        # Check success
        is_dwelling, is_success = self.check_success(dist, ee_vel_magnitude)

        # Compute base reward
        reward, terms = self.reward_computer.compute_reach_reward(
            dist=dist,
            action=action,
            prev_action=prev_action,
            joint_vel=joint_vel,
            is_success=is_success,
        )

        # Add delta distance reward (reward for getting closer)
        if self.prev_dist is not None:
            # Positive if we got closer, negative if we moved away
            delta = self.prev_dist - dist
            # Normalize by max_reach for scale consistency
            delta_normalized = delta / self.reward_computer.config.max_reach
            delta_reward = self.config.w_delta_dist * delta_normalized
            terms["delta_dist"] = delta_reward
            reward += delta_reward
        else:
            terms["delta_dist"] = 0.0

        # Update previous distance for next step
        self.prev_dist = dist

        # Add task-specific info to terms
        terms["is_dwelling"] = float(is_dwelling)
        terms["dwell_count"] = self.dwell_count

        return reward, terms, is_success

    def get_state(self) -> dict:
        """Get current task state for info dict."""
        return {
            "dwell_count": self.dwell_count,
        }


def create_reach_task(
    task_config: dict,
    reward_computer: RewardComputer,
) -> ReachTask:
    """
    Create ReachTask from config dict.

    Args:
        task_config: Dictionary with task parameters
        reward_computer: RewardComputer instance

    Returns:
        ReachTask instance
    """
    config = ReachTaskConfig(
        reach_radius=task_config.get("reach_radius", 0.05),
        dwell_steps=task_config.get("dwell_steps", 5),
        ee_vel_threshold=task_config.get("ee_vel_threshold", 0.1),
        w_delta_dist=task_config.get("w_delta_dist", 5.0),
    )

    return ReachTask(config, reward_computer)
