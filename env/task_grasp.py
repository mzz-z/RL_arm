"""Phase 2: Grasp task for the 2-DOF arm environment."""

import numpy as np
import mujoco
from dataclasses import dataclass
from typing import Optional

from env.rewards import RewardComputer


@dataclass
class GraspTaskConfig:
    """Configuration for grasp task."""

    # Attachment conditions
    attach_radius: float = 0.04  # Distance to trigger attachment
    attach_vel_threshold: float = 0.15  # Max ee velocity to attach

    # Lift/hold conditions
    lift_height: float = 0.1  # Height above table for success
    hold_steps: int = 10  # Steps at height to succeed

    # Table height (for lift calculation)
    table_height: float = 0.4


class GraspTask:
    """
    Phase 2 task: Grasp ball and lift to target height.

    Uses MuJoCo weld constraint for attachment.

    Success requires:
    - Ball attached (via weld constraint)
    - Ball lifted to lift_height above table
    - Maintained for hold_steps consecutive steps
    """

    def __init__(
        self,
        config: GraspTaskConfig,
        reward_computer: RewardComputer,
        weld_id: int,
    ):
        """
        Initialize grasp task.

        Args:
            config: GraspTaskConfig with thresholds
            reward_computer: RewardComputer for reward calculation
            weld_id: ID of the grasp_weld equality constraint
        """
        self.config = config
        self.reward_computer = reward_computer
        self.weld_id = weld_id

        # Task state
        self.attached = False
        self.just_attached = False
        self.hold_count = 0
        self.ever_attached = False
        self.ever_at_height = False

        # Reference to data (set during reset)
        self._data: Optional[mujoco.MjData] = None

    def reset(self, data: mujoco.MjData):
        """
        Reset task state for new episode.

        Args:
            data: MuJoCo data object
        """
        self._data = data
        self.attached = False
        self.just_attached = False
        self.hold_count = 0
        self.ever_attached = False
        self.ever_at_height = False

        # Deactivate weld constraint
        data.eq_active[self.weld_id] = 0

    def check_attach(
        self,
        dist: float,
        ee_vel_magnitude: float,
    ) -> bool:
        """
        Check if attachment conditions are met.

        Args:
            dist: Distance from ee to ball
            ee_vel_magnitude: End-effector speed

        Returns:
            Whether attachment should occur
        """
        if self.attached:
            return False  # Already attached

        return (
            dist < self.config.attach_radius
            and ee_vel_magnitude < self.config.attach_vel_threshold
        )

    def activate_weld(self):
        """Activate the weld constraint to attach ball to ee."""
        if self._data is not None:
            self._data.eq_active[self.weld_id] = 1
        self.attached = True
        self.just_attached = True
        self.ever_attached = True

    def deactivate_weld(self):
        """Deactivate the weld constraint (release ball)."""
        if self._data is not None:
            self._data.eq_active[self.weld_id] = 0
        self.attached = False

    def check_success(
        self,
        ball_z: float,
    ) -> tuple[bool, bool]:
        """
        Check if grasp task succeeded.

        Args:
            ball_z: Ball z position

        Returns:
            (is_at_height, is_success)
            - is_at_height: Ball is at target lift height
            - is_success: Hold completed (task success)
        """
        if not self.attached:
            self.hold_count = 0
            return False, False

        lift = ball_z - self.config.table_height
        is_at_height = lift >= self.config.lift_height

        if is_at_height:
            self.hold_count += 1
            self.ever_at_height = True
        else:
            self.hold_count = 0

        is_success = self.hold_count >= self.config.hold_steps

        return is_at_height, is_success

    def step(
        self,
        dist: float,
        ee_vel_magnitude: float,
        ball_z: float,
    ):
        """
        Update task state for one step.

        Args:
            dist: Distance from ee to ball
            ee_vel_magnitude: End-effector speed
            ball_z: Ball z position
        """
        # Clear just_attached flag from previous step
        self.just_attached = False

        # Check for attachment (if not already attached)
        if not self.attached and self.check_attach(dist, ee_vel_magnitude):
            self.activate_weld()

    def compute_reward(
        self,
        dist: float,
        ee_vel_magnitude: float,
        ball_z: float,
        action: np.ndarray,
        prev_action: Optional[np.ndarray],
        joint_vel: np.ndarray,
    ) -> tuple[float, dict, bool]:
        """
        Compute reward and check for task completion.

        Args:
            dist: Distance from ee to ball
            ee_vel_magnitude: End-effector speed
            ball_z: Ball z position
            action: Current action (2,)
            prev_action: Previous action or None
            joint_vel: Joint velocities (2,)

        Returns:
            (reward, reward_terms, is_success)
        """
        # Update state
        self.step(dist, ee_vel_magnitude, ball_z)

        # Check success
        is_at_height, is_success = self.check_success(ball_z)

        # Compute lift height
        lift_height = ball_z - self.config.table_height if self.attached else 0.0

        # Compute reward
        reward, terms = self.reward_computer.compute_grasp_reward(
            dist=dist,
            lift_height=lift_height,
            action=action,
            prev_action=prev_action,
            joint_vel=joint_vel,
            attached=self.attached,
            just_attached=self.just_attached,
            at_lift_height=is_at_height,
            is_success=is_success,
            table_height=self.config.table_height,
        )

        # Add task-specific info to terms
        terms["attached"] = float(self.attached)
        terms["just_attached"] = float(self.just_attached)
        terms["is_at_height"] = float(is_at_height)
        terms["lift_height"] = lift_height
        terms["hold_count"] = self.hold_count

        return reward, terms, is_success

    def get_state(self) -> dict:
        """Get current task state for info dict."""
        return {
            "attached": self.attached,
            "just_attached": self.just_attached,
            "hold_count": self.hold_count,
            "ever_attached": self.ever_attached,
            "lift_success": self.ever_at_height,
        }


def create_grasp_task(
    task_config: dict,
    reward_computer: RewardComputer,
    weld_id: int,
) -> GraspTask:
    """
    Create GraspTask from config dict.

    Args:
        task_config: Dictionary with task parameters
        reward_computer: RewardComputer instance
        weld_id: ID of grasp_weld constraint

    Returns:
        GraspTask instance
    """
    config = GraspTaskConfig(
        attach_radius=task_config.get("attach_radius", 0.04),
        attach_vel_threshold=task_config.get("attach_vel_threshold", 0.15),
        lift_height=task_config.get("lift_height", 0.1),
        hold_steps=task_config.get("hold_steps", 10),
        table_height=task_config.get("table_height", 0.4),
    )

    return GraspTask(config, reward_computer, weld_id)
