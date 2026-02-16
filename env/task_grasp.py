"""Phase 2: Pick-and-place task for the 4-DOF arm environment."""

import numpy as np
import mujoco
from dataclasses import dataclass
from typing import Optional

from env.rewards import RewardComputer


@dataclass
class GraspTaskConfig:
    """Configuration for grasp-and-place task."""

    # Attachment conditions
    attach_radius: float = 0.04  # Distance to trigger attachment
    attach_vel_threshold: float = 0.15  # Max ee velocity to attach

    # Placement conditions
    place_radius: float = 0.05  # Distance to destination for placement
    hold_steps: int = 10  # Steps at destination to succeed

    # Table height (for reference)
    table_height: float = 0.4


class GraspTask:
    """
    Phase 2 task: Pick ball from source, place at destination.

    Uses MuJoCo weld constraint for attachment.

    Phases:
    1. Approach: Move ee to ball position
    2. Grasp: Attach ball via weld constraint
    3. Transport: Move attached ball toward destination
    4. Place: Hold ball at destination for hold_steps

    Success requires:
    - Ball attached (via weld constraint)
    - Ball brought within place_radius of destination
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
        self.just_placed = False
        self.hold_count = 0
        self.ever_attached = False
        self.ever_at_destination = False

        # Destination position (set during reset)
        self.destination_pos = np.zeros(3)

        # Reference to data (set during reset)
        self._data: Optional[mujoco.MjData] = None

    def reset(self, data: mujoco.MjData, destination_pos: np.ndarray = None):
        """
        Reset task state for new episode.

        Args:
            data: MuJoCo data object
            destination_pos: 3D position of place target
        """
        self._data = data
        self.attached = False
        self.just_attached = False
        self.just_placed = False
        self.hold_count = 0
        self.ever_attached = False
        self.ever_at_destination = False

        if destination_pos is not None:
            self.destination_pos = destination_pos.copy()
        else:
            self.destination_pos = np.zeros(3)

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

    def check_placement(
        self,
        ball_pos: np.ndarray,
    ) -> tuple[bool, bool]:
        """
        Check if ball is at destination.

        Args:
            ball_pos: Ball position (3,)

        Returns:
            (at_destination, is_success)
            - at_destination: Ball is within place_radius of destination
            - is_success: Hold completed (task success)
        """
        if not self.attached:
            self.hold_count = 0
            return False, False

        dist_to_dest = np.linalg.norm(ball_pos - self.destination_pos)
        at_destination = dist_to_dest < self.config.place_radius

        if at_destination:
            if not self.ever_at_destination:
                self.just_placed = True
            self.hold_count += 1
            self.ever_at_destination = True
        else:
            self.hold_count = 0

        is_success = self.hold_count >= self.config.hold_steps

        return at_destination, is_success

    def step(
        self,
        dist: float,
        ee_vel_magnitude: float,
    ):
        """
        Update task state for one step.

        Args:
            dist: Distance from ee to ball
            ee_vel_magnitude: End-effector speed
        """
        # Clear per-step flags from previous step
        self.just_attached = False
        self.just_placed = False

        # Check for attachment (if not already attached)
        if not self.attached and self.check_attach(dist, ee_vel_magnitude):
            self.activate_weld()

    def compute_reward(
        self,
        dist: float,
        ee_vel_magnitude: float,
        ball_pos: np.ndarray,
        action: np.ndarray,
        prev_action: Optional[np.ndarray],
        joint_vel: np.ndarray,
    ) -> tuple[float, dict, bool]:
        """
        Compute reward and check for task completion.

        Args:
            dist: Distance from ee to ball
            ee_vel_magnitude: End-effector speed
            ball_pos: Ball position (3,)
            action: Current action (4,)
            prev_action: Previous action or None
            joint_vel: Joint velocities (4,)

        Returns:
            (reward, reward_terms, is_success)
        """
        # Update state
        self.step(dist, ee_vel_magnitude)

        # Check placement
        at_destination, is_success = self.check_placement(ball_pos)

        # Distance from ball to destination
        dist_to_dest = np.linalg.norm(ball_pos - self.destination_pos)

        # Compute reward
        reward, terms = self.reward_computer.compute_grasp_reward(
            dist=dist,
            dist_to_dest=dist_to_dest,
            action=action,
            prev_action=prev_action,
            joint_vel=joint_vel,
            attached=self.attached,
            just_attached=self.just_attached,
            just_placed=self.just_placed,
            at_destination=at_destination,
            is_success=is_success,
        )

        # Add task-specific info to terms
        terms["attached"] = float(self.attached)
        terms["just_attached"] = float(self.just_attached)
        terms["at_destination"] = float(at_destination)
        terms["dist_to_dest"] = dist_to_dest
        terms["hold_count"] = self.hold_count

        return reward, terms, is_success

    def get_state(self) -> dict:
        """Get current task state for info dict."""
        return {
            "attached": self.attached,
            "just_attached": self.just_attached,
            "hold_count": self.hold_count,
            "ever_attached": self.ever_attached,
            "place_success": self.ever_at_destination,
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
        place_radius=task_config.get("place_radius", 0.05),
        hold_steps=task_config.get("hold_steps", 10),
        table_height=task_config.get("table_height", 0.4),
    )

    return GraspTask(config, reward_computer, weld_id)
