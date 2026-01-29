"""Reward computation for the 2-DOF arm environment."""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class RewardConfig:
    """Configuration for reward computation."""

    # Normalization
    max_reach: float = 0.5  # link1 + link2 length
    normalize_distance: bool = True

    # Smoothness penalties
    w_action_mag: float = 0.01  # Penalty on ||action||^2
    w_action_change: float = 0.005  # Penalty on ||action - prev_action||^2
    w_joint_vel: float = 0.0  # Optional penalty on joint velocities

    # Phase 1: Reach rewards
    w_dist: float = 1.0  # Weight on distance (negative)
    reach_success_bonus: float = 10.0  # Bonus for completing reach

    # Phase 2: Grasp rewards
    attach_bonus: float = 2.0  # One-time bonus for attachment
    w_lift: float = 1.0  # Weight on lift height
    w_hold_per_step: float = 0.1  # Bonus per step at lift height
    grasp_success_bonus: float = 15.0  # Bonus for completing grasp task


class RewardComputer:
    """Computes rewards for the arm environment."""

    def __init__(self, config: RewardConfig):
        """
        Initialize reward computer.

        Args:
            config: RewardConfig with weights and thresholds
        """
        self.config = config

    def compute_reach_reward(
        self,
        dist: float,
        action: np.ndarray,
        prev_action: Optional[np.ndarray],
        joint_vel: np.ndarray,
        is_success: bool,
    ) -> tuple[float, dict]:
        """
        Compute reward for reach task (Phase 1).

        Args:
            dist: Distance from ee to ball
            action: Current action (2,)
            prev_action: Previous action or None
            joint_vel: Joint velocities (2,)
            is_success: Whether reach task succeeded

        Returns:
            (total_reward, reward_terms_dict)
        """
        terms = {}

        # Distance shaping (negative, normalized)
        if self.config.normalize_distance:
            r_dist = -dist / self.config.max_reach
        else:
            r_dist = -dist
        terms["dist"] = self.config.w_dist * r_dist

        # Action magnitude penalty
        r_action_mag = -np.sum(action**2)
        terms["action_mag"] = self.config.w_action_mag * r_action_mag

        # Action change penalty
        if prev_action is not None:
            r_action_change = -np.sum((action - prev_action) ** 2)
            terms["action_change"] = self.config.w_action_change * r_action_change
        else:
            terms["action_change"] = 0.0

        # Joint velocity penalty (optional)
        if self.config.w_joint_vel > 0:
            r_joint_vel = -np.sum(joint_vel**2)
            terms["joint_vel"] = self.config.w_joint_vel * r_joint_vel
        else:
            terms["joint_vel"] = 0.0

        # Success bonus
        if is_success:
            terms["success"] = self.config.reach_success_bonus
        else:
            terms["success"] = 0.0

        # Total reward
        total = sum(terms.values())

        return total, terms

    def compute_grasp_reward(
        self,
        dist: float,
        lift_height: float,
        action: np.ndarray,
        prev_action: Optional[np.ndarray],
        joint_vel: np.ndarray,
        attached: bool,
        just_attached: bool,
        at_lift_height: bool,
        is_success: bool,
        table_height: float,
    ) -> tuple[float, dict]:
        """
        Compute reward for grasp task (Phase 2).

        Uses gating to prevent reward hacking:
        - Before attach: distance reward
        - After attach: lift/hold rewards (no distance)

        Args:
            dist: Distance from ee to ball
            lift_height: Ball z - table z
            action: Current action (2,)
            prev_action: Previous action or None
            joint_vel: Joint velocities (2,)
            attached: Whether ball is currently attached
            just_attached: Whether attachment just happened this step
            at_lift_height: Whether ball is at target lift height
            is_success: Whether grasp task succeeded (held at height)
            table_height: Table z position

        Returns:
            (total_reward, reward_terms_dict)
        """
        terms = {}

        # Distance shaping (only when not attached)
        if not attached:
            if self.config.normalize_distance:
                r_dist = -dist / self.config.max_reach
            else:
                r_dist = -dist
            terms["dist"] = self.config.w_dist * r_dist
        else:
            terms["dist"] = 0.0

        # Attachment bonus (one-time)
        if just_attached:
            terms["attach"] = self.config.attach_bonus
        else:
            terms["attach"] = 0.0

        # Lift shaping (only when attached)
        if attached and lift_height > 0:
            # Normalize lift by max reach
            r_lift = lift_height / self.config.max_reach
            terms["lift"] = self.config.w_lift * r_lift
        else:
            terms["lift"] = 0.0

        # Hold bonus (per step at lift height)
        if attached and at_lift_height:
            terms["hold"] = self.config.w_hold_per_step
        else:
            terms["hold"] = 0.0

        # Action penalties (same as reach)
        r_action_mag = -np.sum(action**2)
        terms["action_mag"] = self.config.w_action_mag * r_action_mag

        if prev_action is not None:
            r_action_change = -np.sum((action - prev_action) ** 2)
            terms["action_change"] = self.config.w_action_change * r_action_change
        else:
            terms["action_change"] = 0.0

        if self.config.w_joint_vel > 0:
            r_joint_vel = -np.sum(joint_vel**2)
            terms["joint_vel"] = self.config.w_joint_vel * r_joint_vel
        else:
            terms["joint_vel"] = 0.0

        # Success bonus
        if is_success:
            terms["success"] = self.config.grasp_success_bonus
        else:
            terms["success"] = 0.0

        # Total reward
        total = sum(terms.values())

        return total, terms


def create_reward_computer(config: dict) -> RewardComputer:
    """
    Create RewardComputer from config dict.

    Args:
        config: Dictionary with reward parameters

    Returns:
        RewardComputer instance
    """
    reward_config = RewardConfig(
        max_reach=config.get("max_reach", 0.5),
        normalize_distance=config.get("normalize_distance", True),
        w_action_mag=config.get("w_action_mag", 0.01),
        w_action_change=config.get("w_action_change", 0.005),
        w_joint_vel=config.get("w_joint_vel", 0.0),
        w_dist=config.get("w_dist", 1.0),
        reach_success_bonus=config.get("reach", {}).get("success_bonus", 10.0),
        attach_bonus=config.get("grasp", {}).get("attach_bonus", 2.0),
        w_lift=config.get("grasp", {}).get("w_lift", 1.0),
        w_hold_per_step=config.get("grasp", {}).get("w_hold_per_step", 0.1),
        grasp_success_bonus=config.get("grasp", {}).get("success_bonus", 15.0),
    )

    return RewardComputer(reward_config)
