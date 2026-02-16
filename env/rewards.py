"""Reward computation for the 4-DOF arm environment."""

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
    w_proximity: float = 0.0  # Exponential proximity bonus (stronger gradient near target)
    proximity_alpha: float = 5.0  # Steepness of proximity curve
    reach_success_bonus: float = 10.0  # Bonus for completing reach

    # Phase 2: Grasp rewards (approach + attach)
    attach_bonus: float = 2.0  # One-time bonus for attachment
    w_lift: float = 1.0  # Weight on lift height

    # Phase 2: Place rewards (transport + place)
    w_place_dist: float = 1.0  # Weight on distance to destination (after attach)
    w_place_proximity: float = 0.5  # Proximity bonus for destination
    place_proximity_alpha: float = 5.0  # Steepness of place proximity curve
    place_bonus: float = 5.0  # One-time bonus for placing
    w_hold_per_step: float = 0.1  # Bonus per step at destination
    grasp_success_bonus: float = 15.0  # Bonus for completing pick-and-place


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

        # Exponential proximity bonus (creates bowl-shaped attractor near target)
        if self.config.w_proximity > 0:
            norm_dist = dist / self.config.max_reach
            r_proximity = np.exp(-self.config.proximity_alpha * norm_dist)
            terms["proximity"] = self.config.w_proximity * r_proximity
        else:
            terms["proximity"] = 0.0

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
        dist_to_dest: float,
        action: np.ndarray,
        prev_action: Optional[np.ndarray],
        joint_vel: np.ndarray,
        attached: bool,
        just_attached: bool,
        just_placed: bool,
        at_destination: bool,
        is_success: bool,
    ) -> tuple[float, dict]:
        """
        Compute reward for grasp-and-place task (Phase 2).

        Uses gating to shape behavior through task phases:
        - Before attach: distance to ball (approach shaping)
        - After attach: distance to destination (transport shaping)
        - At destination: hold bonus + place bonus

        Args:
            dist: Distance from ee to ball
            dist_to_dest: Distance from ball to destination
            action: Current action (4,)
            prev_action: Previous action or None
            joint_vel: Joint velocities (4,)
            attached: Whether ball is currently attached
            just_attached: Whether attachment just happened this step
            just_placed: Whether placement just happened this step
            at_destination: Whether ball is within place radius
            is_success: Whether task succeeded (held at dest)

        Returns:
            (total_reward, reward_terms_dict)
        """
        terms = {}

        # Phase 1: Approach - distance to ball (only when not attached)
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

        # Phase 2: Transport - distance to destination (only when attached)
        if attached:
            if self.config.normalize_distance:
                r_place_dist = -dist_to_dest / self.config.max_reach
            else:
                r_place_dist = -dist_to_dest
            terms["place_dist"] = self.config.w_place_dist * r_place_dist

            # Proximity bonus for destination
            if self.config.w_place_proximity > 0:
                norm_d = dist_to_dest / self.config.max_reach
                r_prox = np.exp(-self.config.place_proximity_alpha * norm_d)
                terms["place_proximity"] = self.config.w_place_proximity * r_prox
            else:
                terms["place_proximity"] = 0.0
        else:
            terms["place_dist"] = 0.0
            terms["place_proximity"] = 0.0

        # Placement bonus (one-time)
        if just_placed:
            terms["place"] = self.config.place_bonus
        else:
            terms["place"] = 0.0

        # Hold bonus (per step at destination while attached)
        if attached and at_destination:
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


def create_reward_computer(config: dict, task_mode: str = "reach") -> RewardComputer:
    """
    Create RewardComputer from config dict.

    Args:
        config: Dictionary with reward parameters
        task_mode: "reach" or "grasp" — selects which section's w_dist to use

    Returns:
        RewardComputer instance
    """
    reach_config = config.get("reach", {})
    grasp_config = config.get("grasp", {})

    # w_dist: use the section matching the active task mode
    if task_mode == "grasp":
        w_dist = grasp_config.get("w_dist", config.get("w_dist", 1.0))
    else:
        w_dist = reach_config.get("w_dist", config.get("w_dist", 1.0))

    reward_config = RewardConfig(
        max_reach=config.get("max_reach", 0.5),
        normalize_distance=config.get("normalize_distance", True),
        w_action_mag=config.get("w_action_mag", 0.01),
        w_action_change=config.get("w_action_change", 0.005),
        w_joint_vel=config.get("w_joint_vel", 0.0),
        w_dist=w_dist,
        w_proximity=reach_config.get("w_proximity", 0.0),
        proximity_alpha=reach_config.get("proximity_alpha", 5.0),
        reach_success_bonus=reach_config.get("success_bonus", 10.0),
        attach_bonus=grasp_config.get("attach_bonus", 2.0),
        w_lift=grasp_config.get("w_lift", 1.0),
        w_place_dist=grasp_config.get("w_place_dist", 1.0),
        w_place_proximity=grasp_config.get("w_place_proximity", 0.5),
        place_proximity_alpha=grasp_config.get("place_proximity_alpha", 5.0),
        place_bonus=grasp_config.get("place_bonus", 5.0),
        w_hold_per_step=grasp_config.get("w_hold_per_step", 0.1),
        grasp_success_bonus=grasp_config.get("success_bonus", 15.0),
    )

    return RewardComputer(reward_config)
