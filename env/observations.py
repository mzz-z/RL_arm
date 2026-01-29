"""Observation builder for the 2-DOF arm environment."""

import numpy as np
import mujoco
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from env.mujoco_env import MujocoArmEnv


class ObservationBuilder:
    """
    Builds the observation vector from MuJoCo state.

    Observation vector (14 dimensions):
        - joint angles (2): shoulder, elbow (normalized by joint limits)
        - joint velocities (2): shoulder_vel, elbow_vel (scaled)
        - end-effector position (3): x, y, z (world frame)
        - end-effector velocity (3): vx, vy, vz
        - ball position (3): x, y, z (world frame)
        - grasp flag (1): 0 or 1
    """

    OBS_DIM = 14

    def __init__(
        self,
        model: mujoco.MjModel,
        ids: dict,
        joint_limits: dict,
        vel_scale: float = 10.0,
    ):
        """
        Initialize observation builder.

        Args:
            model: MuJoCo model
            ids: Dictionary of model element IDs from validation
            joint_limits: Dict with shoulder/elbow limit tuples
            vel_scale: Scale factor for joint velocities
        """
        self.model = model
        self.ids = ids
        self.vel_scale = vel_scale

        # Joint limit normalization
        self.shoulder_min, self.shoulder_max = joint_limits["shoulder"]
        self.elbow_min, self.elbow_max = joint_limits["elbow"]

        # Precompute normalization factors
        self.shoulder_range = self.shoulder_max - self.shoulder_min
        self.elbow_range = self.elbow_max - self.elbow_min

    def get_observation(
        self,
        data: mujoco.MjData,
        attached: bool,
    ) -> np.ndarray:
        """
        Build observation vector from current state.

        Args:
            data: MuJoCo data object
            attached: Whether ball is attached (grasp flag)

        Returns:
            14-dim observation vector
        """
        obs = np.zeros(self.OBS_DIM, dtype=np.float32)

        # Joint angles (normalized to roughly [-1, 1])
        shoulder_pos = data.qpos[self.ids["shoulder_qpos_addr"]]
        elbow_pos = data.qpos[self.ids["elbow_qpos_addr"]]

        obs[0] = 2.0 * (shoulder_pos - self.shoulder_min) / self.shoulder_range - 1.0
        obs[1] = 2.0 * (elbow_pos - self.elbow_min) / self.elbow_range - 1.0

        # Joint velocities (scaled)
        shoulder_vel = data.qvel[self.ids["shoulder_qvel_addr"]]
        elbow_vel = data.qvel[self.ids["elbow_qvel_addr"]]

        obs[2] = shoulder_vel / self.vel_scale
        obs[3] = elbow_vel / self.vel_scale

        # End-effector position (world frame)
        ee_pos = data.site_xpos[self.ids["ee_site"]]
        obs[4:7] = ee_pos

        # End-effector velocity
        ee_vel = self._compute_ee_velocity(data)
        obs[7:10] = ee_vel

        # Ball position (world frame)
        ball_pos = data.xpos[self.ids["ball_body"]]
        obs[10:13] = ball_pos

        # Grasp flag
        obs[13] = float(attached)

        return obs

    def _compute_ee_velocity(self, data: mujoco.MjData) -> np.ndarray:
        """
        Compute end-effector linear velocity using site Jacobian.

        Returns:
            3D velocity vector (vx, vy, vz)
        """
        # Allocate Jacobians for position (3 x nv) and rotation (3 x nv)
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))

        # Compute Jacobian for the ee_site
        mujoco.mj_jacSite(self.model, data, jacp, jacr, self.ids["ee_site"])

        # Linear velocity = Jacobian_pos @ qvel
        ee_vel = jacp @ data.qvel

        return ee_vel

    def get_state_info(
        self,
        data: mujoco.MjData,
        attached: bool,
        dwell_count: int,
        hold_count: int,
        step_count: int,
        max_reach: float,
    ) -> dict:
        """
        Build debug info dictionary.

        Args:
            data: MuJoCo data object
            attached: Whether ball is attached
            dwell_count: Steps within reach radius
            hold_count: Steps at lift height
            step_count: Total steps this episode
            max_reach: Maximum arm reach for normalization

        Returns:
            Info dictionary with debug values
        """
        ee_pos = data.site_xpos[self.ids["ee_site"]].copy()
        ball_pos = data.xpos[self.ids["ball_body"]].copy()
        ee_vel = self._compute_ee_velocity(data)

        dist = np.linalg.norm(ee_pos - ball_pos)
        ee_vel_magnitude = np.linalg.norm(ee_vel)

        info = {
            # State
            "dist": dist,
            "dist_normalized": dist / max_reach,
            "ee_pos": ee_pos,
            "ee_vel": ee_vel.copy(),
            "ee_vel_magnitude": ee_vel_magnitude,
            "ball_pos": ball_pos,
            "ball_z": ball_pos[2],
            "attached": attached,
            # Joint state
            "shoulder_pos": data.qpos[self.ids["shoulder_qpos_addr"]],
            "elbow_pos": data.qpos[self.ids["elbow_qpos_addr"]],
            "shoulder_vel": data.qvel[self.ids["shoulder_qvel_addr"]],
            "elbow_vel": data.qvel[self.ids["elbow_qvel_addr"]],
            # Progress counters
            "dwell_count": dwell_count,
            "hold_count": hold_count,
            "step_count": step_count,
        }

        return info


def compute_ee_position(model: mujoco.MjModel, data: mujoco.MjData, ee_site_id: int) -> np.ndarray:
    """Utility to get end-effector position."""
    mujoco.mj_forward(model, data)
    return data.site_xpos[ee_site_id].copy()


def compute_ball_position(model: mujoco.MjModel, data: mujoco.MjData, ball_body_id: int) -> np.ndarray:
    """Utility to get ball position."""
    mujoco.mj_forward(model, data)
    return data.xpos[ball_body_id].copy()
