"""Observation builder for the 4-DOF arm environment."""

import numpy as np
import mujoco
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from env.mujoco_env import MujocoArmEnv


class ObservationBuilder:
    """
    Builds the observation vector from MuJoCo state.

    Observation vector (24 dimensions):
        - joint angles (4): base, shoulder, elbow, wrist (normalized by joint limits)
        - joint velocities (4): base_vel, shoulder_vel, elbow_vel, wrist_vel (scaled)
        - end-effector position (3): x, y, z (world frame)
        - end-effector velocity (3): vx, vy, vz
        - ball position (3): x, y, z (world frame)
        - relative position (3): ball - ee (direction to target)
        - grasp flag (1): 0 or 1
        - destination position (3): x, y, z (place target; zeros for reach mode)
    """

    OBS_DIM = 24
    GRASP_FLAG_IDX = 20

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
            joint_limits: Dict with base/shoulder/elbow/wrist limit tuples
            vel_scale: Scale factor for joint velocities
        """
        self.model = model
        self.ids = ids
        self.vel_scale = vel_scale

        # Joint limit normalization
        self.base_min, self.base_max = joint_limits["base"]
        self.shoulder_min, self.shoulder_max = joint_limits["shoulder"]
        self.elbow_min, self.elbow_max = joint_limits["elbow"]
        self.wrist_min, self.wrist_max = joint_limits["wrist"]

        # Precompute normalization factors
        self.base_range = self.base_max - self.base_min
        self.shoulder_range = self.shoulder_max - self.shoulder_min
        self.elbow_range = self.elbow_max - self.elbow_min
        self.wrist_range = self.wrist_max - self.wrist_min

    def get_observation(
        self,
        data: mujoco.MjData,
        attached: bool,
        destination_pos: np.ndarray = None,
    ) -> np.ndarray:
        """
        Build observation vector from current state.

        Args:
            data: MuJoCo data object
            attached: Whether ball is attached (grasp flag)
            destination_pos: Place target position (3,) or None for reach mode

        Returns:
            24-dim observation vector
        """
        obs = np.zeros(self.OBS_DIM, dtype=np.float32)

        # Joint angles (normalized to roughly [-1, 1])
        base_pos = data.qpos[self.ids["base_qpos_addr"]]
        shoulder_pos = data.qpos[self.ids["shoulder_qpos_addr"]]
        elbow_pos = data.qpos[self.ids["elbow_qpos_addr"]]
        wrist_pos = data.qpos[self.ids["wrist_qpos_addr"]]

        obs[0] = 2.0 * (base_pos - self.base_min) / self.base_range - 1.0
        obs[1] = 2.0 * (shoulder_pos - self.shoulder_min) / self.shoulder_range - 1.0
        obs[2] = 2.0 * (elbow_pos - self.elbow_min) / self.elbow_range - 1.0
        obs[3] = 2.0 * (wrist_pos - self.wrist_min) / self.wrist_range - 1.0

        # Joint velocities (scaled)
        base_vel = data.qvel[self.ids["base_qvel_addr"]]
        shoulder_vel = data.qvel[self.ids["shoulder_qvel_addr"]]
        elbow_vel = data.qvel[self.ids["elbow_qvel_addr"]]
        wrist_vel = data.qvel[self.ids["wrist_qvel_addr"]]

        obs[4] = base_vel / self.vel_scale
        obs[5] = shoulder_vel / self.vel_scale
        obs[6] = elbow_vel / self.vel_scale
        obs[7] = wrist_vel / self.vel_scale

        # End-effector position (world frame)
        ee_pos = data.site_xpos[self.ids["ee_site"]]
        obs[8:11] = ee_pos

        # End-effector velocity
        ee_vel = self._compute_ee_velocity(data)
        obs[11:14] = ee_vel

        # Ball position (world frame)
        ball_pos = data.xpos[self.ids["ball_body"]]
        obs[14:17] = ball_pos

        # Relative position: ball - ee (direction to target)
        obs[17:20] = ball_pos - ee_pos

        # Grasp flag
        obs[self.GRASP_FLAG_IDX] = float(attached)

        # Destination position (place target; zeros for reach mode)
        if destination_pos is not None:
            obs[21:24] = destination_pos

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
            "base_pos": data.qpos[self.ids["base_qpos_addr"]],
            "shoulder_pos": data.qpos[self.ids["shoulder_qpos_addr"]],
            "elbow_pos": data.qpos[self.ids["elbow_qpos_addr"]],
            "wrist_pos": data.qpos[self.ids["wrist_qpos_addr"]],
            "base_vel": data.qvel[self.ids["base_qvel_addr"]],
            "shoulder_vel": data.qvel[self.ids["shoulder_qvel_addr"]],
            "elbow_vel": data.qvel[self.ids["elbow_qvel_addr"]],
            "wrist_vel": data.qvel[self.ids["wrist_qvel_addr"]],
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
