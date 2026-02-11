"""Main MuJoCo environment for the 3-DOF arm RL task."""

import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
from pathlib import Path
from typing import Optional, Any

from env.validation import validate_model, get_arm_geometry
from env.observations import ObservationBuilder
from env.rewards import RewardConfig, RewardComputer, create_reward_computer
from env.task_reach import ReachTask, create_reach_task
from env.task_grasp import GraspTask, create_grasp_task


class MujocoArmEnv(gym.Env):
    """
    Gymnasium environment for 3-DOF arm reach/grasp task.

    The arm has:
    - Base joint: rotates around z-axis (turret)
    - Shoulder joint: rotates around y-axis
    - Elbow joint: rotates around y-axis

    Supports two task modes:
    - "reach": Move end-effector to ball position
    - "grasp": Attach ball and lift to target height
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        model_path: Optional[str] = None,
        task_mode: str = "reach",
        max_episode_steps: int = 200,
        frame_skip: int = 4,
        render_mode: Optional[str] = None,
        # Spawn parameters
        spawn_radius_min: float = 0.15,
        spawn_radius_max: float = 0.40,
        spawn_angle_min: float = -1.0,
        spawn_angle_max: float = 1.0,
        # Azimuth angle for 3D ball spawning (around z-axis)
        spawn_azimuth_min: float = 0.0,
        spawn_azimuth_max: float = 0.0,
        # Initial joint state
        init_base_range: tuple = (-0.1, 0.1),
        init_shoulder_range: tuple = (-0.3, 0.3),
        init_elbow_range: tuple = (-0.3, 0.3),
        init_vel_noise_std: float = 0.01,
        # Task-specific parameters
        reach_radius: float = 0.05,
        dwell_steps: int = 5,
        ee_vel_threshold: float = 0.1,
        w_delta_dist: float = 5.0,  # Delta distance reward weight
        attach_radius: float = 0.04,
        attach_vel_threshold: float = 0.15,
        lift_height: float = 0.1,
        hold_steps: int = 10,
        # Reward parameters
        reward_config: Optional[dict] = None,
        # Termination parameters
        ball_fell_threshold: float = -0.05,
        unreachable_margin: float = 0.1,
    ):
        """
        Initialize the arm environment.

        Args:
            model_path: Path to MuJoCo XML model (default: assets/arm.xml)
            task_mode: "reach" or "grasp"
            max_episode_steps: Maximum steps per episode
            frame_skip: Number of sim steps per env step
            render_mode: "human" or "rgb_array" or None
            spawn_radius_min/max: Ball spawn distance from base
            spawn_angle_min/max: Ball spawn elevation angle (radians)
            spawn_azimuth_min/max: Ball spawn azimuth angle around z-axis (radians)
            init_base_range: Initial base angle range
            init_shoulder_range: Initial shoulder angle range
            init_elbow_range: Initial elbow angle range
            init_vel_noise_std: Initial velocity noise
            reach_radius: Success distance for reach task
            dwell_steps: Steps to dwell for reach success
            ee_vel_threshold: Max ee velocity for reach success
            attach_radius: Distance for grasp attachment
            attach_vel_threshold: Max ee velocity for attachment
            lift_height: Target lift height for grasp
            hold_steps: Steps to hold for grasp success
            reward_config: Optional reward configuration dict
            ball_fell_threshold: Terminate if ball z < table - this
            unreachable_margin: Terminate if dist > max_reach + this
        """
        super().__init__()

        # Store parameters
        self.task_mode = task_mode
        self.max_episode_steps = max_episode_steps
        self.frame_skip = frame_skip
        self.render_mode = render_mode

        # Spawn parameters
        self.spawn_radius_min = spawn_radius_min
        self.spawn_radius_max = spawn_radius_max
        self.spawn_angle_min = spawn_angle_min
        self.spawn_angle_max = spawn_angle_max
        self.spawn_azimuth_min = spawn_azimuth_min
        self.spawn_azimuth_max = spawn_azimuth_max

        # Initial state parameters
        self.init_base_range = init_base_range
        self.init_shoulder_range = init_shoulder_range
        self.init_elbow_range = init_elbow_range
        self.init_vel_noise_std = init_vel_noise_std

        # Termination parameters
        self.ball_fell_threshold = ball_fell_threshold
        self.unreachable_margin = unreachable_margin

        # Load MuJoCo model
        if model_path is None:
            model_path = Path(__file__).parent.parent / "assets" / "arm.xml"
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)

        # Validate model and get IDs
        self.ids = validate_model(self.model, self.data)
        self.geometry = get_arm_geometry(self.model)

        # Store useful constants
        self.max_reach = self.geometry["max_reach"]
        self.table_height = self.geometry["table_height"]
        self.ball_radius = self.geometry["ball_radius"]

        # Create observation builder
        self.obs_builder = ObservationBuilder(
            model=self.model,
            ids=self.ids,
            joint_limits=self.ids["joint_limits"],
        )

        # Create reward computer
        if reward_config is None:
            reward_config = {}
        reward_config["max_reach"] = self.max_reach
        self.reward_computer = create_reward_computer(reward_config)

        # Create task (reach or grasp)
        if task_mode == "reach":
            task_config = {
                "reach_radius": reach_radius,
                "dwell_steps": dwell_steps,
                "ee_vel_threshold": ee_vel_threshold,
                "w_delta_dist": w_delta_dist,
            }
            self.task = create_reach_task(task_config, self.reward_computer)
        elif task_mode == "grasp":
            task_config = {
                "attach_radius": attach_radius,
                "attach_vel_threshold": attach_vel_threshold,
                "lift_height": lift_height,
                "hold_steps": hold_steps,
                "table_height": self.table_height,
            }
            self.task = create_grasp_task(
                task_config,
                self.reward_computer,
                self.ids["grasp_weld"],
            )
        else:
            raise ValueError(f"Unknown task_mode: {task_mode}")

        # Define action and observation spaces
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(ObservationBuilder.OBS_DIM,),
            dtype=np.float32,
        )

        # Episode state
        self.step_count = 0
        self.prev_action = None
        self._episode_return = 0.0

        # Controller (set externally via set_controller)
        self.controller = None

        # Rendering
        self.viewer = None
        self.renderer = None

    def set_controller(self, controller):
        """
        Set the action controller.

        Args:
            controller: PositionTargetController instance
        """
        self.controller = controller

    def create_controller_from_config(self, control_config: dict):
        """
        Create and set a controller from config dict.

        This is a convenience method that creates a PositionTargetController
        using the environment's validated joint limits and the provided config.

        Args:
            control_config: Control configuration dict with keys:
                - action_scale: float (radians per env step)
                - rate_limit: dict with 'enabled' and 'max_delta'
                - lowpass: dict with 'enabled' and 'alpha'

        Returns:
            The created controller instance
        """
        from control.controllers import create_controller

        controller = create_controller(control_config, self.ids["joint_limits"])
        self.set_controller(controller)
        return controller

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        """
        Reset the environment.

        Args:
            seed: Random seed for reproducibility
            options: Optional reset options

        Returns:
            (observation, info)
        """
        super().reset(seed=seed)

        # Reset MuJoCo state
        mujoco.mj_resetData(self.model, self.data)

        # Initialize arm joints
        base_init = self.np_random.uniform(*self.init_base_range)
        shoulder_init = self.np_random.uniform(*self.init_shoulder_range)
        elbow_init = self.np_random.uniform(*self.init_elbow_range)

        self.data.qpos[self.ids["base_qpos_addr"]] = base_init
        self.data.qpos[self.ids["shoulder_qpos_addr"]] = shoulder_init
        self.data.qpos[self.ids["elbow_qpos_addr"]] = elbow_init

        # Add small velocity noise
        self.data.qvel[self.ids["base_qvel_addr"]] = self.np_random.normal(
            0, self.init_vel_noise_std
        )
        self.data.qvel[self.ids["shoulder_qvel_addr"]] = self.np_random.normal(
            0, self.init_vel_noise_std
        )
        self.data.qvel[self.ids["elbow_qvel_addr"]] = self.np_random.normal(
            0, self.init_vel_noise_std
        )

        # Spawn ball in reachable region
        ball_pos = self._sample_ball_position()
        self._set_ball_position(ball_pos)

        # Reset task state (grasp task needs data for weld constraint)
        if self.task_mode == "grasp":
            self.task.reset(self.data)
        else:
            self.task.reset()

        # Reset episode state
        self.step_count = 0
        self.prev_action = None
        self._episode_return = 0.0

        # Reset controller if available
        if self.controller is not None:
            self.controller.reset(np.array([base_init, shoulder_init, elbow_init]))

        # Forward to update state
        mujoco.mj_forward(self.model, self.data)

        # Get observation
        attached = self._is_attached()
        obs = self.obs_builder.get_observation(self.data, attached)

        # Get info
        info = self._get_info()
        info["is_success"] = False

        return obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Take one step in the environment.

        Args:
            action: Action array (3,) in [-1, 1]

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        action = np.asarray(action, dtype=np.float32)

        # Apply action via controller
        if self.controller is not None:
            self.controller.apply_action(action, self.data)
        else:
            # Fallback: direct control (not recommended)
            self.data.ctrl[:3] = action * 2.0  # Scale to approximate joint range

        # Step simulation
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        self.step_count += 1

        # Get state info
        ee_pos = self.data.site_xpos[self.ids["ee_site"]].copy()
        ball_pos = self.data.xpos[self.ids["ball_body"]].copy()
        ee_vel = self.obs_builder._compute_ee_velocity(self.data)

        dist = np.linalg.norm(ee_pos - ball_pos)
        ee_vel_magnitude = np.linalg.norm(ee_vel)
        ball_z = ball_pos[2]

        joint_vel = np.array([
            self.data.qvel[self.ids["base_qvel_addr"]],
            self.data.qvel[self.ids["shoulder_qvel_addr"]],
            self.data.qvel[self.ids["elbow_qvel_addr"]],
        ])

        # Compute reward and check success
        if self.task_mode == "reach":
            reward, reward_terms, is_success = self.task.compute_reward(
                dist=dist,
                ee_vel_magnitude=ee_vel_magnitude,
                action=action,
                prev_action=self.prev_action,
                joint_vel=joint_vel,
            )
        else:  # grasp
            reward, reward_terms, is_success = self.task.compute_reward(
                dist=dist,
                ee_vel_magnitude=ee_vel_magnitude,
                ball_z=ball_z,
                action=action,
                prev_action=self.prev_action,
                joint_vel=joint_vel,
            )

        self._episode_return += reward
        self.prev_action = action.copy()

        # Check termination conditions
        terminated = False
        truncated = False

        if is_success:
            terminated = True

        # Safety termination: ball fell off table
        if ball_z < self.table_height + self.ball_fell_threshold:
            terminated = True

        # Safety termination: ball unreachable
        if dist > self.max_reach + self.unreachable_margin:
            terminated = True

        # Safety termination: simulation instability
        if np.any(np.isnan(self.data.qpos)) or np.any(np.isnan(self.data.qvel)):
            terminated = True
            reward = -10.0  # Penalty for instability

        # Time limit truncation
        if self.step_count >= self.max_episode_steps:
            truncated = True

        # Get observation
        attached = self._is_attached()
        obs = self.obs_builder.get_observation(self.data, attached)

        # Get info
        info = self._get_info()
        info["is_success"] = is_success
        info["reward_terms"] = reward_terms
        info["episode_return"] = self._episode_return

        # Add episode info on terminal
        if terminated or truncated:
            info["episode"] = {
                "r": self._episode_return,
                "l": self.step_count,
                "is_success": is_success,
            }

        return obs, reward, terminated, truncated, info

    def _sample_ball_position(self) -> np.ndarray:
        """
        Sample ball position in reachable region using spherical coordinates.

        Uses azimuth angle for 3D spawning around the base.

        Returns:
            3D position (x, y, z)
        """
        # Sample radial distance
        r = self.np_random.uniform(self.spawn_radius_min, self.spawn_radius_max)

        # Sample elevation angle (0 = forward along horizontal, positive = upward)
        elevation = self.np_random.uniform(self.spawn_angle_min, self.spawn_angle_max)

        # Sample azimuth angle (around z-axis)
        azimuth = self.np_random.uniform(self.spawn_azimuth_min, self.spawn_azimuth_max)

        # Convert to Cartesian
        horiz_dist = r * np.cos(elevation)
        x = horiz_dist * np.cos(azimuth)
        y = horiz_dist * np.sin(azimuth)
        z = self.table_height + self.ball_radius + r * np.sin(elevation)

        # Ensure ball is above table
        z = max(z, self.table_height + self.ball_radius)

        return np.array([x, y, z])

    def _set_ball_position(self, pos: np.ndarray):
        """
        Set ball position in simulation.

        Args:
            pos: 3D position (x, y, z)
        """
        # Ball has freejoint, so qpos is [x, y, z, qw, qx, qy, qz]
        addr = self.ids["ball_qpos_addr"]
        self.data.qpos[addr:addr + 3] = pos
        self.data.qpos[addr + 3:addr + 7] = [1, 0, 0, 0]  # Identity quaternion

        # Zero ball velocity
        vel_addr = self.ids["ball_qvel_addr"]
        self.data.qvel[vel_addr:vel_addr + 6] = 0

    def _is_attached(self) -> bool:
        """Check if ball is attached (weld active)."""
        return bool(self.data.eq_active[self.ids["grasp_weld"]])

    def _get_info(self) -> dict:
        """Build info dictionary."""
        attached = self._is_attached()

        info = self.obs_builder.get_state_info(
            data=self.data,
            attached=attached,
            dwell_count=getattr(self.task, "dwell_count", 0),
            hold_count=getattr(self.task, "hold_count", 0),
            step_count=self.step_count,
            max_reach=self.max_reach,
        )

        # Add task-specific state
        info.update(self.task.get_state())

        # Add controller debug info
        if self.controller is not None and hasattr(self.controller, "get_debug_info"):
            info.update(self.controller.get_debug_info())

        return info

    def render(self):
        """Render the environment."""
        if self.render_mode is None:
            return None

        if self.render_mode == "human":
            if self.viewer is None:
                from mujoco import viewer
                self.viewer = viewer.launch_passive(self.model, self.data)
            self.viewer.sync()
            return None
        else:
            if self.renderer is None:
                self.renderer = mujoco.Renderer(self.model, height=480, width=640)
            self.renderer.update_scene(self.data)
            return self.renderer.render()

    def close(self):
        """Clean up resources."""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    def set_spawn_params(
        self,
        radius_range: Optional[tuple] = None,
        angle_range: Optional[tuple] = None,
        azimuth_range: Optional[tuple] = None,
    ):
        """
        Update spawn parameters (for curriculum learning).

        Args:
            radius_range: (min, max) spawn radius
            angle_range: (min, max) spawn elevation angle
            azimuth_range: (min, max) azimuth angle around z-axis
        """
        if radius_range is not None:
            self.spawn_radius_min, self.spawn_radius_max = radius_range
        if angle_range is not None:
            self.spawn_angle_min, self.spawn_angle_max = angle_range
        if azimuth_range is not None:
            self.spawn_azimuth_min, self.spawn_azimuth_max = azimuth_range
