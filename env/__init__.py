"""Environment module for the 2-DOF arm RL task."""

from env.mujoco_env import MujocoArmEnv
from env.validation import validate_model, get_arm_geometry
from env.observations import ObservationBuilder
from env.rewards import RewardConfig, RewardComputer, create_reward_computer
from env.task_reach import ReachTask, create_reach_task
from env.task_grasp import GraspTask, create_grasp_task

__all__ = [
    "MujocoArmEnv",
    "validate_model",
    "get_arm_geometry",
    "ObservationBuilder",
    "RewardConfig",
    "RewardComputer",
    "create_reward_computer",
    "ReachTask",
    "create_reach_task",
    "GraspTask",
    "create_grasp_task",
]
