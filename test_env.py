"""Test script for the 2-DOF arm environment."""

import numpy as np
from pathlib import Path


def test_environment():
    """Test basic environment functionality."""
    print("=" * 60)
    print("Testing 2-DOF Arm Environment")
    print("=" * 60)

    # Import environment
    from env.mujoco_env import MujocoArmEnv
    from control.controllers import PositionTargetController

    # Test 1: Create reach environment
    print("\n[1] Creating reach environment...")
    env = MujocoArmEnv(task_mode="reach")
    print(f"    Observation space: {env.observation_space}")
    print(f"    Action space: {env.action_space}")
    print(f"    Max reach: {env.max_reach:.3f}m")
    print(f"    Table height: {env.table_height:.3f}m")

    # Create and attach controller
    print("\n[2] Creating controller...")
    controller = PositionTargetController(
        action_scale=0.1,
        joint_limits=env.ids["joint_limits"],
    )
    env.set_controller(controller)
    print("    Controller attached")

    # Test 2: Reset environment
    print("\n[3] Testing reset...")
    obs, info = env.reset(seed=42)
    print(f"    Observation shape: {obs.shape}")
    print(f"    Observation: {obs[:4]}... (first 4 values)")
    print(f"    Info keys: {list(info.keys())}")
    print(f"    Initial distance: {info['dist']:.4f}m")

    # Test 3: Take random steps
    print("\n[4] Testing random steps...")
    total_reward = 0
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if i == 0:
            print(f"    Step 1: reward={reward:.4f}, dist={info['dist']:.4f}")

    print(f"    Total reward (10 steps): {total_reward:.4f}")
    print(f"    Final distance: {info['dist']:.4f}m")

    # Test 4: Run full episode
    print("\n[5] Running full episode (max 200 steps)...")
    obs, info = env.reset(seed=123)
    episode_return = 0
    steps = 0

    while True:
        # Simple policy: move toward ball (use observation)
        # obs[4:7] = ee_pos, obs[10:13] = ball_pos
        ee_pos = obs[4:7]
        ball_pos = obs[10:13]
        direction = ball_pos - ee_pos

        # Convert to simple action (this won't work well, just testing)
        action = np.clip(direction[:2] * 5, -1, 1)  # Simplified

        obs, reward, terminated, truncated, info = env.step(action)
        episode_return += reward
        steps += 1

        if terminated or truncated:
            break

    print(f"    Episode finished in {steps} steps")
    print(f"    Episode return: {episode_return:.4f}")
    print(f"    Success: {info.get('is_success', False)}")
    print(f"    Final distance: {info['dist']:.4f}m")

    env.close()
    print("\n[6] Reach environment test completed!")

    # Test 5: Create grasp environment
    print("\n" + "=" * 60)
    print("[7] Testing grasp environment...")
    env = MujocoArmEnv(task_mode="grasp")
    controller = PositionTargetController(
        action_scale=0.1,
        joint_limits=env.ids["joint_limits"],
    )
    env.set_controller(controller)

    obs, info = env.reset(seed=42)
    print(f"    Grasp env created")
    print(f"    Initial attached: {info.get('attached', False)}")

    # Take some steps
    for _ in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    print(f"    After 50 steps - attached: {info.get('attached', False)}")

    env.close()
    print("\n[8] Grasp environment test completed!")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


def test_config():
    """Test configuration loading."""
    print("\n" + "=" * 60)
    print("Testing Configuration Loading")
    print("=" * 60)

    from config import load_config, validate_config

    # Test loading default config
    print("\n[1] Loading default.yaml...")
    config = load_config(Path(__file__).parent / "config" / "default.yaml")
    print(f"    task_mode: {config['env']['task_mode']}")
    print(f"    num_envs: {config['ppo']['num_envs']}")

    # Test loading reach config
    print("\n[2] Loading reach.yaml...")
    config = load_config(Path(__file__).parent / "config" / "reach.yaml")
    print(f"    task_mode: {config['env']['task_mode']}")
    print(f"    curriculum enabled: {config['curriculum']['enabled']}")
    print(f"    curriculum stages: {len(config['curriculum']['stages'])}")

    # Test validation
    print("\n[3] Validating config...")
    validate_config(config)

    print("\nConfiguration tests passed!")


def test_vectorized_env():
    """Test vectorized environment creation."""
    print("\n" + "=" * 60)
    print("Testing Vectorized Environment")
    print("=" * 60)

    try:
        from gymnasium.vector import SyncVectorEnv
        from env.mujoco_env import MujocoArmEnv
        from control.controllers import PositionTargetController

        def make_env(seed, rank):
            def _init():
                env = MujocoArmEnv(task_mode="reach")
                controller = PositionTargetController(
                    action_scale=0.1,
                    joint_limits=env.ids["joint_limits"],
                )
                env.set_controller(controller)
                env.reset(seed=seed + rank)
                return env
            return _init

        print("\n[1] Creating 4 parallel environments...")
        num_envs = 4
        vec_env = SyncVectorEnv([make_env(seed=42, rank=i) for i in range(num_envs)])

        print(f"    Num envs: {vec_env.num_envs}")
        print(f"    Single obs space: {vec_env.single_observation_space}")
        print(f"    Single action space: {vec_env.single_action_space}")

        # Test reset
        print("\n[2] Testing vectorized reset...")
        obs, infos = vec_env.reset()
        print(f"    Obs shape: {obs.shape}")

        # Test step
        print("\n[3] Testing vectorized step...")
        actions = np.random.uniform(-1, 1, (num_envs, 2)).astype(np.float32)
        obs, rewards, terminateds, truncateds, infos = vec_env.step(actions)
        print(f"    Rewards: {rewards}")
        print(f"    Terminateds: {terminateds}")

        vec_env.close()
        print("\nVectorized environment test passed!")

    except ImportError as e:
        print(f"Skipping vectorized test (missing dependency): {e}")


if __name__ == "__main__":
    test_config()
    test_environment()
    test_vectorized_env()
