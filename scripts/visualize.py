#!/usr/bin/env python
"""
Visualize a trained policy in real-time.

Usage:
    python scripts/visualize.py --checkpoint runs/<run>/checkpoint_best.pt
    python scripts/visualize.py --checkpoint runs/<run>/checkpoint_best.pt --episodes 5
    python scripts/visualize.py --random  # Watch random policy
"""

import sys
import time
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from env.mujoco_env import MujocoArmEnv
from config import load_config


def create_env_with_viewer(config: dict) -> MujocoArmEnv:
    """Create environment with human rendering enabled."""
    env_config = config.get("env", {})
    control_config = config.get("control", {})
    reward_config = config.get("reward", {})

    env = MujocoArmEnv(
        task_mode=env_config.get("task_mode", "reach"),
        max_episode_steps=env_config.get("max_episode_steps", 200),
        frame_skip=env_config.get("frame_skip", 4),
        render_mode="human",
        # Spawn parameters
        spawn_radius_min=env_config.get("spawn", {}).get("radius_min", 0.15),
        spawn_radius_max=env_config.get("spawn", {}).get("radius_max", 0.40),
        spawn_angle_min=env_config.get("spawn", {}).get("angle_min", -1.0),
        spawn_angle_max=env_config.get("spawn", {}).get("angle_max", 1.0),
        spawn_y_min=env_config.get("spawn", {}).get("y_min", -0.15),
        spawn_y_max=env_config.get("spawn", {}).get("y_max", 0.15),
        # Task parameters
        reach_radius=env_config.get("reach", {}).get("reach_radius", 0.05),
        dwell_steps=env_config.get("reach", {}).get("dwell_steps", 5),
        ee_vel_threshold=env_config.get("reach", {}).get("ee_vel_threshold", 0.1),
        attach_radius=env_config.get("magnet", {}).get("attach_radius", 0.04),
        attach_vel_threshold=env_config.get("magnet", {}).get("attach_vel_threshold", 0.15),
        lift_height=env_config.get("lift", {}).get("lift_height", 0.1),
        hold_steps=env_config.get("lift", {}).get("hold_steps", 10),
        reward_config=reward_config,
    )
    env.create_controller_from_config(control_config)

    return env


def load_policy(checkpoint_path: str, config: dict, device: str = "cpu"):
    """Load trained policy from checkpoint."""
    from rl.networks import create_actor_critic
    from rl.buffer import create_buffer
    from rl.ppo import create_ppo

    obs_dim = 14
    action_dim = 2

    policy = create_actor_critic(config.get("model", {}), obs_dim, action_dim)
    buffer = create_buffer(config.get("ppo", {}), obs_dim, action_dim, device)

    ppo_config = config.get("ppo", {})
    experiment_config = config.get("experiment", {})
    total_env_steps = experiment_config.get("total_env_steps", 2_000_000)
    steps_per_update = ppo_config.get("rollout_steps", 2048) * ppo_config.get("num_envs", 8)
    total_updates = total_env_steps // steps_per_update

    ppo = create_ppo(
        policy=policy,
        buffer=buffer,
        ppo_config=ppo_config,
        model_config=config.get("model", {}),
        obs_dim=obs_dim,
        total_updates=total_updates,
        device=device,
    )

    ppo.load(checkpoint_path)
    return ppo


def run_episode(env, policy_fn, render_delay: float = 0.02, verbose: bool = True):
    """
    Run a single episode with visualization.

    Args:
        env: Environment with render_mode="human"
        policy_fn: Function that takes obs and returns action
        render_delay: Delay between frames (seconds)
        verbose: Print step information

    Returns:
        Episode statistics dict
    """
    obs, info = env.reset()
    env.render()

    episode_return = 0.0
    episode_length = 0
    done = False

    if verbose:
        print(f"\n{'='*50}")
        print("Starting episode...")
        print(f"Ball position: {info.get('ball_pos', 'N/A')}")
        print(f"{'='*50}")

    while not done:
        # Get action
        action = policy_fn(obs)

        # Step
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        episode_return += reward
        episode_length += 1

        # Render
        env.render()

        # Small delay for visualization
        if render_delay > 0:
            time.sleep(render_delay)

        # Print progress
        if verbose and episode_length % 20 == 0:
            dist = info.get("dist", 0)
            attached = "[ATTACHED]" if info.get("attached", False) else ""
            print(f"  Step {episode_length}: dist={dist:.3f} return={episode_return:.2f} {attached}")

    # Print result
    if verbose:
        success_str = "SUCCESS!" if info.get("is_success", False) else "Failed"
        print(f"\n{success_str}")
        print(f"  Return: {episode_return:.2f}")
        print(f"  Length: {episode_length}")
        print(f"  Final distance: {info.get('dist', 0):.3f}")

    return {
        "episode_return": episode_return,
        "episode_length": episode_length,
        "is_success": info.get("is_success", False),
    }


def visualize_trained_policy(
    checkpoint_path: str,
    config_path: str = None,
    num_episodes: int = 5,
    device: str = "cpu",
    render_delay: float = 0.02,
):
    """
    Visualize a trained policy.

    Args:
        checkpoint_path: Path to checkpoint file
        config_path: Path to config (auto-detect if None)
        num_episodes: Number of episodes to run
        device: Device for policy
        render_delay: Delay between frames
    """
    checkpoint_path = Path(checkpoint_path)

    # Auto-detect config
    if config_path is None:
        config_path = checkpoint_path.parent / "config.yaml"
        if not config_path.exists():
            config_path = project_root / "config" / "reach.yaml"
            print(f"Warning: Using default config {config_path}")

    config = load_config(str(config_path))
    print(f"Loaded config: {config_path}")
    print(f"Task mode: {config['env']['task_mode']}")

    # Load policy
    print(f"Loading checkpoint: {checkpoint_path}")
    ppo = load_policy(str(checkpoint_path), config, device)

    # Create environment
    env = create_env_with_viewer(config)
    print("Created environment with viewer")

    # Define policy function
    def policy_fn(obs):
        action, _, _ = ppo.get_action(obs, deterministic=True)
        return action

    # Run episodes
    print(f"\nRunning {num_episodes} episodes...")
    print("Press Ctrl+C to stop early\n")

    results = []
    try:
        for i in range(num_episodes):
            print(f"\n--- Episode {i+1}/{num_episodes} ---")
            result = run_episode(env, policy_fn, render_delay=render_delay)
            results.append(result)

            # Brief pause between episodes
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n\nStopped by user")

    # Summary
    if results:
        successes = sum(r["is_success"] for r in results)
        avg_return = np.mean([r["episode_return"] for r in results])
        avg_length = np.mean([r["episode_length"] for r in results])

        print(f"\n{'='*50}")
        print("Summary:")
        print(f"  Episodes: {len(results)}")
        print(f"  Success rate: {successes}/{len(results)} ({100*successes/len(results):.1f}%)")
        print(f"  Avg return: {avg_return:.2f}")
        print(f"  Avg length: {avg_length:.1f}")
        print(f"{'='*50}")

    env.close()


def visualize_random_policy(
    config_path: str = None,
    num_episodes: int = 3,
    render_delay: float = 0.02,
):
    """
    Visualize random policy (for testing environment).

    Args:
        config_path: Path to config
        num_episodes: Number of episodes to run
        render_delay: Delay between frames
    """
    if config_path is None:
        config_path = project_root / "config" / "reach.yaml"

    config = load_config(str(config_path))
    print(f"Loaded config: {config_path}")
    print(f"Task mode: {config['env']['task_mode']}")

    env = create_env_with_viewer(config)
    print("Created environment with viewer")

    def random_policy(obs):
        return env.action_space.sample()

    print(f"\nRunning {num_episodes} episodes with random policy...")
    print("Press Ctrl+C to stop early\n")

    try:
        for i in range(num_episodes):
            print(f"\n--- Episode {i+1}/{num_episodes} ---")
            run_episode(env, random_policy, render_delay=render_delay)
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n\nStopped by user")

    env.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize trained policy")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint")
    parser.add_argument("--config", type=str, default=None, help="Path to config")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes")
    parser.add_argument("--device", type=str, default="cpu", help="Device")
    parser.add_argument("--delay", type=float, default=0.02, help="Render delay (seconds)")
    parser.add_argument("--random", action="store_true", help="Use random policy")
    args = parser.parse_args()

    if args.random:
        visualize_random_policy(
            config_path=args.config,
            num_episodes=args.episodes,
            render_delay=args.delay,
        )
    elif args.checkpoint:
        visualize_trained_policy(
            checkpoint_path=args.checkpoint,
            config_path=args.config,
            num_episodes=args.episodes,
            device=args.device,
            render_delay=args.delay,
        )
    else:
        print("Error: Must specify --checkpoint or --random")
        print("Examples:")
        print("  python scripts/visualize.py --checkpoint runs/reach_xxx/checkpoint_best.pt")
        print("  python scripts/visualize.py --random")
        sys.exit(1)
