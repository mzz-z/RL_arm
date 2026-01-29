"""Evaluation runner for trained policies."""

import numpy as np
import torch
from typing import Dict, List, Optional, Any
from pathlib import Path

from env.mujoco_env import MujocoArmEnv
from rl.ppo import PPO


def make_eval_env(config: dict) -> MujocoArmEnv:
    """
    Create a single environment for evaluation.

    Args:
        config: Configuration dictionary

    Returns:
        MujocoArmEnv instance
    """
    env_config = config.get("env", {})
    control_config = config.get("control", {})
    reward_config = config.get("reward", {})

    # Create environment
    env = MujocoArmEnv(
        task_mode=env_config.get("task_mode", "reach"),
        max_episode_steps=env_config.get("max_episode_steps", 200),
        frame_skip=env_config.get("frame_skip", 4),
        # Spawn parameters
        spawn_radius_min=env_config.get("spawn", {}).get("radius_min", 0.15),
        spawn_radius_max=env_config.get("spawn", {}).get("radius_max", 0.40),
        spawn_angle_min=env_config.get("spawn", {}).get("angle_min", -1.0),
        spawn_angle_max=env_config.get("spawn", {}).get("angle_max", 1.0),
        spawn_y_min=env_config.get("spawn", {}).get("y_min", -0.15),
        spawn_y_max=env_config.get("spawn", {}).get("y_max", 0.15),
        # Initial joint state
        init_shoulder_range=tuple(env_config.get("init_joints", {}).get("shoulder_range", [-0.3, 0.3])),
        init_elbow_range=tuple(env_config.get("init_joints", {}).get("elbow_range", [-0.3, 0.3])),
        init_vel_noise_std=env_config.get("init_joints", {}).get("vel_noise_std", 0.01),
        # Task parameters
        reach_radius=env_config.get("reach", {}).get("reach_radius", 0.05),
        dwell_steps=env_config.get("reach", {}).get("dwell_steps", 5),
        ee_vel_threshold=env_config.get("reach", {}).get("ee_vel_threshold", 0.1),
        attach_radius=env_config.get("magnet", {}).get("attach_radius", 0.04),
        attach_vel_threshold=env_config.get("magnet", {}).get("attach_vel_threshold", 0.15),
        lift_height=env_config.get("lift", {}).get("lift_height", 0.1),
        hold_steps=env_config.get("lift", {}).get("hold_steps", 10),
        # Reward config
        reward_config=reward_config,
        # Termination
        ball_fell_threshold=env_config.get("termination", {}).get("ball_fell_threshold", -0.05),
        unreachable_margin=env_config.get("termination", {}).get("unreachable_margin", 0.1),
    )

    # Create and set controller
    env.create_controller_from_config(control_config)

    return env


def evaluate(
    ppo: PPO,
    config: dict,
    num_episodes: int = 20,
    seeds: Optional[List[int]] = None,
    verbose: bool = False,
) -> Dict[str, float]:
    """
    Run deterministic evaluation episodes.

    Args:
        ppo: Trained PPO agent
        config: Configuration dictionary
        num_episodes: Number of evaluation episodes
        seeds: Fixed seeds for reproducibility (cycles if fewer than num_episodes)
        verbose: Whether to print per-episode info

    Returns:
        Dictionary of aggregated metrics
    """
    # Default seeds
    if seeds is None:
        seeds = list(range(100, 100 + num_episodes))

    # Create fresh environment for evaluation
    eval_env = make_eval_env(config)

    # Tracking
    all_returns = []
    all_lengths = []
    all_successes = []
    all_time_to_success = []

    # Phase 2 specific
    all_attached = []
    all_lift_success = []
    all_dropped = []

    for i in range(num_episodes):
        seed = seeds[i % len(seeds)]
        obs, _ = eval_env.reset(seed=seed)

        episode_return = 0.0
        episode_length = 0
        done = False

        while not done:
            # Deterministic action (mean of policy)
            action, _, _ = ppo.get_action(obs, deterministic=True)

            obs, reward, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated

            episode_return += reward
            episode_length += 1

        # Extract final info
        success = info.get("is_success", False)

        all_returns.append(episode_return)
        all_lengths.append(episode_length)
        all_successes.append(float(success))

        if success:
            all_time_to_success.append(episode_length)

        # Phase 2 metrics
        if "ever_attached" in info:
            all_attached.append(float(info.get("ever_attached", False)))
        if "lift_success" in info:
            all_lift_success.append(float(info.get("lift_success", False)))
        if "dropped" in info:
            all_dropped.append(float(info.get("dropped", False)))

        if verbose:
            success_str = "+" if success else "-"
            print(f"  Episode {i+1}/{num_episodes}: {success_str} "
                  f"R={episode_return:.2f} L={episode_length}")

    eval_env.close()

    # Aggregate metrics
    metrics = {
        "success_rate": np.mean(all_successes),
        "avg_return": np.mean(all_returns),
        "std_return": np.std(all_returns),
        "avg_length": np.mean(all_lengths),
    }

    if len(all_time_to_success) > 0:
        metrics["time_to_success_mean"] = np.mean(all_time_to_success)
        metrics["time_to_success_std"] = np.std(all_time_to_success)

    # Phase 2 metrics
    if len(all_attached) > 0:
        metrics["attach_rate"] = np.mean(all_attached)
    if len(all_lift_success) > 0:
        metrics["lift_success_rate"] = np.mean(all_lift_success)
    if len(all_dropped) > 0:
        metrics["drop_rate"] = np.mean(all_dropped)

    return metrics


def evaluate_checkpoint(
    checkpoint_path: str,
    config: dict,
    num_episodes: int = 20,
    seeds: Optional[List[int]] = None,
    device: str = "cpu",
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Evaluate a saved checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        config: Configuration dictionary
        num_episodes: Number of evaluation episodes
        seeds: Fixed seeds for reproducibility
        device: Device for inference
        verbose: Whether to print results

    Returns:
        Dictionary of evaluation metrics
    """
    from rl.networks import create_actor_critic
    from rl.buffer import create_buffer

    # Get dimensions
    obs_dim = 14  # ObservationBuilder.OBS_DIM
    action_dim = 2

    # Create policy
    policy = create_actor_critic(config.get("model", {}), obs_dim, action_dim)

    # Create buffer (needed for PPO)
    buffer = create_buffer(config.get("ppo", {}), obs_dim, action_dim, device)

    # Calculate total updates (needed for LR scheduler)
    ppo_config = config.get("ppo", {})
    experiment_config = config.get("experiment", {})
    total_env_steps = experiment_config.get("total_env_steps", 2_000_000)
    steps_per_update = ppo_config.get("rollout_steps", 2048) * ppo_config.get("num_envs", 8)
    total_updates = total_env_steps // steps_per_update

    # Create PPO
    from rl.ppo import create_ppo
    ppo = create_ppo(
        policy=policy,
        buffer=buffer,
        ppo_config=ppo_config,
        model_config=config.get("model", {}),
        obs_dim=obs_dim,
        total_updates=total_updates,
        device=device,
    )

    # Load checkpoint
    ppo.load(checkpoint_path)

    if verbose:
        print(f"Loaded checkpoint: {checkpoint_path}")
        print(f"Evaluating {num_episodes} episodes...")

    # Run evaluation
    metrics = evaluate(
        ppo=ppo,
        config=config,
        num_episodes=num_episodes,
        seeds=seeds,
        verbose=verbose,
    )

    if verbose:
        print(f"\nEvaluation Results:")
        print(f"  Success rate: {metrics['success_rate']*100:.1f}%")
        print(f"  Avg return:   {metrics['avg_return']:.2f} +/- {metrics['std_return']:.2f}")
        print(f"  Avg length:   {metrics['avg_length']:.1f}")
        if "time_to_success_mean" in metrics:
            print(f"  Time to success: {metrics['time_to_success_mean']:.1f} +/- {metrics.get('time_to_success_std', 0):.1f}")

    return metrics


if __name__ == "__main__":
    import argparse
    from config import load_config

    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--episodes", type=int, default=20, help="Number of episodes")
    parser.add_argument("--device", type=str, default="cpu", help="Device")
    args = parser.parse_args()

    config = load_config(args.config)
    metrics = evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        config=config,
        num_episodes=args.episodes,
        device=args.device,
    )
