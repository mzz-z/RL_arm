"""Main training script for PPO."""

import os
import random
import subprocess
import numpy as np
import torch
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Optional, Tuple, Union
from gymnasium.vector import SyncVectorEnv, AsyncVectorEnv

from env.mujoco_env import MujocoArmEnv
from env.observations import ObservationBuilder
from rl.networks import create_actor_critic
from rl.buffer import create_buffer
from rl.ppo import create_ppo, PPO
from config import load_config, validate_config, save_config
from train_eval.logger import Logger, TrainingStats, create_logger
from train_eval.curriculum import CurriculumManager, create_curriculum, update_env_curriculum
from train_eval.eval import evaluate


def set_all_seeds(seed: int, fast_mode: bool = True) -> None:
    """
    Set seeds for all random number generators.

    Args:
        seed: Random seed
        fast_mode: If True, use fast non-deterministic settings for better performance.
                   If False, use deterministic settings for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if fast_mode:
        # Fast mode: allow non-deterministic operations for speed
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        # Enable TF32 for faster matmuls on Ampere+ GPUs (minor precision loss, big speedup)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Set matmul precision for PyTorch 2.x (high = TF32 on supported hardware)
        if hasattr(torch, 'set_float32_matmul_precision'):
            torch.set_float32_matmul_precision('high')
    else:
        # Deterministic mode: slower but reproducible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def _create_single_env(
    env_config: dict,
    control_config: dict,
    reward_config: dict,
    seed: int,
    rank: int,
) -> MujocoArmEnv:
    """
    Top-level function to create a single environment instance.
    This is a standalone function (not a closure) so it can be pickled for AsyncVectorEnv.

    Args:
        env_config: Environment configuration dict
        control_config: Controller configuration dict
        reward_config: Reward configuration dict
        seed: Base seed
        rank: Environment index (for seed offset)

    Returns:
        MujocoArmEnv instance
    """
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

    # Seed with rank offset for diversity
    env.reset(seed=seed + rank)

    return env


def make_env(config: dict, seed: int, rank: int):
    """
    Factory function for creating a single env instance (for SyncVectorEnv).

    Args:
        config: Full configuration dict
        seed: Base seed
        rank: Environment index (for seed offset)

    Returns:
        Callable that creates the environment
    """
    env_config = config.get("env", {})
    control_config = config.get("control", {})
    reward_config = config.get("reward", {})

    # Return a partial application of the top-level function (picklable)
    return partial(
        _create_single_env,
        env_config=env_config,
        control_config=control_config,
        reward_config=reward_config,
        seed=seed,
        rank=rank,
    )


def build_vectorized_env(
    config: dict,
    use_async: bool = True,
) -> Union[SyncVectorEnv, AsyncVectorEnv]:
    """
    Build vectorized environment with N parallel workers.

    Args:
        config: Configuration dictionary
        use_async: If True, use AsyncVectorEnv for parallel stepping (faster).
                   If False, use SyncVectorEnv (sequential, for debugging).

    Returns:
        Vectorized environment instance
    """
    ppo_config = config.get("ppo", {})
    experiment_config = config.get("experiment", {})

    num_envs = ppo_config.get("num_envs", 8)
    base_seed = experiment_config.get("seed", 42)

    env_fns = [
        make_env(config, seed=base_seed, rank=i)
        for i in range(num_envs)
    ]

    if use_async:
        # AsyncVectorEnv: each env runs in its own subprocess
        # This enables true parallelism for MuJoCo stepping
        vec_env = AsyncVectorEnv(env_fns, shared_memory=False)
    else:
        # SyncVectorEnv: sequential stepping (useful for debugging)
        vec_env = SyncVectorEnv(env_fns)

    return vec_env


def setup_experiment(config_path: str) -> Tuple[dict, Path]:
    """
    Initialize experiment with proper reproducibility.

    Args:
        config_path: Path to config file

    Returns:
        (config, run_dir)
    """
    # Load and validate config
    config = load_config(config_path)
    validate_config(config)

    # Create run directory
    experiment_config = config.get("experiment", {})
    run_name = experiment_config.get("run_name")

    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        task_mode = config.get("env", {}).get("task_mode", "reach")
        seed = experiment_config.get("seed", 42)
        run_name = f"{task_mode}_{timestamp}_{seed}"

    run_dir = Path("runs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save config copy
    save_config(config, run_dir / "config.yaml")

    # Save git commit hash (if available)
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        (run_dir / "git_hash.txt").write_text(git_hash)
    except Exception:
        pass

    # Save environment info
    env_info = f"Python: {os.sys.version}\n"
    env_info += f"PyTorch: {torch.__version__}\n"
    env_info += f"NumPy: {np.__version__}\n"
    (run_dir / "env_info.txt").write_text(env_info)

    return config, run_dir


def collect_rollout(
    vec_env: Union[SyncVectorEnv, AsyncVectorEnv],
    ppo: PPO,
    rollout_steps: int,
    initial_obs: np.ndarray,
    stats: TrainingStats,
    logger: Optional[Logger] = None,
    verbose_episodes: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Collect rollout from vectorized env, tracking episode completions.

    Args:
        vec_env: Vectorized environment
        ppo: PPO agent
        rollout_steps: Steps to collect per environment
        initial_obs: Initial observations
        stats: Training statistics
        logger: Optional logger for episode info
        verbose_episodes: Whether to print episode ends

    Returns:
        (final_obs, final_dones)
    """
    obs = initial_obs
    num_envs = vec_env.num_envs

    # Reset buffer
    ppo.buffer.reset()

    for step in range(rollout_steps):
        # Get action from policy (now also returns raw_action for log-prob consistency)
        action, log_prob, value, raw_action = ppo.get_action(obs, deterministic=False)

        # Update observation normalizer
        if ppo.obs_normalizer is not None:
            ppo.obs_normalizer.update(obs)

        # Step all environments
        next_obs, rewards, terminateds, truncateds, infos = vec_env.step(action)
        dones = terminateds | truncateds

        # Store transition (including raw_action to avoid atanh reconstruction errors)
        ppo.buffer.add(
            obs=obs,
            action=action,
            log_prob=log_prob,
            reward=rewards,
            done=dones.astype(np.float32),
            value=value,
            raw_action=raw_action,
        )

        # Track completed episodes
        # In SyncVectorEnv, when an episode ends, infos["final_info"][i] contains
        # the final info dict for that env. If env didn't terminate, it's None.
        final_infos = infos.get("final_info", [None] * num_envs)

        for i, done in enumerate(dones):
            if done and final_infos[i] is not None:
                info = final_infos[i]

                # Extract episode return and length from our env's info
                # Our env returns episode info in info["episode"] on termination
                if "episode" in info:
                    ep_return = float(info["episode"].get("r", 0))
                    ep_length = int(info["episode"].get("l", 0))
                    is_success = bool(info["episode"].get("is_success", False))
                else:
                    # Fallback
                    ep_return = float(info.get("episode_return", 0))
                    ep_length = int(info.get("step_count", 0))
                    is_success = bool(info.get("is_success", False))

                # Update stats
                stats.recent_returns.append(ep_return)
                stats.recent_lengths.append(ep_length)
                stats.recent_successes.append(float(is_success))
                stats.episodes_completed += 1

                # Phase 2 metrics
                if "ever_attached" in info:
                    stats.recent_attach_rate.append(float(info.get("ever_attached", False)))
                if "lift_success" in info:
                    stats.recent_lift_success.append(float(info.get("lift_success", False)))
                if "dropped" in info:
                    stats.recent_drop_rate.append(float(info.get("dropped", False)))

                # Print episode end
                if verbose_episodes and logger is not None:
                    logger.print_episode_end(info, ep_return, ep_length, i)

        obs = next_obs

    # Get final values for bootstrapping
    final_value = ppo.get_value(obs)
    final_dones = dones.astype(np.float32)

    # Compute GAE
    ppo.buffer.compute_gae(final_value, final_dones)

    return obs, final_dones


def train(
    config_path: str,
    resume_from: Optional[str] = None,
    verbose_episodes: bool = False,
) -> None:
    """
    Main training function.

    Args:
        config_path: Path to configuration file
        resume_from: Optional checkpoint path to resume from
        verbose_episodes: Whether to print episode completions
    """
    # Setup experiment
    config, run_dir = setup_experiment(config_path)
    experiment_config = config.get("experiment", {})
    ppo_config = config.get("ppo", {})
    model_config = config.get("model", {})

    # Set seeds (fast_mode enables non-deterministic but faster operations)
    seed = experiment_config.get("seed", 42)
    fast_mode = experiment_config.get("fast_mode", True)
    set_all_seeds(seed, fast_mode=fast_mode)

    # Get device
    device = experiment_config.get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        device = "cpu"

    # Check if async envs are enabled (default True for performance)
    use_async_envs = experiment_config.get("async_envs", True)

    # Check if curriculum is enabled - if so, prefer sync envs for reliable updates
    curriculum_config = config.get("curriculum", {})
    curriculum_enabled = curriculum_config.get("enabled", False)
    if curriculum_enabled and use_async_envs:
        print("Note: Curriculum enabled - using sync envs for reliable curriculum updates")
        use_async_envs = False

    print(f"\n{'='*60}")
    print(f"Training: {config.get('env', {}).get('task_mode', 'reach')}")
    print(f"Device: {device}")
    print(f"Fast mode: {fast_mode}")
    print(f"Async envs: {use_async_envs}")
    print(f"Run directory: {run_dir}")
    print(f"{'='*60}\n")

    # Build vectorized environment
    vec_env = build_vectorized_env(config, use_async=use_async_envs)
    num_envs = ppo_config.get("num_envs", 8)
    env_type = "async" if use_async_envs else "sync"
    print(f"Created {num_envs} parallel environments ({env_type})")

    # Get dimensions
    obs_dim = ObservationBuilder.OBS_DIM  # 14
    action_dim = 2

    # Create policy
    policy = create_actor_critic(model_config, obs_dim, action_dim)

    # Create buffer
    buffer = create_buffer(ppo_config, obs_dim, action_dim, device)

    # Calculate total updates
    total_env_steps = experiment_config.get("total_env_steps", 2_000_000)
    rollout_steps = ppo_config.get("rollout_steps", 2048)
    steps_per_update = rollout_steps * num_envs
    total_updates = total_env_steps // steps_per_update

    # Create PPO
    ppo = create_ppo(
        policy=policy,
        buffer=buffer,
        ppo_config=ppo_config,
        model_config=model_config,
        obs_dim=obs_dim,
        total_updates=total_updates,
        device=device,
    )

    # Create logger
    logger = create_logger(run_dir, config)

    # Create curriculum manager
    curriculum = create_curriculum(config)
    if curriculum is not None:
        print(f"Curriculum enabled: {curriculum.num_stages} stages")
        # Apply initial curriculum stage
        update_env_curriculum(vec_env, curriculum.current_stage)

    # Initialize training stats
    stats = TrainingStats()

    # Resume from checkpoint if specified
    if resume_from is not None:
        print(f"Resuming from: {resume_from}")
        update_count = ppo.load(resume_from)
        stats.update_count = update_count
        stats.global_step = update_count * steps_per_update

    # Training parameters
    log_interval = experiment_config.get("log_interval_updates", 10)
    eval_interval = experiment_config.get("eval_interval_updates", 50)
    checkpoint_interval = experiment_config.get("checkpoint_interval_updates", 100)
    num_eval_episodes = experiment_config.get("num_eval_episodes", 20)
    eval_seeds = experiment_config.get("eval_seeds", list(range(100, 120)))

    # Video recording parameters
    record_video = experiment_config.get("record_video", False)
    video_interval = experiment_config.get("video_interval_updates", 200)

    # Initial reset
    obs, _ = vec_env.reset(seed=seed)

    print(f"\nStarting training: {total_updates} updates, {total_env_steps:,} env steps")
    print(f"Rollout: {rollout_steps} steps x {num_envs} envs = {steps_per_update:,} samples/update\n")

    # Training loop
    while stats.global_step < total_env_steps:
        # Collect rollout
        obs, _ = collect_rollout(
            vec_env=vec_env,
            ppo=ppo,
            rollout_steps=rollout_steps,
            initial_obs=obs,
            stats=stats,
            logger=logger,
            verbose_episodes=verbose_episodes,
        )

        stats.global_step += steps_per_update

        # Update PPO
        update_metrics = ppo.update()
        stats.update_count += 1

        # Get action std for logging
        action_std = np.exp(ppo.policy.log_std.detach().cpu().numpy()).mean()

        # Log training metrics
        if stats.update_count % log_interval == 0:
            logger.log_training_update(stats, update_metrics, action_std)
            logger.print_training_summary(stats, update_metrics)

        # Periodic evaluation
        if stats.update_count % eval_interval == 0:
            print(f"\n>>> Running evaluation ({num_eval_episodes} episodes)...")
            eval_metrics = evaluate(
                ppo=ppo,
                config=config,
                num_episodes=num_eval_episodes,
                seeds=eval_seeds,
                verbose=False,
            )
            logger.log_eval(eval_metrics, stats.global_step)

            print(f"    Success: {eval_metrics['success_rate']*100:.1f}%")
            print(f"    Return:  {eval_metrics['avg_return']:.2f}")

            # Check for best model
            if eval_metrics["success_rate"] > stats.best_eval_score:
                stats.best_eval_score = eval_metrics["success_rate"]
                ppo.save(str(run_dir / "checkpoint_best.pt"))
                print(f"    New best! Saved checkpoint_best.pt")

            # Curriculum update
            if curriculum is not None:
                advanced = curriculum.maybe_advance(eval_metrics)
                if advanced:
                    update_env_curriculum(vec_env, curriculum.current_stage)
                    logger.log("curriculum/stage", curriculum.stage_idx, stats.global_step)

        # Periodic video recording
        if record_video and stats.update_count % video_interval == 0:
            from train_eval.video import record_evaluation_videos
            print(f"\n>>> Recording evaluation videos...")
            video_dir = run_dir / "videos"
            record_evaluation_videos(
                ppo=ppo,
                config=config,
                output_dir=video_dir,
                num_episodes=3,
                step=stats.global_step,
            )

        # Periodic checkpoint
        if stats.update_count % checkpoint_interval == 0:
            ppo.save(str(run_dir / "checkpoint_latest.pt"))
            print(f"Saved checkpoint_latest.pt")

    # Final checkpoint
    ppo.save(str(run_dir / "checkpoint_final.pt"))
    print(f"\nTraining complete! Final checkpoint saved.")

    # Final evaluation
    print(f"\n>>> Final evaluation ({num_eval_episodes} episodes)...")
    final_metrics = evaluate(
        ppo=ppo,
        config=config,
        num_episodes=num_eval_episodes,
        seeds=eval_seeds,
        verbose=False,
    )
    print(f"Final Success: {final_metrics['success_rate']*100:.1f}%")
    print(f"Final Return:  {final_metrics['avg_return']:.2f}")

    # Cleanup
    vec_env.close()
    logger.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train PPO on arm reach/grasp task")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
    parser.add_argument("--verbose", action="store_true", help="Print episode completions")
    args = parser.parse_args()

    train(
        config_path=args.config,
        resume_from=args.resume,
        verbose_episodes=args.verbose,
    )
