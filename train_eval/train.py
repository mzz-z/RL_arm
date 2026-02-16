"""Main training script for PPO."""

import os
import sys
import random
import subprocess
import numpy as np
import torch
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Optional, Tuple, Union

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from gymnasium.vector import SyncVectorEnv, AsyncVectorEnv

from env.mujoco_env import MujocoArmEnv
from env.observations import ObservationBuilder
from rl.networks import create_actor_critic
from rl.buffer import create_buffer
from rl.ppo import create_ppo, PPO
import torch.nn.functional as F

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
        spawn_azimuth_min=env_config.get("spawn", {}).get("azimuth_min", 0.0),
        spawn_azimuth_max=env_config.get("spawn", {}).get("azimuth_max", 0.0),
        # Initial joint state
        init_base_range=tuple(env_config.get("init_joints", {}).get("base_range", [-0.1, 0.1])),
        init_shoulder_range=tuple(env_config.get("init_joints", {}).get("shoulder_range", [-0.3, 0.3])),
        init_elbow_range=tuple(env_config.get("init_joints", {}).get("elbow_range", [-0.3, 0.3])),
        init_wrist_range=tuple(env_config.get("init_joints", {}).get("wrist_range", [-0.3, 0.3])),
        init_vel_noise_std=env_config.get("init_joints", {}).get("vel_noise_std", 0.01),
        # Task parameters
        reach_radius=env_config.get("reach", {}).get("reach_radius", 0.05),
        dwell_steps=env_config.get("reach", {}).get("dwell_steps", 5),
        ee_vel_threshold=env_config.get("reach", {}).get("ee_vel_threshold", 0.1),
        w_delta_dist=env_config.get("reach", {}).get("w_delta_dist", 5.0),
        attach_radius=env_config.get("magnet", {}).get("attach_radius", 0.04),
        attach_vel_threshold=env_config.get("magnet", {}).get("attach_vel_threshold", 0.15),
        place_radius=env_config.get("place", {}).get("place_radius", 0.05),
        hold_steps=env_config.get("place", {}).get("hold_steps", 10),
        # Reward config
        reward_config=reward_config,
        # Termination
        ball_fell_threshold=env_config.get("termination", {}).get("ball_fell_threshold", -0.05),
        unreachable_margin=env_config.get("termination", {}).get("unreachable_margin", 0.1),
        # Guided action
        guide_alpha=env_config.get("guide", {}).get("alpha_initial", 0.0) if env_config.get("guide", {}).get("enabled", False) else 0.0,
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
    env_info = f"Python: {sys.version}\n"
    env_info += f"PyTorch: {torch.__version__}\n"
    env_info += f"NumPy: {np.__version__}\n"
    (run_dir / "env_info.txt").write_text(env_info)

    return config, run_dir


def bc_warmup(
    vec_env: SyncVectorEnv,
    ppo: PPO,
    num_steps: int,
    device: str = "cpu",
) -> np.ndarray:
    """
    Behavioral cloning warmup: pre-train policy to imitate Jacobian guide.

    Collects observations from the environment, computes the analytical
    Jacobian-transpose guide action, and trains the policy to match via MSE.
    This gives the policy a warm start before PPO fine-tuning.

    Args:
        vec_env: SyncVectorEnv (must have .envs for guide access)
        ppo: PPO agent (policy will be modified in-place)
        num_steps: Number of BC training steps
        device: Torch device

    Returns:
        Final observations after warmup
    """
    policy = ppo.policy
    policy.train()

    bc_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    obs, _ = vec_env.reset()
    running_loss = 0.0
    log_interval = max(num_steps // 10, 1)

    for step in range(num_steps):
        # Get guide actions from all envs
        guide_actions = []
        for env in vec_env.envs:
            unwrapped = env
            while hasattr(unwrapped, 'env'):
                unwrapped = unwrapped.env
            guide_actions.append(unwrapped._compute_guide_action())
        guide_actions_np = np.array(guide_actions)

        # Update and normalize observations
        if ppo.obs_normalizer is not None:
            ppo.obs_normalizer.update(obs)
            obs_norm = ppo.obs_normalizer.normalize(obs)
        else:
            obs_norm = obs

        obs_tensor = torch.FloatTensor(obs_norm).to(device)
        target_tensor = torch.FloatTensor(guide_actions_np).to(device)

        # Forward: get pre-tanh mean from actor
        mean, _, _ = policy.forward(obs_tensor)

        # Loss: MSE between tanh(mean) and guide action
        policy_action = torch.tanh(mean)
        loss = F.mse_loss(policy_action, target_tensor)

        bc_optimizer.zero_grad()
        loss.backward()
        bc_optimizer.step()

        running_loss += loss.item()

        # Step env with guide actions to explore trajectory states
        obs, _, _, _, _ = vec_env.step(guide_actions_np)

        if (step + 1) % log_interval == 0:
            avg_loss = running_loss / log_interval
            print(f"    BC warmup [{step+1}/{num_steps}]: loss={avg_loss:.4f}")
            running_loss = 0.0

    # Reset env to clean state after warmup
    obs, _ = vec_env.reset()
    print(f"    BC warmup complete")
    return obs


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

    # Manual episode accumulators (robust regardless of Gymnasium version)
    ep_returns = np.zeros(num_envs, dtype=np.float64)
    ep_lengths = np.zeros(num_envs, dtype=np.int32)

    for step in range(rollout_steps):
        # Get action from policy (now also returns raw_action for log-prob consistency)
        action, log_prob, value, raw_action = ppo.get_action(obs, deterministic=False)

        # Normalize obs for buffer storage using the SAME normalizer stats
        # that get_action() just used. This ensures old_log_probs and the stored
        # obs are consistent, avoiding the mismatch where PPO.update() would
        # re-normalize with drifted stats.
        if ppo.obs_normalizer is not None:
            obs_normalized = ppo.obs_normalizer.normalize(obs)
        else:
            obs_normalized = obs

        # Update observation normalizer AFTER normalizing for buffer
        if ppo.obs_normalizer is not None:
            ppo.obs_normalizer.update(obs)

        # Step all environments
        next_obs, rewards, terminateds, truncateds, infos = vec_env.step(action)
        dones = terminateds | truncateds

        # Accumulate episode stats (rewards from terminal step are included)
        ep_returns += rewards
        ep_lengths += 1

        # Store transition with normalized obs (matching get_action's input)
        ppo.buffer.add(
            obs=obs_normalized,
            action=action,
            log_prob=log_prob,
            reward=rewards,
            done=dones.astype(np.float32),
            value=value,
            raw_action=raw_action,
        )

        # Track completed episodes using manual accumulators
        # This is robust against Gymnasium version differences in final_info handling
        final_infos = infos.get("final_info", [None] * num_envs)

        for i, done in enumerate(dones):
            if not done:
                continue

            # Record episode stats from accumulators (always works)
            stats.recent_returns.append(float(ep_returns[i]))
            stats.recent_lengths.append(int(ep_lengths[i]))
            stats.episodes_completed += 1

            # Try to get success from final_info (best-effort)
            is_success = False
            info = None
            if final_infos is not None and i < len(final_infos):
                info = final_infos[i]
            if info is not None:
                if "episode" in info:
                    is_success = bool(info["episode"].get("is_success", False))
                else:
                    is_success = bool(info.get("is_success", False))

                # Phase 2 metrics
                if "ever_attached" in info:
                    stats.recent_attach_rate.append(float(info.get("ever_attached", False)))
                if "place_success" in info:
                    stats.recent_place_success.append(float(info.get("place_success", False)))
                if "dropped" in info:
                    stats.recent_drop_rate.append(float(info.get("dropped", False)))

            stats.recent_successes.append(float(is_success))

            # Print only successful episodes (verbose mode)
            if verbose_episodes and is_success and logger is not None:
                ep_info = info if info is not None else {}
                logger.print_episode_end(ep_info, float(ep_returns[i]), int(ep_lengths[i]), i)

            # Reset accumulators for this env
            ep_returns[i] = 0.0
            ep_lengths[i] = 0

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
    transfer_from: Optional[str] = None,
    verbose_episodes: bool = False,
) -> None:
    """
    Main training function.

    Args:
        config_path: Path to configuration file
        resume_from: Optional checkpoint path to resume training (loads full state)
        transfer_from: Optional checkpoint path for transfer learning (loads weights only)
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
    obs_dim = ObservationBuilder.OBS_DIM
    action_dim = 4

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
        update_count = ppo.load(resume_from, weights_only=False)
        stats.update_count = update_count
        stats.global_step = update_count * steps_per_update

    # Transfer learning: load weights only, start training fresh
    if transfer_from is not None:
        print(f"Transfer learning from: {transfer_from}")
        ppo.load(transfer_from, weights_only=True)
        print("  Loaded policy weights and observation normalizer")
        print("  Fresh optimizer and learning rate schedule")

        # Fix normalizer for task transfer: reset collapsed dimensions.
        # The grasp flag (obs dim 20) was always 0 during reach training,
        # so its variance collapsed to ~0. Without this fix, obs[20]=1
        # would normalize to ~10000 (clipped to 10.0), injecting a huge
        # random signal since the policy never learned this dimension.
        if ppo.obs_normalizer is not None:
            grasp_flag_idx = ObservationBuilder.GRASP_FLAG_IDX  # dim 20
            ppo.obs_normalizer.var[grasp_flag_idx] = 1.0
            ppo.obs_normalizer.mean[grasp_flag_idx] = 0.0
            print(f"  Reset normalizer stats for grasp flag (dim {grasp_flag_idx})")

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

    # Training assist: behavioral cloning warmup
    assist_config = config.get("training_assist", {})
    bc_steps = assist_config.get("bc_warmup_steps", 0)
    if bc_steps > 0 and hasattr(vec_env, 'envs'):
        print(f"\n>>> Behavioral cloning warmup ({bc_steps} steps)...")
        obs = bc_warmup(vec_env, ppo, bc_steps, device)

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

        # Anneal guide_alpha if guided mode is enabled
        guide_cfg = config.get("env", {}).get("guide", {})
        if guide_cfg.get("enabled", False):
            alpha_init = guide_cfg.get("alpha_initial", 0.5)
            alpha_final = guide_cfg.get("alpha_final", 0.0)
            anneal_frac = guide_cfg.get("anneal_fraction", 0.8)
            progress = min(stats.global_step / total_env_steps / anneal_frac, 1.0)
            current_alpha = alpha_init + (alpha_final - alpha_init) * progress
            # Set on all envs (SyncVectorEnv only — async envs use initial value)
            if hasattr(vec_env, 'envs'):
                for env in vec_env.envs:
                    unwrapped = env
                    while hasattr(unwrapped, 'env'):
                        unwrapped = unwrapped.env
                    unwrapped.guide_alpha = current_alpha
            if stats.update_count % log_interval == 0:
                logger.log("guide/alpha", current_alpha, stats.global_step)

        # Log training metrics
        if stats.update_count % log_interval == 0:
            logger.log_training_update(stats, update_metrics, action_std)
            logger.print_training_summary(stats, update_metrics)

        # Periodic evaluation
        if stats.update_count % eval_interval == 0:
            # Compute current guide_alpha for eval envs
            _guide_cfg = config.get("env", {}).get("guide", {})
            if _guide_cfg.get("enabled", False):
                _a_init = _guide_cfg.get("alpha_initial", 0.5)
                _a_final = _guide_cfg.get("alpha_final", 0.0)
                _a_frac = _guide_cfg.get("anneal_fraction", 0.8)
                _prog = min(stats.global_step / total_env_steps / _a_frac, 1.0)
                eval_guide_alpha = _a_init + (_a_final - _a_init) * _prog
            else:
                eval_guide_alpha = None

            # Get current curriculum spawn params (if curriculum is active)
            curriculum_spawn = curriculum.get_spawn_params() if curriculum is not None else None
            curriculum_reach_radius = curriculum.current_stage.get("reach_radius") if curriculum is not None else None

            # Eval on curriculum-stage distribution (what training actually sees)
            if curriculum_spawn:
                print(f"\n>>> Curriculum eval (stage {curriculum.stage_idx + 1}, {num_eval_episodes} episodes)...")
                curriculum_eval = evaluate(
                    ppo=ppo,
                    config=config,
                    num_episodes=num_eval_episodes,
                    seeds=eval_seeds,
                    verbose=False,
                    spawn_params=curriculum_spawn,
                    guide_alpha=eval_guide_alpha,
                    reach_radius=curriculum_reach_radius,
                )
                print(f"    Curriculum Success: {curriculum_eval['success_rate']*100:.1f}%")
                print(f"    Curriculum Return:  {curriculum_eval['avg_return']:.2f}")
                if "attach_rate" in curriculum_eval:
                    print(f"    Attach Rate:        {curriculum_eval['attach_rate']*100:.1f}%")
                if "place_success_rate" in curriculum_eval:
                    print(f"    Place Rate:         {curriculum_eval['place_success_rate']*100:.1f}%")
                logger.log("eval_curriculum/success_rate", curriculum_eval["success_rate"], stats.global_step)
                logger.log("eval_curriculum/avg_return", curriculum_eval["avg_return"], stats.global_step)

                # Also run stochastic eval on curriculum stage for diagnostics
                stochastic_eval = evaluate(
                    ppo=ppo,
                    config=config,
                    num_episodes=num_eval_episodes,
                    seeds=eval_seeds,
                    verbose=False,
                    spawn_params=curriculum_spawn,
                    deterministic=False,
                    guide_alpha=eval_guide_alpha,
                    reach_radius=curriculum_reach_radius,
                )
                print(f"    Stochastic Success: {stochastic_eval['success_rate']*100:.1f}%")
                logger.log("eval_stochastic/success_rate", stochastic_eval["success_rate"], stats.global_step)

                # Use curriculum eval for curriculum advancement
                eval_for_curriculum = curriculum_eval
            else:
                eval_for_curriculum = None

            # Eval on full/default distribution (tracks true target performance)
            print(f"\n>>> Full eval ({num_eval_episodes} episodes)...")
            eval_metrics = evaluate(
                ppo=ppo,
                config=config,
                num_episodes=num_eval_episodes,
                seeds=eval_seeds,
                verbose=False,
                guide_alpha=eval_guide_alpha,
            )
            logger.log_eval(eval_metrics, stats.global_step)

            print(f"    Full Success: {eval_metrics['success_rate']*100:.1f}%")
            print(f"    Full Return:  {eval_metrics['avg_return']:.2f}")
            if "attach_rate" in eval_metrics:
                print(f"    Attach Rate:  {eval_metrics['attach_rate']*100:.1f}%")
            if "place_success_rate" in eval_metrics:
                print(f"    Place Rate:   {eval_metrics['place_success_rate']*100:.1f}%")

            # Check for best model (based on full distribution)
            if eval_metrics["success_rate"] > stats.best_eval_score:
                stats.best_eval_score = eval_metrics["success_rate"]
                ppo.save(str(run_dir / "checkpoint_best.pt"))
                print(f"    New best! Saved checkpoint_best.pt")

            # Curriculum update (based on curriculum-stage eval, not full eval)
            if curriculum is not None and eval_for_curriculum is not None:
                advanced = curriculum.maybe_advance(eval_for_curriculum)
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

    # Final evaluation (use final guide_alpha value)
    _final_guide_cfg = config.get("env", {}).get("guide", {})
    final_guide_alpha = _final_guide_cfg.get("alpha_final", 0.0) if _final_guide_cfg.get("enabled", False) else None
    print(f"\n>>> Final evaluation ({num_eval_episodes} episodes)...")
    final_metrics = evaluate(
        ppo=ppo,
        config=config,
        num_episodes=num_eval_episodes,
        seeds=eval_seeds,
        verbose=False,
        guide_alpha=final_guide_alpha,
    )
    print(f"Final Success: {final_metrics['success_rate']*100:.1f}%")
    print(f"Final Return:  {final_metrics['avg_return']:.2f}")
    if "attach_rate" in final_metrics:
        print(f"Attach Rate:   {final_metrics['attach_rate']*100:.1f}%")
    if "place_success_rate" in final_metrics:
        print(f"Place Rate:    {final_metrics['place_success_rate']*100:.1f}%")

    # Cleanup
    vec_env.close()
    logger.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train PPO on arm reach/grasp task")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from (full state)")
    parser.add_argument("--transfer", type=str, default=None, help="Checkpoint for transfer learning (weights only)")
    parser.add_argument("--verbose", action="store_true", help="Print episode completions")
    args = parser.parse_args()

    if args.resume and args.transfer:
        print("Error: Cannot specify both --resume and --transfer")
        sys.exit(1)

    train(
        config_path=args.config,
        resume_from=args.resume,
        transfer_from=args.transfer,
        verbose_episodes=args.verbose,
    )
