#!/usr/bin/env python
"""
Record videos of a trained policy.

Usage:
    python scripts/record_video.py --checkpoint runs/<run>/checkpoint_best.pt
    python scripts/record_video.py --checkpoint runs/<run>/checkpoint_best.pt --episodes 5 --output videos/
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import load_config
from train_eval.video import record_evaluation_videos


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


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Record videos of trained policy")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--config", type=str, default=None, help="Path to config")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to record")
    parser.add_argument("--device", type=str, default="cpu", help="Device")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="Seeds for episodes")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)

    # Auto-detect config
    if args.config is None:
        config_path = checkpoint_path.parent / "config.yaml"
        if not config_path.exists():
            config_path = project_root / "config" / "reach.yaml"
            print(f"Warning: Using default config {config_path}")
    else:
        config_path = Path(args.config)

    # Auto-detect output directory
    if args.output is None:
        output_dir = checkpoint_path.parent / "videos"
    else:
        output_dir = Path(args.output)

    config = load_config(str(config_path))
    print(f"Config: {config_path}")
    print(f"Task mode: {config['env']['task_mode']}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output: {output_dir}")

    # Load policy
    print("\nLoading policy...")
    ppo = load_policy(str(checkpoint_path), config, args.device)

    # Record videos
    print(f"\nRecording {args.episodes} episodes...")
    results = record_evaluation_videos(
        ppo=ppo,
        config=config,
        output_dir=output_dir,
        num_episodes=args.episodes,
        seeds=args.seeds,
        step=0,
    )

    # Summary
    successes = sum(r["is_success"] for r in results)
    print(f"\nRecorded {len(results)} videos")
    print(f"Success rate: {successes}/{len(results)} ({100*successes/len(results):.1f}%)")
    print(f"Videos saved to: {output_dir}")


if __name__ == "__main__":
    main()
