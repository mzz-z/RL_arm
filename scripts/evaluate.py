#!/usr/bin/env python
"""Convenience script to evaluate a trained checkpoint."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from train_eval.eval import evaluate_checkpoint
from config import load_config

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate trained checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--config", type=str, default=None, help="Path to config (auto-detect if not specified)")
    parser.add_argument("--episodes", type=int, default=20, help="Number of episodes")
    parser.add_argument("--device", type=str, default="cpu", help="Device")
    args = parser.parse_args()

    # Try to auto-detect config from checkpoint directory
    if args.config is None:
        checkpoint_dir = Path(args.checkpoint).parent
        config_path = checkpoint_dir / "config.yaml"
        if config_path.exists():
            args.config = str(config_path)
        else:
            # Default to reach config
            args.config = str(project_root / "config" / "reach.yaml")
            print(f"Warning: No config found in checkpoint dir, using {args.config}")

    config = load_config(args.config)

    metrics = evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        config=config,
        num_episodes=args.episodes,
        device=args.device,
        verbose=True,
    )
