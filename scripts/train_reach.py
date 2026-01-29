#!/usr/bin/env python
"""Convenience script to train the reach task."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from train_eval.train import train

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train reach task")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
    parser.add_argument("--verbose", action="store_true", help="Print episode completions")
    args = parser.parse_args()

    config_path = project_root / "config" / "reach.yaml"

    train(
        config_path=str(config_path),
        resume_from=args.resume,
        verbose_episodes=args.verbose,
    )
