"""Logging utilities for training metrics."""

import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class TrainingStats:
    """Tracks training statistics with rolling windows."""

    global_step: int = 0
    update_count: int = 0
    episodes_completed: int = 0
    best_eval_score: float = float("-inf")

    # Rolling windows for smoothed metrics
    recent_returns: deque = field(default_factory=lambda: deque(maxlen=100))
    recent_lengths: deque = field(default_factory=lambda: deque(maxlen=100))
    recent_successes: deque = field(default_factory=lambda: deque(maxlen=100))

    # Phase 2 specific
    recent_attach_rate: deque = field(default_factory=lambda: deque(maxlen=100))
    recent_lift_success: deque = field(default_factory=lambda: deque(maxlen=100))
    recent_drop_rate: deque = field(default_factory=lambda: deque(maxlen=100))


class Logger:
    """
    Central logging utility for training metrics.

    Outputs to:
    - Console (formatted summaries)
    - TensorBoard (scalars for visualization)
    """

    def __init__(
        self,
        run_dir: Path,
        use_tensorboard: bool = True,
        log_interval: int = 10,
    ):
        """
        Initialize logger.

        Args:
            run_dir: Directory for run outputs
            use_tensorboard: Whether to use TensorBoard
            log_interval: How often to print console summaries
        """
        self.run_dir = Path(run_dir)
        self.log_interval = log_interval
        self.use_tensorboard = use_tensorboard

        # TensorBoard writer
        self.writer = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir=str(self.run_dir / "tensorboard"))
            except ImportError:
                print("Warning: TensorBoard not available. Logging to console only.")
                self.use_tensorboard = False

    def log(self, tag: str, value: float, step: int) -> None:
        """
        Log a scalar value.

        Args:
            tag: Metric name (e.g., "train/return")
            value: Metric value
            step: Global step
        """
        if self.writer is not None:
            self.writer.add_scalar(tag, value, step)

    def log_dict(self, metrics: Dict[str, float], step: int, prefix: str = "") -> None:
        """
        Log multiple scalar values.

        Args:
            metrics: Dictionary of metric name -> value
            step: Global step
            prefix: Optional prefix for all tags
        """
        for name, value in metrics.items():
            tag = f"{prefix}/{name}" if prefix else name
            self.log(tag, value, step)

    def log_training_update(
        self,
        stats: TrainingStats,
        update_metrics: Dict[str, float],
        action_std: float,
    ) -> None:
        """
        Log comprehensive training diagnostics.

        Args:
            stats: Training statistics
            update_metrics: Metrics from PPO update
            action_std: Current action standard deviation
        """
        step = stats.global_step

        # Episode statistics (rolling averages)
        if len(stats.recent_returns) > 0:
            self.log("train/episode_return_mean", np.mean(stats.recent_returns), step)
            self.log("train/episode_return_std", np.std(stats.recent_returns), step)
        if len(stats.recent_lengths) > 0:
            self.log("train/episode_length_mean", np.mean(stats.recent_lengths), step)
        if len(stats.recent_successes) > 0:
            self.log("train/success_rate", np.mean(stats.recent_successes), step)
        self.log("train/episodes_completed", stats.episodes_completed, step)

        # PPO loss components
        self.log("loss/policy", update_metrics["loss/policy"], step)
        self.log("loss/value", update_metrics["loss/value"], step)
        self.log("loss/entropy", update_metrics["loss/entropy"], step)

        # PPO diagnostics
        self.log("stats/approx_kl", update_metrics["stats/approx_kl"], step)
        self.log("stats/clip_frac", update_metrics["stats/clip_frac"], step)
        self.log("stats/explained_var", update_metrics["stats/explained_var"], step)
        self.log("stats/learning_rate", update_metrics["stats/learning_rate"], step)
        self.log("stats/std", action_std, step)

        # Phase 2 metrics
        if len(stats.recent_attach_rate) > 0:
            self.log("train/attach_rate", np.mean(stats.recent_attach_rate), step)
        if len(stats.recent_lift_success) > 0:
            self.log("train/lift_success_rate", np.mean(stats.recent_lift_success), step)
        if len(stats.recent_drop_rate) > 0:
            self.log("train/drop_rate", np.mean(stats.recent_drop_rate), step)

    def log_eval(self, eval_metrics: Dict[str, float], step: int) -> None:
        """
        Log evaluation metrics.

        Args:
            eval_metrics: Dictionary of evaluation metrics
            step: Global step
        """
        for name, value in eval_metrics.items():
            self.log(f"eval/{name}", value, step)

    def print_training_summary(
        self,
        stats: TrainingStats,
        update_metrics: Dict[str, float],
        eval_metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Print formatted training summary to console.

        Args:
            stats: Training statistics
            update_metrics: Metrics from PPO update
            eval_metrics: Optional evaluation metrics
        """
        print(f"\n{'='*60}")
        print(f"Update {stats.update_count} | Steps: {stats.global_step:,}")
        print(f"{'='*60}")

        # Episode stats
        print(f"Episodes: {stats.episodes_completed}")
        if len(stats.recent_returns) > 0:
            print(f"Return:   {np.mean(stats.recent_returns):.2f} +/- {np.std(stats.recent_returns):.2f}")
        if len(stats.recent_lengths) > 0:
            print(f"Length:   {np.mean(stats.recent_lengths):.1f}")
        if len(stats.recent_successes) > 0:
            print(f"Success:  {np.mean(stats.recent_successes)*100:.1f}%")

        # PPO stats
        print(f"\nPPO Losses:")
        print(f"  Policy:  {update_metrics['loss/policy']:.4f}")
        print(f"  Value:   {update_metrics['loss/value']:.4f}")
        print(f"  Entropy: {update_metrics['loss/entropy']:.4f}")

        # Diagnostics
        print(f"\nDiagnostics:")
        print(f"  KL:         {update_metrics['stats/approx_kl']:.4f}")
        print(f"  Clip:       {update_metrics['stats/clip_frac']*100:.1f}%")
        print(f"  Explained:  {update_metrics['stats/explained_var']:.3f}")
        print(f"  LR:         {update_metrics['stats/learning_rate']:.2e}")
        print(f"  Std:        {update_metrics['stats/std']:.3f}")

        # Phase 2 metrics
        if len(stats.recent_attach_rate) > 0:
            print(f"\nPhase 2 Metrics:")
            print(f"  Attach:     {np.mean(stats.recent_attach_rate)*100:.1f}%")
        if len(stats.recent_lift_success) > 0:
            print(f"  Lift:       {np.mean(stats.recent_lift_success)*100:.1f}%")

        if eval_metrics:
            print(f"\nEvaluation:")
            print(f"  Success: {eval_metrics['success_rate']*100:.1f}%")
            print(f"  Return:  {eval_metrics['avg_return']:.2f}")

    def print_episode_end(
        self,
        info: Dict[str, Any],
        episode_return: float,
        episode_length: int,
        env_idx: int,
    ) -> None:
        """
        Print episode completion info for debugging.

        Args:
            info: Environment info dict
            episode_return: Total episode return
            episode_length: Episode length
            env_idx: Environment index
        """
        success_str = "+" if info.get("is_success", False) else "-"
        dist = info.get("dist", 0)
        attached = "[A]" if info.get("attached", False) else ""

        print(f"  Env {env_idx}: {success_str} R={episode_return:.2f} "
              f"L={episode_length} D={dist:.3f} {attached}")

    def close(self) -> None:
        """Close the logger and flush writers."""
        if self.writer is not None:
            self.writer.close()


def create_logger(run_dir: Path, config: dict) -> Logger:
    """
    Create Logger from config.

    Args:
        run_dir: Run output directory
        config: Experiment config

    Returns:
        Logger instance
    """
    return Logger(
        run_dir=run_dir,
        use_tensorboard=True,
        log_interval=config.get("experiment", {}).get("log_interval_updates", 10),
    )
