"""Curriculum learning manager for progressive difficulty."""

import warnings
from typing import Dict, List, Optional, Any
from gymnasium.vector import SyncVectorEnv, AsyncVectorEnv


class CurriculumManager:
    """
    Manages curriculum stages based on evaluation performance.

    Progressively increases task difficulty as the agent improves.
    """

    def __init__(
        self,
        stages: List[Dict[str, Any]],
        metric: str = "success_rate",
        patience: int = 3,
    ):
        """
        Initialize curriculum manager.

        Args:
            stages: List of stage configs, each with:
                - threshold: Performance needed to advance
                - spawn: Dict with spawn parameters to update
            metric: Which eval metric to track for advancement
            patience: Evals above threshold before advancing
        """
        self.stages = stages
        self.metric = metric
        self.patience = patience

        self.stage_idx = 0
        self.updates_above_threshold = 0

    @property
    def current_stage(self) -> Dict[str, Any]:
        """Get current curriculum stage config."""
        return self.stages[self.stage_idx]

    @property
    def is_final_stage(self) -> bool:
        """Check if we're at the final stage."""
        return self.stage_idx >= len(self.stages) - 1

    @property
    def num_stages(self) -> int:
        """Total number of curriculum stages."""
        return len(self.stages)

    def maybe_advance(self, eval_metrics: Dict[str, float]) -> bool:
        """
        Check if we should advance to next curriculum stage.

        Args:
            eval_metrics: Dictionary of evaluation metrics

        Returns:
            True if we advanced to a new stage
        """
        if self.is_final_stage:
            return False

        current_performance = eval_metrics.get(self.metric, 0)
        threshold = self.current_stage.get("threshold", 1.0)

        if current_performance >= threshold:
            self.updates_above_threshold += 1

            if self.updates_above_threshold >= self.patience:
                self.stage_idx += 1
                self.updates_above_threshold = 0
                print(f"\n>>> Curriculum advanced to stage {self.stage_idx + 1}/{len(self.stages)}")
                print(f"    Previous performance: {current_performance:.2%}")
                print(f"    New spawn range: {self.current_stage.get('spawn', {})}")
                return True
        else:
            # Reset patience counter if we drop below threshold
            self.updates_above_threshold = 0

        return False

    def get_spawn_params(self) -> Dict[str, Any]:
        """
        Get spawn parameters for current stage.

        Returns:
            Dictionary with spawn configuration
        """
        return self.current_stage.get("spawn", {})

    def state_dict(self) -> Dict[str, Any]:
        """Get curriculum state for checkpointing."""
        return {
            "stage_idx": self.stage_idx,
            "updates_above_threshold": self.updates_above_threshold,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Load curriculum state from checkpoint."""
        self.stage_idx = state.get("stage_idx", 0)
        self.updates_above_threshold = state.get("updates_above_threshold", 0)


def update_env_curriculum(vec_env, stage_config: Dict[str, Any]) -> None:
    """
    Update vectorized env with new curriculum parameters.

    Args:
        vec_env: Gymnasium vectorized environment (SyncVectorEnv or AsyncVectorEnv)
        stage_config: Curriculum stage configuration

    Note:
        For AsyncVectorEnv, this updates the envs attribute which contains
        the subprocess connections. The actual parameter updates happen
        through method calls that are serialized to the subprocesses.
    """
    spawn_config = stage_config.get("spawn", {})

    if not spawn_config:
        return

    # Get radius range from config
    radius_min = spawn_config.get("radius_min")
    radius_max = spawn_config.get("radius_max")

    if radius_min is not None and radius_max is not None:
        radius_range = (radius_min, radius_max)
    else:
        radius_range = None

    # Get angle range (optional)
    angle_min = spawn_config.get("angle_min")
    angle_max = spawn_config.get("angle_max")
    angle_range = (angle_min, angle_max) if angle_min is not None else None

    # Get azimuth range (optional)
    azimuth_min = spawn_config.get("azimuth_min")
    azimuth_max = spawn_config.get("azimuth_max")
    azimuth_range = (azimuth_min, azimuth_max) if azimuth_min is not None else None

    # Update each environment in the vectorized env
    if isinstance(vec_env, AsyncVectorEnv):
        # For AsyncVectorEnv, envs are in worker subprocesses
        # We can use call_async to invoke methods on all workers
        # However, this requires the method to exist and be callable
        # For now, warn that curriculum may not work perfectly with async envs
        warnings.warn(
            "Curriculum updates with AsyncVectorEnv may not take effect until env reset. "
            "Consider using async_envs: false in config if curriculum is critical.",
            RuntimeWarning,
        )
        # Try to update via call if the vec_env supports it
        try:
            # This attempts to call set_spawn_params on all envs
            vec_env.call("set_spawn_params",
                        radius_range=radius_range,
                        angle_range=angle_range,
                        azimuth_range=azimuth_range)
        except Exception:
            # If call doesn't work, the curriculum update will happen on next reset
            pass
    else:
        # SyncVectorEnv: direct access to envs
        for env in vec_env.envs:
            # Get the unwrapped environment
            unwrapped = env
            while hasattr(unwrapped, "env"):
                unwrapped = unwrapped.env

            # Update spawn parameters
            if hasattr(unwrapped, "set_spawn_params"):
                unwrapped.set_spawn_params(
                    radius_range=radius_range,
                    angle_range=angle_range,
                    azimuth_range=azimuth_range,
                )


def create_curriculum(config: dict) -> Optional[CurriculumManager]:
    """
    Create CurriculumManager from config.

    Args:
        config: Full config dict

    Returns:
        CurriculumManager if enabled, None otherwise
    """
    curriculum_config = config.get("curriculum", {})

    if not curriculum_config.get("enabled", False):
        return None

    stages = curriculum_config.get("stages", [])
    if not stages:
        print("Warning: Curriculum enabled but no stages defined")
        return None

    return CurriculumManager(
        stages=stages,
        metric=curriculum_config.get("metric", "success_rate"),
        patience=curriculum_config.get("patience", 3),
    )
