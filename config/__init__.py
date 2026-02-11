"""Configuration loading utilities."""

import yaml
from pathlib import Path
from typing import Union


def deep_merge(base: dict, override: dict) -> dict:
    """
    Recursively merge override into base.

    Args:
        base: Base dictionary
        override: Override dictionary

    Returns:
        Merged dictionary
    """
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value

    return result


def load_config(config_path: Union[str, Path]) -> dict:
    """
    Load config with inheritance from default.yaml.

    Args:
        config_path: Path to task-specific config (reach.yaml or grasp.yaml)

    Returns:
        Merged config dict
    """
    config_path = Path(config_path)
    config_dir = config_path.parent

    # Load default config if it exists in the same directory
    default_path = config_dir / "default.yaml"
    if default_path.exists() and config_path.name != "default.yaml":
        with open(default_path) as f:
            config = yaml.safe_load(f)

        # Load and merge task-specific overrides
        with open(config_path) as f:
            overrides = yaml.safe_load(f)
        if overrides:
            config = deep_merge(config, overrides)
    else:
        # Standalone config (e.g., saved in a run directory) — load directly
        with open(config_path) as f:
            config = yaml.safe_load(f)

    return config


def validate_config(config: dict) -> None:
    """
    Validate config before training starts.

    Args:
        config: Configuration dictionary

    Raises:
        AssertionError: If validation fails
    """
    # Required keys exist
    assert config.get("env", {}).get("task_mode") is not None, "Missing env.task_mode"
    assert config.get("env", {}).get("max_episode_steps") is not None, "Missing env.max_episode_steps"
    assert config.get("control", {}).get("action_scale") is not None, "Missing control.action_scale"
    assert config.get("ppo", {}).get("lr") is not None, "Missing ppo.lr"
    assert config.get("ppo", {}).get("num_envs") is not None, "Missing ppo.num_envs"

    # Spawn range is reachable
    arm = config.get("env", {}).get("arm", {})
    max_reach = arm.get("link1_length", 0.25) + arm.get("link2_length", 0.25)
    spawn_max = config.get("env", {}).get("spawn", {}).get("radius_max", 0.4)
    assert spawn_max < max_reach, f"spawn.radius_max ({spawn_max}) >= arm reach ({max_reach})"

    # Reward weights match task mode
    task_mode = config["env"]["task_mode"]
    if task_mode == "reach":
        assert "reach" in config.get("reward", {}), "Missing reward.reach for reach task"
    elif task_mode == "grasp":
        assert "grasp" in config.get("reward", {}), "Missing reward.grasp for grasp task"

    # Minibatch size is reasonable
    ppo = config.get("ppo", {})
    rollout_size = ppo.get("rollout_steps", 2048) * ppo.get("num_envs", 8)
    minibatch = ppo.get("minibatch_size", 64)
    assert minibatch <= rollout_size, f"minibatch_size ({minibatch}) > rollout size ({rollout_size})"

    # LR schedule is valid
    lr_schedule = ppo.get("lr_schedule", "linear")
    assert lr_schedule in ["linear", "cosine", "constant"], f"Invalid lr_schedule: {lr_schedule}"

    print("Config validation passed")


def save_config(config: dict, path: Union[str, Path]) -> None:
    """
    Save config to YAML file.

    Args:
        config: Configuration dictionary
        path: Output path
    """
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
