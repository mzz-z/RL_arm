"""Video recording utilities for training visualization."""

import numpy as np
from pathlib import Path
from typing import Optional, List
import time


class VideoRecorder:
    """
    Records environment frames to video files.

    Supports MP4 output using imageio or OpenCV.
    """

    def __init__(
        self,
        output_dir: Path,
        fps: int = 30,
        width: int = 640,
        height: int = 480,
    ):
        """
        Initialize video recorder.

        Args:
            output_dir: Directory to save videos
            fps: Frames per second
            width: Video width
            height: Video height
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.width = width
        self.height = height

        self.frames: List[np.ndarray] = []
        self.recording = False
        self.current_video_path: Optional[Path] = None

    def start_recording(self, filename: str) -> None:
        """
        Start recording a new video.

        Args:
            filename: Video filename (without extension)
        """
        self.frames = []
        self.recording = True
        self.current_video_path = self.output_dir / f"{filename}.mp4"

    def add_frame(self, frame: np.ndarray) -> None:
        """
        Add a frame to the current recording.

        Args:
            frame: RGB image array (H, W, 3)
        """
        if self.recording and frame is not None:
            # Ensure frame is the right size
            if frame.shape[:2] != (self.height, self.width):
                frame = self._resize_frame(frame)
            self.frames.append(frame)

    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame to target dimensions."""
        try:
            import cv2
            return cv2.resize(frame, (self.width, self.height))
        except ImportError:
            # Fallback: simple nearest-neighbor resize with numpy
            h, w = frame.shape[:2]
            y_indices = (np.arange(self.height) * h / self.height).astype(int)
            x_indices = (np.arange(self.width) * w / self.width).astype(int)
            return frame[y_indices][:, x_indices]

    def stop_recording(self) -> Optional[Path]:
        """
        Stop recording and save the video.

        Returns:
            Path to saved video, or None if no frames were recorded
        """
        if not self.recording or len(self.frames) == 0:
            self.recording = False
            return None

        self.recording = False
        video_path = self.current_video_path

        # Try to save with imageio first, then opencv
        try:
            self._save_with_imageio(video_path)
        except ImportError:
            try:
                self._save_with_opencv(video_path)
            except ImportError:
                print("Warning: Neither imageio nor opencv available. Saving as numpy array.")
                np_path = video_path.with_suffix('.npy')
                np.save(np_path, np.array(self.frames))
                return np_path

        self.frames = []
        return video_path

    def _save_with_imageio(self, path: Path) -> None:
        """Save video using imageio."""
        import imageio

        writer = imageio.get_writer(str(path), fps=self.fps, codec='libx264')
        for frame in self.frames:
            writer.append_data(frame)
        writer.close()
        print(f"Saved video: {path}")

    def _save_with_opencv(self, path: Path) -> None:
        """Save video using OpenCV."""
        import cv2

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(path), fourcc, self.fps, (self.width, self.height))

        for frame in self.frames:
            # Convert RGB to BGR for OpenCV
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.write(bgr_frame)

        writer.release()
        print(f"Saved video: {path}")

    def is_recording(self) -> bool:
        """Check if currently recording."""
        return self.recording

    @property
    def frame_count(self) -> int:
        """Number of frames recorded so far."""
        return len(self.frames)


def record_episode(
    env,
    policy_fn,
    video_path: Path,
    max_steps: int = 500,
    fps: int = 30,
    seed: Optional[int] = None,
) -> dict:
    """
    Record a single episode to video.

    Args:
        env: Environment with render_mode="rgb_array"
        policy_fn: Function that takes obs and returns action
        video_path: Output path for video
        max_steps: Maximum steps per episode
        fps: Video framerate
        seed: Random seed for reset

    Returns:
        Episode statistics dict
    """
    recorder = VideoRecorder(
        output_dir=video_path.parent,
        fps=fps,
    )
    recorder.start_recording(video_path.stem)

    # Reset environment
    if seed is not None:
        obs, info = env.reset(seed=seed)
    else:
        obs, info = env.reset()

    episode_return = 0.0
    episode_length = 0
    done = False

    while not done and episode_length < max_steps:
        # Record frame
        frame = env.render()
        recorder.add_frame(frame)

        # Get action
        action = policy_fn(obs)

        # Step
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        episode_return += reward
        episode_length += 1

    # Record final frame
    frame = env.render()
    recorder.add_frame(frame)

    # Save video
    saved_path = recorder.stop_recording()

    return {
        "video_path": saved_path,
        "episode_return": episode_return,
        "episode_length": episode_length,
        "is_success": info.get("is_success", False),
    }


def record_evaluation_videos(
    ppo,
    config: dict,
    output_dir: Path,
    num_episodes: int = 3,
    seeds: Optional[List[int]] = None,
    step: int = 0,
) -> List[dict]:
    """
    Record multiple evaluation episodes.

    Args:
        ppo: PPO agent
        config: Configuration dict
        output_dir: Directory to save videos
        num_episodes: Number of episodes to record
        seeds: Seeds for each episode
        step: Current training step (for filename)

    Returns:
        List of episode statistics
    """
    from train_eval.eval import make_eval_env

    # Create environment with rgb_array rendering
    env_config = config.get("env", {}).copy()
    control_config = config.get("control", {})
    reward_config = config.get("reward", {})

    from env.mujoco_env import MujocoArmEnv

    env = MujocoArmEnv(
        task_mode=env_config.get("task_mode", "reach"),
        max_episode_steps=env_config.get("max_episode_steps", 200),
        frame_skip=env_config.get("frame_skip", 4),
        render_mode="rgb_array",
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

    # Default seeds
    if seeds is None:
        seeds = list(range(200, 200 + num_episodes))

    # Record episodes
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for i in range(num_episodes):
        seed = seeds[i % len(seeds)]
        video_path = output_dir / f"eval_step{step}_ep{i}"

        def policy_fn(obs):
            action, _, _, _ = ppo.get_action(obs, deterministic=True)
            return action

        result = record_episode(
            env=env,
            policy_fn=policy_fn,
            video_path=video_path,
            seed=seed,
        )
        results.append(result)

        success_str = "+" if result["is_success"] else "-"
        print(f"  Recorded episode {i+1}/{num_episodes}: {success_str} "
              f"R={result['episode_return']:.2f} L={result['episode_length']}")

    env.close()
    return results
