# RL Arm

A reinforcement learning system that trains a 4-DOF robotic arm to reach target positions and perform pick-and-place tasks in a simulated environment.

The arm is simulated in MuJoCo and trained using Proximal Policy Optimization (PPO). Training is split into two phases: first the arm learns to reach arbitrary targets in 3D space, then it transfers that knowledge to pick up a ball from one platform and place it on another.

## How It Works

The arm has four joints (base rotation, shoulder, elbow, wrist) and is controlled through position targets. At each step the policy outputs a 4-dimensional action representing joint target deltas, which a PD controller tracks. The policy is a two-layer MLP with a tanh-squashed Gaussian output distribution.

**Phase 1 (Reach):** The arm learns to move its end-effector to a ball spawned at random positions within its workspace. A curriculum gradually increases the spawn range and tightens the success radius as the agent improves. A Jacobian-transpose guide provides initial demonstrations through behavioral cloning warmup and residual assistance that anneals over training.

**Phase 2 (Pick-and-Place):** The reach policy is transferred via weight loading. The arm must now approach the ball, trigger a magnetic attachment when close enough, transport it to a destination platform, and hold it there. The reward switches from distance-to-ball shaping (before attachment) to distance-to-destination shaping (after attachment).

## Project Structure

```
RL_arm/
  assets/
    arm.xml              MuJoCo model (arm, table, ball, platforms, lighting)
  config/
    default.yaml          Base configuration (all parameters with defaults)
    reach.yaml            Phase 1 overrides (reward, curriculum, guide)
    grasp.yaml            Phase 2 overrides (place params, curriculum)
  env/
    mujoco_env.py         Gymnasium environment wrapping MuJoCo
    observations.py       24-dim observation vector construction
    rewards.py            Reward computation for both task modes
    task_reach.py         Reach task logic (distance check, dwell timer)
    task_grasp.py         Pick-and-place task logic (attach, transport, place)
    validation.py         Model sanity checks on load
  control/
    controllers.py        PD position-target controller
    action_space.py       Action scaling and joint limit enforcement
  rl/
    ppo.py                PPO algorithm with clipping and GAE
    networks.py           Actor-critic MLP with state-independent log_std
    buffer.py             Rollout buffer with GAE computation
    normalizer.py         Running mean/variance observation normalizer
  train_eval/
    train.py              Main training loop (vectorized envs, curriculum, logging)
    eval.py               Evaluation runner
    curriculum.py         Staged curriculum manager
    logger.py             TensorBoard + console logging
    video.py              Video recording utilities
  scripts/
    train_reach.py        Convenience script for phase 1
    train_grasp.py        Convenience script for phase 2
    evaluate.py           Evaluate a saved checkpoint
    visualize.py          Real-time visualization with MuJoCo viewer
    record_video.py       Record evaluation videos to MP4
```

## Requirements

- Python 3.10+
- PyTorch
- MuJoCo (via `mujoco` Python bindings)
- Gymnasium
- NumPy
- TensorBoard (optional, for logging)
- imageio or OpenCV (optional, for video recording)

```
pip install torch mujoco gymnasium numpy tensorboard imageio
```

## Usage

### Train the reach task (Phase 1)

```
python scripts/train_reach.py
```

This uses `config/reach.yaml`. Training progress logs to TensorBoard under `runs/<run_name>/tensorboard/`. Checkpoints are saved periodically and the best model is tracked by evaluation success rate.

To resume from a checkpoint:

```
python scripts/train_reach.py --resume runs/<run_name>/checkpoint_latest.pt
```

### Train pick-and-place (Phase 2)

Transfer the reach policy weights into the grasp task:

```
python scripts/train_grasp.py --transfer runs/<reach_run>/checkpoint_best.pt
```

This loads the trained reach weights and observation normalizer, then trains with the grasp reward and curriculum defined in `config/grasp.yaml`.

### Evaluate a checkpoint

```
python scripts/evaluate.py --checkpoint runs/<run_name>/checkpoint_best.pt
```

### Visualize in real-time

```
python scripts/visualize.py --checkpoint runs/<run_name>/checkpoint_best.pt
```

Flags:
- `--episodes N` to set episode count
- `--success-only` to pre-screen and only display successful episodes
- `--random` to watch a random policy (useful for testing the environment)

### Record videos

```
python scripts/record_video.py --checkpoint runs/<run_name>/checkpoint_best.pt --output videos/
```

### Use the general training script directly

For full control over config paths and options:

```
python train_eval/train.py --config config/reach.yaml
python train_eval/train.py --config config/grasp.yaml --transfer runs/<run>/checkpoint_best.pt
```

### Monitor training

```
tensorboard --logdir runs/
```

## Configuration

All parameters are defined in `config/default.yaml` with sensible defaults. Phase-specific configs (`reach.yaml`, `grasp.yaml`) override only what they need to change.

Key parameters worth adjusting:

| Parameter | File | Description |
|---|---|---|
| `ppo.num_envs` | default.yaml | Parallel environments (higher = faster, more memory) |
| `ppo.rollout_steps` | default.yaml | Steps per env per update |
| `experiment.total_env_steps` | default.yaml | Total training budget |
| `reward.reach.proximity_alpha` | reach.yaml | Steepness of proximity reward near target |
| `curriculum.stages` | reach.yaml | Spawn ranges and thresholds per stage |
| `env.guide.alpha_initial` | reach.yaml | Jacobian guide strength at start of training |
| `reward.grasp.attach_bonus` | grasp.yaml | One-time bonus for grasping the ball |
| `reward.grasp.place_bonus` | grasp.yaml | One-time bonus for reaching the destination |

## Observation Space (24 dimensions)

| Index | Description |
|---|---|
| 0-3 | Joint positions (base, shoulder, elbow, wrist) |
| 4-7 | Joint velocities |
| 8-10 | End-effector position (x, y, z) |
| 11-13 | Vector from end-effector to ball |
| 14 | Distance to ball (scalar) |
| 15-17 | Ball position (x, y, z) |
| 18-19 | Sin/cos of base joint angle |
| 20 | Grasp flag (1.0 if ball attached, 0.0 otherwise) |
| 21-23 | Destination position (zeros during reach) |

## Action Space

4 continuous values in [-1, 1], representing target joint angle deltas scaled by `control.action_scale` (default 0.15 radians per step). The PD controller tracks these targets.
