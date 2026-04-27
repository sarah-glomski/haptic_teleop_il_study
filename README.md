# Haptic Teleop IL — Diffusion Policy Pipeline

This document describes the full data collection → training → inference pipeline
for the HoloLens 2 + Kinova Gen3 + PieZense imitation learning study. 

---

## 1. System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        HARDWARE                                  │
│                                                                  │
│  HoloLens 2                Kinova Gen3 (7-DOF)                   │
│  (hand tracking)  ──────►  TCP pose + gripper control            │
│                                                                  │
│  ZED M (front)             DJI Osmo Action 4 (wrist)             │
│  (640×720 @ 30 Hz)         (USB V4L2 @ 30 Hz)                    │
│                                                                  │
│  Piezense (2-ch pressure sensor, system 0, channels 2 & 3)       │
└───────────────────────────┬─────────────────────────────────────┘
                            │ ROS2 (Jazzy)
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                   PIPELINE STAGES                                │
│                                                                  │
│  1. data_collection/   — teleop + synchronized recording         │
│  2. training/          — HDF5 → zarr conversion, training        │
│  3. testing/           — policy rollout on real robot            │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. ROS2 Topic Map

| Topic | Type | Publisher | Subscribers |
|---|---|---|---|
| `robot_obs/pose` | `PoseStamped` | kinova_state_publisher | hdf5_data_collector, inference |
| `robot_obs/gripper` | `Float32` | kinova_state_publisher | hdf5_data_collector, inference |
| `robot_obs/joint_states` | `JointState` | kinova_state_publisher | hdf5_data_collector |
| `robot_action/pose` | `PoseStamped` | kinova_hand_controller | hdf5_data_collector |
| `robot_action/gripper` | `Float32` | kinova_hand_controller | hdf5_data_collector |
| `hand/pose` | `PoseStamped` | hololens_hand_node | kinova_hand_controller, hdf5_data_collector |
| `hand/gripper_cmd` | `Float32` | hololens_hand_node | kinova_hand_controller |
| `/zed_front/zed_node/left/image_rect_color` | `Image` | zed_wrapper | hdf5_data_collector, inference |
| `/dji_wrist/dji_wrist/color/image_raw` | `Image` | dji_camera_node | hdf5_data_collector, inference |
| `piezense/data` | `PiezenseSystemArray` | piezense_driver | hdf5_data_collector, inference |

---

## 3. Data Collection

### 3.1 Launch

```bash
python3 data_collection/launch_data_collection.py \
    --robot-ip 192.168.1.10 \
    --zed-serial 17875187 \
    --dji-device 0
```

Pygame window controls: **R** Reset · **S** Start · **D** Done/Save · **P** Pause · **U** Unpause · **Q** Quit

### 3.2 Synchronization Architecture

`hdf5_data_collector.py` runs a 7-stream `ApproximateTimeSynchronizer` over:

- `robot_action/pose`, `robot_action/gripper`
- `robot_obs/pose`, `robot_obs/gripper`
- `hand/pose`
- `/zed_front/...`, `/dji_wrist/...`

**Slop:** 120 ms. This is intentionally wide because ZED, DJI, and Kinova poll on
independent clocks. Tighten if synchronization quality is a concern.

Side-channel data (updated asynchronously at each sync tick, not filtered):
`robot_obs/joint_states`, `hand/gripper_cmd`, `hand/hand_width`, `hand/finger_tips`,
`/hololens/palm/right`, `/hololens/thumb/right`, `/hololens/index/right`,
`/hololens/gaze`, `piezense/data`.

### 3.3 HDF5 Schema (per `episode_N.hdf5`)

```
episode_N.hdf5
├── action/
│   ├── pose:               (T, 7)   float32  [x, y, z, qx, qy, qz, qw]  commanded TCP
│   └── gripper:            (T,)     float32  0=open, 1=closed
├── observation/
│   ├── pose:               (T, 7)   float32  observed TCP
│   ├── gripper:            (T,)     float32
│   └── joint_states:       (T, 7)   float32  rad
├── hololens/
│   ├── palm_pose:          (T, 7)   float32  [xyz, qxyzw] Unity/ROS frame
│   ├── thumb_pose:         (T, 7)   float32
│   ├── index_pose:         (T, 7)   float32
│   ├── gaze_pose:          (T, 7)   float32
│   ├── finger_tips:        (T, 15)  float32  [thumb(3), index(3), ..., pinky(3)]
│   ├── hand_width:         (T,)     float32  thumb-index distance (m)
│   └── hand_pose_robot_frame: (T, 7) float32
├── piezense/
│   └── pressure_input:     (T, 2)   float32  Pa — channels 2 & 3 of system 0
└── images/
    ├── zed_front:          (T, 3, H, W) uint8  CHW, LZF-compressed
    └── dji_wrist:           (T, 3, H, W) uint8  CHW, LZF-compressed
```

**Collection rate:** ~30 Hz (set by ApproximateTimeSynchronizer throughput).
**Attributes:** `num_frames`, `collection_rate_hz`, `episode_index`.

---

## 4. Data Conversion

```bash
cd training/
python convert_data.py \
    --input ../data_collection/demo_data \
    --output kinova_teleop.zarr
```

### 4.1 Pose Representation: Rotation 6D

Quaternion poses `[x, y, z, qx, qy, qz, qw]` are converted to a 10D vector:

```
[x, y, z,  r1x, r1y, r1z, r2x, r2y, r2z,  gripper]
 ───────    ─────────────────────────────    ───────
  3D pos         rotation 6D (rot6d)         scalar
```

`rot6d` = first two columns of the rotation matrix (Zhou et al. 2019,
*"On the Continuity of Rotation Representations in Neural Networks"*).
This avoids discontinuities in quaternion/euler spaces and is network-friendly.

Conversion (scipy, no pytorch3d dependency):
```python
rot_mat = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()  # 3×3
rot6d   = [rot_mat[:, 0], rot_mat[:, 1]]                     # 6D
```

Inverse (for decoding policy actions in inference):
```python
c1 = rot6d[:3] / norm(rot6d[:3])
c2 = (rot6d[3:] - dot(rot6d[3:], c1)*c1); c2 /= norm(c2)
c3 = cross(c1, c2)
euler_xyz_deg = Rotation.from_matrix([c1,c2,c3]).as_euler('xyz', degrees=True)
```

**Gripper:** normalized scalar from Kortex API. `0.0` = fully open, `1.0` = fully closed.

### 4.2 Zarr Schema (per-episode groups)

```
kinova_teleop.zarr/
├── episode_0/
│   ├── zed_front_rgb:      (T, 224, 224, 3)  uint8   HWC
│   ├── dji_wrist_rgb:       (T, 224, 224, 3)  uint8   HWC
│   ├── pose:               (T, 10)            float32 [xyz, rot6d, gripper_obs]
│   ├── action:             (T, 10)            float32 [xyz, rot6d, gripper_cmd]
│   └── piezense_pressure:  (T, 2)             float32 Pa
├── episode_1/
...
```

Images are stored HWC (converted from CHW in HDF5) because the dataset loads them
into `(obs_horizon, H, W, 3)` tensors and permutes to CHW for the encoder.
Chunked at `(1, 224, 224, 3)` — one frame per chunk for efficient random access.

---

## 5. Training

### 5.1 Launch

```bash
cd training/

# Point to the zarr dataset (or set env var)
export KINOVA_ZARR_PATH=kinova_teleop.zarr

python train.py --config-name=train_diffusion_unet_timm_kinova

# Options
python train.py --config-name=train_diffusion_unet_timm_kinova training.debug=True
python train.py --config-name=train_diffusion_unet_timm_kinova logging.mode=disabled
python train.py --config-name=train_diffusion_unet_timm_kinova training.num_epochs=200
```

### 5.2 Policy Architecture

**`DiffusionUnetTimmPolicy`** from the UMI / dt_ag-main diffusion_policy codebase.

```
Observations ──► TimmObsEncoder ──► conditioning vector
                                          │
Noisy action ──► 1D-UNet (temporal) ─────┘──► denoised action chunk
```

#### Observation Encoder: TimmObsEncoder

| Key | Shape | Role |
|---|---|---|
| `zed_front_rgb` | `(B, 2, 3, 224, 224)` | front camera, obs_horizon=2 |
| `dji_wrist_rgb` | `(B, 2, 3, 224, 224)` | wrist camera, obs_horizon=2 |
| `pose` | `(B, 2, 10)` | TCP pose history, obs_horizon=2 |
| `piezense_pressure` | `(B, 2, 2)` | pressure history, obs_horizon=2 |

- **Image backbone:** `vit_base_patch16_clip_224.openai` (ViT-B/16, pretrained on CLIP)
- **Feature aggregation:** `attention_pool_2d` over patch tokens
- **Downsample ratio:** 32 (patch size)
- **`share_rgb_model: False`** — separate encoder weights per camera. Set `True` to halve
  encoder parameters if GPU memory is tight; loses camera-specific feature specialization.
- **`frozen: False`** — encoder fine-tuned end-to-end. Set `True` + `freeze_encoder: True`
  in training config to lock the backbone (faster, less overfitting on small datasets).

#### Diffusion UNet

- **Architecture:** 1D temporal UNet over the action sequence
- **`down_dims: [256, 512, 1024]`** — channel depths at each resolution level
- **`kernel_size: 5`** — temporal convolution kernel
- **`n_groups: 8`** — GroupNorm groups
- **`diffusion_step_embed_dim: 128`** — sinusoidal diffusion step embedding dim
- **`cond_predict_scale: True`** — FiLM conditioning (scale+shift) from observations

#### Noise Scheduler

**DDIM** (`DDIMScheduler`):

| Parameter | Value | Notes |
|---|---|---|
| `num_train_timesteps` | 50 | Number of diffusion steps during training |
| `num_inference_steps` | 16 | DDIM steps at rollout (skips; tunable) |
| `beta_schedule` | `squaredcos_cap_v2` | Cosine schedule; more stable than linear |
| `prediction_type` | `epsilon` | Predicts noise, not x0 |
| `clip_sample` | True | Clips denoised sample to [-1, 1] during inference |

**Latency impact:** Inference steps is the dominant GPU latency knob.
At 16 steps on an RTX 3090, expect ~80–150 ms per inference call.
Reducing to 8 steps roughly halves this with mild quality degradation.

### 5.3 Temporal Parameters

```
obs_horizon    = 2    frames of history fed as observation
action_horizon = 16   frames predicted per inference call
n_action_steps = 8    frames executed before re-querying the policy
horizon        = 16   sliding window length in dataset (max(obs_h, action_h))
```

The relationship follows the **receding horizon** / **action chunking** design from
Chi et al. (2023) *Diffusion Policy*:

```
Time:  t-1  t0  t1  t2  t3  t4  t5  t6  t7  ...  t15
       ──────────────────────────────────────────────────
Obs:   [   obs_horizon=2   ]
Act:              [────────── action_horizon=16 ──────────]
Exec:             [── n_action_steps=8 ──]
                                          ^ re-plan here
```

**Key tradeoffs:**
- **Larger `action_horizon`:** smoother motions, less reactive to disturbances
- **Smaller `n_action_steps`:** more responsive, more GPU inference calls
- **`n_action_steps = action_horizon`:** execute all predicted actions before re-planning
  (least reactive, most efficient GPU use)
- **`obs_horizon = 2`:** provides implicit velocity information to the policy;
  increasing to 3–4 adds more history but increases conditioning vector size

### 5.4 Training Hyperparameters

| Parameter | Value | Notes |
|---|---|---|
| `optimizer` | AdamW | |
| `lr` | 3e-4 | Standard for ViT fine-tuning |
| `betas` | [0.95, 0.999] | High β₁ for stable diffusion training |
| `weight_decay` | 1e-6 | |
| `lr_scheduler` | cosine | With `lr_warmup_steps=500` |
| `batch_size` | 1 | Increase if GPU memory allows; quality improves with larger batches |
| `num_epochs` | 120 | |
| `gradient_accumulate_every` | 1 | Effective batch = batch_size × this |
| `use_ema` | True | EMA of policy weights used for inference |
| `train_diffusion_n_samples` | 1 | Diffusion samples per training step (1 is standard) |
| `input_pertub` | 0.1 | Action input perturbation during training (regularization) |

**Augmentation (applied during training only):**
- `RandomCrop` at 0.95 ratio
- `ColorJitter`: brightness±0.3, contrast±0.4, saturation±0.5, hue±0.08

**Normalization:** `LinearNormalizer` with `mode='limits'` (min/max normalization)
applied to `pose`, `action`, and `piezense_pressure`. Image pixels are normalized
to [0,1] in the dataset `__getitem__` and then `imagenet_norm: True` in the encoder
applies ImageNet mean/std.

**Checkpointing:** Top-20 by `train_loss` (min) + last checkpoint always saved.
Outputs: `training/data/outputs/<date>/<time>_train_diffusion_unet_timm_kinova_kinova_teleop/`

**WandB:** Project `"Haptic Teleop IL Study"`, mode `online`. Set `logging.mode=disabled`
to skip.

---

## 6. Inference / Policy Rollout

### 6.1 Launch

```bash
# With explicit checkpoint:
python testing/launch_inference.py --model training/data/outputs/.../last.ckpt

# Auto-find latest checkpoint:
python testing/launch_inference.py --latest

# Key options:
#   --dt              0.033    action step period (s); default 30 Hz to match training
#   --action-horizon  6        actions executed per inference cycle
#   --diffusion-steps 16       DDIM steps (trade latency vs quality)
#   --robot-ip        192.168.1.10
#   --no-pygame              disable control window (headless)
```

Keyboard controls (pygame window): **S** Start · **D** Pause · **R** Reset home · **Q** Quit

### 6.2 Process Architecture

```
Main process (ROS2 PolicyNode)          GPU process (inference_loop)
──────────────────────────────          ────────────────────────────
30 Hz obs update timer                  Load policy from checkpoint
  ↓ pack shared_obs["obs"]    ──────►   Poll shared_obs["obs"] for new data
                                        Run policy.predict_action(obs)
30 Hz control timer           ◄──────   Put (action, timestamp) into action_queue
  ↓ pop action_queue
  ↓ compute P-loop velocity
  ↓ SendTwistCommand to Kinova

pygame thread
  ↓ S/D/R/Q keyboard events
  ↓ set shared_obs["paused"]
```

The GPU process is spawned via `multiprocessing.Process(start_method="spawn")` to
isolate CUDA context from the ROS2 spin thread.

### 6.3 Latency Chain

```
sensor capture          ──►  ~0 ms   (hardware)
ROS2 message delivery   ──►  ~5 ms   (loopback)
ApproxTimeSynchronizer  ──►  ~0–40 ms (slop buffer)
obs pack + IPC          ──►  ~2 ms
GPU inference (16 steps)──►  ~80–200 ms  ← dominant latency
action_queue drain      ──►  ~1 ms
P-loop velocity command ──►  ~2 ms
Kortex SDK delivery     ──►  ~5 ms
robot execution         ──►  ~33 ms  (1/30 Hz control tick)
```

**Total end-to-end:** ~130–280 ms from observation capture to robot motion.

The `action_exec_latency` parameter (not currently exposed via CLI, defaults implicitly
via `dt` scheduling) controls the minimum time between observation and the first executed
action. Actions timestamped in the past when they arrive are skipped.

### 6.4 Kinova Velocity Controller (SINGLE_LEVEL_SERVOING)

The policy outputs 10D **target poses** (not delta actions). These are tracked via a
30 Hz Cartesian P-loop sending `TwistCommand` in `SINGLE_LEVEL_SERVOING` mode:

```python
vel = P_GAIN * (target_xyz - current_xyz)        # P-loop
vel = clip(vel, MAX_LINEAR_SPEED)                 # 0.10 m/s hard cap
vel = VEL_ALPHA * vel + (1-VEL_ALPHA) * prev_vel  # EWA smoothing (α=0.4)
ang_vel_z = P_GAIN * wrap(target_tz - current_tz) # θ_z P-loop only
```

**Watchdog:** `duration = 200 ms` on every `TwistCommand`. If the control loop
misses ~6 ticks (node crash/freeze), the Kortex SDK stops the robot automatically.

**Workspace bounds** (hard-clipped before sending to controller):

| Axis | Min (m) | Max (m) |
|---|---|---|
| X | 0.40 | 0.50 |
| Y | -0.27 | 0.27 |
| Z | 0.025 | 0.30 |

**Home position:** `(0.44, 0.00, 0.43)` m, `(180°, 0°, 0°)` Euler XYZ.

### 6.5 Piezense at Inference

Piezense (`piezense/data`) is subscribed as a side-channel (not in the synchronized
filter). If the driver is not running, `piezense_pressure` defaults to zeros, which
the policy receives as a valid (if uninformative) observation. The piezense driver
must be running for tactile feedback to inform the policy.

---

## 7. Key Parameters for Tuning

### Latency / Responsiveness

| Parameter | Where | Effect |
|---|---|---|
| `--diffusion-steps` | inference CLI | GPU inference time; 8 ≈ 50 ms, 16 ≈ 120 ms, 32 ≈ 250 ms |
| `--action-horizon` | inference CLI | Actions executed per cycle; fewer = more reactive |
| `--dt` | inference CLI | Action step duration; should match training data collection rate (0.033 s = 30 Hz) |
| `P_GAIN` (inference.py:95) | hardcoded | Velocity controller gain; too high → overshoot/oscillation |
| `VEL_ALPHA` (inference.py:97) | hardcoded | Smoothing; lower = more responsive, higher = smoother |
| `MAX_LINEAR_SPEED` (inference.py:94) | hardcoded | Hard velocity cap (m/s) |

### Training Quality

| Parameter | Where | Effect |
|---|---|---|
| `obs_horizon` | task yaml | Temporal context (2 = velocity implicit); increasing > 2 rarely helps |
| `action_horizon` | task yaml | Must retrain if changed; determines chunk length in zarr |
| `batch_size` | training yaml | 1 is very small; try 4–8 if VRAM allows |
| `num_epochs` | training yaml | Monitor val_loss; 120 may be insufficient for < 50 demos |
| `frozen: False` | training yaml | Fine-tune backbone; consider `frozen: True` + more epochs for small datasets |
| `share_rgb_model` | training yaml | False = separate encoders per camera (recommended) |
| `lr_warmup_steps` | training yaml | 500 steps; reduce if dataset is small (< 1000 total frames) |
| `val_ratio` | task yaml | 0.05 = 5% held out; with < 20 episodes consider 0.1 |
| `slop` | hdf5_data_collector.py | Synchronizer tolerance; 120 ms is wide — effects data alignment quality |

### Data Collection Quality

| Parameter | Where | Effect |
|---|---|---|
| `slop=0.12` | hdf5_data_collector.py:136 | ApproxTimeSync tolerance; tighter = better alignment, more dropped frames |
| `IMG_SIZE=224` | convert_data.py | Resize target; matches ViT-B/16 input — do not change without adjusting encoder |
| `COLLECTIONS` | (N/A) | Dataset split/grouping — managed via `--input` directory |

---

## 8. File Map

```
haptic_teleop_il_study/
├── data_collection/
│   ├── launch_data_collection.py    — full pipeline launch
│   ├── hdf5_data_collector.py       — synchronized recorder (pygame UI)
│   ├── kinova_state_publisher.py    — Kortex → ROS2 state (read-only)
│   ├── kinova_hand_controller.py    — HoloLens → Kortex velocity control
│   ├── hololens_hand_node.py        — hand joint → hand/pose, hand/gripper_cmd
│   ├── hololens_tf_publisher_ros2.py— HoloLens joints → TF2
│   ├── hdf5_to_zarr.py              — HDF5 → flat (UMI-style) zarr (alt format)
│   ├── dji_camera_node.py           — DJI Osmo → ROS2 Image
│   ├── dji_camera_validate.py       — verify DJI device index
│   ├── launch_teleop.py             — teleoperation only (no recording)
│   └── preflight_check.py           — topic/sensor health check
│
├── training/
│   ├── convert_data.py              — HDF5 → per-episode zarr (for training)
│   ├── kinova_dataset.py            — PyTorch Dataset class (KinovaImageDataset)
│   ├── train.py                     — Hydra entry point
│   └── config/
│       ├── train_diffusion_unet_timm_kinova.yaml  — main training config
│       └── task/kinova_teleop.yaml                — shape_meta, dataset path
│
├── testing/
│   ├── inference.py                 — policy rollout node (ROS2 + GPU process)
│   └── launch_inference.py          — full inference pipeline launch
│
└── PIPELINE.md                      — this document
```

**Note on `hdf5_to_zarr.py`:** This creates a *flat* (UMI-style ReplayBuffer) zarr
format with `data/` and `meta/episode_ends` — compatible with UMI training
infrastructure. The `training/convert_data.py` creates a *per-episode* zarr
format (compatible with `KinovaImageDataset`). Both represent the same data;
choose based on which training infrastructure you want to use.

---

## 9. Dependencies

```
# ROS2 packages
ros-jazzy-rosbridge-suite
ros-jazzy-v4l2-camera         # optional; dji_camera_node.py uses OpenCV directly
zed-ros2-wrapper               # ZED M camera

# Python
kortex-api        # Kinova Kortex Python SDK
zarr              # zarr storage
h5py              # HDF5 reading
natsort           # natural-sort episode files
scipy             # rotation conversion (no pytorch3d required)
cv2               # image resize
pygame            # control window
dill              # checkpoint loading
hydra-core        # config management
omegaconf
diffusers         # DDIMScheduler
torch torchvision

# UMI / diffusion_policy codebase (shared with Robomimic):
# Robomimic/dt_ag-main/universal_manipulation_interface/
#   — diffusion_policy.*  (policy, workspace, dataset, model)
#   — dt_ag.rotation_transformer  (used by Robomimic scripts; not used here)
```
