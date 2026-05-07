# Haptic Teleop IL Study

ROS2 data collection pipeline for robotic imitation learning.  
A HoloLens 2 (hand tracking) teleoperates a Kinova Gen3 arm while synchronized  
demonstrations are recorded to HDF5 and converted to Zarr for diffusion policy training.

## System Overview

```
HoloLens 2  ──rosbridge──▶  hololens_tf_publisher
                        ──▶  hololens_hand_node ──▶  hand/pose
                                                 ──▶  hand/gripper_cmd
                                                 ──▶  hand/tracking_active

Kinova Gen3 ◀── kinova_hand_controller ◀── hand/pose
            ──▶  kinova_state_publisher ──▶  robot_obs/pose

ZED M (front)   ──▶  /zed_front/zed_node/left/image_rect_color
DJI Osmo Action 4 ──▶  /dji_wrist/dji_wrist/color/image_raw

hdf5_data_collector  ──▶  episode_N.hdf5
hdf5_to_zarr         ──▶  output.zarr  (for diffusion policy training)
```

## Hardware

| Component | Details |
|---|---|
| Headset | HoloLens 2 (MRTK hand tracking) |
| Robot arm | Kinova Gen3 7-DoF |
| Front camera | ZED M (ZED Mini stereo) |
| Wrist camera | DJI Osmo Action 4 (USB-C UVC mode) |
| Host machine | Ubuntu 24.04, ROS2 Jazzy, RTX 4090 |

## Prerequisites

### ROS2 packages
```bash
sudo apt install ros-jazzy-rosbridge-suite
sudo apt install ros-jazzy-zed-ros2-wrapper   # or build from source
sudo apt install ros-jazzy-v4l2-camera  # or use dji_camera_node.py (bundled)
```

### Kinova Kortex API
```bash
# Install the bundled wheel into system Python 3.12 (used by all ROS2 nodes)
/usr/bin/python3.12 -m pip install --break-system-packages \
    kortex_api-2.6.0.post3-py3-none-any.whl \
    'protobuf>=3.20.0,<5' pygame opencv-python h5py zarr natsort scipy
```

### Python environment for data post-processing and training
Create the `umi` conda environment from the UMI codebase:
```bash
mamba env create -f universal_manipulation_interface/conda_environment.yaml
conda activate umi
pip install 'protobuf>=3.15.0,<5' h5py natsort pygame pytorch3d
```

> **Important**: ROS2 nodes must run with system Python 3.12 (`/usr/bin/python3.12`),  
> not a conda environment. Always `conda deactivate` before launching.

## Running Teleop

```bash
conda deactivate
source /opt/ros/jazzy/setup.bash
cd haptic_teleop_il_study/
/usr/bin/python3.12 launch_teleop.py --robot-ip 192.168.1.10
```

HoloLens startup checklist:
- Robot powered on, e-stop released
- Host machine and HoloLens on the same WiFi
- In HoloLens app: set RosConnector URL to `ws://<HOST_IP>:9090`
- Press **Arm** to enable wrist tracking
- Press **Gripper** to enable thumb-index gripper control

> **Wrist orientation settling (important)**: After you press **Arm**, the robot stays
> completely still for **0.5 s** while MRTK hand tracking stabilises. MRTK can return
> a valid-looking but identity quaternion for the first 1–2 frames of hand acquisition;
> capturing that as the reference causes incorrect wrist commands. After settling, the
> controller snapshots your actual palm yaw and the robot's current `theta_z`. All
> subsequent wrist rotation is tracked as a *delta* from that reference.
> Hold your hand in your intended neutral pose (palm facing down) **before** pressing **Arm**
> and hold it still for the 0.5 s settling window. The log line
> `Wrist reference captured (after settling): palm_yaw=X° robot_theta_z=Y° holo_pos=(x, y, z)`
> confirms the snapshot. If `palm_yaw` is exactly `0.0°` it means identity was still captured —
> try again. If the robot drifts rotationally after a tracking dropout, re-press **Arm**.

> **Robot fault recovery**: If the arm enters a fault state (joint limit, collision, or
> e-stop), the controller immediately disables arm motion and attempts `ClearFaults()` +
> re-enters servoing mode every 3 s automatically. You will see:
> `ROBOT FAULT DETECTED — arm motion disabled.`  then  `Robot fault cleared — re-entering servoing mode.`
> After the fault clears, **re-press Arm** in the HoloLens app before motion resumes.
> If faults recur, reduce `max_angular_speed_dps` or `max_linear_speed_mps` via `ros2 param set`.

> **Z axis not moving — workspace calibration**: The z axis is the vertical (up/down) axis.
> If height movement isn't reflected on the robot, the most likely cause is that
> `workspace_z_offset` (default 0.2 m) doesn't map your hand's natural height into the
> robot workspace [0.025, 0.30 m]. Calibrate by running (robot NOT enabled):
> `ros2 topic echo /hololens/palm/right` — note the `position.z` value at your neutral
> hand height. Set `workspace_z_offset = home_z (0.107) − holo_z_at_neutral`.
> Also move your hand up/down and verify `position.z` changes (not `position.x`).
> The `holo_pos` values in the settling log confirm this each session.

Stop with `Ctrl-C` — the watchdog halts the robot within 200 ms.

## Preflight Check

Run in a second terminal while `launch_teleop.py` is running to verify all  
data streams before the robot responds to hand commands:

```bash
source /opt/ros/jazzy/setup.bash
/usr/bin/python3.12 preflight_check.py
```

USB + robot ping only (no ROS needed):
```bash
/usr/bin/python3.12 preflight_check.py --usb-only
```

The check reports:
- ZED M USB connection
- Kinova TCP reachability
- Hz + live values for every ROS topic
- Whether `hand/tracking_active` is True (Arm button pressed)
- Current hand pose vs. workspace bounds

## Running Data Collection

```bash
conda deactivate
source /opt/ros/jazzy/setup.bash
/usr/bin/python3.12 launch_data_collection.py \
    --robot-ip 192.168.1.10 \
    --zed-serial <ZED_SERIAL> \
    --rs-wrist <REALSENSE_SERIAL>
```

Keyboard controls (pygame window):

| Key | Action |
|-----|--------|
| R | Reset robot to home position |
| S | Start recording episode |
| D | Done — end episode and save HDF5 |
| P | Pause |
| U | Unpause |
| Q | Quit |

Episodes are saved to `demo_data/episode_N.hdf5`.

## HDF5 Schema

```
episode_N.hdf5
├── action/
│   ├── pose       (T, 7)  float32  [x, y, z, qx, qy, qz, qw]
│   └── gripper    (T,)    float32  [0–1]
├── observation/
│   ├── pose       (T, 7)  float32
│   ├── gripper    (T,)    float32
│   └── joint_states (T, 7) float32  [rad]
├── hololens/
│   ├── palm_pose       (T, 7)   float32
│   ├── thumb_pose      (T, 7)   float32
│   ├── index_pose      (T, 7)   float32
│   ├── gaze_pose       (T, 7)   float32
│   ├── finger_tips     (T, 15)  float32
│   └── hand_width      (T,)     float32
└── images/
    ├── zed_front  (T, 3, H, W)  uint8  LZF-compressed, CHW
    └── dji_wrist   (T, 3, H, W)  uint8  LZF-compressed, CHW
```

Metadata: `num_frames`, `collection_rate_hz=30`, `episode_index`

## Converting to Zarr

After collecting episodes:
```bash
conda activate umi
python hdf5_to_zarr.py demo_data/ output.zarr
```

Zarr output (UMI-style flat concatenation):
```
output.zarr/
├── data/
│   ├── zed_front_rgb    (N, H, W, 3)  uint8   HWC
│   ├── dji_wrist_rgb     (N, H, W, 3)  uint8   HWC
│   ├── pose             (N, 10)       float32  [x,y,z, rot6d(6), gripper]
│   ├── action           (N, 10)       float32
│   ├── joint_states     (N, 7)        float32
│   ├── holo_palm_pose   (N, 7)        float32
│   ├── holo_hand_width  (N,)          float32
│   └── holo_finger_tips (N, 15)       float32
└── meta/
    └── episode_ends     (num_episodes,) int64
```

## Workspace Calibration

Edit these parameters in `kinova_hand_controller.py` to match your setup:

```python
workspace_x_min = 0.40   # metres from robot base
workspace_x_max = 0.50
workspace_y_min = -0.27
workspace_y_max =  0.27
workspace_z_min = 0.025
workspace_z_max = 0.30
```

Edit these in `hololens_hand_node.py` to map HoloLens world space → robot space:

```python
workspace_x_offset = 0.0   # metres added to scaled HoloLens position
workspace_y_offset = 0.0
workspace_z_offset = 0.0
workspace_x_scale  = 1.0   # multiplier
workspace_y_scale  = 1.0
workspace_z_scale  = 1.0
```

## File Reference

| File | Purpose |
|---|---|
| `launch_teleop.py` | Minimal launch: rosbridge + HoloLens nodes + Kinova controller |
| `launch_data_collection.py` | Full launch: adds cameras + HDF5 recorder |
| `preflight_check.py` | Verify USB devices, robot ping, and topic streams |
| `hololens_tf_publisher_ros2.py` | Broadcasts HoloLens joint TF transforms |
| `hololens_hand_node.py` | Converts raw joints → `hand/pose`, `hand/gripper_cmd` |
| `kinova_state_publisher.py` | Reads Kinova state → `robot_obs/*` at 30 Hz |
| `kinova_hand_controller.py` | P-loop velocity controller with workspace safety limits |
| `hdf5_data_collector.py` | 7-stream synchronized recording with pygame UI |
| `hdf5_to_zarr.py` | Converts HDF5 episodes to UMI-style Zarr for training |
