# haptic_teleop_il_study

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
RealSense D435i ──▶  /rs_wrist/rs_wrist/color/image_raw

hdf5_data_collector  ──▶  episode_N.hdf5
hdf5_to_zarr         ──▶  output.zarr  (for diffusion policy training)
```

## Hardware

| Component | Details |
|---|---|
| Headset | HoloLens 2 (MRTK hand tracking) |
| Robot arm | Kinova Gen3 7-DoF |
| Front camera | ZED M (ZED Mini stereo) |
| Wrist camera | Intel RealSense D435i |
| Host machine | Ubuntu 24.04, ROS2 Jazzy, RTX 4090 |

## Prerequisites

### ROS2 packages
```bash
sudo apt install ros-jazzy-rosbridge-suite
sudo apt install ros-jazzy-zed-ros2-wrapper   # or build from source
sudo apt install ros-jazzy-realsense2-camera
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
    └── rs_wrist   (T, 3, H, W)  uint8  LZF-compressed, CHW
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
│   ├── rs_wrist_rgb     (N, H, W, 3)  uint8   HWC
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
