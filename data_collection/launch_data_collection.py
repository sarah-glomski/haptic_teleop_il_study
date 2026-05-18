#!/usr/bin/python3.12
"""
Launch file — Haptic Teleop IL Study data collection system

Launches the full pipeline:
  1. rosbridge_websocket   — WebSocket bridge so the HoloLens Unity app can talk to ROS2
  2. hololens_tf_publisher — Broadcasts HoloLens joint poses to the TF2 tree
  3. hololens_hand_node    — Converts HoloLens joints → hand/pose, hand/gripper_cmd, …
  4. kinova_state_publisher— Reads Kinova Gen3 state → robot_obs/*
  5. kinova_hand_controller— Hand-tracking → Kinova velocity commands → robot_action/*
  6. hdf5_data_collector   — Synchronized data recording with pygame UI
  7. ZED M camera          — /zed_isometric/zed_node/left/image_rect_color
  8. DJI Osmo Action 4      — /dji_wrist/dji_wrist/color/image_raw (wrist camera)

Prerequisites (install once):
  sudo apt install ros-$ROS_DISTRO-rosbridge-suite
  sudo apt install ros-$ROS_DISTRO-zed-ros2-wrapper   # or build from source
  pip install kortex-api natsort zarr h5py pygame opencv-python scipy

Usage:
  python3 launch_data_collection.py [options]

  --robot-ip       ROBOT_IP       Kinova Gen3 IP (default 192.168.1.10)
  --zed-serial     SERIAL_NO      ZED M serial number as string (default '')
  --dji-device     N              V4L2 device index for DJI camera (default 0)

Keyboard controls (in the pygame window):
  R - Reset robot to home
  S - Start recording episode
  D - Done / end recording and save HDF5
  P - Pause
  U - Unpause
  Q - Quit
"""

import argparse
import os
import subprocess
import sys

from launch import LaunchDescription, LaunchService
from launch.actions import ExecuteProcess
from launch_ros.actions import Node


_PYTHON = '/usr/bin/python3.12'


ZED_SERIAL = '17875187'


def generate_launch_description(
    robot_ip: str = '192.168.1.10',
    zed_serial: str = ZED_SERIAL,
    dji_device: int = 0,
    no_zed: bool = False,
    no_cameras: bool = False,
    no_piezense: bool = False,
    no_rosbridge: bool = False,
) -> LaunchDescription:

    script_dir = os.path.dirname(os.path.abspath(__file__))

    def script(name):
        return os.path.join(script_dir, name)

    return LaunchDescription([

        # ── 1. rosbridge WebSocket server ─────────────────────────────────────
        # The HoloLens Unity app connects to ws://<host-ip>:9090.
        # Skip with --no-rosbridge when rosbridge is already running so the
        # HoloLens stays connected across pipeline restarts.
        # Force system python3 in PATH so the #!/usr/bin/env python3 shebang
        # doesn't resolve to miniforge Python 3.13 (which lacks rclpy C extensions).
        *([Node(
            package='rosbridge_server',
            executable='rosbridge_websocket',
            name='rosbridge_websocket',
            output='screen',
            parameters=[{
                'port': 9090,
                'address': '',
            }],
            additional_env={
                'PATH': ':'.join(
                    p for p in os.environ.get('PATH', '').split(':')
                    if 'miniforge' not in p and 'conda' not in p
                ),
            },
        )] if not no_rosbridge else []),

        # ── 2. HoloLens TF publisher ──────────────────────────────────────────
        ExecuteProcess(
            cmd=[_PYTHON, script('hololens_tf_publisher_ros2.py')],
            name='hololens_tf_publisher',
            output='screen',
        ),

        # ── 3. HoloLens hand processing node ─────────────────────────────────
        ExecuteProcess(
            cmd=[_PYTHON, script('hololens_hand_node.py')],
            name='hololens_hand_node',
            output='screen',
        ),

        # ── 4. Kinova Gen3 state publisher ────────────────────────────────────
        ExecuteProcess(
            cmd=[
                _PYTHON, script('kinova_state_publisher.py'),
                '--ros-args', '-p', f'robot_ip:={robot_ip}',
            ],
            name='kinova_state_publisher',
            output='screen',
        ),

        # ── 5. Kinova Gen3 hand controller ────────────────────────────────────
        ExecuteProcess(
            cmd=[
                _PYTHON, script('kinova_hand_controller.py'),
                '--ros-args', '-p', f'robot_ip:={robot_ip}',
            ],
            name='kinova_hand_controller',
            output='screen',
        ),

        # ── 6. Piezense pressure sensor controller ────────────────────────────
        *([ExecuteProcess(
            cmd=['ros2', 'launch', 'piezense_ros', 'ar_teleop_piezense_launch.py'],
            name='piezense_driver',
            output='screen',
        )] if not no_piezense else []),

        # ── 7. HDF5 data collector (pygame UI runs here) ──────────────────────
        ExecuteProcess(
            cmd=[
                _PYTHON, script('hdf5_data_collector.py'),
                '--ros-args',
                '-p', f'enable_zed:={str(not (no_zed or no_cameras)).lower()}',
                '-p', f'enable_dji:={str(not no_cameras).lower()}',
                '-p', f'enable_piezense:={str(not no_piezense).lower()}',
            ],
            name='hdf5_data_collector',
            output='screen',
        ),

        # ── 7. ZED M camera — front view ──────────────────────────────────────
        *([Node(
            package='zed_wrapper',
            executable='zed_wrapper',
            name='zed_node',
            namespace='zed_isometric',
            output='screen',
            parameters=[{
                'camera_model':          'zedm',          # ZED Mini / ZED M
                'camera_name':           'zed_isometric',
                'serial_number':         int(zed_serial) if zed_serial else 0,
                'grab_resolution':       'HD720',          # 1280×720
                'grab_frame_rate':       30,
                'pub_frame_rate':        30.0,
                'general.grab_frame_rate': 30,
                # Enable only what we need
                'depth.depth_mode':      1,               # PERFORMANCE
                'video.extrinsic_in_camera_frame': False,
            }],
        )] if not no_zed else []),

        # ── 8. DJI Osmo Action 4 — wrist-mounted camera ──────────────────────
        *([ExecuteProcess(
            cmd=[
                _PYTHON, script('dji_camera_node.py'),
                '--ros-args',
                '-p', f'device_index:={dji_device}',
                '-r', '/wrist_cam/image_raw:=/dji_wrist/dji_wrist/color/image_raw',
            ],
            name='dji_wrist_camera',
            output='screen',
        )] if not no_cameras else []),
    ])


def main(argv=sys.argv[1:]):
    parser = argparse.ArgumentParser(
        description='Launch Haptic Teleop IL data collection pipeline',
    )
    parser.add_argument('--robot-ip',   default='192.168.1.10',
                        help='Kinova Gen3 IP address (default: 192.168.1.10)')
    parser.add_argument('--zed-serial', default=ZED_SERIAL,
                        help=f'ZED M serial number (default: {ZED_SERIAL})')
    parser.add_argument('--dji-device', type=int, default=0,
                        help='V4L2 device index for DJI Osmo Action 4 (default: 0). '
                             'Run dji_camera_validate.py first to confirm.')
    parser.add_argument('--no-zed', action='store_true',
                        help='Skip launching the ZED M camera node (e.g. if ZED SDK is not installed).')
    parser.add_argument('--no-cameras', action='store_true',
                        help='Skip all camera nodes and disable camera sync in the data collector. '
                             'Implies --no-zed. Use when cameras are unavailable.')
    parser.add_argument('--no-piezense', action='store_true',
                        help='Skip piezense driver and disable piezense recording.')
    parser.add_argument('--no-rosbridge', action='store_true',
                        help='Skip launching rosbridge (use when it is already running '
                             'so the HoloLens stays connected across pipeline restarts).')
    args, launch_argv = parser.parse_known_args(argv)

    print('=' * 60)
    print('Haptic Teleop IL — Data Collection System')
    print('=' * 60)
    print(f'  Robot IP:       {args.robot_ip}')
    print(f'  ZED serial:     {args.zed_serial or "(auto-detect first found)"}')
    print(f'  DJI wrist cam:  /dev/video{args.dji_device}')
    print()
    print('HoloLens:')
    print('  Make sure the HoloLens app is pointed at ws://<this-machine-ip>:9090')
    print()
    print('Keyboard controls (pygame window):')
    print('  R - Reset robot   S - Start   D - Done/Save')
    print('  P - Pause         U - Unpause Q - Quit')
    print('=' * 60)

    # Kill any stale piezense driver processes from previous launches.
    if not args.no_piezense:
        result = subprocess.run(['pkill', '-f', 'piezense_driver'],
                                stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        if result.returncode == 0:
            print('Killed stale piezense_driver process(es)')

    # Auto-detect a running rosbridge so we don't kill it.
    # If --no-rosbridge was not explicitly passed but something is already on
    # port 9090, treat it as an intentional persistent rosbridge and leave it alone.
    if not args.no_rosbridge:
        probe = subprocess.run(['fuser', '9090/tcp'],
                               stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        if probe.returncode == 0:
            print('Rosbridge already running on port 9090 — skipping launch (HoloLens connection preserved)')
            args.no_rosbridge = True

    ld = generate_launch_description(
        robot_ip=args.robot_ip,
        zed_serial=args.zed_serial,
        dji_device=args.dji_device,
        no_zed=args.no_zed or args.no_cameras,
        no_cameras=args.no_cameras,
        no_piezense=args.no_piezense,
        no_rosbridge=args.no_rosbridge,
    )
    ls = LaunchService(argv=launch_argv)
    ls.include_launch_description(ld)
    return ls.run()


if __name__ == '__main__':
    sys.exit(main())
