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
  7. ZED M camera          — /zed_front/zed_node/left/image_rect_color
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
import sys

from launch import LaunchDescription, LaunchService
from launch.actions import ExecuteProcess
from launch_ros.actions import Node


_PYTHON = '/usr/bin/python3.12'

# Piezense ROS2 workspace overlay
_PIEZENSE_WS = '/home/piezense/ros2_ws/install'
_PIEZENSE_DRIVER = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'Piezense-ROS', 'ros2_ws', 'src', 'piezense_ros', 'piezense_ros', 'piezense_driver.py',
)

def _piezense_env():
    env = dict(os.environ)
    prefix = env.get('AMENT_PREFIX_PATH', '')
    env['AMENT_PREFIX_PATH'] = f'{_PIEZENSE_WS}:{prefix}' if prefix else _PIEZENSE_WS
    pypath = env.get('PYTHONPATH', '')
    piezense_pypath = os.path.join(_PIEZENSE_WS, 'piezense_interfaces', 'local', 'lib',
                                   'python3.12', 'dist-packages')
    env['PYTHONPATH'] = f'{piezense_pypath}:{pypath}' if pypath else piezense_pypath
    return env


ZED_SERIAL = '17875187'


def generate_launch_description(
    robot_ip: str = '192.168.1.10',
    zed_serial: str = ZED_SERIAL,
    dji_device: int = 0,
) -> LaunchDescription:

    script_dir = os.path.dirname(os.path.abspath(__file__))

    def script(name):
        return os.path.join(script_dir, name)

    return LaunchDescription([

        # ── 1. rosbridge WebSocket server ─────────────────────────────────────
        # The HoloLens Unity app connects to ws://<host-ip>:9090
        Node(
            package='rosbridge_server',
            executable='rosbridge_websocket',
            name='rosbridge_websocket',
            output='screen',
            parameters=[{
                'port': 9090,
                'address': '',
            }],
        ),

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

        # ── 6. Piezense pressure sensor driver ────────────────────────────────
        # system 0, 4 total channels; input sensors on channels 2 and 3
        ExecuteProcess(
            cmd=[
                _PYTHON, _PIEZENSE_DRIVER,
                '--ros-args',
                '-p', 'systems:=[Piezense:4]',
                '-p', 'hz:=30.0',
            ],
            name='piezense_driver',
            output='screen',
            additional_env=_piezense_env(),
        ),

        # ── 7. HDF5 data collector (pygame UI runs here) ──────────────────────
        ExecuteProcess(
            cmd=[_PYTHON, script('hdf5_data_collector.py')],
            name='hdf5_data_collector',
            output='screen',
        ),

        # ── 7. ZED M camera — front view ──────────────────────────────────────
        # Topics produced:
        #   /zed_front/zed_node/left/image_rect_color   (used by data collector)
        #   /zed_front/zed_node/depth/depth_registered  (available for future use)
        Node(
            package='zed_wrapper',
            executable='zed_wrapper',
            name='zed_node',
            namespace='zed_front',
            output='screen',
            parameters=[{
                'camera_model':          'zedm',          # ZED Mini / ZED M
                'camera_name':           'zed_front',
                'serial_number':         int(zed_serial) if zed_serial else 0,
                'grab_resolution':       'HD720',          # 1280×720
                'grab_frame_rate':       30,
                'pub_frame_rate':        30.0,
                'general.grab_frame_rate': 30,
                # Enable only what we need
                'depth.depth_mode':      1,               # PERFORMANCE
                'video.extrinsic_in_camera_frame': False,
            }],
        ),

        # ── 8. DJI Osmo Action 4 — wrist-mounted camera ──────────────────────
        # Publishes on /dji_wrist/dji_wrist/color/image_raw to match what the
        # HDF5 data collector expects. Set --dji-device if /dev/video0 is wrong
        # (run dji_camera_validate.py first to confirm the device index).
        ExecuteProcess(
            cmd=[
                _PYTHON, script('dji_camera_node.py'),
                '--ros-args',
                '-p', f'device_index:={dji_device}',
                '-r', '/wrist_cam/image_raw:=/dji_wrist/dji_wrist/color/image_raw',
            ],
            name='dji_wrist_camera',
            output='screen',
        ),
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

    ld = generate_launch_description(
        robot_ip=args.robot_ip,
        zed_serial=args.zed_serial,
        dji_device=args.dji_device,
    )
    ls = LaunchService(argv=launch_argv)
    ls.include_launch_description(ld)
    return ls.run()


if __name__ == '__main__':
    sys.exit(main())
