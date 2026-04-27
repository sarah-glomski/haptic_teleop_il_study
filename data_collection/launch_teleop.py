#!/usr/bin/python3.12
"""
Minimal teleop launch — HoloLens → Kinova Gen3 control only.

Use this for testing/calibration. No cameras, no data recording.
Launches:
  1. rosbridge_websocket   — HoloLens WebSocket bridge (port 9090)
  2. hololens_tf_publisher — Broadcasts hand joint TF transforms
  3. hololens_hand_node    — Converts joints → hand/pose + gripper command
  4. kinova_state_publisher— Reads and publishes robot state at 30 Hz
  5. kinova_hand_controller— P-loop velocity controller

Usage:
  /usr/bin/python3.12 launch_teleop.py
  /usr/bin/python3.12 launch_teleop.py --robot-ip 192.168.1.10

Startup checklist:
  □ Kinova Gen3 powered on, e-stop released
  □ Ubuntu machine on same network as robot (ping 192.168.1.10)
  □ HoloLens on same WiFi as this machine
  □ HoloLens app: RosConnector URL = ws://<THIS_MACHINE_IP>:9090
  □ Press 'Arm' button in HoloLens app to start sending poses
  □ Press 'Gripper' button to enable thumb-index gripper control

Stopping:
  Ctrl-C in this terminal → all nodes stop → robot auto-stops
  (watchdog: robot halts within 200 ms if no command arrives)
"""

import argparse
import os
import sys

from launch import LaunchDescription, LaunchService
from launch.actions import ExecuteProcess
from launch_ros.actions import Node


_PYTHON = '/usr/bin/python3.12'

# Piezense ROS2 workspace overlay (provides piezense_interfaces + piezense_ros)
_PIEZENSE_WS = '/home/piezense/ros2_ws/install'

# Piezense driver script
_PIEZENSE_DRIVER = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'Piezense-ROS', 'ros2_ws', 'src', 'piezense_ros', 'piezense_ros', 'piezense_driver.py',
)

def _piezense_env():
    """Environment with piezense workspace overlaid onto the ROS2 environment."""
    env = dict(os.environ)
    prefix = env.get('AMENT_PREFIX_PATH', '')
    env['AMENT_PREFIX_PATH'] = f'{_PIEZENSE_WS}:{prefix}' if prefix else _PIEZENSE_WS
    pypath = env.get('PYTHONPATH', '')
    piezense_pypath = os.path.join(_PIEZENSE_WS, 'piezense_interfaces', 'local', 'lib',
                                   'python3.12', 'dist-packages')
    env['PYTHONPATH'] = f'{piezense_pypath}:{pypath}' if pypath else piezense_pypath
    return env


def generate_launch_description(robot_ip: str) -> LaunchDescription:
    d = os.path.dirname(os.path.abspath(__file__))

    def script(name):
        return os.path.join(d, name)

    return LaunchDescription([
        # ── rosbridge WebSocket (HoloLens → ROS2) ─────────────────────────────
        Node(
            package='rosbridge_server',
            executable='rosbridge_websocket',
            name='rosbridge_websocket',
            output='screen',
            parameters=[{'port': 9090}],
        ),

        # ── HoloLens TF publisher ──────────────────────────────────────────────
        ExecuteProcess(
            cmd=[_PYTHON, script('hololens_tf_publisher_ros2.py')],
            name='hololens_tf_publisher',
            output='screen',
        ),

        # ── HoloLens hand processing node ─────────────────────────────────────
        ExecuteProcess(
            cmd=[_PYTHON, script('hololens_hand_node.py')],
            name='hololens_hand_node',
            output='screen',
        ),

        # ── Kinova state publisher ────────────────────────────────────────────
        ExecuteProcess(
            cmd=[
                _PYTHON, script('kinova_state_publisher.py'),
                '--ros-args', '-p', f'robot_ip:={robot_ip}',
            ],
            name='kinova_state_publisher',
            output='screen',
        ),

        # ── Kinova hand controller ────────────────────────────────────────────
        ExecuteProcess(
            cmd=[
                _PYTHON, script('kinova_hand_controller.py'),
                '--ros-args', '-p', f'robot_ip:={robot_ip}',
            ],
            name='kinova_hand_controller',
            output='screen',
        ),

        # ── Piezense pressure sensor driver ───────────────────────────────────
        # Publishes PiezenseSystemArray on piezense/data at 30 Hz.
        # Input sensors are on system 0, channels 2 and 3.
        # ExecuteProcess(
        #     cmd=[
        #         _PYTHON, _PIEZENSE_DRIVER,
        #         '--ros-args',
        #         '-p', 'systems:=[Piezense:4]',
        #         '-p', 'hz:=30.0',
        #     ],
        #     name='piezense_driver',
        #     output='screen',
        #     additional_env=_piezense_env(),
        # ),
    ])


def main(argv=sys.argv[1:]):
    parser = argparse.ArgumentParser(description='Minimal HoloLens → Kinova teleop')
    parser.add_argument('--robot-ip', default='192.168.1.10',
                        help='Kinova Gen3 IP address')
    args, launch_argv = parser.parse_known_args(argv)

    # Print your machine's IP so the user knows what to enter in the HoloLens app
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        local_ip = s.getsockname()[0]
        s.close()
    except Exception:
        local_ip = '<unknown>'

    print('=' * 60)
    print('HoloLens → Kinova Gen3 Teleoperation')
    print('=' * 60)
    print(f'  Robot IP      : {args.robot_ip}')
    print(f'  This machine  : {local_ip}')
    print()
    print('  In the HoloLens app set RosConnector URL to:')
    print(f'    ws://{local_ip}:9090')
    print()
    print('  Press "Arm" in HoloLens to enable arm tracking.')
    print('  Press "Gripper" to enable thumb-index gripper control.')
    print()
    print('  Ctrl-C stops all nodes and halts the robot.')
    print('=' * 60)

    # Strip conda/miniforge from PATH so that executables with `#!/usr/bin/env python3`
    # (e.g. rosbridge_websocket) resolve to the system Python 3.12, not conda's 3.13.
    path_parts = os.environ.get('PATH', '').split(':')
    os.environ['PATH'] = ':'.join(
        p for p in path_parts if 'miniforge' not in p and 'conda' not in p
    )

    ld = generate_launch_description(robot_ip=args.robot_ip)
    ls = LaunchService(argv=launch_argv)
    ls.include_launch_description(ld)
    return ls.run()


if __name__ == '__main__':
    sys.exit(main())
