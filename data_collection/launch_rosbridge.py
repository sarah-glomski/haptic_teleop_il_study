#!/usr/bin/python3.12
"""
Standalone rosbridge launcher — run this once at the start of the day.

Keeps the WebSocket server alive on port 9090 so the HoloLens app can
connect at any time, even before the main data-collection pipeline starts.
Once this is running, use --no-rosbridge with launch_data_collection.py
to avoid disconnecting the headset on pipeline restarts.

Usage:
  /usr/bin/python3.12 launch_rosbridge.py
  # then in another terminal:
  python3.12 launch_data_collection.py --no-zed --no-rosbridge
"""

import sys

from launch import LaunchDescription, LaunchService
from launch_ros.actions import Node


def main():
    import os
    ld = LaunchDescription([
        Node(
            package='rosbridge_server',
            executable='rosbridge_websocket',
            name='rosbridge_websocket',
            output='screen',
            parameters=[{'port': 9090, 'address': ''}],
            additional_env={
                'PATH': ':'.join(
                    p for p in os.environ.get('PATH', '').split(':')
                    if 'miniforge' not in p and 'conda' not in p
                ),
            },
        ),
    ])
    ls = LaunchService()
    ls.include_launch_description(ld)
    return ls.run()


if __name__ == '__main__':
    sys.exit(main())
