#!/usr/bin/python3.12
"""
Standalone rosbridge launcher — run this once at the start of the day.

Keeps the WebSocket server alive on port 9090 so the HoloLens app can
connect at any time, even before the main data-collection pipeline starts.
Once this is running, use --no-rosbridge with launch_data_collection.py
to avoid disconnecting the headset on pipeline restarts.

Also broadcasts the rosbridge URL over UDP every 2 s so the HoloLens Unity
app can discover this machine's IP automatically (RosBridgeDiscovery.cs).
If no broadcast is received within 5 s the app falls back to its hardcoded URL.

Usage:
  /usr/bin/python3.12 launch_rosbridge.py
  # then in another terminal:
  python3.12 launch_data_collection.py --no-zed --no-rosbridge
"""

import os
import socket
import sys
import threading
import time

from launch import LaunchDescription, LaunchService
from launch_ros.actions import Node

ROSBRIDGE_PORT     = 9090
DISCOVERY_PORT     = 9091  # UDP port the HoloLens listens on for broadcasts
BROADCAST_INTERVAL = 2     # seconds between broadcasts


def _get_local_ip():
    """Return the primary outbound IP — the address the HoloLens will reach us on."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return '127.0.0.1'


def start_discovery_broadcaster():
    """Broadcast ws://<ip>:9090 on UDP 9091 every 2 s (daemon thread).
    The HoloLens app listens for this and uses it instead of the hardcoded IP.
    Falls back to the hardcoded URL if no broadcast is received within 5 s."""
    def _loop():
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        while True:
            ip  = _get_local_ip()
            url = f'ws://{ip}:{ROSBRIDGE_PORT}'
            try:
                sock.sendto(url.encode(), ('255.255.255.255', DISCOVERY_PORT))
            except Exception:
                pass
            time.sleep(BROADCAST_INTERVAL)

    t = threading.Thread(target=_loop, daemon=True)
    t.start()
    ip = _get_local_ip()
    print(f'[rosbridge] Broadcasting discovery URL: ws://{ip}:{ROSBRIDGE_PORT}  (UDP {DISCOVERY_PORT})')


def make_rosbridge_node():
    """The rosbridge_websocket Node, shared by launch_rosbridge.py and
    launch_data_collection.py so the two definitions never diverge.

    Force system python3 in PATH so the #!/usr/bin/env python3 shebang doesn't
    resolve to miniforge Python (which lacks the rclpy C extensions)."""
    return Node(
        package='rosbridge_server',
        executable='rosbridge_websocket',
        name='rosbridge_websocket',
        output='screen',
        parameters=[{'port': ROSBRIDGE_PORT, 'address': ''}],
        additional_env={
            'PATH': ':'.join(
                p for p in os.environ.get('PATH', '').split(':')
                if 'miniforge' not in p and 'conda' not in p
            ),
        },
    )


def _local_ip():
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # No packets are sent; this just picks the interface used for the route.
        s.connect(('8.8.8.8', 80))
        return s.getsockname()[0]
    except OSError:
        return '127.0.0.1'
    finally:
        s.close()


def main():
    print(f'[rosbridge] WebSocket URL: ws://{_local_ip()}:9090', flush=True)
    start_discovery_broadcaster()
    ld = LaunchDescription([make_rosbridge_node()])
    ls = LaunchService()
    ls.include_launch_description(ld)
    return ls.run()


if __name__ == '__main__':
    sys.exit(main())
