#!/usr/bin/env python3
"""Standalone viewer for the data collection cameras.

Starts the ZED M and DJI camera nodes, then displays their live feeds.
Do not run simultaneously with launch_data_collection.py.

Usage:
  source /opt/ros/jazzy/setup.bash
  python3.12 view_cameras.py [--dji-device N] [--zed-serial SERIAL]

Press 'q' to quit.
"""

import argparse
import os
import subprocess
import threading
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool

# Defaults matching launch_data_collection.py
ZED_SERIAL = '17875187'
DJI_DEVICE = 0

CAMERA_STREAMS = {
    'zed_isometric': '/zed_isometric/zed_node/left/image_rect_color',
    'dji_wrist':  '/dji_wrist/dji_wrist/color/image_raw',
}

PLACEHOLDER_W, PLACEHOLDER_H = 640, 360
WARN_AFTER_S = 12.0  # ZED wrapper takes several seconds to initialise
_PYTHON = '/usr/bin/python3.12'


# ── Image decoding (no cv_bridge — avoids NumPy 1.x/2.x incompatibility) ──────

def _imgmsg_to_bgr(msg: Image) -> np.ndarray:
    data = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.step))
    enc = msg.encoding.lower()
    if enc in ('bgr8', 'rgb8'):
        img = data[:, :msg.width * 3].reshape((msg.height, msg.width, 3))
        if enc == 'rgb8':
            img = img[:, :, ::-1]
    elif enc in ('bgra8', 'rgba8'):
        img = data[:, :msg.width * 4].reshape((msg.height, msg.width, 4))
        if enc == 'rgba8':
            img = img[:, :, [2, 1, 0, 3]]
        img = img[:, :, :3]
    elif enc == 'mono8':
        img = data[:, :msg.width]
    else:
        raise ValueError(f'Unsupported encoding: {msg.encoding}')
    return np.ascontiguousarray(img)


# ── ROS2 subscriber node ───────────────────────────────────────────────────────

class CameraViewer(Node):

    def __init__(self):
        super().__init__('camera_viewer')
        self._lock = threading.Lock()
        self._frames = {name: None for name in CAMERA_STREAMS}

        sensor_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
        )
        for name, topic in CAMERA_STREAMS.items():
            self.create_subscription(
                Image, topic,
                lambda msg, n=name: self._image_cb(msg, n),
                qos_profile=sensor_qos,
            )

        # The DJI node starts idle and only streams once it receives True on
        # /dji_camera/enable (the data collector normally sends this). Match its
        # RELIABLE + TRANSIENT_LOCAL QoS so the latched message reaches it.
        enable_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
        )
        self._enable_pub = self.create_publisher(Bool, '/dji_camera/enable', enable_qos)

    def set_dji_enabled(self, enabled: bool):
        self._enable_pub.publish(Bool(data=enabled))

    def _image_cb(self, msg: Image, name: str):
        try:
            bgr = _imgmsg_to_bgr(msg)
        except Exception as e:
            self.get_logger().warn(f'{name}: {e}', throttle_duration_sec=5.0)
            return
        with self._lock:
            self._frames[name] = bgr

    def get_frames(self) -> dict:
        with self._lock:
            return {k: v.copy() for k, v in self._frames.items() if v is not None}


# ── Camera node management ─────────────────────────────────────────────────────

def start_camera_nodes(zed_serial: str, dji_device: int) -> list:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    procs = []

    zed_cmd = [
        'ros2', 'run', 'zed_wrapper', 'zed_wrapper',
        '--ros-args',
        '--remap', '__ns:=/zed_isometric',
        '--remap', '__node:=zed_node',
        '-p', 'camera_model:=zedm',
        '-p', 'camera_name:=zed_isometric',
        '-p', f'serial_number:={int(zed_serial) if zed_serial else 0}',
        '-p', 'grab_resolution:=HD720',
        '-p', 'grab_frame_rate:=30',
        '-p', 'pub_frame_rate:=30.0',
        '-p', 'depth.depth_mode:=1',
    ]
    procs.append(subprocess.Popen(zed_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
    print(f'Started ZED wrapper  (serial {zed_serial or "auto-detect"})')

    dji_cmd = [
        _PYTHON, os.path.join(script_dir, 'dji_camera_node.py'),
        '--ros-args',
        '-p', f'device_index:={dji_device}',
        '-r', '/wrist_cam/image_raw:=/dji_wrist/dji_wrist/color/image_raw',
    ]
    procs.append(subprocess.Popen(dji_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
    print(f'Started DJI camera node  (/dev/video{dji_device})')

    return procs


def stop_camera_nodes(procs: list):
    for p in procs:
        p.terminate()
    for p in procs:
        try:
            p.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            p.kill()


# ── Display ────────────────────────────────────────────────────────────────────

def _placeholder(label: str) -> np.ndarray:
    img = np.zeros((PLACEHOLDER_H, PLACEHOLDER_W, 3), dtype=np.uint8)
    cv2.putText(img, f'Waiting for {label}...', (20, PLACEHOLDER_H // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 2)
    return img


def main():
    parser = argparse.ArgumentParser(description='View data collection camera feeds')
    parser.add_argument('--zed-serial', default=ZED_SERIAL,
                        help=f'ZED M serial number (default: {ZED_SERIAL})')
    parser.add_argument('--dji-device', type=int, default=DJI_DEVICE,
                        help=f'V4L2 device index for DJI camera (default: {DJI_DEVICE})')
    args = parser.parse_args()

    procs = start_camera_nodes(args.zed_serial, args.dji_device)

    rclpy.init()
    node = CameraViewer()
    ros_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    ros_thread.start()

    node.set_dji_enabled(True)  # DJI node idles until enabled; the collector isn't running here

    print("Press 'q' to quit.")

    for name in CAMERA_STREAMS:
        cv2.imshow(name, _placeholder(name))
    cv2.waitKey(1)

    start = time.monotonic()
    warned = set()

    try:
        while rclpy.ok():
            frames = node.get_frames()
            for name in CAMERA_STREAMS:
                if name in frames:
                    cv2.imshow(name, frames[name])
                else:
                    elapsed = time.monotonic() - start
                    if elapsed > WARN_AFTER_S and name not in warned:
                        warned.add(name)
                        print(f'[WARN] No frames from {name} after {elapsed:.0f}s — '
                              f'check camera connection and device index')

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        node.set_dji_enabled(False)  # release the camera so the next run starts clean
        time.sleep(0.2)              # let the latched disable flush before teardown
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()
        stop_camera_nodes(procs)
        print('Camera nodes stopped.')


if __name__ == '__main__':
    main()
