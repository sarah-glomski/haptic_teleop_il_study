#!/usr/bin/env python3
"""
DJI Osmo Action 4 Camera ROS2 Node

Streams the DJI Osmo Action 4 (connected via USB-C in UVC/webcam mode) as a
ROS2 sensor_msgs/Image topic for use in the data collection pipeline.

Camera setup (do once on the DJI camera):
  Menu → Settings → Control Method → UVC Camera

Published topics:
  /wrist_cam/image_raw    sensor_msgs/Image   BGR8, configurable rate

ROS2 Parameters:
  device_index   int     0       V4L2 device index (/dev/videoN)
  width          int   640       Requested capture width
  height         int   480       Requested capture height
  fps            float 30.0      Target publish rate
  frame_id       str  'wrist_cam'  TF frame for image header

Usage:
  source /opt/ros/jazzy/setup.bash
  /usr/bin/python3.12 dji_camera_node.py

  # Override device index:
  /usr/bin/python3.12 dji_camera_node.py --ros-args -p device_index:=2

  # Remap to the topic expected by the data collector:
  /usr/bin/python3.12 dji_camera_node.py --ros-args -r /wrist_cam/image_raw:=/dji_wrist/dji_wrist/color/image_raw
"""

import time
import threading

import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image


class DJICameraNode(Node):

    def __init__(self):
        super().__init__('dji_camera_node')

        self.device_index = self.declare_parameter('device_index', 0).value
        self.width        = self.declare_parameter('width',        640).value
        self.height       = self.declare_parameter('height',       480).value
        fps               = self.declare_parameter('fps',           30.0).value
        self.frame_id     = self.declare_parameter('frame_id',     'wrist_cam').value

        # Sensor-data QoS: best-effort, keep-last-1 — matches camera drivers
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self._pub = self.create_publisher(Image, '/wrist_cam/image_raw', sensor_qos)
        self._cap = None

        # Latest frame shared between capture thread and publish timer
        self._latest_frame  = None
        self._actual_width  = self.width
        self._actual_height = self.height
        self._frame_lock    = threading.Lock()
        self._running       = True

        self._open_camera()

        # Capture thread reads continuously so cap.read() never blocks the timer.
        # Auto-reconnects when the UVC stream silently restarts.
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()

        self.create_timer(1.0 / fps, self._publish_frame)
        self.get_logger().info(
            f'DJI camera node started — /dev/video{self.device_index} '
            f'{self._actual_width}×{self._actual_height} @ {fps:.0f} Hz'
        )

    def _open_camera(self):
        if self._cap is not None:
            self._cap.release()

        self._cap = cv2.VideoCapture(self.device_index, cv2.CAP_V4L2)
        if not self._cap.isOpened():
            self.get_logger().fatal(
                f'Cannot open /dev/video{self.device_index}. '
                'Check that the camera is in UVC mode and the device index is correct '
                '(set ros param device_index).'
            )
            raise RuntimeError(f'Failed to open /dev/video{self.device_index}')

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

        self._actual_width  = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._actual_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.get_logger().info(
            f'Camera opened: {self._actual_width}×{self._actual_height} '
            f'(requested {self.width}×{self.height})'
        )

    def _capture_loop(self):
        """Runs in a daemon thread. Drains V4L2 buffer and auto-reconnects on failure."""
        consecutive_failures = 0

        while self._running:
            if self._cap is None or not self._cap.isOpened():
                time.sleep(1.0)
                continue

            ret, frame = self._cap.read()

            if ret:
                consecutive_failures = 0
                if frame.shape[1] != self.width or frame.shape[0] != self.height:
                    frame = cv2.resize(frame, (self.width, self.height))
                with self._frame_lock:
                    self._latest_frame = frame
                # Yield briefly — avoids spinning the CPU at 100% between frames
                time.sleep(0.001)
            else:
                consecutive_failures += 1
                if consecutive_failures == 1:
                    self.get_logger().warn(
                        f'Frame grab failed on /dev/video{self.device_index} '
                        '— attempting reconnect …'
                    )
                try:
                    self._open_camera()
                    consecutive_failures = 0
                    self.get_logger().info('Camera reconnected successfully')
                except Exception:
                    time.sleep(1.0)

    def _publish_frame(self):
        with self._frame_lock:
            frame = self._latest_frame
        if frame is None:
            return

        msg                 = Image()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.height          = frame.shape[0]
        msg.width           = frame.shape[1]
        msg.encoding        = 'bgr8'
        msg.is_bigendian    = False
        msg.step            = frame.shape[1] * frame.shape[2]
        msg.data            = frame.tobytes()
        self._pub.publish(msg)

    def destroy_node(self):
        self._running = False
        if self._cap is not None:
            self._cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = DJICameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
