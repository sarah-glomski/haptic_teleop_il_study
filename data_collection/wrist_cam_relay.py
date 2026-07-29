#!/usr/bin/env python3
"""
Wrist-cam relay — JPEG-compresses the DJI wrist camera stream for the HoloLens.

Subscribes to the raw DJI image topic and republishes it as a throttled
sensor_msgs/CompressedImage, small enough to ship over rosbridge to the
HoloLens app (which decodes the JPEG into a floating wrist-cam window).

Raw 224x224 BGR @30 Hz over rosbridge would be ~4.5 MB/s of base64 JSON;
JPEG at ~15 Hz / quality 60 is ~150 KB/s — comfortable over WiFi.

Published topics:
  /dji_wrist/compressed      sensor_msgs/CompressedImage (format='jpeg')

Also (optionally, default on) keeps the DJI camera enabled by latching
/dji_camera/enable True every second — same approach as inference.py — so the
wrist view works during teleop, not only while the data collector records.

ROS2 Parameters:
  rate            float 15.0   publish rate (Hz)
  jpeg_quality    int   60     cv2 JPEG quality (0-100)
  keep_enabled    bool  True   re-publish /dji_camera/enable True at 1 Hz

Usage:
  source /opt/ros/jazzy/setup.bash
  /usr/bin/python3.12 wrist_cam_relay.py
"""

import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Bool

DJI_TOPIC = '/dji_wrist/dji_wrist/color/image_raw'
OUT_TOPIC = '/dji_wrist/compressed'


class WristCamRelay(Node):

    def __init__(self):
        super().__init__('wrist_cam_relay')

        rate               = self.declare_parameter('rate',         15.0).value
        self.jpeg_quality  = self.declare_parameter('jpeg_quality', 60).value
        keep_enabled       = self.declare_parameter('keep_enabled', True).value

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.create_subscription(Image, DJI_TOPIC, self._image_cb, sensor_qos)
        self._pub = self.create_publisher(CompressedImage, OUT_TOPIC, sensor_qos)

        self._latest = None
        self.create_timer(1.0 / rate, self._publish)

        if keep_enabled:
            enable_qos = QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                history=HistoryPolicy.KEEP_LAST,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
                depth=1,
            )
            self._enable_pub = self.create_publisher(Bool, '/dji_camera/enable', enable_qos)
            self.create_timer(1.0, self._keep_enabled)

        self.get_logger().info(
            f'Wrist-cam relay ready — {DJI_TOPIC} → {OUT_TOPIC} '
            f'@ {rate:.0f} Hz, JPEG q={self.jpeg_quality}'
        )

    def _keep_enabled(self):
        msg = Bool()
        msg.data = True
        self._enable_pub.publish(msg)

    def _image_cb(self, msg: Image):
        # Decode raw Image without cv_bridge (same pattern as the collector).
        frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        enc = msg.encoding.lower()
        if enc in ('rgb8', 'rgba8'):
            frame = frame[:, :, [2, 1, 0]] if enc == 'rgb8' else frame[:, :, [2, 1, 0, 3]][:, :, :3]
        elif enc in ('bgra8',):
            frame = frame[:, :, :3]
        # bgr8 passes through — cv2.imencode expects BGR.
        self._latest = np.ascontiguousarray(frame[:, :, :3])

    def _publish(self):
        if self._latest is None:
            return
        ok, buf = cv2.imencode('.jpg', self._latest,
                               [int(cv2.IMWRITE_JPEG_QUALITY), int(self.jpeg_quality)])
        if not ok:
            return
        msg = CompressedImage()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'wrist_cam'
        msg.format = 'jpeg'
        msg.data = buf.tobytes()
        self._pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = WristCamRelay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
