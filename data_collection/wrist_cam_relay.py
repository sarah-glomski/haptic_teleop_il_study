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

GATED ON THE HEADSET'S OWN TOGGLE. The app publishes /camera_feed
("true"/"false") for its voice-toggled Camera window, and nothing is sent
unless that reads true.

This is not an optimisation, it is a correctness fix. Measured 2026-08-18 with
the window CLOSED, this relay was still pushing ~59 KB/s of base64 JPEG at the
headset, which could not drain it: over 1 MB of standing backlog in the socket
send queue. rosbridge is ONE tcp connection, so that backlog head-of-line
blocks the pose stream coming the other way, and every /hololens/* topic
stalls together for ~2 s at a time — taking hand tracking with it. The wrist
view is a convenience; continuous hand tracking is the data.

If /camera_feed is never heard (older app build), the relay stays quiet and
says so once, rather than defaulting to the behaviour that caused the stalls.

Also keeps the DJI camera enabled while the view is open by latching
/dji_camera/enable True every second. That is gated too — with the view shut
the data collector regains sole control of the camera, which is what its
per-episode enable/disable assumed all along.

ROS2 Parameters:
  rate            float 15.0   publish rate (Hz) while the view is open
  jpeg_quality    int   60     cv2 JPEG quality (0-100)
  keep_enabled    bool  True   hold /dji_camera/enable True while view is open
  require_toggle  bool  True   False = stream regardless (the old behaviour)
  toggle_debounce_s float 0.4  hold a new toggle value this long before acting

Usage:
  source /opt/ros/jazzy/setup.bash
  /usr/bin/python3.12 wrist_cam_relay.py
"""

import time

import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Bool, String

DJI_TOPIC = '/dji_wrist/dji_wrist/color/image_raw'
OUT_TOPIC = '/dji_wrist/compressed'
TOGGLE_TOPIC = '/camera_feed'      # the headset's Camera window, "true"/"false"


class WristCamRelay(Node):

    def __init__(self):
        super().__init__('wrist_cam_relay')

        rate               = self.declare_parameter('rate',         15.0).value
        self.jpeg_quality  = self.declare_parameter('jpeg_quality', 60).value
        keep_enabled       = self.declare_parameter('keep_enabled', True).value
        self._require_toggle = self.declare_parameter('require_toggle', True).value
        self._debounce = self.declare_parameter('toggle_debounce_s', 0.4).value

        # Nothing goes out until the headset says its Camera window is open.
        self._view_open  = not self._require_toggle
        self._pending = None            # (value, monotonic time it first appeared)
        self._toggle_seen = False
        self._warned_no_toggle = False
        self._start = self.get_clock().now()
        self.create_subscription(String, TOGGLE_TOPIC, self._toggle_cb, 10)

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
            + (f'; idle until {TOGGLE_TOPIC} is true' if self._require_toggle
               else '; UNGATED (require_toggle:=false)')
        )

    def _toggle_cb(self, msg: String):
        """Act on a settled value, not on every message.

        The app republishes this ~50 Hz, so a single dropped or reordered
        message must not flip the stream. Requiring the new value to hold for
        toggle_debounce_s also means one log line per real toggle instead of
        one per message if anything ever contends on the topic.
        """
        self._toggle_seen = True
        want = msg.data.strip().lower() == 'true'
        if want == self._view_open:
            self._pending = None
            return
        now = time.monotonic()
        if self._pending is None or self._pending[0] != want:
            self._pending = (want, now)
            return
        if now - self._pending[1] < self._debounce:
            return
        self._view_open = want
        self._pending = None
        self.get_logger().info(
            f'Headset camera window {"OPENED — streaming wrist cam" if want else "CLOSED — relay quiet"}')

    def _streaming(self) -> bool:
        """True only while the headset actually wants the picture."""
        if not self._require_toggle:
            return True
        if not self._toggle_seen and not self._warned_no_toggle:
            age = (self.get_clock().now() - self._start).nanoseconds / 1e9
            if age > 10.0:
                self._warned_no_toggle = True
                self.get_logger().warn(
                    f'No {TOGGLE_TOPIC} in {age:.0f}s — staying quiet. The headset app '
                    f'publishes it for its Camera window; if this build does not, run '
                    f'with -p require_toggle:=false to stream unconditionally (and see '
                    f'this module docstring for why that stalls hand tracking).')
        return self._view_open

    def _keep_enabled(self):
        # Only hold the camera on while the view is open; otherwise the data
        # collector's per-episode enable/disable is the only writer.
        if not self._streaming():
            return
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
        if self._latest is None or not self._streaming():
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
