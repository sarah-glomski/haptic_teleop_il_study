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
  width          int   224       Published (output) width  — model input; INTER_AREA
  height         int   224       Published (output) height — model input; INTER_AREA
  capture_width  int  1280       Capture width requested from the camera
  capture_height int   720       Capture height requested from the camera
                                  (DJI webcam only offers 1280x720 / 1920x1080)
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
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool


class DJICameraNode(Node):

    def __init__(self):
        super().__init__('dji_camera_node')

        self.device_index = self.declare_parameter('device_index', 0).value
        # Published / output size. Frames are resized down to this once, here, so
        # both data collection and inference consume the exact same pixels with
        # the same latency (no per-consumer resize). Defaults to the 224x224
        # model input. Resize uses INTER_AREA to match the training pipeline.
        self.width        = self.declare_parameter('width',        224).value
        self.height       = self.declare_parameter('height',       224).value
        # Capture size requested from the camera. The DJI Osmo Action 4 webcam
        # only offers 1280x720 and 1920x1080 over UVC; anything else is silently
        # clamped by the driver. Request a real native mode (720p = lowest
        # bandwidth) instead of relying on that clamp.
        self.capture_width  = self.declare_parameter('capture_width',  1280).value
        self.capture_height = self.declare_parameter('capture_height',  720).value
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
        self._cap_lock      = threading.Lock()
        self._running       = True
        self._enabled       = False  # starts disabled; collector publishes True to stream

        # TransientLocal so a late-subscribing collector still gets the current state
        enable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=1,
        )
        self.create_subscription(Bool, '/dji_camera/enable', self._enable_cb, enable_qos)

        # Capture thread idles until enabled, then reads continuously and auto-reconnects.
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()

        self.create_timer(1.0 / fps, self._publish_frame)
        self.get_logger().info(
            f'DJI camera node ready — /dev/video{self.device_index} '
            f'capture {self.capture_width}×{self.capture_height} → publish '
            f'{self.width}×{self.height} @ {fps:.0f} Hz '
            f'(idle; waiting for /dji_camera/enable)'
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

        # Force MJPG, then request a native capture resolution. BUFFERSIZE=2 is
        # deliberate: on the Osmo Action 4 over V4L2, BUFFERSIZE=1 halves the
        # delivered rate to 15 fps; >=2 gives the full 30 fps.
        self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.capture_width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.capture_height)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

        self._actual_width  = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._actual_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.get_logger().info(
            f'Camera opened: {self._actual_width}×{self._actual_height} '
            f'(requested {self.capture_width}×{self.capture_height})'
        )

    def _enable_cb(self, msg: Bool):
        if msg.data and not self._enabled:
            try:
                with self._cap_lock:
                    self._open_camera()
                self._enabled = True
                self.get_logger().info(
                    f'DJI camera enabled — {self._actual_width}×{self._actual_height}'
                )
            except Exception as e:
                self.get_logger().error(f'DJI camera enable failed: {e}')
        elif not msg.data and self._enabled:
            self._enabled = False          # capture loop checks this before acquiring cap_lock
            with self._cap_lock:
                if self._cap is not None:
                    self._cap.release()
                    self._cap = None
            with self._frame_lock:
                self._latest_frame = None
            self.get_logger().info('DJI camera disabled')

    def _capture_loop(self):
        """Runs in a daemon thread. Idles when disabled; auto-reconnects on failure."""
        consecutive_failures = 0

        while self._running:
            if not self._enabled:
                time.sleep(0.1)
                consecutive_failures = 0
                continue

            with self._cap_lock:
                if not self._enabled or self._cap is None or not self._cap.isOpened():
                    ret, frame = False, None
                else:
                    ret, frame = self._cap.read()

            if not self._enabled or frame is None:
                time.sleep(0.01)
                continue

            if ret:
                consecutive_failures = 0
                if frame.shape[1] != self.width or frame.shape[0] != self.height:
                    # INTER_AREA must match the resize used in inference.py and the
                    # offline conversion (convert_data.py / hdf5_to_zarr.py) so the
                    # policy sees identical pixels in training and at inference.
                    frame = cv2.resize(frame, (self.width, self.height),
                                       interpolation=cv2.INTER_AREA)
                with self._frame_lock:
                    self._latest_frame = frame
                time.sleep(0.001)
            else:
                consecutive_failures += 1
                if consecutive_failures == 1:
                    self.get_logger().warn(
                        f'Frame grab failed on /dev/video{self.device_index} '
                        '— attempting reconnect …'
                    )
                try:
                    with self._cap_lock:
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
        self._enabled = False
        with self._cap_lock:
            if self._cap is not None:
                self._cap.release()
                self._cap = None
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
