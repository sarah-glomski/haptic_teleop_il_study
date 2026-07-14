#!/usr/bin/env python3
"""
ZED M Camera ROS2 Node — UVC / "zed-camera mode" (no ZED SDK required)

The ZED M enumerates as a standard UVC webcam that streams both stereo images
side-by-side in a single frame (left | right, concatenated horizontally). This
node reads that raw UVC stream, crops out the LEFT image, and publishes it on the
same topic the data-collection pipeline already expects from the full ZED SDK
wrapper — so the rest of the pipeline is unchanged.

This deliberately avoids the ZED SDK + zed-ros2-wrapper (CUDA, multi-GB build).
The pipeline's only ZED consumer is the left RGB image, so a UVC read is
sufficient. Caveat: the image is the raw (unrectified) left sensor image, not
SDK-rectified. For an RGB-only policy input that is fine; if true rectification /
depth is ever needed, install the SDK and use the zed_wrapper node instead.

Camera setup: none — plug the ZED M into a USB 3.0 (SuperSpeed) port. Over USB 2.0
only the HID interface enumerates and NO /dev/video node appears for the ZED.

Published topics:
  /zed_isometric/zed_node/left/image_rect_color   sensor_msgs/Image  BGR8

ROS2 Parameters:
  device_index    int    -1      V4L2 device index (/dev/videoN). -1 = auto-detect
                                  the ZED by its by-id name (STEREOLABS/ZED).
  capture_width   int   2560     Side-by-side capture width  (HD720 stereo = 2560)
  capture_height  int    720     Capture height              (HD720 stereo = 720)
  fps             float 30.0     Target publish rate
  frame_id        str  'zed_isometric_left_camera_optical_frame'

Usage:
  source /opt/ros/jazzy/setup.bash
  /usr/bin/python3.12 zed_uvc_node.py
  /usr/bin/python3.12 zed_uvc_node.py --ros-args -p device_index:=2
"""

import glob
import os
import threading
import time

import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image


LEFT_TOPIC = '/zed_isometric/zed_node/left/image_rect_color'


def find_zed_devices() -> list:
    """Return all /dev/videoN indices belonging to the ZED's UVC interface, sorted.

    Matches the by-id symlink name (STEREOLABS / ZED / "Technologies, Inc. ZED-M").
    The ZED exposes MULTIPLE video nodes — typically a streaming node plus a
    metadata-only node that cannot be opened for capture — so the caller must try
    each in order and keep the one that actually yields frames. Never matches the
    DJI camera (different by-id name).
    """
    candidates = set()
    for link in glob.glob('/dev/v4l/by-id/*'):
        name = os.path.basename(link).lower()
        if 'zed' in name or 'stereolabs' in name:
            target = os.path.realpath(link)
            base = os.path.basename(target)          # e.g. 'video2'
            if base.startswith('video') and base[5:].isdigit():
                candidates.add(int(base[5:]))
    return sorted(candidates)


def find_zed_device() -> int:
    """Lowest ZED video index, or -1 if absent (used for a quick presence check)."""
    devs = find_zed_devices()
    return devs[0] if devs else -1


class ZedUvcNode(Node):

    def __init__(self):
        super().__init__('zed_uvc_node')

        self.device_index   = self.declare_parameter('device_index',   -1).value
        self.capture_width  = self.declare_parameter('capture_width',  2560).value
        self.capture_height = self.declare_parameter('capture_height',  720).value
        fps                 = self.declare_parameter('fps',            30.0).value
        self.frame_id       = self.declare_parameter(
            'frame_id', 'zed_isometric_left_camera_optical_frame').value

        # Candidate video nodes to try, in order. An explicit device_index param
        # pins one; otherwise probe every ZED node and pick the streaming one.
        if self.device_index < 0:
            self._candidates = find_zed_devices()
        else:
            self._candidates = [self.device_index]
        self.device_index = self._candidates[0] if self._candidates else -1

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self._pub = self.create_publisher(Image, LEFT_TOPIC, sensor_qos)

        self._cap           = None
        self._latest_frame  = None
        self._frame_lock    = threading.Lock()
        self._cap_lock      = threading.Lock()
        self._running       = True

        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()

        self.create_timer(1.0 / fps, self._publish_frame)

        if self.device_index < 0:
            self.get_logger().fatal(
                'No ZED UVC device found under /dev/v4l/by-id (STEREOLABS/ZED). '
                'The ZED M is likely on a USB 2.0 port — only its HID interface '
                'enumerates there. Plug it into a USB 3.0 (SuperSpeed) port so the '
                'video interface appears, or pass -p device_index:=N explicitly.'
            )
        else:
            self.get_logger().info(
                f'ZED UVC node ready — /dev/video{self.device_index} '
                f'capture {self.capture_width}×{self.capture_height} (side-by-side) '
                f'→ left {self.capture_width // 2}×{self.capture_height} @ {fps:.0f} Hz '
                f'on {LEFT_TOPIC}'
            )

    def _open_camera(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None

        # Re-probe in case the node numbers changed across a reconnect.
        candidates = self._candidates or find_zed_devices()

        last_err = None
        for idx in candidates:
            cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
            if not cap.isOpened():
                last_err = f'/dev/video{idx}: cannot open'
                cap.release()
                continue
            # The ZED M streams raw YUYV side-by-side stereo. Request the native
            # HD720 stereo mode; the driver clamps to a supported mode otherwise.
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV'))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.capture_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.capture_height)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
            # Verify this node actually streams (the ZED also exposes a
            # metadata-only node that opens but never yields frames).
            ok, frame = cap.read()
            if not ok or frame is None:
                last_err = f'/dev/video{idx}: opened but no frames (metadata node?)'
                cap.release()
                continue
            self._cap = cap
            self.device_index = idx
            aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.get_logger().info(
                f'ZED opened on /dev/video{idx}: {aw}×{ah} side-by-side '
                f'(requested {self.capture_width}×{self.capture_height})'
            )
            return

        raise RuntimeError(f'No streaming ZED video node found ({last_err})')

    def _capture_loop(self):
        consecutive_failures = 0
        while self._running:
            if self.device_index < 0:
                time.sleep(0.5)
                continue

            with self._cap_lock:
                if self._cap is None or not self._cap.isOpened():
                    ret, frame = False, None
                else:
                    ret, frame = self._cap.read()

            if ret and frame is not None:
                consecutive_failures = 0
                # Crop the LEFT half of the side-by-side stereo frame.
                w = frame.shape[1]
                left = frame[:, : w // 2]
                with self._frame_lock:
                    self._latest_frame = left
                time.sleep(0.001)
            else:
                consecutive_failures += 1
                if consecutive_failures == 1:
                    self.get_logger().warn(
                        f'ZED frame grab failed on /dev/video{self.device_index} '
                        '— attempting reconnect …'
                    )
                try:
                    with self._cap_lock:
                        self._open_camera()
                    consecutive_failures = 0
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
        with self._cap_lock:
            if self._cap is not None:
                self._cap.release()
                self._cap = None
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ZedUvcNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
