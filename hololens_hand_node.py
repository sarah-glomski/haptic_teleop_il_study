#!/usr/bin/env python3
"""
HoloLens Hand Processing Node (ROS2)

Aggregates raw HoloLens 2 joint poses (arriving via rosbridge_websocket) and
publishes processed hand tracking data for robot control and data logging.

Published topics:
  hand/pose           geometry_msgs/PoseStamped  - right palm in robot base frame
  hand/gripper_cmd    std_msgs/Float32            - 0=open, 1=closed (thumb-index pinch)
  hand/hand_width     std_msgs/Float32            - raw thumb-index distance (metres)
  hand/finger_tips    std_msgs/Float32MultiArray  - 15 floats: [thumb(xyz), index(xyz),
                                                     middle(xyz), ring(xyz), pinky(xyz)]
  hand/tracking_active std_msgs/Bool              - True if hand data is arriving

Subscribed topics (from HoloLens Unity app via rosbridge):
  /hololens/palm/right     geometry_msgs/PoseStamped  (required for hand/pose)
  /hololens/thumb/right    geometry_msgs/PoseStamped  (required for gripper/hand_width)
  /hololens/index/right    geometry_msgs/PoseStamped  (required for gripper/hand_width)
  /hololens/middle/right   geometry_msgs/PoseStamped  (optional — finger_tips[6:9])
  /hololens/ring/right     geometry_msgs/PoseStamped  (optional — finger_tips[9:12])
  /hololens/pinky/right    geometry_msgs/PoseStamped  (optional — finger_tips[12:15])
  /hololens/gaze           geometry_msgs/PoseStamped  (stored for logging)

ROS2 Parameters:
  pinch_close_m   float  0.025  distance (m) below which gripper fully closes
  pinch_open_m    float  0.10   distance (m) above which gripper fully opens
  filter_alpha    float  0.3    low-pass position filter coefficient (0=heavy, 1=raw)
  grip_alpha      float  0.2    low-pass gripper filter coefficient

  Workspace transform — maps HoloLens Unity world frame to robot base_link frame.
  The HoloLens publishes poses in the Unity world frame (already converted to ROS
  axis convention by the Unity publishers). You need to calibrate these offsets
  so that the workspace in front of the participant maps to the robot's reachable
  workspace. See the calibration guide in the README.

  workspace_x_offset  float  0.4   (m) forward offset from Unity origin to robot workspace
  workspace_y_offset  float  0.0   (m) lateral offset
  workspace_z_offset  float  0.2   (m) vertical offset
  workspace_x_scale   float  1.0   scale factor for X axis
  workspace_y_scale   float  1.0   scale factor for Y axis
  workspace_z_scale   float  1.0   scale factor for Z axis

Usage:
  ros2 run haptic_teleop_il_study hololens_hand_node
"""

import time
import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32, Bool, Float32MultiArray


class LowPassFilter:
    """Exponential moving average filter."""
    def __init__(self, alpha: float = 0.3):
        self.alpha = alpha
        self._val = None

    def filter(self, x: np.ndarray) -> np.ndarray:
        if self._val is None:
            self._val = x.copy()
        else:
            self._val = self.alpha * x + (1.0 - self.alpha) * self._val
        return self._val.copy()

    def reset(self):
        self._val = None


class HoloLensHandNode(Node):
    """
    Processes raw HoloLens PoseStamped topics into robot-space hand commands
    and finger-tracking data ready for robot control and HDF5 logging.
    """

    # If no palm message arrives within this many seconds, mark tracking as inactive
    TRACKING_TIMEOUT_S = 0.5

    def __init__(self):
        super().__init__('hololens_hand_node')

        # ── Parameters ─────────────────────────────────────────────────────────
        self.pinch_close_m = self.declare_parameter('pinch_close_m', 0.025).value
        self.pinch_open_m  = self.declare_parameter('pinch_open_m',  0.10).value
        filter_alpha       = self.declare_parameter('filter_alpha',  0.3).value
        grip_alpha         = self.declare_parameter('grip_alpha',    0.2).value

        self.ws_x_offset = self.declare_parameter('workspace_x_offset', 0.4).value
        self.ws_y_offset = self.declare_parameter('workspace_y_offset', 0.0).value
        self.ws_z_offset = self.declare_parameter('workspace_z_offset', 0.2).value
        self.ws_x_scale  = self.declare_parameter('workspace_x_scale',  1.0).value
        self.ws_y_scale  = self.declare_parameter('workspace_y_scale',  1.0).value
        self.ws_z_scale  = self.declare_parameter('workspace_z_scale',  1.0).value

        # ── Filters ─────────────────────────────────────────────────────────────
        self._pos_filter  = LowPassFilter(filter_alpha)
        self._grip_filter = LowPassFilter(grip_alpha)

        # ── Latest messages from HoloLens ──────────────────────────────────────
        self._palm_msg   = None
        self._thumb_msg  = None
        self._index_msg  = None
        self._middle_msg = None
        self._ring_msg   = None
        self._pinky_msg  = None
        self._gaze_msg   = None
        self._last_palm_t = None  # monotonic time of last palm message

        # ── Publishers ──────────────────────────────────────────────────────────
        self.hand_pose_pub    = self.create_publisher(PoseStamped,      'hand/pose',            10)
        self.gripper_pub      = self.create_publisher(Float32,          'hand/gripper_cmd',     10)
        self.hand_width_pub   = self.create_publisher(Float32,          'hand/hand_width',      10)
        self.finger_tips_pub  = self.create_publisher(Float32MultiArray,'hand/finger_tips',     10)
        self.tracking_pub     = self.create_publisher(Bool,             'hand/tracking_active', 10)

        # ── Subscriptions ───────────────────────────────────────────────────────
        self.create_subscription(PoseStamped, '/hololens/palm/right',
                                 self._palm_cb,   10)
        self.create_subscription(PoseStamped, '/hololens/thumb/right',
                                 self._thumb_cb,  10)
        self.create_subscription(PoseStamped, '/hololens/index/right',
                                 self._index_cb,  10)
        self.create_subscription(PoseStamped, '/hololens/middle/right',
                                 self._middle_cb, 10)
        self.create_subscription(PoseStamped, '/hololens/ring/right',
                                 self._ring_cb,   10)
        self.create_subscription(PoseStamped, '/hololens/pinky/right',
                                 self._pinky_cb,  10)
        self.create_subscription(PoseStamped, '/hololens/gaze',
                                 self._gaze_cb,   10)

        # ── Processing timer at 30 Hz ───────────────────────────────────────────
        self.create_timer(1.0 / 30.0, self._process_and_publish)

        self.get_logger().info('HoloLens Hand Node initialized')

    # ── Callbacks (just store latest message) ───────────────────────────────
    def _palm_cb(self, msg: PoseStamped):
        self._palm_msg = msg
        self._last_palm_t = time.monotonic()

    def _thumb_cb(self,  msg: PoseStamped): self._thumb_msg  = msg
    def _index_cb(self,  msg: PoseStamped): self._index_msg  = msg
    def _middle_cb(self, msg: PoseStamped): self._middle_msg = msg
    def _ring_cb(self,   msg: PoseStamped): self._ring_msg   = msg
    def _pinky_cb(self,  msg: PoseStamped): self._pinky_msg  = msg
    def _gaze_cb(self,   msg: PoseStamped): self._gaze_msg   = msg

    # ── Helpers ──────────────────────────────────────────────────────────────
    def _is_tracking(self) -> bool:
        if self._last_palm_t is None:
            return False
        return (time.monotonic() - self._last_palm_t) < self.TRACKING_TIMEOUT_S

    @staticmethod
    def _pos_from_msg(msg: PoseStamped) -> np.ndarray:
        return np.array([msg.pose.position.x,
                         msg.pose.position.y,
                         msg.pose.position.z], dtype=np.float64)

    def _to_robot_frame(self, holo_pos: np.ndarray) -> np.ndarray:
        """Scale and offset HoloLens world position into the robot base frame.

        The HoloLens Unity publishers already convert from Unity left-handed
        coordinates to ROS right-handed convention. This function applies a
        further per-axis scale + offset to map the HoloLens workspace
        (in front of the participant) to the robot's reachable workspace.

        Calibration procedure (rough):
          1. Have the participant hold their palm at the robot home position.
          2. Note the published /hololens/palm/right values.
          3. Set workspace_{xyz}_offset = home_robot_xyz - holo_xyz_at_home.
          4. Adjust _scale if the motion range feels too large/small.

        For a more precise calibration, collect several reference points and
        fit a rigid-body transform using TF2 or scipy.
        """
        return np.array([
            holo_pos[0] * self.ws_x_scale + self.ws_x_offset,
            holo_pos[1] * self.ws_y_scale + self.ws_y_offset,
            holo_pos[2] * self.ws_z_scale + self.ws_z_offset,
        ], dtype=np.float32)

    # ── Main processing loop ─────────────────────────────────────────────────
    def _process_and_publish(self):
        tracking = self._is_tracking()
        self.tracking_pub.publish(Bool(data=tracking))

        if not tracking or self._palm_msg is None:
            return

        now = self.get_clock().now().to_msg()

        # ── hand/pose — right palm transformed to robot base frame ──────────
        palm_holo = self._pos_from_msg(self._palm_msg)
        robot_pos = self._to_robot_frame(palm_holo)
        robot_pos = self._pos_filter.filter(robot_pos)

        pose_msg = PoseStamped()
        pose_msg.header.stamp = now
        pose_msg.header.frame_id = 'base_link'
        pose_msg.pose.position.x = float(robot_pos[0])
        pose_msg.pose.position.y = float(robot_pos[1])
        pose_msg.pose.position.z = float(robot_pos[2])
        # Keep the palm orientation from HoloLens for wrist rotation tracking
        pose_msg.pose.orientation = self._palm_msg.pose.orientation
        self.hand_pose_pub.publish(pose_msg)

        # ── hand/hand_width + hand/gripper_cmd — from pinch gesture ─────────
        if self._thumb_msg is not None and self._index_msg is not None:
            thumb_pos = self._pos_from_msg(self._thumb_msg)
            index_pos = self._pos_from_msg(self._index_msg)
            hand_width = float(np.linalg.norm(thumb_pos - index_pos))

            self.hand_width_pub.publish(Float32(data=hand_width))

            # Map hand_width to gripper command with linear interpolation
            if hand_width <= self.pinch_close_m:
                raw_grip = 1.0
            elif hand_width >= self.pinch_open_m:
                raw_grip = 0.0
            else:
                span = self.pinch_open_m - self.pinch_close_m
                raw_grip = 1.0 - (hand_width - self.pinch_close_m) / span

            gripper_cmd = float(
                self._grip_filter.filter(np.array([raw_grip]))[0]
            )
            self.gripper_pub.publish(Float32(data=gripper_cmd))

        # ── hand/finger_tips — 5 × 3 = 15 floats (zeros if not tracked) ────
        tips = np.zeros(15, dtype=np.float32)
        for i, msg in enumerate([
            self._thumb_msg,
            self._index_msg,
            self._middle_msg,
            self._ring_msg,
            self._pinky_msg,
        ]):
            if msg is not None:
                tips[i * 3: i * 3 + 3] = self._pos_from_msg(msg)

        fa_msg = Float32MultiArray()
        fa_msg.data = tips.tolist()
        self.finger_tips_pub.publish(fa_msg)


def main(args=None):
    rclpy.init(args=args)
    node = HoloLensHandNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
