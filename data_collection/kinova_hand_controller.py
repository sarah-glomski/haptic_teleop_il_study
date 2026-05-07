#!/usr/bin/env python3
"""
Kinova Gen3 Hand Controller Node (ROS2)

Subscribes to hand tracking output from hololens_hand_node and controls the
Kinova Gen3 arm using a P-loop velocity controller (SendTwistCommand in
SINGLE_LEVEL_SERVOING mode). Mirrors the XArm mode-7 approach in
Robomimic/data_collection/xarm_hand_controller.py.

═══════════════════════════════════════════════════════════════════════════════
SAFETY ARCHITECTURE
═══════════════════════════════════════════════════════════════════════════════

1. TwistCommand watchdog (most critical)
   Every command sets duration = TWIST_WATCHDOG_MS. If this node crashes,
   hangs, or misses the control timer, the Kortex SDK stops the robot
   automatically after that many milliseconds. Never use duration=0.

2. Workspace position bounds
   Target position is hard-clipped to [x/y/z_min, x/y/z_max].
   A separate "soft zone" (workspace_soft_margin_m) reduces max speed
   linearly as the TCP approaches any wall. At the wall itself, velocity
   toward that wall is zero (still allows motion parallel / away from wall).

3. Linear velocity cap  (max_linear_speed_mps)
4. Angular velocity cap  (max_angular_speed_dps)

5. Velocity smoothing (vel_alpha)
   Exponential low-pass on the commanded velocity vector suppresses sudden
   large accelerations that would stress the arm mechanically.

6. Tracking-loss stop
   If hand/tracking_active goes False the target is cleared and a zero-
   velocity command is sent immediately.

7. Pause / Reset
   /pause_kinova (Bool): stops motion; resumes on False.
   /reset_kinova (Bool): moves to home position and opens gripper.

═══════════════════════════════════════════════════════════════════════════════

Subscribed topics:
  hand/pose            geometry_msgs/PoseStamped
  hand/gripper_cmd     std_msgs/Float32
  hand/tracking_active std_msgs/Bool
  /reset_kinova        std_msgs/Bool
  /pause_kinova        std_msgs/Bool
  /wrist_tracking      std_msgs/String    — "true"/"false"; gates arm movement (HoloLens Arm button)
  /gripper_movement    std_msgs/String    — "true"/"false"; gates gripper (HoloLens Gripper button)
  /vertical_only       std_msgs/String    — "true"/"false"; restricts motion to Z-axis only

Published topics:
  robot_goal/pose      geometry_msgs/PoseStamped  — live goal pose (always, arm need not be active)
  robot_goal/gripper   std_msgs/Float32           — live gripper target (always, gripper need not be active)
  robot_action/pose    geometry_msgs/PoseStamped  — goal pose echoed only while arm is active
  robot_action/gripper std_msgs/Float32

ROS2 Parameters:
  robot_ip                    str    '192.168.1.10'
  username / password         str    'admin' / 'admin'
  control_rate                float  30.0   Hz
  ── Safety ──────────────────────────────────────────────────────────────
  max_linear_speed_mps        float  0.10   m/s   hard velocity cap
  max_angular_speed_dps       float  15.0   deg/s hard angular cap
  vel_alpha                   float  0.4    velocity smoothing (0=very smooth)
  workspace_x_min / x_max     float  0.40 / 0.50  m
  workspace_y_min / y_max     float -0.27 / 0.27  m
  workspace_z_min / z_max     float  0.025 / 0.30  m
  workspace_soft_margin_m     float  0.04   soft deceleration zone (m from wall)
  ── EEF orientation ──────────────────────────────────────────────────────
  fixed_theta_x_deg           float  180.0
  fixed_theta_y_deg           float  0.0
  fixed_theta_z_offset_deg    float  0.0
  position_scale              float  1.0   hand-motion → robot-motion scale (all axes)
  p_gain                      float  2.0
  ── Home position (m / deg) ──────────────────────────────────────────────
  home_x / home_y / home_z    float  0.474 / 0.02 / 0.107
  home_tx / home_ty / home_tz float  -179.3 / -0.4 / 89.3
"""

import math
import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation as R
from rcl_interfaces.msg import SetParametersResult
from std_msgs.msg import Bool, Float32, String

from kortex_api.TCPTransport import TCPTransport
from kortex_api.RouterClient import RouterClient
from kortex_api.SessionManager import SessionManager
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.messages import Session_pb2, Base_pb2


# If the node misses this many ms worth of control ticks the Kortex SDK stops
# the robot automatically. Must be > 1/(control_rate) * 1000 to avoid false
# trips but short enough to stop the arm quickly on node crash.
# At 30 Hz one tick ≈ 33 ms → 200 ms gives ~6 missed cycles of margin.
TWIST_WATCHDOG_MS = 200


class KinovaHandController(Node):
    """
    ROS2 teleoperation controller for the Kinova Gen3.

    Safety: see module docstring.
    """

    # Hard boundary margin — clip target this far inside the software bounds
    # so we stay away from the Kortex firmware's own position limits.
    _HARD_MARGIN = 0.005  # 5 mm

    def __init__(self):
        super().__init__('kinova_hand_controller')

        # ── Parameters ─────────────────────────────────────────────────────────
        self.robot_ip  = self.declare_parameter('robot_ip',  '192.168.1.10').value
        self.username  = self.declare_parameter('username',  'admin').value
        self.password  = self.declare_parameter('password',  'admin').value

        self.control_rate          = self.declare_parameter('control_rate',            30.0).value
        self.max_linear_speed      = self.declare_parameter('max_linear_speed_mps',    0.10).value
        self.max_angular_speed     = self.declare_parameter('max_angular_speed_dps',   15.0).value
        self.vel_alpha             = self.declare_parameter('vel_alpha',               0.4).value

        self.x_min = self.declare_parameter('workspace_x_min',  0.40).value
        self.x_max = self.declare_parameter('workspace_x_max',  0.50).value
        self.y_min = self.declare_parameter('workspace_y_min', -0.27).value
        self.y_max = self.declare_parameter('workspace_y_max',  0.27).value
        self.z_min = self.declare_parameter('workspace_z_min',  0.025).value
        self.z_max = self.declare_parameter('workspace_z_max',  0.30).value
        self.soft_margin = self.declare_parameter('workspace_soft_margin_m', 0.01).value

        self.fixed_theta_x        = self.declare_parameter('fixed_theta_x_deg',        180.0).value
        self.fixed_theta_y        = self.declare_parameter('fixed_theta_y_deg',          0.0).value
        self.fixed_theta_z_offset = self.declare_parameter('fixed_theta_z_offset_deg',   0.0).value
        self.position_scale       = self.declare_parameter('position_scale',              1.0).value
        self.p_gain               = self.declare_parameter('p_gain',                    2.0).value

        self.home_x  = self.declare_parameter('home_x',   0.474).value
        self.home_y  = self.declare_parameter('home_y',   0.02).value
        self.home_z  = self.declare_parameter('home_z',   0.107).value
        self.home_tx = self.declare_parameter('home_tx', -179.3).value
        self.home_ty = self.declare_parameter('home_ty',   -0.4).value
        self.home_tz = self.declare_parameter('home_tz',   89.3).value

        # ── Connect and configure robot ─────────────────────────────────────────
        self.get_logger().info(f'Connecting to Kinova Gen3 at {self.robot_ip} …')
        self._connect()
        self._setup_servoing()

        # ── Safety — log configured limits at startup ───────────────────────────
        self.get_logger().info(
            f'\n'
            f'  ╔══ SAFETY LIMITS ══════════════════════════════╗\n'
            f'  ║ Workspace  X [{self.x_min:.3f}, {self.x_max:.3f}] m          ║\n'
            f'  ║            Y [{self.y_min:.3f}, {self.y_max:.3f}] m           ║\n'
            f'  ║            Z [{self.z_min:.3f}, {self.z_max:.3f}] m            ║\n'
            f'  ║ Soft zone  {self.soft_margin * 1000:.0f} mm from each wall        ║\n'
            f'  ║ Max linear {self.max_linear_speed * 1000:.0f} mm/s               ║\n'
            f'  ║ Max angular {self.max_angular_speed:.0f} deg/s               ║\n'
            f'  ║ Watchdog   {TWIST_WATCHDOG_MS} ms                        ║\n'
            f'  ╚══════════════════════════════════════════════╝'
        )

        # ── Controller state ────────────────────────────────────────────────────
        self.target_position       = None   # np.ndarray (3,) metres, already clipped
        self.target_theta_z_deg    = self.fixed_theta_z_offset
        self.gripper_cmd           = 0.0
        self.is_paused             = False
        self.is_resetting          = False
        self.hand_tracking_active  = False
        self._smoothed_vel         = np.zeros(3)  # exponentially smoothed velocity

        # References captured at arm-enable (after settling).
        # Position uses a delta approach: target = ref_robot_pos + (holo - ref_holo) * scale,
        # so workspace offsets in hololens_hand_node cancel out and no calibration is needed.
        self._ref_yaw: float | None = None
        self._ref_theta_z: float = self.home_tz
        self._ref_holo_pos: np.ndarray | None = None
        self._ref_robot_pos: np.ndarray | None = None
        self._identity_warned = False  # suppress repeated identity-rejection log spam
        # Monotonic time when arm tracking was last enabled.  Reference capture is
        # deferred 0.5 s to let MRTK hand tracking stabilise — MRTK can return
        # tracked=True with identity rotation for 1-2 frames on first acquisition,
        # and capturing that as the reference produces incorrect wrist commands.
        self._arm_enabled_at: float = 0.0

        # Fault state — set when the Kortex SDK reports ROBOT_IN_FAULT.
        # While faulted, all motion commands are suppressed and ClearFaults() is
        # retried every 3 s. arm_enabled is also cleared so the user must explicitly
        # re-press Arm in the HoloLens app after recovery.
        self._is_faulted = False
        self._last_fault_clear_t = 0.0

        # ── HoloLens safety gates (all off by default) ──────────────────────────
        self.arm_enabled     = False  # /wrist_tracking "true" to enable arm movement
        self.gripper_enabled = False  # /gripper_movement "true" to enable gripper
        self.vertical_only   = False  # /vertical_only "true" to restrict to Z-axis

        # ── Subscriptions ───────────────────────────────────────────────────────
        self.create_subscription(PoseStamped, 'hand/pose',            self._hand_pose_cb,       10)
        self.create_subscription(Float32,     'hand/gripper_cmd',     self._gripper_cb,         10)
        self.create_subscription(Bool,        'hand/tracking_active', self._tracking_status_cb, 10)
        self.create_subscription(Bool,        '/reset_kinova',        self._reset_cb,           10)
        self.create_subscription(Bool,        '/pause_kinova',        self._pause_cb,           10)
        self.create_subscription(String,      '/wrist_tracking',      self._arm_toggle_cb,      10)
        self.create_subscription(String,      '/gripper_movement',    self._gripper_toggle_cb,  10)
        self.create_subscription(String,      '/vertical_only',       self._vertical_toggle_cb, 10)

        # ── Publishers ──────────────────────────────────────────────────────────
        self.goal_pose_pub      = self.create_publisher(PoseStamped, 'robot_goal/pose',      10)
        self.goal_gripper_pub   = self.create_publisher(Float32,     'robot_goal/gripper',   10)
        self.action_pose_pub    = self.create_publisher(PoseStamped, 'robot_action/pose',    10)
        self.action_gripper_pub = self.create_publisher(Float32,     'robot_action/gripper', 10)

        # ── Control loop ────────────────────────────────────────────────────────
        self.create_timer(1.0 / self.control_rate, self._control_loop)

        # ── Live parameter tuning ───────────────────────────────────────────────
        self.add_on_set_parameters_callback(self._on_parameter_change)

    # ── Robot connection ──────────────────────────────────────────────────────
    def _connect(self):
        self._transport = TCPTransport()
        self._router = RouterClient(
            self._transport,
            lambda ex: self.get_logger().error(f'Kortex transport error: {ex}'),
        )
        self._transport.connect(self.robot_ip, 10000)

        session_info = Session_pb2.CreateSessionInfo()
        session_info.username = self.username
        session_info.password = self.password
        session_info.session_inactivity_timeout = 60000
        session_info.connection_inactivity_timeout = 2000

        self._session_manager = SessionManager(self._router)
        self._session_manager.CreateSession(session_info)

        self._base        = BaseClient(self._router)
        self._base_cyclic = BaseCyclicClient(self._router)
        self.get_logger().info('Connected to Kinova Gen3')

    def _setup_servoing(self):
        """Set SINGLE_LEVEL_SERVOING mode (enables SendTwistCommand)."""
        mode = Base_pb2.ServoingModeInformation()
        mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
        self._base.SetServoingMode(mode)
        time.sleep(0.2)
        self.get_logger().info('Kinova set to SINGLE_LEVEL_SERVOING (velocity control)')

    # ── Safety helpers ────────────────────────────────────────────────────────
    def _clip_to_workspace(self, pos: np.ndarray) -> np.ndarray:
        """Hard-clip position to safe workspace bounds."""
        m = self._HARD_MARGIN
        return np.array([
            np.clip(pos[0], self.x_min + m, self.x_max - m),
            np.clip(pos[1], self.y_min + m, self.y_max - m),
            np.clip(pos[2], self.z_min + m, self.z_max - m),
        ])

    def _boundary_speed_scale(self, current_pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
        """
        Soft boundary: reduce velocity component-wise as the TCP approaches
        any workspace wall.  When the TCP is at the wall the outward velocity
        component is zeroed; at soft_margin distance from the wall it is full.

        For each axis and each direction we compute a scale factor in [0, 1]:
          scale = clamp(dist_to_wall / soft_margin, 0, 1)
        and only apply it to the velocity *toward* that wall (the sign check
        ensures we can always move away from a boundary we've somehow entered).
        """
        if self.soft_margin <= 0:
            return vel

        out = vel.copy()
        bounds = [
            (0, self.x_min, self.x_max),
            (1, self.y_min, self.y_max),
            (2, self.z_min, self.z_max),
        ]
        for axis, lo, hi in bounds:
            pos = current_pos[axis]
            v   = out[axis]

            # Distance to lower wall: penalise negative (inward) velocity
            dist_lo = pos - lo
            if v < 0 and dist_lo < self.soft_margin:
                scale = max(0.0, dist_lo / self.soft_margin)
                out[axis] = v * scale

            # Distance to upper wall: penalise positive (inward) velocity
            dist_hi = hi - pos
            if v > 0 and dist_hi < self.soft_margin:
                scale = max(0.0, dist_hi / self.soft_margin)
                out[axis] = v * scale

        return out

    # ── Fault handling ────────────────────────────────────────────────────────
    def _enter_fault_state(self):
        """Called the first time a ROBOT_IN_FAULT error is detected."""
        if self._is_faulted:
            return
        self._is_faulted = True
        self.arm_enabled = False       # require explicit re-press of Arm after recovery
        self.target_position = None
        self._smoothed_vel[:] = 0.0
        self._ref_yaw = None
        self._ref_holo_pos = None
        self._ref_robot_pos = None
        self.get_logger().error(
            'ROBOT FAULT DETECTED — arm motion disabled. '
            'Auto-clear will be attempted every 3 s. '
            'Re-press Arm in HoloLens after fault clears to resume.'
        )

    def _try_clear_fault(self):
        """Attempt ClearFaults() + re-enter servoing mode. Rate-limited to 1 try / 3 s."""
        now = time.monotonic()
        if now - self._last_fault_clear_t < 3.0:
            return
        self._last_fault_clear_t = now
        try:
            self._base.ClearFaults()
            time.sleep(0.5)
            self._setup_servoing()
            self._is_faulted = False
            self.get_logger().warn(
                'Robot fault cleared — re-entering servoing mode. '
                'Re-press Arm in HoloLens to resume movement.'
            )
        except Exception as e:
            self.get_logger().error(f'Fault clear attempt failed: {e}')

    # ── Callbacks ─────────────────────────────────────────────────────────────
    @staticmethod
    def _palm_yaw_deg(o) -> float:
        """Extract yaw (rotation about ROS z-axis) from a quaternion message."""
        return math.degrees(math.atan2(
            2.0 * (o.w * o.z + o.x * o.y),
            1.0 - 2.0 * (o.y * o.y + o.z * o.z),
        ))

    def _hand_pose_cb(self, msg: PoseStamped):
        current_holo_pos = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])

        # ── Always publish goal pose for visualization ────────────────────────
        # Uses the delta approach so the goal reflects where the robot would be,
        # not the raw (uncalibrated) HoloLens world position.
        raw_yaw = self._palm_yaw_deg(msg.pose.orientation)
        if self._ref_holo_pos is not None and self._ref_robot_pos is not None:
            delta = (current_holo_pos - self._ref_holo_pos) * self.position_scale
            goal_pos = self._clip_to_workspace(self._ref_robot_pos + delta)
        else:
            goal_pos = np.array([self.home_x, self.home_y, self.home_z])
        if self._ref_yaw is not None:
            d = (raw_yaw - self._ref_yaw + 180.0) % 360.0 - 180.0
            goal_theta_z = self.fixed_theta_z_offset + self._ref_theta_z + d
        else:
            goal_theta_z = self.fixed_theta_z_offset + self.home_tz
        self._publish_pose(self.goal_pose_pub, goal_pos, goal_theta_z, msg.header.stamp)

        if self.is_resetting or self.is_paused or not self.arm_enabled or self._is_faulted:
            return

        # ── Settling period ───────────────────────────────────────────────────
        # Hold robot completely still for 0.5 s after arm tracking is enabled.
        # MRTK can return tracked=True with identity rotation for the first 1-2
        # frames on hand acquisition; capturing that as the yaw reference produces
        # wrong wrist commands.  During settling no targets are updated.
        if self._ref_yaw is None:
            if time.monotonic() - self._arm_enabled_at < 0.5:
                return  # still settling — don't move yet
            # Reject near-identity quaternion: MRTK can return tracked=True with
            # w≈1, xyz≈0 for several frames after first acquisition.  Any real hand
            # pose has a rotation angle well above 5° from identity.
            o = msg.pose.orientation
            angle_deg = math.degrees(2.0 * math.acos(min(1.0, abs(o.w))))
            if angle_deg < 5.0:
                if not self._identity_warned:
                    self._identity_warned = True
                    self.get_logger().warn(
                        f'Palm orientation still near-identity ({angle_deg:.1f}°) after settling — '
                        'waiting for real MRTK tracking. Re-raise your hand if this persists.'
                    )
                return
            self._identity_warned = False
            # Capture all references from the first stable pose
            yaw_deg = self._palm_yaw_deg(msg.pose.orientation)
            self._ref_yaw = yaw_deg
            self._ref_holo_pos = current_holo_pos.copy()
            try:
                fb = self._base_cyclic.RefreshFeedback()
                self._ref_theta_z = fb.base.tool_pose_theta_z
                self._ref_robot_pos = np.array([
                    fb.base.tool_pose_x,
                    fb.base.tool_pose_y,
                    fb.base.tool_pose_z,
                ])
            except Exception:
                self._ref_theta_z = self.home_tz
                self._ref_robot_pos = np.array([self.home_x, self.home_y, self.home_z])
            self.get_logger().info(
                f'Reference captured (after settling): '
                f'palm_yaw={yaw_deg:.1f}°  robot_theta_z={self._ref_theta_z:.1f}°  '
                f'robot_pos=({self._ref_robot_pos[0]:.3f}, {self._ref_robot_pos[1]:.3f}, {self._ref_robot_pos[2]:.3f})  '
                f'holo_pos=({current_holo_pos[0]:.3f}, {current_holo_pos[1]:.3f}, {current_holo_pos[2]:.3f})'
            )

        # ── Position target (delta from reference) ────────────────────────────
        delta = (current_holo_pos - self._ref_holo_pos) * self.position_scale
        clipped = self._clip_to_workspace(self._ref_robot_pos + delta)

        if self.vertical_only and self.target_position is not None:
            self.target_position[2] = clipped[2]
        else:
            self.target_position = clipped

        # ── Orientation target (delta wrist yaw) ──────────────────────────────
        yaw_deg = self._palm_yaw_deg(msg.pose.orientation)
        delta_yaw = (yaw_deg - self._ref_yaw + 180.0) % 360.0 - 180.0
        self.target_theta_z_deg = self.fixed_theta_z_offset + self._ref_theta_z + delta_yaw

        self._publish_pose(self.action_pose_pub, self.target_position, self.target_theta_z_deg, msg.header.stamp)

    def _gripper_cb(self, msg: Float32):
        # Always publish goal gripper for visualization, regardless of enabled state.
        goal_gripper = float(np.clip(msg.data, 0.0, 1.0))
        self.goal_gripper_pub.publish(Float32(data=goal_gripper))

        if self.is_resetting or self.is_paused or not self.gripper_enabled or self._is_faulted:
            return

        self.gripper_cmd = goal_gripper

        try:
            gripper_cmd = Base_pb2.GripperCommand()
            gripper_cmd.mode = Base_pb2.GRIPPER_POSITION
            finger = gripper_cmd.gripper.finger.add()
            finger.finger_identifier = 1
            finger.value = self.gripper_cmd
            self._base.SendGripperCommand(gripper_cmd)
        except Exception as e:
            err_str = str(e)
            if 'ROBOT_IN_FAULT' in err_str or 'IN_FAULT' in err_str:
                self._enter_fault_state()
            else:
                self.get_logger().error(f'Gripper command error: {e}')
            return

        # Read back actual position (may differ from command if grasping)
        try:
            fb = self._base.GetMeasuredGripperMovement(
                Base_pb2.GripperRequest(mode=Base_pb2.GRIPPER_POSITION)
            )
            actual = float(fb.finger[0].value) if fb.finger else self.gripper_cmd
        except Exception:
            actual = self.gripper_cmd

        self.action_gripper_pub.publish(Float32(data=actual))

    # ── Live parameter updates ────────────────────────────────────────────────
    def _on_parameter_change(self, params):
        for p in params:
            if p.name == 'vel_alpha':
                self.vel_alpha = float(p.value)
                self.get_logger().info(f'vel_alpha → {p.value}')
            elif p.name == 'p_gain':
                self.p_gain = float(p.value)
                self.get_logger().info(f'p_gain → {p.value}')
            elif p.name == 'position_scale':
                self.position_scale = float(p.value)
                self.get_logger().info(f'position_scale → {p.value}')
        return SetParametersResult(successful=True)

    def _arm_toggle_cb(self, msg: String):
        enabled = msg.data.strip().lower() == 'true'
        if self.arm_enabled and not enabled:
            self.target_position = None
            self._smoothed_vel[:] = 0.0
            self._ref_yaw = None
            self._ref_holo_pos = None
            self._ref_robot_pos = None
            self._send_zero_twist()
            self.get_logger().info('Arm tracking disabled — robot stopped')
        elif not self.arm_enabled and enabled:
            self._ref_yaw = None
            self._ref_holo_pos = None
            self._ref_robot_pos = None
            self._identity_warned = False
            self._arm_enabled_at = time.monotonic()
            self.get_logger().info(
                'Arm tracking enabled — robot held still for 0.5 s while MRTK tracking settles, '
                'then wrist reference will be captured'
            )
        self.arm_enabled = enabled

    def _gripper_toggle_cb(self, msg: String):
        enabled = msg.data.strip().lower() == 'true'
        if enabled != self.gripper_enabled:
            self.get_logger().info(f'Gripper control {"enabled" if enabled else "disabled"}')
        self.gripper_enabled = enabled

    def _vertical_toggle_cb(self, msg: String):
        v_only = msg.data.strip().lower() == 'true'
        if v_only != self.vertical_only:
            self.get_logger().info(f'Vertical-only mode {"enabled" if v_only else "disabled"}')
        self.vertical_only = v_only

    def _tracking_status_cb(self, msg: Bool):
        was_active = self.hand_tracking_active
        self.hand_tracking_active = msg.data
        if was_active and not self.hand_tracking_active:
            self.target_position = None
            self._smoothed_vel[:] = 0.0
            self._ref_yaw = None
            self._ref_holo_pos = None
            self._ref_robot_pos = None
            self._arm_enabled_at = time.monotonic()  # re-settle when tracking resumes
            self._send_zero_twist()
            self.get_logger().warn('Hand tracking lost — robot stopped; will re-settle on re-acquisition')

    def _reset_cb(self, msg: Bool):
        if not msg.data or self.is_resetting:
            return
        self.get_logger().info('Resetting Kinova Gen3 to home …')
        self.is_resetting = True
        self.target_position = None
        self._smoothed_vel[:] = 0.0
        self._send_zero_twist()
        threading.Thread(target=self._do_reset, daemon=True).start()

    def _do_reset(self):
        try:
            action = Base_pb2.Action()
            action.name = 'Home'
            action.application_data = ''

            speed = Base_pb2.CartesianSpeed()
            speed.translation = 0.08  # m/s — conservative reset speed
            speed.orientation = 12.0  # deg/s
            action.reach_pose.constraint.speed.CopyFrom(speed)

            pose = action.reach_pose.target_pose
            pose.x = self.home_x
            pose.y = self.home_y
            pose.z = self.home_z
            pose.theta_x = self.home_tx
            pose.theta_y = self.home_ty
            pose.theta_z = self.home_tz

            finished = threading.Event()

            def _on_action(notif, ev=finished):
                if notif.action_event in (Base_pb2.ACTION_END, Base_pb2.ACTION_ABORT):
                    ev.set()

            self._base.OnNotificationActionTopic(_on_action, Base_pb2.NotificationOptions())
            self._base.ExecuteAction(action)

            if not finished.wait(timeout=30.0):
                self.get_logger().warn('Home reset timed out — aborting')
                self._base.StopAction()
            else:
                # Open gripper
                gc = Base_pb2.GripperCommand()
                gc.mode = Base_pb2.GRIPPER_POSITION
                f = gc.gripper.finger.add()
                f.finger_identifier = 1
                f.value = 0.0
                self._base.SendGripperCommand(gc)
                time.sleep(1.0)
                self.get_logger().info('Reset complete — gripper open')

        except Exception as e:
            self.get_logger().error(f'Reset error: {e}')
        finally:
            try:
                self._setup_servoing()
            except Exception:
                pass
            self.is_resetting = False

    def _pause_cb(self, msg: Bool):
        self.is_paused = msg.data
        if self.is_paused:
            self._smoothed_vel[:] = 0.0
            self._send_zero_twist()
            self.get_logger().info('Kinova paused')
        else:
            self.get_logger().info('Kinova resumed')

    # ── Publishers ────────────────────────────────────────────────────────────
    def _publish_pose(self, pub, position: np.ndarray, theta_z_deg: float, stamp):
        msg = PoseStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = 'base_link'
        msg.pose.position.x = float(position[0])
        msg.pose.position.y = float(position[1])
        msg.pose.position.z = float(position[2])

        quat = R.from_euler('xyz', [
            math.radians(self.fixed_theta_x),
            math.radians(self.fixed_theta_y),
            math.radians(theta_z_deg),
        ]).as_quat()
        msg.pose.orientation.x = float(quat[0])
        msg.pose.orientation.y = float(quat[1])
        msg.pose.orientation.z = float(quat[2])
        msg.pose.orientation.w = float(quat[3])
        pub.publish(msg)

    # ── Control loop ──────────────────────────────────────────────────────────
    def _control_loop(self):
        """
        30 Hz P-loop velocity controller with:
          - velocity smoothing (exponential filter)
          - soft boundary deceleration
          - hard velocity cap
          - TwistCommand watchdog duration
          - fault detection + auto-recovery
        """
        # Fault takes priority: suppress all commands and attempt periodic clear.
        if self._is_faulted:
            self._try_clear_fault()
            return

        if self.is_paused or self.is_resetting or self.target_position is None:
            self._smoothed_vel[:] = 0.0
            self._send_zero_twist()
            return

        try:
            feedback = self._base_cyclic.RefreshFeedback()
            current_pos = np.array([
                feedback.base.tool_pose_x,
                feedback.base.tool_pose_y,
                feedback.base.tool_pose_z,
            ])
            current_theta_z = feedback.base.tool_pose_theta_z

            # P-loop: raw velocity from position error
            pos_error = self.target_position - current_pos
            raw_vel = self.p_gain * pos_error

            # 1 — Hard cap before smoothing (safety)
            speed = float(np.linalg.norm(raw_vel))
            if speed > self.max_linear_speed:
                raw_vel = raw_vel * (self.max_linear_speed / speed)

            # 2 — Soft boundary deceleration (uses current TCP position)
            raw_vel = self._boundary_speed_scale(current_pos, raw_vel)

            # 3 — Velocity smoothing (limits effective acceleration)
            self._smoothed_vel = (
                self.vel_alpha * raw_vel
                + (1.0 - self.vel_alpha) * self._smoothed_vel
            )

            # 4 — Final hard cap on smoothed velocity
            smooth_speed = float(np.linalg.norm(self._smoothed_vel))
            send_vel = self._smoothed_vel.copy()
            if smooth_speed > self.max_linear_speed:
                send_vel = send_vel * (self.max_linear_speed / smooth_speed)

            # Angular P-loop (theta_z only; theta_x/y are kept fixed)
            theta_z_err = self.target_theta_z_deg - current_theta_z
            theta_z_err = (theta_z_err + 180.0) % 360.0 - 180.0  # wrap [-180, 180]
            ang_vel_z = float(
                np.clip(self.p_gain * theta_z_err,
                        -self.max_angular_speed, self.max_angular_speed)
            )

            # ── Send TwistCommand ─────────────────────────────────────────────
            # duration = TWIST_WATCHDOG_MS: robot auto-stops if we miss this
            # many ms worth of ticks (node crash / freeze protection).
            cmd = Base_pb2.TwistCommand()
            cmd.reference_frame = Base_pb2.CARTESIAN_REFERENCE_FRAME_MIXED
            cmd.duration = TWIST_WATCHDOG_MS
            cmd.twist.linear_x  = float(send_vel[0])
            cmd.twist.linear_y  = float(send_vel[1])
            cmd.twist.linear_z  = float(send_vel[2])
            cmd.twist.angular_x = 0.0
            cmd.twist.angular_y = 0.0
            cmd.twist.angular_z = ang_vel_z
            self._base.SendTwistCommand(cmd)

        except Exception as e:
            err_str = str(e)
            if 'ROBOT_IN_FAULT' in err_str or 'IN_FAULT' in err_str:
                self._enter_fault_state()
            else:
                self.get_logger().error(f'Control loop error: {e}')
                self._send_zero_twist()

    def _send_zero_twist(self):
        """Zero-velocity command with watchdog duration (safe stop)."""
        try:
            cmd = Base_pb2.TwistCommand()
            cmd.reference_frame = Base_pb2.CARTESIAN_REFERENCE_FRAME_MIXED
            cmd.duration = TWIST_WATCHDOG_MS
            self._base.SendTwistCommand(cmd)
        except Exception:
            pass

    # ── Cleanup ────────────────────────────────────────────────────────────────
    def destroy_node(self):
        self.get_logger().info('Disconnecting from Kinova Gen3 …')
        try:
            self._send_zero_twist()
            self._session_manager.CloseSession()
            self._transport.disconnect()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = KinovaHandController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
