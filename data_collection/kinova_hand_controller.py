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
  All tunables (robot connection, safety caps, workspace bounds, gains, home
  pose, orientation clamps) are declared via declare_parameter() in __init__ —
  see there for the authoritative names, defaults, and inline notes. Override
  any of them at launch with -p <name>:=<value>; vel_alpha, p_gain,
  position_scale and max_linear_speed_mps are also live-tunable at runtime
  (see _on_parameter_change).
"""

import math
import signal
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

# Shared safety limits and home posture, also used by testing/inference.py, so
# a workspace bound or speed cap edited there moves both consumers.
# NOTE: this node still owns its own Kortex session and control loop (the
# imports above) — only the LIMITS are shared so far. Rewiring _connect and the
# control loop onto KinovaArm is the remaining half of that refactor.
from kinova_arm import ArmLimits, HOME_JOINTS_DEG, TWIST_WATCHDOG_MS

# Optional dependency for tracking_mode:=ff_ruckig (pip install ruckig).
try:
    from ruckig import InputParameter, OutputParameter, Result, Ruckig
    RUCKIG_AVAILABLE = True
except ImportError:
    RUCKIG_AVAILABLE = False


def spring_reference_step(ref_pos, ref_vel, target, wn, dt):
    """One semi-implicit Euler step of a critically-damped 2nd-order reference
    model chasing `target`. Returns the new (ref_pos, ref_vel).

    Used by tracking_mode:=ff_spring: the model provides a smooth position
    reference AND its consistent velocity (the feed-forward term) — replacing
    the legacy EMA filters, whose smoothing came at the cost of pure lag.
    """
    acc = wn * wn * (target - ref_pos) - 2.0 * wn * ref_vel
    ref_vel = ref_vel + acc * dt
    ref_pos = ref_pos + ref_vel * dt
    return ref_pos, ref_vel


# TWIST_WATCHDOG_MS and HOME_JOINTS_DEG are imported from kinova_arm — change
# them there and BOTH teleop and inference pick the new values up.


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

        # ── Shared safety limits ─────────────────────────────────────────────
        # Every default below comes from kinova_arm.ArmLimits, which
        # testing/inference.py loads too. Edit a workspace bound, a speed cap,
        # an orientation clamp or the home posture THERE and both teleop and
        # policy rollout move together — that coupling is the entire point of
        # the shared module. These stay ROS params so a single session can be
        # overridden at launch without editing code.
        _lim = ArmLimits()

        self.control_rate          = self.declare_parameter('control_rate',            30.0).value
        self.max_linear_speed      = self.declare_parameter('max_linear_speed_mps',    _lim.max_linear_speed_mps).value
        self.max_angular_speed     = self.declare_parameter('max_angular_speed_dps',   _lim.max_angular_speed_dps).value
        self.vel_alpha             = self.declare_parameter('vel_alpha',               _lim.vel_alpha).value

        self.x_min = self.declare_parameter('workspace_x_min',  _lim.x[0]).value
        self.x_max = self.declare_parameter('workspace_x_max',  _lim.x[1]).value
        self.y_min = self.declare_parameter('workspace_y_min',  _lim.y[0]).value
        self.y_max = self.declare_parameter('workspace_y_max',  _lim.y[1]).value
        self.z_min = self.declare_parameter('workspace_z_min',  _lim.z[0]).value
        self.z_max = self.declare_parameter('workspace_z_max',  _lim.z[1]).value
        self.soft_margin = self.declare_parameter('workspace_soft_margin_m', _lim.soft_margin_m).value

        self.position_scale       = self.declare_parameter('position_scale',              1.0).value
        self.p_gain               = self.declare_parameter('p_gain',                    _lim.p_gain).value

        # ── Tracking mode (latency work, 2026-07) ────────────────────────────
        # legacy    : original P + EMA path, byte-identical (default).
        # ff_spring : critically-damped 2nd-order reference model supplies
        #             (x_ref, v_ref); law v = ff_gain·v_ref + p_gain·(x_ref−x).
        #             Kills the v/p_gain tracking gap. Knob: spring_wn.
        # ff_ruckig : same law, reference from Ruckig time-optimal OTG under
        #             explicit accel/jerk limits. Knobs: ruckig_amax/jmax.
        # Once one mode wins on hardware, prune the others.
        self.tracking_mode = self.declare_parameter('tracking_mode', 'legacy').value
        self.ff_gain       = self.declare_parameter('ff_gain',       1.0).value
        self.spring_wn     = self.declare_parameter('spring_wn',     8.0).value
        self.ruckig_amax   = self.declare_parameter('ruckig_amax',   2.0).value
        self.ruckig_jmax   = self.declare_parameter('ruckig_jmax',   20.0).value
        # Damping against measured robot velocity (ff modes only). Counters the
        # reversal overshoot caused by the arm's internal twist-execution lag
        # (measured 2026-07-27: x1.2-1.7 amplitude gain, worst on z). 0 = off.
        self.d_gain        = self.declare_parameter('d_gain',        0.5).value
        if self.tracking_mode == 'ff_ruckig' and not RUCKIG_AVAILABLE:
            self.get_logger().warn('ruckig not installed — falling back to ff_spring')
            self.tracking_mode = 'ff_spring'
        # Reject hand-tracking glitches: a target jump implying hand speed above
        # this (m/s) is clamped to it (ff modes only). Measured 2026-07-27: the
        # HoloLens invented ~19 cm of vertical motion while the hand was still.
        self.hand_speed_limit = self.declare_parameter('hand_speed_limit', 1.5).value
        # Reference-generator state (None = re-anchor to the robot on next tick)
        self._ref_pos = None
        self._ref_vel = np.zeros(3)
        self._rk = None   # (Ruckig otg, InputParameter, OutputParameter) triple
        # Robot-velocity estimate (feedback deltas) — used by the ff damping
        # term and by the stall guard.
        self._fb_prev_pos = None
        self._fb_prev_t = None
        self._robot_vel_est = np.zeros(3)

        # ── Stall guard (see _check_stall) ───────────────────────────────────
        # Trips when the arm is commanded above stall_cmd_mps but measured
        # moving below stall_move_mps for stall_timeout_s continuously, and
        # re-anchors the clutch instead of letting the hand/arm offset build up
        # into a max-speed dash. Set stall_timeout_s to 0 to disable.
        self.stall_timeout_s = self.declare_parameter('stall_timeout_s', _lim.stall_timeout_s).value
        self.stall_cmd_mps   = self.declare_parameter('stall_cmd_mps',   _lim.stall_cmd_mps).value
        self.stall_move_mps  = self.declare_parameter('stall_move_mps',  _lim.stall_move_mps).value
        self._stall_since = None

        # ── Orientation teleop (default ON; --no-orientation locks the wrist) ──
        # launch_data_collection.py / launch_teleop.py always pass this
        # explicitly; --no-orientation is what sends false, pinning the
        # end-effector at the home orientation for translation-only teleop.
        # The wrist tracks the palm orientation as a CLUTCHED DELTA from the
        # enable-time reference — same scheme as position, so it starts from the
        # robot's current orientation with no jump. Control uses
        # quaternion/rotation-matrix error (rotvec), NEVER raw Kortex Euler
        # error: the Kortex ZYX decomposition has a gimbal singularity at
        # theta_x≈±180° (the home orientation) that makes Euler components flip;
        # rotation-matrix error is immune to representation flips.
        self.enable_orientation   = self.declare_parameter('enable_orientation',      True).value
        # Safety clamps, applied to the target's rotation FROM HOME. The
        # rotation-from-home rotvec is split into three INDEPENDENT per-axis
        # components (base frame) and each is clamped separately:
        #   - roll  = component about base x   (max_roll_deg, symmetric)
        #   - pitch = component about base y   (pitch_min_deg..pitch_max_deg, asymmetric)
        #   - yaw   = component about base z   (yaw_min_deg..yaw_max_deg, asymmetric)
        # Roll/pitch keep the gripper from tilting into the table (the XYZ
        # workspace box assumes a roughly downward tool; tilting sweeps the
        # fingers below the TCP). Yaw must also stay clear of the wrist joint's
        # travel limits (sustained chase into a joint limit faults the arm —
        # seen at ~38 s in earlier testing). NOTE: roll↔base-x, pitch↔base-y,
        # yaw↔base-z is a naming convention; relabel the limits if your tool
        # axes are named the other way.
        # Current defaults live in kinova_arm.ArmLimits and are shared with
        # inference: roll locked to ±3°, PITCH open to -48° for tool tilt, YAW
        # open to +93° toward tool-forward, each with only 3° the other way.
        # See the WARNING there — tilting sweeps the fingers below the TCP, and
        # the workspace z floor is enforced on the TCP only.
        self.max_roll_deg         = self.declare_parameter('max_roll_deg',      _lim.max_roll_deg).value
        self.pitch_min_deg        = self.declare_parameter('pitch_min_deg',     _lim.pitch_min_deg).value
        self.pitch_max_deg        = self.declare_parameter('pitch_max_deg',     _lim.pitch_max_deg).value
        self.yaw_min_deg          = self.declare_parameter('yaw_min_deg',       _lim.yaw_min_deg).value
        self.yaw_max_deg          = self.declare_parameter('yaw_max_deg',       _lim.yaw_max_deg).value
        self.orientation_p_gain   = self.declare_parameter('orientation_p_gain', _lim.orientation_p_gain).value
        # Yaw offset (deg) between the HoloLens world frame and the robot base
        # frame. The orientation delta is expressed in HoloLens-world axes; if the
        # operator does not face along the robot's +x, conjugate the delta by this
        # yaw so hand tilts map to matching robot tilts. 0 = frames assumed aligned.
        self.orientation_yaw_offset_deg = self.declare_parameter('orientation_yaw_offset_deg', 0.0).value

        # Cartesian home — kept consistent with HOME_JOINTS_DEG (the joint-space
        # reset target); used for the workspace/orientation reference math.
        self.home_x  = self.declare_parameter('home_x',   0.5715).value
        self.home_y  = self.declare_parameter('home_y',   0.0137).value
        self.home_z  = self.declare_parameter('home_z',   0.1068).value
        self.home_tx = self.declare_parameter('home_tx', -180.0).value
        self.home_ty = self.declare_parameter('home_ty',   0.0).value
        self.home_tz = self.declare_parameter('home_tz',   90.0).value

        # ── Connect and configure robot ─────────────────────────────────────────
        self.get_logger().info(f'Connecting to Kinova Gen3 at {self.robot_ip} …')
        self._connect()
        self._setup_servoing()

        # ── Safety — log configured limits at startup ───────────────────────────
        _rows = [
            f' Workspace  X [{self.x_min:.3f}, {self.x_max:.3f}] m',
            f'            Y [{self.y_min:.3f}, {self.y_max:.3f}] m',
            f'            Z [{self.z_min:.3f}, {self.z_max:.3f}] m',
            f' Soft zone  {self.soft_margin * 1000:.0f} mm from each wall',
            f' Max linear {self.max_linear_speed * 1000:.0f} mm/s',
            f' Max angular {self.max_angular_speed:.0f} deg/s',
            f' Watchdog   {TWIST_WATCHDOG_MS} ms',
            (f' Orientation ON — roll ≤ ±{self.max_roll_deg:.0f}°, '
             f'pitch ∈ [{self.pitch_min_deg:.0f}°, {self.pitch_max_deg:.0f}°], '
             f'yaw ∈ [{self.yaw_min_deg:.0f}°, {self.yaw_max_deg:.0f}°] from home'
             if self.enable_orientation else
             ' Orientation LOCKED at home (translation-only; --no-orientation)'),
        ]
        _W = max(len(r) for r in _rows) + 2
        self.get_logger().info(
            '\n'
            f'  ╔══ SAFETY LIMITS {"═" * (_W - 17)}╗\n' +
            ''.join(f'  ║{r:<{_W}}║\n' for r in _rows) +
            f'  ╚{"═" * _W}╝'
        )

        # ── Controller state ────────────────────────────────────────────────────
        self.target_position       = None   # np.ndarray (3,) metres, already clipped
        self.target_theta_z_deg    = self.home_tz
        self.gripper_cmd           = 0.0
        self.is_paused             = False
        self.is_resetting          = False
        self.hand_tracking_active  = False
        self._smoothed_vel         = np.zeros(3)  # exponentially smoothed velocity

        # Orientation-mode state (only used when enable_orientation)
        self.target_rot            = None   # scipy Rotation — clamped target orientation
        self._smoothed_ang_vel     = np.zeros(3)  # smoothed angular velocity (deg/s, base frame)
        self._home_rot             = R.from_euler('xyz', [
            math.radians(self.home_tx),
            math.radians(self.home_ty),
            math.radians(self.home_tz),
        ])

        # References captured at arm-enable (after settling).
        # Position uses a delta approach: target = ref_robot_pos + (holo - ref_holo) * scale,
        # so workspace offsets in hololens_hand_node cancel out and no calibration is needed.
        # Yaw is fixed at home_tz — not tracked from hand orientation.
        self._ref_holo_pos: np.ndarray | None = None
        self._ref_robot_pos: np.ndarray | None = None
        # Orientation references, captured in the SAME settling block as position
        # (so anything that clears _ref_robot_pos re-captures these too).
        self._ref_holo_rot = None    # scipy Rotation of the palm at enable
        self._ref_robot_rot = None   # scipy Rotation of the TCP at enable
        # Monotonic time when arm tracking was last enabled. Reference capture is
        # deferred `reference_settle_seconds` to let MRTK hand tracking stabilise —
        # MRTK can return tracked=True with identity rotation for 1-2 frames on
        # first acquisition, and capturing that as the reference can cause a wrist
        # jump. Default 0.0 = capture immediately (no start lag); raise it (e.g.
        # 0.2-0.5) only if you see an orientation jump right after pressing Arm.
        self.settle_seconds = self.declare_parameter('reference_settle_seconds', 0.0).value
        self._arm_enabled_at: float = 0.0

        # Fault state — set when the Kortex SDK reports ROBOT_IN_FAULT.
        # LATCHED for the life of the process: all motion commands are suppressed
        # and there is no auto-clear, because recovering silently makes the arm
        # dash to catch up with the hand (see _enter_fault_state).
        self._is_faulted = False
        self._last_fault_clear_t = 0.0   # also throttles the periodic fault nag

        # ── HoloLens safety gates (all off by default) ──────────────────────────
        self.arm_enabled     = False  # /wrist_tracking "true" to enable arm movement
        self.gripper_enabled = False  # /gripper_movement "true" to enable gripper
        self.vertical_only   = False  # /vertical_only "true" to restrict to Z-axis

        # ── Subscriptions ───────────────────────────────────────────────────────
        self.create_subscription(PoseStamped, 'hand/pose',            self._hand_pose_cb,       10)
        self.create_subscription(Float32,     'hand/gripper_cmd',     self._gripper_cb,         1)
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
        """Latch into fault and STAY stopped. There is deliberately no recovery.

        A fault means the arm stopped while the operator's hand kept moving, so
        the clutch offset between hand and robot is now arbitrarily large.
        Bringing the arm back online silently makes the P-loop dash to close
        that gap at max_linear_speed.

        That is not hypothetical: on 2026-07-28 a fault auto-cleared 0.8 s after
        it was raised and the arm dashed into the table, destroying the task
        objects. Two safeties failed together —
          1. auto-clear restored servoing without any human decision, and
          2. `arm_enabled = False` below did NOT hold, because the HoloLens
             republishes arm_enable=true continuously: the operator had not
             touched the toggle, so _arm_toggle_cb re-armed 1 ms later.
        So the fault now latches for the life of the process, and
        _arm_toggle_cb ignores the HoloLens while it is latched. Recovery is
        deliberate: clear the fault in the Kinova web app, then restart teleop
        — which re-anchors the clutch from scratch.
        """
        if self._is_faulted:
            return
        self._is_faulted = True
        self.arm_enabled = False
        self.target_position = None
        self.target_rot = None
        self._smoothed_vel[:] = 0.0
        self._smoothed_ang_vel[:] = 0.0
        self._ref_holo_pos = None
        self._ref_robot_pos = None
        self._ref_holo_rot = None
        self._ref_robot_rot = None
        self._ref_pos = None
        self._rk = None
        self.get_logger().error(
            'ROBOT FAULT DETECTED — arm motion disabled and LATCHED. '
            'No auto-clear: re-enabling would make the arm dash to catch up '
            'with your hand. To recover: clear the fault in the Kinova web app, '
            'then restart the teleop stack.'
        )

    def _reanchor_clutch(self, reason: str):
        """Drop every clutch reference so tracking restarts from where the arm
        ACTUALLY is, with zero offset.

        Setting the references to None makes _hand_pose_cb re-capture them from
        the current robot pose and the current hand pose on its next message —
        the same path used when Arm is first pressed. _arm_enabled_at is reset
        too so the settling window (reference_settle_seconds) applies again.
        """
        self.target_position = None
        self.target_rot = None
        self._ref_holo_pos = None
        self._ref_robot_pos = None
        self._ref_holo_rot = None
        self._ref_robot_rot = None
        self._smoothed_vel[:] = 0.0
        self._smoothed_ang_vel[:] = 0.0
        self._ref_pos = None
        self._rk = None
        self._arm_enabled_at = time.monotonic()
        self.get_logger().warn(f'Clutch re-anchored — {reason}')

    def _check_stall(self, current_pos, send_vel) -> bool:
        """True when the arm is commanded to move but isn't. Re-anchors on trip.

        The P-loop chases target_position with no bound on how far that target
        may drift from the arm's real position. Whenever the arm stops for a
        reason this node cannot observe — a fault, a manual-control takeover,
        a twist-watchdog expiry, an RPC stall — the operator's hand keeps going
        and the offset grows for as long as the stall lasts. The instant the arm
        frees up, that offset is a position error, and the P-loop closes it at
        max_linear_speed. On 2026-07-28 a ~3 s stall did exactly this and threw
        the arm across the table.

        Detecting the STALL rather than the gap size is deliberate: in legacy
        (pure-P) mode the steady-state tracking error is speed-proportional
        (error = v / p_gain — at p_gain 2.0 a brisk 0.3 m/s hand move legitimately
        sits 15 cm behind), so a plain gap threshold would fire constantly during
        normal fast teleop. "Commanded but not moving" has no such false positive.
        """
        if self.stall_timeout_s <= 0.0:
            return False

        commanded = float(np.linalg.norm(send_vel))
        measured  = float(np.linalg.norm(self._robot_vel_est))

        if commanded < self.stall_cmd_mps or measured > self.stall_move_mps:
            self._stall_since = None
            return False

        now = time.monotonic()
        if self._stall_since is None:
            self._stall_since = now
            return False
        if now - self._stall_since < self.stall_timeout_s:
            return False

        stalled_for = now - self._stall_since
        self._stall_since = None
        gap = (float(np.linalg.norm(self.target_position - current_pos))
               if self.target_position is not None else 0.0)
        self.get_logger().error(
            f'ARM STALLED — commanded {commanded * 1000:.0f} mm/s but measured '
            f'{measured * 1000:.0f} mm/s for {stalled_for:.1f} s '
            f'(hand is now {gap * 100:.1f} cm ahead of the arm). '
            'Refusing to chase — hold still while tracking re-anchors.'
        )
        self._reanchor_clutch('recovering from a stall, not chasing the backlog')
        return True

    def _nag_fault(self):
        """Re-state the fault every 5 s so it cannot be missed mid-session."""
        now = time.monotonic()
        if now - self._last_fault_clear_t < 5.0:
            return
        self._last_fault_clear_t = now
        self.get_logger().error(
            'STILL IN FAULT — arm motion disabled. Clear the fault in the '
            'Kinova web app, then restart the teleop stack.'
        )

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
        if self._ref_holo_pos is not None and self._ref_robot_pos is not None:
            delta = (current_holo_pos - self._ref_holo_pos) * self.position_scale
            goal_pos = self._clip_to_workspace(self._ref_robot_pos + delta)
        else:
            goal_pos = np.array([self.home_x, self.home_y, self.home_z])
        self._publish_pose(self.goal_pose_pub, goal_pos, self.target_theta_z_deg, msg.header.stamp)

        if self.is_resetting or self.is_paused or not self.arm_enabled or self._is_faulted:
            return

        # ── Settling period ───────────────────────────────────────────────────
        if self._ref_robot_pos is None:
            if time.monotonic() - self._arm_enabled_at < self.settle_seconds:
                return  # still settling — don't move yet
            self._ref_holo_pos = current_holo_pos.copy()
            try:
                fb = self._base_cyclic.RefreshFeedback()
                self._ref_robot_pos = np.array([
                    fb.base.tool_pose_x,
                    fb.base.tool_pose_y,
                    fb.base.tool_pose_z,
                ])
                self._ref_robot_rot = R.from_euler('xyz', [
                    math.radians(fb.base.tool_pose_theta_x),
                    math.radians(fb.base.tool_pose_theta_y),
                    math.radians(fb.base.tool_pose_theta_z),
                ])
            except Exception:
                self._ref_robot_pos = np.array([self.home_x, self.home_y, self.home_z])
                self._ref_robot_rot = self._home_rot
            # Palm orientation reference. With reference_settle_seconds > 0 the
            # MRTK first-frame identity glitch has passed by now; at the default 0
            # a glitched identity quaternion here just starts the delta from
            # identity (raise reference_settle_seconds if that ever bites).
            o = msg.pose.orientation
            self._ref_holo_rot = R.from_quat([o.x, o.y, o.z, o.w])
            self.get_logger().info(
                f'Reference captured (after settling): '
                f'robot_pos=({self._ref_robot_pos[0]:.3f}, {self._ref_robot_pos[1]:.3f}, {self._ref_robot_pos[2]:.3f})  '
                f'holo_pos=({current_holo_pos[0]:.3f}, {current_holo_pos[1]:.3f}, {current_holo_pos[2]:.3f})'
                + (f'  holo_quat=({o.x:.2f},{o.y:.2f},{o.z:.2f},{o.w:.2f})' if self.enable_orientation else '')
            )

        # ── Position target (delta from reference) ────────────────────────────
        delta = (current_holo_pos - self._ref_holo_pos) * self.position_scale
        clipped = self._clip_to_workspace(self._ref_robot_pos + delta)

        # Hand-glitch rejection (ff modes only; legacy path stays byte-identical):
        # clamp target jumps implying hand speed above hand_speed_limit. Tracking
        # glitches can teleport the wrist estimate tens of cm in one frame.
        if (self.tracking_mode != 'legacy' and self.target_position is not None
                and not self.vertical_only):
            step = clipped - self.target_position
            step_norm = float(np.linalg.norm(step))
            max_step = self.hand_speed_limit * (1.0 / 30.0)   # per hand-msg tick
            if step_norm > max_step:
                clipped = self.target_position + step * (max_step / step_norm)

        if self.vertical_only and self.target_position is not None:
            self.target_position[2] = clipped[2]
        else:
            self.target_position = clipped

        # ── Orientation target (clutched delta, clamped) ──────────────────────
        # Skipped in vertical_only mode (freeze orientation with position x/y).
        if self.enable_orientation and self._ref_holo_rot is not None and not self.vertical_only:
            o = msg.pose.orientation
            q = np.array([o.x, o.y, o.z, o.w])
            if np.linalg.norm(q) > 0.5:  # guard against zero/garbage quaternions
                hand_rot = R.from_quat(q / np.linalg.norm(q))
                # Delta since clutch, in HoloLens-world axes …
                delta_rot = hand_rot * self._ref_holo_rot.inv()
                # … optionally conjugated into robot-base axes by a fixed yaw
                if abs(self.orientation_yaw_offset_deg) > 1e-6:
                    rz = R.from_euler('z', math.radians(self.orientation_yaw_offset_deg))
                    delta_rot = rz * delta_rot * rz.inv()
                target_raw = delta_rot * self._ref_robot_rot

                # Safety clamp relative to HOME orientation: split the rotation-
                # from-home rotvec into three INDEPENDENT per-axis components
                # (roll about base x, pitch about base y, yaw about base z) and
                # clamp each on its own axis.
                r_home = (target_raw * self._home_rot.inv()).as_rotvec()
                roll  = float(np.clip(r_home[0], -math.radians(self.max_roll_deg),
                                                  math.radians(self.max_roll_deg)))
                pitch = float(np.clip(r_home[1],  math.radians(self.pitch_min_deg),
                                                  math.radians(self.pitch_max_deg)))
                yaw   = float(np.clip(r_home[2],  math.radians(self.yaw_min_deg),
                                                  math.radians(self.yaw_max_deg)))
                r_clamped = np.array([roll, pitch, yaw])
                self.target_rot = R.from_rotvec(r_clamped) * self._home_rot

                # Anti-windup: if any axis was clamped, re-anchor the clutch
                # reference so the CURRENT hand pose maps to the (clamped) target.
                # Without this, over-rotating past the cone builds a dead-band —
                # the robot stays pinned at the limit until the hand rotates back
                # within the cone, which feels like lag on reversal. Re-anchoring
                # makes the wrist track instantly the moment the hand comes back;
                # the ±deg limits themselves are unchanged (still a hard wall).
                if not np.allclose(r_clamped, r_home, atol=1e-6):
                    # Solve for the ref_holo that maps hand_rot → target_rot,
                    # inverting the forward chain (delta → yaw-conjugation → target).
                    delta_conj = self.target_rot * self._ref_robot_rot.inv()
                    if abs(self.orientation_yaw_offset_deg) > 1e-6:
                        delta_hold = rz.inv() * delta_conj * rz
                    else:
                        delta_hold = delta_conj
                    self._ref_holo_rot = delta_hold.inv() * hand_rot

        self._publish_pose(self.action_pose_pub, self.target_position, self.target_theta_z_deg, msg.header.stamp,
                           rot=self.target_rot if self.enable_orientation else None)

    def _gripper_cb(self, msg: Float32):
        # Always publish goal gripper for visualization, regardless of enabled state.
        goal_gripper = float(np.clip(msg.data, 0.0, 1.0))
        self.goal_gripper_pub.publish(Float32(data=goal_gripper))

        if self.is_resetting or self.is_paused or not self.gripper_enabled or self._is_faulted:
            return

        self.gripper_cmd = goal_gripper
        self.action_gripper_pub.publish(Float32(data=self.gripper_cmd))

        # Send immediately — no timer wait — to eliminate up to 33 ms of scheduling jitter.
        try:
            gc = Base_pb2.GripperCommand()
            gc.mode = Base_pb2.GRIPPER_POSITION
            f = gc.gripper.finger.add()
            f.finger_identifier = 1
            f.value = self.gripper_cmd
            self._base.SendGripperCommand(gc)
        except Exception as e:
            err_str = str(e)
            if 'ROBOT_IN_FAULT' in err_str or 'IN_FAULT' in err_str:
                self._enter_fault_state()
            elif 'SESSION_NOT_IN_CONTROL' in err_str:
                pass  # transient during servoing-mode transition; next callback retries
            else:
                self.get_logger().error(f'Gripper callback error: {e}')

    # ── Live parameter updates ────────────────────────────────────────────────
    def _on_parameter_change(self, params):
        for p in params:
            if p.name == 'vel_alpha':
                self.vel_alpha = float(p.value)
                self.get_logger().info(f'vel_alpha → {p.value}')
            elif p.name == 'p_gain':
                self.p_gain = float(p.value)
                self.get_logger().info(f'p_gain → {p.value}')
            elif p.name == 'tracking_mode':
                mode = str(p.value)
                if mode not in ('legacy', 'ff_spring', 'ff_ruckig'):
                    self.get_logger().warn(f'unknown tracking_mode "{mode}" — ignored')
                    continue
                if mode == 'ff_ruckig' and not RUCKIG_AVAILABLE:
                    self.get_logger().warn('ruckig not installed — using ff_spring')
                    mode = 'ff_spring'
                self.tracking_mode = mode
                self._ref_pos = None      # re-anchor cleanly in the new mode
                self._rk = None
                self.get_logger().info(f'tracking_mode → {mode}')
            elif p.name == 'ff_gain':
                self.ff_gain = float(p.value)
                self.get_logger().info(f'ff_gain → {p.value}')
            elif p.name == 'spring_wn':
                self.spring_wn = float(p.value)
                self.get_logger().info(f'spring_wn → {p.value}')
            elif p.name == 'd_gain':
                self.d_gain = float(p.value)
                self.get_logger().info(f'd_gain → {p.value}')
            elif p.name == 'hand_speed_limit':
                self.hand_speed_limit = float(p.value)
                self.get_logger().info(f'hand_speed_limit → {p.value}')
            elif p.name == 'max_roll_deg':
                self.max_roll_deg = float(p.value)
                self.get_logger().info(f'max_roll_deg → {p.value}')
            elif p.name == 'pitch_min_deg':
                self.pitch_min_deg = float(p.value)
                self.get_logger().info(f'pitch_min_deg → {p.value}')
            elif p.name == 'pitch_max_deg':
                self.pitch_max_deg = float(p.value)
                self.get_logger().info(f'pitch_max_deg → {p.value}')
            elif p.name == 'yaw_min_deg':
                self.yaw_min_deg = float(p.value)
                self.get_logger().info(f'yaw_min_deg → {p.value}')
            elif p.name == 'yaw_max_deg':
                self.yaw_max_deg = float(p.value)
                self.get_logger().info(f'yaw_max_deg → {p.value}')
            elif p.name == 'max_angular_speed_dps':
                self.max_angular_speed = float(p.value)
                self.get_logger().info(f'max_angular_speed_dps → {p.value}')
            elif p.name == 'position_scale':
                self.position_scale = float(p.value)
                self.get_logger().info(f'position_scale → {p.value}')
            elif p.name == 'max_linear_speed_mps':
                self.max_linear_speed = float(p.value)
                self.get_logger().info(f'max_linear_speed_mps → {p.value}')
        return SetParametersResult(successful=True)

    def _arm_toggle_cb(self, msg: String):
        # The HoloLens republishes this topic continuously rather than only on
        # press, so once a fault forces arm_enabled False the next message would
        # look like a fresh enable and silently re-arm the robot (measured: 1 ms
        # after the fault on 2026-07-28, which preceded the crash). While the
        # fault is latched the headset gets no say.
        if self._is_faulted:
            return
        enabled = msg.data.strip().lower() == 'true'
        if self.arm_enabled and not enabled:
            self.target_position = None
            self._smoothed_vel[:] = 0.0
            self._ref_holo_pos = None
            self._ref_robot_pos = None
            self.target_rot = None
            self._ref_holo_rot = None
            self._ref_robot_rot = None
            self._smoothed_ang_vel[:] = 0.0
            self._send_zero_twist()
            self.get_logger().info('Arm tracking disabled — robot stopped')
        elif not self.arm_enabled and enabled:
            self._ref_holo_pos = None
            self._ref_robot_pos = None
            self._arm_enabled_at = time.monotonic()
            if self.settle_seconds > 0:
                self.get_logger().info(
                    f'Arm tracking enabled — robot held still for {self.settle_seconds:.2f} s '
                    'while MRTK tracking settles, then wrist reference will be captured'
                )
            else:
                self.get_logger().info('Arm tracking enabled — capturing wrist reference immediately')
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
            self._ref_holo_pos = None
            self._ref_robot_pos = None
            self.target_rot = None
            self._ref_holo_rot = None
            self._ref_robot_rot = None
            self._smoothed_ang_vel[:] = 0.0
            self._arm_enabled_at = time.monotonic()  # re-settle when tracking resumes
            self._send_zero_twist()  # no-op during reset (reset owns the arm)
            if self.is_resetting:
                self.get_logger().warn('Hand tracking lost during reset — reset continues')
            else:
                self.get_logger().warn('Hand tracking lost — robot stopped; will re-settle on re-acquisition')

    def _reset_cb(self, msg: Bool):
        if not msg.data or self.is_resetting:
            return
        self.get_logger().info('Resetting Kinova Gen3 to home …')
        self.target_position = None
        self._smoothed_vel[:] = 0.0
        self._send_zero_twist()      # prompt stop — sent while is_resetting is still False
        self.is_resetting = True     # now lock out async twist senders for the whole reset
        threading.Thread(target=self._do_reset, daemon=True).start()

    def _do_reset(self):
        try:
            # Wait for the twist watchdog (200 ms) to fire and the arm to reach a
            # controlled stop before issuing the position action.
            time.sleep(1.0)

            # JOINT-SPACE home (2026-07-27): a Cartesian reach_pose is
            # IK-ambiguous on a 7-DOF arm — the elbow/posture could settle
            # differently on every reset. reach_joint_angles pins the exact
            # measured home configuration, so every episode starts from an
            # identical posture. Angles captured from the physical arm at the
            # canonical home TCP (0.350, 0.000, 0.120).
            action = Base_pb2.Action()
            action.name = 'Home'
            action.application_data = ''

            # NOTE: no speed constraint — JOINT_CONSTRAINT_SPEED on
            # reach_joint_angles is rejected by this firmware (verified
            # 2026-07-27: ACTION_ABORT/METHOD_FAILED via notification).
            # Without a constraint the firmware's trajectory planner picks a
            # moderate profile (same as the web app's Home action).
            for i, ang in enumerate(HOME_JOINTS_DEG):
                ja = action.reach_joint_angles.joint_angles.joint_angles.add()
                ja.joint_identifier = i
                ja.value = ang

            self._base.ExecuteAction(action)

            # Confirm arrival by polling JOINT angles rather than relying on
            # action notifications (which get missed or arrive as stale events,
            # causing false timeouts). The control loop stays off the router
            # during reset, so _do_reset owns it and RefreshFeedback is safe.
            # NOTE: wrap-aware diff — joints 1/5 sit at ~359.6°, right on the
            # 0/360 seam.
            def _joint_err_deg(fb):
                errs = []
                for i, tgt in enumerate(HOME_JOINTS_DEG):
                    cur = fb.actuators[i].position
                    d = abs(cur - tgt) % 360.0
                    errs.append(min(d, 360.0 - d))
                return max(errs)

            deadline  = time.monotonic() + 30.0
            start_pos = None
            reached   = False
            moved     = False
            while time.monotonic() < deadline:
                try:
                    fb  = self._base_cyclic.RefreshFeedback()
                    cur = np.array([fb.base.tool_pose_x, fb.base.tool_pose_y, fb.base.tool_pose_z])
                except Exception:
                    time.sleep(0.1)
                    continue
                if start_pos is None:
                    start_pos = cur
                elif np.linalg.norm(cur - start_pos) > 0.01:
                    moved = True
                if _joint_err_deg(fb) < 2.0:
                    reached = True
                    break
                time.sleep(0.1)

            if reached:
                # Open gripper
                gc = Base_pb2.GripperCommand()
                gc.mode = Base_pb2.GRIPPER_POSITION
                f = gc.gripper.finger.add()
                f.finger_identifier = 1
                f.value = 0.0
                self._base.SendGripperCommand(gc)
                time.sleep(1.0)
                self.get_logger().info('Reset complete — gripper open')
            elif moved:
                self.get_logger().warn('Home reset timed out — arm moved but never reached home')
            else:
                self.get_logger().warn(
                    'Home reset failed — arm never moved; ExecuteAction was likely '
                    'rejected (check for a robot fault or unreachable pose)'
                )
            if not reached:
                try:
                    self._base.StopAction()
                except Exception:
                    pass

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
    def _publish_pose(self, pub, position: np.ndarray, theta_z_deg: float, stamp, rot=None):
        """Publish a pose. If `rot` (scipy Rotation) is given — orientation mode —
        it is used verbatim, so robot_action/pose records the true commanded
        orientation. Otherwise the legacy fixed home_tx/home_ty + theta_z is used."""
        msg = PoseStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = 'base_link'
        msg.pose.position.x = float(position[0])
        msg.pose.position.y = float(position[1])
        msg.pose.position.z = float(position[2])

        quat = (rot.as_quat() if rot is not None else R.from_euler('xyz', [
            math.radians(self.home_tx),
            math.radians(self.home_ty),
            math.radians(theta_z_deg),
        ]).as_quat())
        msg.pose.orientation.x = float(quat[0])
        msg.pose.orientation.y = float(quat[1])
        msg.pose.orientation.z = float(quat[2])
        msg.pose.orientation.w = float(quat[3])
        pub.publish(msg)

    # ── Control loop ──────────────────────────────────────────────────────────
    def _ruckig_step(self, target, dt):
        """One Ruckig OTG step toward `target`. Maintains its own state chain;
        returns (ref_pos, ref_vel). Re-created whenever the chain was reset."""
        if self._rk is None:
            otg = Ruckig(3, dt)
            inp = InputParameter(3)
            out = OutputParameter(3)
            inp.current_position     = list(self._ref_pos)
            inp.current_velocity     = list(self._ref_vel)
            inp.current_acceleration = [0.0, 0.0, 0.0]
            inp.max_velocity     = [self.max_linear_speed] * 3
            inp.max_acceleration = [self.ruckig_amax] * 3
            inp.max_jerk         = [self.ruckig_jmax] * 3
            self._rk = (otg, inp, out)
        otg, inp, out = self._rk
        inp.target_position = list(target)
        inp.target_velocity = [0.0, 0.0, 0.0]
        res = otg.update(inp, out)
        if res not in (Result.Working, Result.Finished):
            # Constraint solver rejected the state (shouldn't happen with sane
            # limits) — reset the chain and hold position this tick.
            self._rk = None
            return self._ref_pos, np.zeros(3)
        out.pass_to_input(inp)
        return np.array(out.new_position), np.array(out.new_velocity)

    def _control_loop(self):
        """
        30 Hz P-loop velocity controller with:
          - velocity smoothing (exponential filter)
          - soft boundary deceleration
          - hard velocity cap
          - TwistCommand watchdog duration
          - fault detection (latching — no auto-recovery, by design)
        """
        # Fault takes priority and is LATCHED — suppress every command, stay off
        # the router, and do NOT try to recover (see _enter_fault_state).
        if self._is_faulted:
            self._nag_fault()
            return

        # ── Arm control ───────────────────────────────────────────────────────
        if self.is_resetting:
            # _do_reset() drives the arm via ExecuteAction on its own thread, using
            # the same Kortex RouterClient. The router is NOT thread-safe — issuing
            # SendTwistCommand here concurrently corrupts RPC framing and makes
            # ExecuteAction block forever (reset silently hangs). Stay off the
            # router entirely; the twist watchdog (TWIST_WATCHDOG_MS) has already
            # stopped the arm after the zero-twist sent in _reset_cb.
            self._smoothed_vel[:] = 0.0
            self._ref_pos = None
            self._rk = None
            self._fb_prev_pos = None
            self._robot_vel_est[:] = 0.0
            self._stall_since = None
            return
        if self.is_paused or self.target_position is None:
            self._smoothed_vel[:] = 0.0
            self._ref_pos = None
            self._rk = None
            self._fb_prev_pos = None
            self._robot_vel_est[:] = 0.0
            self._stall_since = None
            self._send_zero_twist()
        else:
            try:
                feedback = self._base_cyclic.RefreshFeedback()
                current_pos = np.array([
                    feedback.base.tool_pose_x,
                    feedback.base.tool_pose_y,
                    feedback.base.tool_pose_z,
                ])

                # Robot-velocity estimate from feedback deltas (light EMA).
                # Computed for EVERY tracking mode, before the control law: the
                # ff path uses it for its damping term, and the stall guard
                # below needs it in legacy mode too.
                now_t = time.monotonic()
                if self._fb_prev_pos is not None and self._fb_prev_t is not None:
                    fb_dt = max(now_t - self._fb_prev_t, 1e-3)
                    v_inst = (current_pos - self._fb_prev_pos) / fb_dt
                    self._robot_vel_est = 0.5 * v_inst + 0.5 * self._robot_vel_est
                self._fb_prev_pos = current_pos.copy()
                self._fb_prev_t = now_t

                if self.tracking_mode in ('ff_spring', 'ff_ruckig'):
                    # ── Feed-forward tracking (latency fix) ─────────────────
                    # A reference generator smooths the clutched hand target
                    # into a consistent (x_ref, v_ref); the law tracks it with
                    # feed-forward, so no speed-proportional error is needed to
                    # sustain motion (unlike the legacy pure-P path below).
                    dt = 1.0 / self.control_rate
                    if self._ref_pos is None:      # (re-)anchor on the robot
                        self._ref_pos = current_pos.copy()
                        self._ref_vel = np.zeros(3)
                        self._rk = None
                    if self.tracking_mode == 'ff_spring':
                        self._ref_pos, self._ref_vel = spring_reference_step(
                            self._ref_pos, self._ref_vel,
                            self.target_position, self.spring_wn, dt)
                    else:
                        self._ref_pos, self._ref_vel = self._ruckig_step(
                            self.target_position, dt)

                    raw_vel = (self.ff_gain * self._ref_vel
                               + self.p_gain * (self._ref_pos - current_pos)
                               - self.d_gain * (self._robot_vel_est - self._ref_vel))

                    # Same safety chain as legacy: hard cap → soft walls → cap.
                    speed = float(np.linalg.norm(raw_vel))
                    if speed > self.max_linear_speed:
                        raw_vel = raw_vel * (self.max_linear_speed / speed)
                    raw_vel = self._boundary_speed_scale(current_pos, raw_vel)
                    send_vel = raw_vel.copy()
                    speed = float(np.linalg.norm(send_vel))
                    if speed > self.max_linear_speed:
                        send_vel = send_vel * (self.max_linear_speed / speed)
                    # Keep the EMA state warm so live mode switches are smooth.
                    self._smoothed_vel = send_vel.copy()
                else:
                    # ── Legacy path (byte-identical to the original) ─────────
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

                # ── Stall guard ──────────────────────────────────────────────
                # Commanding motion while the arm is NOT moving means it has
                # stopped for a reason this node cannot see. Bail out before
                # the clutch offset grows any further.
                if self._check_stall(current_pos, send_vel):
                    self._send_zero_twist()
                    return

                # ── Angular velocity ─────────────────────────────────────────
                # Translation-only mode (default): all angular velocities zero —
                # SINGLE_LEVEL_SERVOING holds the current EEF orientation
                # passively. We historically avoided an angular P-loop because
                # the Kortex ZYX Euler decomposition has a gimbal singularity
                # near theta_x=±180° (the arm's home orientation), causing
                # theta_z to flip between 89.3° and ≈0°/180°.
                #
                # Orientation mode (enable_orientation): P-loop on the FULL
                # ROTATION error instead of Euler components. Euler feedback is
                # converted to a rotation matrix first — different Euler triplets
                # near the singularity represent the SAME rotation, so the matrix
                # (and thus the error rotvec) is immune to the flips that made
                # Euler-based control unusable.
                ang_send = np.zeros(3)
                if self.enable_orientation and self.target_rot is not None:
                    current_rot = R.from_euler('xyz', [
                        math.radians(feedback.base.tool_pose_theta_x),
                        math.radians(feedback.base.tool_pose_theta_y),
                        math.radians(feedback.base.tool_pose_theta_z),
                    ])
                    # Base-frame rotation error target ∘ current⁻¹, as rotvec (rad)
                    err_deg = np.degrees(
                        (self.target_rot * current_rot.inv()).as_rotvec())
                    raw_ang = self.orientation_p_gain * err_deg  # deg/s
                    ang_speed = float(np.linalg.norm(raw_ang))
                    if ang_speed > self.max_angular_speed:
                        raw_ang *= self.max_angular_speed / ang_speed
                    self._smoothed_ang_vel = (
                        self.vel_alpha * raw_ang
                        + (1.0 - self.vel_alpha) * self._smoothed_ang_vel
                    )
                    ang_send = self._smoothed_ang_vel.copy()
                    ang_speed = float(np.linalg.norm(ang_send))
                    if ang_speed > self.max_angular_speed:
                        ang_send *= self.max_angular_speed / ang_speed
                else:
                    self._smoothed_ang_vel[:] = 0.0

                cmd = Base_pb2.TwistCommand()
                # BASE frame when commanding angular velocity (our error rotvec is
                # base-frame; this matches testing/inference.py). Keep the legacy
                # MIXED frame in translation-only mode — with zero angulars the
                # two are equivalent for linear velocity, so behavior is unchanged.
                cmd.reference_frame = (Base_pb2.CARTESIAN_REFERENCE_FRAME_BASE
                                       if self.enable_orientation
                                       else Base_pb2.CARTESIAN_REFERENCE_FRAME_MIXED)
                cmd.duration = TWIST_WATCHDOG_MS
                cmd.twist.linear_x  = float(send_vel[0])
                cmd.twist.linear_y  = float(send_vel[1])
                cmd.twist.linear_z  = float(send_vel[2])
                cmd.twist.angular_x = float(ang_send[0])
                cmd.twist.angular_y = float(ang_send[1])
                cmd.twist.angular_z = float(ang_send[2])
                self._base.SendTwistCommand(cmd)

            except Exception as e:
                err_str = str(e)
                if 'ROBOT_IN_FAULT' in err_str or 'IN_FAULT' in err_str:
                    self._enter_fault_state()
                elif 'SESSION_NOT_IN_CONTROL' in err_str:
                    pass  # transient during servoing-mode transition; next tick retries
                else:
                    self.get_logger().error(f'Arm control loop error: {e}')
                    self._send_zero_twist()

    def _send_zero_twist(self):
        """Zero-velocity command with watchdog duration (safe stop)."""
        if self.is_resetting:
            # A home reset (ExecuteAction) owns the arm and the Kortex router. A
            # twist command here would cancel the in-flight reach action and race
            # the reset's RPCs — this is why losing hand tracking mid-reset used to
            # abort it. The single intended pre-reset stop is sent in _reset_cb
            # before is_resetting is set; the twist watchdog holds the arm still.
            return
        try:
            cmd = Base_pb2.TwistCommand()
            cmd.reference_frame = Base_pb2.CARTESIAN_REFERENCE_FRAME_MIXED
            cmd.duration = TWIST_WATCHDOG_MS
            self._base.SendTwistCommand(cmd)
        except Exception:
            pass

    # ── Cleanup ────────────────────────────────────────────────────────────────
    def destroy_node(self):
        """Tear the Kortex session down step by step.

        Each step gets its OWN try/except: sharing one block meant that when the
        arm was already faulted, _send_zero_twist() raised ROBOT_IN_FAULT on the
        first line and CloseSession()/disconnect() were silently skipped. The
        abandoned session makes the arm latch a NETWORK_ERROR fault, so the next
        run starts in fault — a loop that only ends with a manual clear.
        Failures are logged rather than swallowed for the same reason.
        """
        self.get_logger().info('Disconnecting from Kinova Gen3 …')
        for label, step in (
            ('zero twist',    self._send_zero_twist),
            ('close session', lambda: self._session_manager.CloseSession()),
            ('disconnect',    lambda: self._transport.disconnect()),
        ):
            try:
                step()
            except Exception as e:
                # Expected for 'zero twist' when the arm is faulted; the session
                # still MUST be closed, so keep going.
                self.get_logger().warn(f'Shutdown step "{label}" failed: {e}')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = KinovaHandController()

    # launch_data_collection.py clears stale nodes with `pkill -f
    # kinova_hand_controller`, which sends SIGTERM — and Python's default
    # SIGTERM disposition kills the process WITHOUT running the finally block,
    # so the Kortex session would be abandoned and the arm would latch a
    # NETWORK_ERROR fault. Translate it into the same orderly path as Ctrl-C.
    def _term(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, _term)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
