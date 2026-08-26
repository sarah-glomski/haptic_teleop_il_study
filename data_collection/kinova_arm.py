#!/usr/bin/env python3
"""
Shared Cartesian velocity-control layer for the Kinova Gen3.

Teleop (kinova_hand_controller.py) and policy rollout (testing/inference.py)
drive the SAME arm through the same Kortex API with the same safety needs, but
each used to carry its own copy of the session handling, the P-loop, the speed
caps, the workspace bounds and the twist plumbing. The copies drifted badly:

    2026-07-29 audit          teleop            inference
    max linear speed          0.50 m/s          0.10 m/s
    workspace bounds          x 0.25-0.70       x 0.25-0.45  (stale)
    soft boundary decel       yes               no
    orientation               full quaternion   yaw only
    fault handling            latching          none
    stall guard               yes               none

The orientation work and the 2026-07-28 crash fixes both landed in teleop only,
because there was no single place to put them. This module is that place. Each
consumer keeps what is genuinely its own — teleop owns the HoloLens clutch, the
feed-forward tracking modes and the gates; inference owns the policy, the obs
buffers and the action queue — and both share everything from the target pose
down to the wire.

Nothing here knows about ROS, HoloLens or policies: it takes a target pose and
drives the arm to it safely.
"""

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R

from kortex_api.TCPTransport import TCPTransport
from kortex_api.RouterClient import RouterClient
from kortex_api.SessionManager import SessionManager
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.messages import Session_pb2, Base_pb2


# If a consumer misses this many ms worth of control ticks the Kortex SDK stops
# the robot automatically. Must be > 1/(control_rate) * 1000 to avoid false
# trips but short enough to stop the arm quickly on a crash. At 30 Hz one tick
# is ~33 ms, so 200 ms gives ~6 missed cycles of margin. Never use duration=0.
TWIST_WATCHDOG_MS = 200

# Joint-space home configuration (degrees, joints 1-7). Measured from the
# physical arm 2026-07-27 at the far-forward home posture (TCP 0.5715, 0.0137,
# 0.1068, tool down). Used with reach_joint_angles so the 7-DOF arm returns to
# an IDENTICAL posture every episode — a Cartesian home is IK-ambiguous and the
# elbow could settle differently each time.
HOME_JOINTS_DEG = [358.258, 47.071, 181.685, 284.279, 357.708, 303.052, 89.735]


@dataclass
class ArmLimits:
    """Every safety-relevant number in one place.

    These defaults ARE the safe configuration — a consumer that wants something
    different should say so explicitly rather than redefining constants in its
    own file, which is exactly how the two implementations drifted apart.
    """
    # Velocity
    max_linear_speed_mps: float = 0.50
    max_angular_speed_dps: float = 45.0
    p_gain: float = 2.0
    orientation_p_gain: float = 2.0
    vel_alpha: float = 0.7                 # EMA on commanded velocity

    # Workspace box (metres)
    x: Tuple[float, float] = (0.25, 0.70)
    y: Tuple[float, float] = (-0.35, 0.35)
    z: Tuple[float, float] = (0.025, 0.25)
    soft_margin_m: float = 0.01            # soft deceleration zone from a wall
    hard_margin_m: float = 0.005           # clip this far inside the box

    # Orientation clamps, applied to the rotation FROM HOME split into three
    # independent per-axis rotvec components in the base frame.
    #
    # Yaw is the free axis — open through +93 deg toward tool-forward (plug
    # insertion at the far table edge), 3 deg the other way. Roll and pitch stay
    # locked to a small +/-3 deg tolerance: a downward-pointing tool sweeps its
    # fingers BELOW the TCP as it tilts, and the workspace z floor (z[0]) is
    # enforced on the TCP only, so any real tilt breaks the table-clearance
    # guarantee that box gives.
    max_roll_deg: float = 3.0              # about base x, symmetric
    pitch_min_deg: float = -3.0            # about base y
    pitch_max_deg: float = 3.0
    yaw_min_deg: float = -3.0              # about base z
    yaw_max_deg: float = 93.0              # open toward tool-forward

    # Stall guard (see StallGuard). stall_timeout_s <= 0 disables it.
    stall_timeout_s: float = 0.4
    stall_cmd_mps: float = 0.03
    stall_move_mps: float = 0.01

    # ── Tilted-grip mode (opt-in) ─────────────────────────────────────────
    # Normally roll/pitch are locked to +-3 deg because a downward tool sweeps
    # its fingers BELOW the TCP as it tilts while the z floor is enforced on the
    # TCP only. Tilting is nonetheless useful: with the fingers stacked rather
    # than side by side, the lower pad carries the payload's weight on top of
    # the grip force, so ch2-ch3 encodes mass — which side-by-side pads cannot
    # sense at all (measured: weight separates at 0.3-0.7 SD, compliance at 4.5).
    #
    # allow_tilt() re-opens roll and pitch to `deg` and raises the z floor by
    # the FULL gripper span times sin(deg) — twice the half-span the geometry
    # implies — so the clearance the box guarantees survives the sweep.
    # GRIPPER_SPAN_M is the assumption to check against the real hardware
    # before trusting this; if the TCP is not at fingertip height the floor
    # needs raising further.
    GRIPPER_SPAN_M: float = 0.085          # Robotiq 2F-85 max opening

    def allow_tilt(self, deg: float):
        """Open roll/pitch to +-deg and raise the z floor to match. Returns self."""
        if deg <= 0:
            return self
        sweep = self.GRIPPER_SPAN_M * math.sin(math.radians(min(deg, 90.0)))
        self.max_roll_deg = float(deg)
        self.pitch_min_deg = -float(deg)
        self.pitch_max_deg = float(deg)
        self.z = (self.z[0] + sweep, self.z[1])
        return self

    def home_rotation(self, tx=-180.0, ty=0.0, tz=90.0) -> R:
        """Home orientation as a scipy Rotation (Kortex Euler XYZ, degrees)."""
        return R.from_euler('xyz', [math.radians(tx), math.radians(ty),
                                    math.radians(tz)])


class _NullLog:
    """Stand-in when no ROS logger is supplied (tests, scripts)."""
    def info(self, m):  print(f'[info] {m}')
    def warn(self, m):  print(f'[WARN] {m}')
    def error(self, m): print(f'[ERROR] {m}')


class FaultLatch:
    """Latching robot-fault state. There is deliberately no auto-recovery.

    A fault means the arm stopped while its commander kept advancing targets, so
    the offset between commanded and actual pose is now arbitrarily large.
    Bringing the arm back online silently makes the P-loop close that gap at
    max_linear_speed.

    That is not hypothetical: on 2026-07-28 a fault auto-cleared 0.8 s after it
    was raised and the arm dashed into the table, destroying the task objects.
    Recovery is deliberate: clear the fault in the Kinova web app and restart
    the process, which re-anchors everything from scratch.
    """

    def __init__(self, log, on_fault: Optional[Callable[[], None]] = None,
                 nag_seconds: float = 5.0, recovery_hint: str = 'restart the stack'):
        self._log = log
        self._on_fault = on_fault
        self._nag_seconds = nag_seconds
        self._hint = recovery_hint
        self._latched = False
        self._last_nag_t = 0.0

    @property
    def latched(self) -> bool:
        return self._latched

    def trip(self, detail: str = ''):
        """Latch the fault. Idempotent — only the first call acts."""
        if self._latched:
            return
        self._latched = True
        self._log.error(
            'ROBOT FAULT DETECTED — arm motion disabled and LATCHED. '
            'No auto-clear: re-enabling would make the arm dash to catch up. '
            f'To recover: clear the fault in the Kinova web app, then {self._hint}.'
            + (f' ({detail})' if detail else '')
        )
        if self._on_fault is not None:
            self._on_fault()

    def nag(self):
        """Re-state the fault periodically so it cannot be missed mid-session."""
        now = time.monotonic()
        if now - self._last_nag_t < self._nag_seconds:
            return
        self._last_nag_t = now
        self._log.error(
            'STILL IN FAULT — arm motion disabled. Clear the fault in the '
            f'Kinova web app, then {self._hint}.')

    def classify(self, exc: Exception) -> str:
        """Map a Kortex RPC exception to 'fault' | 'transient' | 'other'.

        Latches on a fault as a side effect, so callers can simply branch on the
        returned string.
        """
        s = str(exc)
        if 'ROBOT_IN_FAULT' in s or 'IN_FAULT' in s:
            self.trip()
            return 'fault'
        if 'SESSION_NOT_IN_CONTROL' in s:
            # Transient during a servoing-mode transition; the next tick retries.
            return 'transient'
        return 'other'


class StallGuard:
    """Detects "commanded to move but not moving" and asks for a re-anchor.

    The P-loop chases a target with no bound on how far that target may drift
    from the arm's real position. Whenever the arm stops for a reason the
    commander cannot observe — a fault, a manual-control takeover, a
    twist-watchdog expiry, an RPC stall — targets keep advancing and the offset
    grows for as long as the stall lasts. The instant the arm frees up, that
    offset is a position error and the P-loop closes it at max_linear_speed. On
    2026-07-28 a ~3 s stall did exactly this and threw the arm across the table.

    Detecting the STALL rather than the gap size is deliberate: under a pure-P
    law the steady-state tracking error is speed-proportional (error = v/p_gain
    — at p_gain 2.0 a brisk 0.3 m/s move legitimately sits 15 cm behind), so a
    plain gap threshold fires constantly during normal fast motion. "Commanded
    but not moving" has no such false positive.
    """

    def __init__(self, limits: ArmLimits, log,
                 on_stall: Optional[Callable[[str], None]] = None):
        self._lim = limits
        self._log = log
        self._on_stall = on_stall
        self._stall_since = None
        self._prev_pos = None
        self._prev_t = None
        self.velocity_est = np.zeros(3)

    def reset(self):
        """Forget history — call whenever the commander stops commanding."""
        self._stall_since = None
        self._prev_pos = None
        self._prev_t = None
        self.velocity_est[:] = 0.0

    def update_velocity(self, current_pos: np.ndarray):
        """Robot-velocity estimate from feedback deltas (light EMA).

        Must be called every control tick, before check(), in EVERY control
        mode — this estimate is the only way the commander can tell that the
        arm has stopped.
        """
        now = time.monotonic()
        if self._prev_pos is not None and self._prev_t is not None:
            dt = max(now - self._prev_t, 1e-3)
            v_inst = (current_pos - self._prev_pos) / dt
            self.velocity_est = 0.5 * v_inst + 0.5 * self.velocity_est
        self._prev_pos = np.asarray(current_pos).copy()
        self._prev_t = now

    def check(self, commanded_vel: np.ndarray, gap_m: float = 0.0) -> bool:
        """True when a stall is confirmed. Fires on_stall and resets on trip."""
        if self._lim.stall_timeout_s <= 0.0:
            return False

        commanded = float(np.linalg.norm(commanded_vel))
        measured = float(np.linalg.norm(self.velocity_est))

        if commanded < self._lim.stall_cmd_mps or measured > self._lim.stall_move_mps:
            self._stall_since = None
            return False

        now = time.monotonic()
        if self._stall_since is None:
            self._stall_since = now
            return False
        if now - self._stall_since < self._lim.stall_timeout_s:
            return False

        stalled_for = now - self._stall_since
        self._stall_since = None
        self._log.error(
            f'ARM STALLED — commanded {commanded * 1000:.0f} mm/s but measured '
            f'{measured * 1000:.0f} mm/s for {stalled_for:.1f} s '
            f'(target is now {gap_m * 100:.1f} cm ahead of the arm). '
            'Refusing to chase — re-anchoring instead.')
        if self._on_stall is not None:
            self._on_stall('recovering from a stall, not chasing the backlog')
        return True


class KinovaArm:
    """Kortex session + safety-checked Cartesian velocity control.

    Owns the connection, the speed/workspace/orientation limits, the fault latch
    and the stall guard. Consumers supply target poses; this class decides what
    is safe to send.
    """

    def __init__(self, robot_ip: str = '192.168.1.10',
                 username: str = 'admin', password: str = 'admin',
                 limits: Optional[ArmLimits] = None,
                 log=None,
                 on_stall: Optional[Callable[[str], None]] = None,
                 on_fault: Optional[Callable[[], None]] = None,
                 recovery_hint: str = 'restart the stack'):
        self.robot_ip = robot_ip
        self.username = username
        self.password = password
        self.limits = limits or ArmLimits()
        self.log = log or _NullLog()

        self.fault = FaultLatch(self.log, on_fault=on_fault,
                                recovery_hint=recovery_hint)
        self.stall = StallGuard(self.limits, self.log, on_stall=on_stall)

        # Set while a reach action owns the router: a twist here would cancel
        # the in-flight action and race its RPCs.
        self.twists_suppressed = False

        self._smoothed_vel = np.zeros(3)
        self._smoothed_ang_vel = np.zeros(3)

        self._transport = None
        self._router = None
        self._session_manager = None
        self.base = None
        self.base_cyclic = None

    # ── Session lifecycle ─────────────────────────────────────────────────────
    def connect(self):
        self._transport = TCPTransport()
        self._router = RouterClient(
            self._transport,
            lambda ex: self.log.error(f'Kortex transport error: {ex}'))
        self._transport.connect(self.robot_ip, 10000)

        session_info = Session_pb2.CreateSessionInfo()
        session_info.username = self.username
        session_info.password = self.password
        session_info.session_inactivity_timeout = 60000
        session_info.connection_inactivity_timeout = 2000

        self._session_manager = SessionManager(self._router)
        self._session_manager.CreateSession(session_info)

        self.base = BaseClient(self._router)
        self.base_cyclic = BaseCyclicClient(self._router)
        self.log.info('Connected to Kinova Gen3')

    def setup_servoing(self):
        """Set SINGLE_LEVEL_SERVOING mode (enables SendTwistCommand)."""
        mode = Base_pb2.ServoingModeInformation()
        mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
        self.base.SetServoingMode(mode)
        time.sleep(0.2)
        self.log.info('Kinova set to SINGLE_LEVEL_SERVOING (velocity control)')

    def disconnect(self):
        """Tear the session down step by step.

        Each step gets its OWN try/except. Sharing one block meant that when the
        arm was already faulted, the zero twist raised ROBOT_IN_FAULT on the
        first line and CloseSession()/disconnect() were silently skipped. The
        abandoned session makes the arm latch NETWORK_ERROR, so the next run
        starts in fault — a loop that only ends with a manual clear.
        """
        self.log.info('Disconnecting from Kinova Gen3 …')
        for label, step in (
            ('zero twist',    self.send_zero_twist),
            ('close session', lambda: self._session_manager.CloseSession()),
            ('disconnect',    lambda: self._transport.disconnect()),
        ):
            try:
                step()
            except Exception as e:
                self.log.warn(f'Shutdown step "{label}" failed: {e}')

    # ── Feedback ──────────────────────────────────────────────────────────────
    def refresh_feedback(self):
        return self.base_cyclic.RefreshFeedback()

    @staticmethod
    def tcp_position(fb) -> np.ndarray:
        return np.array([fb.base.tool_pose_x, fb.base.tool_pose_y,
                         fb.base.tool_pose_z])

    @staticmethod
    def tcp_rotation(fb) -> R:
        """TCP orientation as a Rotation.

        Kortex reports Euler XYZ, whose ZYX decomposition has a gimbal
        singularity near theta_x = +/-180 deg — the arm's home orientation —
        where the components flip. Converting straight to a Rotation makes all
        downstream error maths immune to those representation flips.
        """
        return R.from_euler('xyz', [
            math.radians(fb.base.tool_pose_theta_x),
            math.radians(fb.base.tool_pose_theta_y),
            math.radians(fb.base.tool_pose_theta_z),
        ])

    # ── Workspace safety ──────────────────────────────────────────────────────
    def clip_to_workspace(self, pos: np.ndarray) -> np.ndarray:
        """Hard-clip a target to the workspace box, inside a small margin."""
        lim, m = self.limits, self.limits.hard_margin_m
        return np.array([
            np.clip(pos[0], lim.x[0] + m, lim.x[1] - m),
            np.clip(pos[1], lim.y[0] + m, lim.y[1] - m),
            np.clip(pos[2], lim.z[0] + m, lim.z[1] - m),
        ])

    def boundary_speed_scale(self, current_pos: np.ndarray,
                             vel: np.ndarray) -> np.ndarray:
        """Soft boundary: fade out velocity as the TCP nears a workspace wall.

        Per axis and direction, scale = clamp(dist_to_wall / soft_margin, 0, 1),
        applied only to the component moving TOWARD that wall — so it is always
        possible to move away from a boundary already crossed.
        """
        lim = self.limits
        if lim.soft_margin_m <= 0:
            return vel

        out = vel.copy()
        for axis, (lo, hi) in enumerate((lim.x, lim.y, lim.z)):
            pos = current_pos[axis]
            v = out[axis]

            dist_lo = pos - lo
            if v < 0 and dist_lo < lim.soft_margin_m:
                out[axis] = v * max(0.0, dist_lo / lim.soft_margin_m)

            dist_hi = hi - pos
            if v > 0 and dist_hi < lim.soft_margin_m:
                out[axis] = v * max(0.0, dist_hi / lim.soft_margin_m)

        return out

    def cap_linear(self, vel: np.ndarray) -> np.ndarray:
        speed = float(np.linalg.norm(vel))
        if speed > self.limits.max_linear_speed_mps:
            return vel * (self.limits.max_linear_speed_mps / speed)
        return vel

    # ── Orientation safety ────────────────────────────────────────────────────
    def clamp_orientation(self, target_rot: R, home_rot: R) -> Tuple[R, bool]:
        """Clamp a target orientation relative to home. Returns (rot, clamped).

        The rotation FROM HOME is split into three INDEPENDENT per-axis rotvec
        components in the base frame and each is clamped on its own axis:
        roll about x, pitch about y, yaw about z. Roll/pitch keep the gripper
        from tilting into the table (the workspace box assumes a roughly
        downward tool, and tilting sweeps the fingers below the TCP); the yaw
        range is the one opened up for tool-forward reach, and its far side
        still has to stay clear of the wrist joint's travel limit — a sustained
        chase into a joint limit faults the arm.
        """
        lim = self.limits
        r_home = (target_rot * home_rot.inv()).as_rotvec()
        roll = float(np.clip(r_home[0], -math.radians(lim.max_roll_deg),
                             math.radians(lim.max_roll_deg)))
        pitch = float(np.clip(r_home[1], math.radians(lim.pitch_min_deg),
                              math.radians(lim.pitch_max_deg)))
        yaw = float(np.clip(r_home[2], math.radians(lim.yaw_min_deg),
                            math.radians(lim.yaw_max_deg)))
        clamped_vec = np.array([roll, pitch, yaw])
        was_clamped = not np.allclose(clamped_vec, r_home, atol=1e-6)
        return R.from_rotvec(clamped_vec) * home_rot, was_clamped

    # ── Control laws ──────────────────────────────────────────────────────────
    def linear_velocity(self, current_pos: np.ndarray,
                        target_pos: np.ndarray) -> np.ndarray:
        """P-loop -> hard cap -> soft walls -> EMA smoothing -> hard cap."""
        raw = self.limits.p_gain * (target_pos - current_pos)
        raw = self.cap_linear(raw)
        raw = self.boundary_speed_scale(current_pos, raw)

        a = self.limits.vel_alpha
        self._smoothed_vel = a * raw + (1.0 - a) * self._smoothed_vel
        return self.cap_linear(self._smoothed_vel.copy())

    def angular_velocity(self, current_rot: R,
                         target_rot: Optional[R]) -> np.ndarray:
        """Base-frame angular velocity (deg/s) from a full rotation error.

        Uses the rotvec of target * current^-1 rather than per-component Euler
        error: near the home orientation the Kortex Euler decomposition is
        singular and its components flip, which made Euler-based wrist control
        unusable. A rotation-matrix error is immune to representation flips.
        """
        if target_rot is None:
            self._smoothed_ang_vel[:] = 0.0
            return np.zeros(3)

        err_deg = np.degrees((target_rot * current_rot.inv()).as_rotvec())
        raw = self.limits.orientation_p_gain * err_deg

        cap = self.limits.max_angular_speed_dps
        speed = float(np.linalg.norm(raw))
        if speed > cap:
            raw = raw * (cap / speed)

        a = self.limits.vel_alpha
        self._smoothed_ang_vel = a * raw + (1.0 - a) * self._smoothed_ang_vel

        out = self._smoothed_ang_vel.copy()
        speed = float(np.linalg.norm(out))
        if speed > cap:
            out = out * (cap / speed)
        return out

    def reset_velocity_state(self):
        """Zero the smoothing filters and the stall history."""
        self._smoothed_vel[:] = 0.0
        self._smoothed_ang_vel[:] = 0.0
        self.stall.reset()

    # ── Twist output ──────────────────────────────────────────────────────────
    def send_twist(self, linear: np.ndarray, angular: Optional[np.ndarray] = None,
                   base_frame: bool = True):
        """Send one twist with the watchdog duration set. Never duration=0."""
        if self.twists_suppressed:
            return
        cmd = Base_pb2.TwistCommand()
        cmd.reference_frame = (Base_pb2.CARTESIAN_REFERENCE_FRAME_BASE
                               if base_frame else
                               Base_pb2.CARTESIAN_REFERENCE_FRAME_MIXED)
        cmd.duration = TWIST_WATCHDOG_MS
        cmd.twist.linear_x = float(linear[0])
        cmd.twist.linear_y = float(linear[1])
        cmd.twist.linear_z = float(linear[2])
        if angular is not None:
            cmd.twist.angular_x = float(angular[0])
            cmd.twist.angular_y = float(angular[1])
            cmd.twist.angular_z = float(angular[2])
        self.base.SendTwistCommand(cmd)

    def send_zero_twist(self, base_frame: bool = False):
        """Zero-velocity command with watchdog duration (safe stop)."""
        if self.twists_suppressed:
            return
        try:
            self.send_twist(np.zeros(3), np.zeros(3), base_frame=base_frame)
        except Exception:
            pass

    # ── Gripper ───────────────────────────────────────────────────────────────
    def send_gripper(self, value: float):
        """Position command, 0 = open .. 1 = closed. Raises on RPC failure."""
        gc = Base_pb2.GripperCommand()
        gc.mode = Base_pb2.GRIPPER_POSITION
        f = gc.gripper.finger.add()
        f.finger_identifier = 1
        f.value = float(np.clip(value, 0.0, 1.0))
        self.base.SendGripperCommand(gc)

    # ── Home reset ────────────────────────────────────────────────────────────
    def reach_home_joints(self, joints_deg=None, tolerance_deg: float = 2.0,
                          timeout_s: float = 30.0) -> bool:
        """Drive to the joint-space home and wait for arrival. True on success.

        Joint space, not Cartesian: a Cartesian reach is IK-ambiguous on a 7-DOF
        arm, so the elbow could settle differently every episode.

        No speed constraint is attached — JOINT_CONSTRAINT_SPEED on
        reach_joint_angles is rejected by this firmware (verified 2026-07-27:
        ACTION_ABORT/METHOD_FAILED via notification). Without one the firmware
        planner picks a moderate profile, the same as the web app's Home action.

        Arrival is confirmed by POLLING joint angles rather than by action
        notifications, which get missed or arrive stale and cause false
        timeouts. The caller must own the router for the duration (set
        twists_suppressed).
        """
        joints_deg = joints_deg or HOME_JOINTS_DEG

        action = Base_pb2.Action()
        action.name = 'Home'
        action.application_data = ''
        for i, ang in enumerate(joints_deg):
            ja = action.reach_joint_angles.joint_angles.joint_angles.add()
            ja.joint_identifier = i
            ja.value = ang
        self.base.ExecuteAction(action)

        def joint_err_deg(fb):
            # Wrap-aware: joints 1/5 sit at ~359.6 deg, right on the 0/360 seam.
            errs = []
            for i, tgt in enumerate(joints_deg):
                d = abs(fb.actuators[i].position - tgt) % 360.0
                errs.append(min(d, 360.0 - d))
            return max(errs)

        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            try:
                if joint_err_deg(self.refresh_feedback()) < tolerance_deg:
                    return True
            except Exception:
                pass
            time.sleep(0.1)
        return False
