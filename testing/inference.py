#!/usr/bin/env python3
"""
Real-time diffusion policy inference for Kinova Gen3 with ZED front + DJI wrist cameras.

Adapted from Robomimic/testing/inference.py for the HoloLens + Kinova Gen3 setup.

Two-process architecture:
  Main process  — ROS2 PolicyNode: collects observations, tracks actions via Kortex P-loop
  GPU process   — Loads model, runs predict_action(), writes target poses to shared memory

The Kinova is controlled via a 30 Hz P-loop velocity controller (SendTwistCommand in
SINGLE_LEVEL_SERVOING mode), matching the kinova_hand_controller.py approach.

Usage:
    python inference.py --model /path/to/checkpoint.ckpt
    python inference.py --model /path/to/checkpoint.ckpt --dt 0.033 --action-horizon 6
"""

import math
import multiprocessing
import os
import pathlib
import sys
import threading
import time
from multiprocessing import Manager, Process, Queue

import cv2
import numpy as np
import pygame
import torch
import dill
import hydra

# Path setup: UMI / diffusion_policy codebase
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_UMI_ROOT = os.path.join(_THIS_DIR, "..", "..", "Robomimic", "dt_ag-main")
_UMI_DP   = os.path.join(_UMI_ROOT, "universal_manipulation_interface")
_TRAINING_DIR = os.path.join(_THIS_DIR, "..", "training")

for p in [_UMI_DP, _UMI_ROOT, _TRAINING_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

os.chdir(_UMI_DP)

from omegaconf import OmegaConf
OmegaConf.register_new_resolver("eval", eval, use_cache=True)

from diffusion_policy.workspace.base_workspace import BaseWorkspace

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
import message_filters
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, Bool
from cv_bridge import CvBridge
from scipy.spatial.transform import Rotation as R
from piezense_interfaces.msg import PiezenseSystemArray

from kortex_api.TCPTransport import TCPTransport
from kortex_api.RouterClient import RouterClient
from kortex_api.SessionManager import SessionManager
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.messages import Session_pb2, Base_pb2

# ── Constants ──────────────────────────────────────────────────────────────────

POSE_TOPIC      = "robot_obs/pose"
GRIPPER_TOPIC   = "robot_obs/gripper"
ZED_FRONT_TOPIC = "/zed_front/zed_node/left/image_rect_color"
DJI_WRIST_TOPIC  = "/dji_wrist/dji_wrist/color/image_raw"
PIEZENSE_TOPIC  = "piezense/data"

PIEZENSE_SYSTEM_ID      = 0
PIEZENSE_INPUT_CHANNELS = 2
PIEZENSE_INPUT_CHAN_IDS = [2, 3]

KINOVA_IP = "192.168.1.10"

# Workspace bounds (metres) — must match data collection controller params
WS_X = (0.25, 0.45)
WS_Y = (-0.35, 0.35)
WS_Z = (0.025, 0.25)

HOME_X, HOME_Y, HOME_Z   = 0.35, 0.0, 0.12
HOME_TX, HOME_TY, HOME_TZ = -180.0, 0.0, 90.0   # Euler XYZ degrees

MAX_LINEAR_SPEED  = 0.10   # m/s
MAX_ANGULAR_SPEED = 15.0   # deg/s
P_GAIN            = 2.0
VEL_ALPHA         = 0.4    # smoothing factor (0 = no smoothing)
TWIST_WATCHDOG_MS = 200    # ms — Kortex auto-stops if we miss this many ms

IMG_SIZE     = 224
OBS_HORIZON  = 2

# CAMERA_KEYS = ["zed_front_rgb", "dji_wrist_rgb"]
CAMERA_KEYS = ["dji_wrist_rgb"]
# CAMERA_TOPICS = {
#     "zed_front_rgb": ZED_FRONT_TOPIC,
#     "dji_wrist_rgb":  DJI_WRIST_TOPIC,
# }
CAMERA_TOPICS = {
    "dji_wrist_rgb":  DJI_WRIST_TOPIC,
}


# ── Rotation helpers ───────────────────────────────────────────────────────────

def pose_msg_to_10d(pose_msg: PoseStamped, gripper: float) -> np.ndarray:
    """Convert PoseStamped + gripper scalar to 10D [xyz, rot6d, grip].

    Encoding MUST match training/convert_data.py (what the zarr was built with):
      - scipy from_quat([qx,qy,qz,qw]) directly — the physical quaternion, no reorder.
      - rot6d = first two COLUMNS of the rotation matrix (Zhou et al. 2019).
    (Note: data_collection/hdf5_to_zarr.py uses pytorch3d rows + a misread quat,
     a DIFFERENT convention — do not mix the two converters.)
    """
    pos = np.array([pose_msg.pose.position.x,
                    pose_msg.pose.position.y,
                    pose_msg.pose.position.z])
    q = np.array([pose_msg.pose.orientation.x,
                  pose_msg.pose.orientation.y,
                  pose_msg.pose.orientation.z,
                  pose_msg.pose.orientation.w])  # [qx, qy, qz, qw]
    rot_mat = R.from_quat(q).as_matrix()                     # (3, 3), physical
    rot6d = np.concatenate([rot_mat[:, 0], rot_mat[:, 1]])   # first two COLUMNS  (6,)
    return np.concatenate([pos, rot6d, [gripper]]).astype(np.float32)  # (10,)


def rot6d_to_euler_xyz(rot6d: np.ndarray) -> np.ndarray:
    """Decode training-encoded rot6d → Euler XYZ degrees.

    Inverse of pose_msg_to_10d / convert_data.py (COLUMNS convention):
      rot6d = [col0, col1] of the physical rotation matrix.
    Gram-Schmidt to re-orthonormalize (network output isn't exactly orthonormal),
    rebuild the matrix from its columns, then read physical Euler XYZ via scipy.
    """
    c0 = rot6d[:3] / (np.linalg.norm(rot6d[:3]) + 1e-8)
    c1 = rot6d[3:6] - np.dot(rot6d[3:6], c0) * c0
    c1 = c1 / (np.linalg.norm(c1) + 1e-8)
    c2 = np.cross(c0, c1)
    mat = np.column_stack([c0, c1, c2])                     # columns (physical R)
    return R.from_matrix(mat).as_euler("xyz", degrees=True)


# ── Model loading ──────────────────────────────────────────────────────────────

def load_policy(model_path: str, num_inference_steps: int = 16):
    path = pathlib.Path(model_path)
    payload = torch.load(path.open("rb"), pickle_module=dill, map_location="cpu")
    cfg = payload["cfg"]
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    policy.num_inference_steps = num_inference_steps
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy.eval().to(device)
    print(f"Policy loaded on {device} (EMA={cfg.training.use_ema}, "
          f"diffusion_steps={num_inference_steps})")
    return policy


def load_obs_keys(model_path: str) -> list[str]:
    path = pathlib.Path(model_path)
    payload = torch.load(path.open("rb"), pickle_module=dill, map_location="cpu")
    return list(payload["cfg"].policy.shape_meta["obs"].keys())


# ── PolicyNode ─────────────────────────────────────────────────────────────────

class PolicyNode(Node):
    """ROS2 node: collects observations, tracks policy targets via Kinova P-loop."""

    def __init__(self, shared_obs: dict, start_time: float, model_path: str):
        super().__init__("kinova_policy_node")
        np.set_printoptions(suppress=True, precision=4)

        self.shared_obs = shared_obs
        self.start_time = start_time
        self.obs_keys = load_obs_keys(model_path)
        self.get_logger().info(f"Policy obs keys: {self.obs_keys}")

        # ── Kinova connection ────────────────────────────────────────────────
        self.get_logger().info(f"Connecting to Kinova Gen3 at {KINOVA_IP}...")
        self._connect_kinova()
        self._setup_servoing()

        # ── Controller state ─────────────────────────────────────────────────
        self.current_target_xyz = None      # np.ndarray (3,) metres
        self.current_target_euler = None    # np.ndarray (3,) degrees XYZ
        self.current_gripper_cmd  = 0.0
        self._smoothed_vel = np.zeros(3)
        self.paused = True
        self.is_resetting = False

        # ── QoS ──────────────────────────────────────────────────────────────
        sensor_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
        )

        # ── Synchronized observation subscribers ─────────────────────────────
        self._bridge = CvBridge()
        self.gripper_state = 0.0

        self.pose_sub     = message_filters.Subscriber(self, PoseStamped, POSE_TOPIC,      qos_profile=sensor_qos)
        self.gripper_sub  = message_filters.Subscriber(self, Float32,     GRIPPER_TOPIC,   qos_profile=sensor_qos)
        # self.zed_sub      = message_filters.Subscriber(self, Image,       ZED_FRONT_TOPIC, qos_profile=sensor_qos)
        self.wrist_sub    = message_filters.Subscriber(self, Image,       DJI_WRIST_TOPIC,  qos_profile=sensor_qos)

        self.sync = message_filters.ApproximateTimeSynchronizer(
            # [self.pose_sub, self.gripper_sub, self.zed_sub, self.wrist_sub],
            [self.pose_sub, self.gripper_sub, self.wrist_sub],
            queue_size=100,
            slop=0.05,
            allow_headerless=True,
        )
        self.sync.registerCallback(self.synced_obs_callback)

        # ── Enable DJI camera ────────────────────────────────────────────────
        _dji_enable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=1,
        )
        self._dji_enable_pub = self.create_publisher(Bool, '/dji_camera/enable', _dji_enable_qos)
        self.create_timer(0.5, self._enable_dji_camera)

        # ── Observation buffers ──────────────────────────────────────────────
        self.pose_buffer = []
        self.cam_buffers = {k: [] for k in CAMERA_KEYS}

        # ── Piezense side-channel (latest-value, not synchronized) ───────────
        self._latest_piezense = np.array([111337.0, 110375.0], dtype=np.float32)  # training mean fallback
        self.piezense_buffer  = []
        self.create_subscription(
            PiezenseSystemArray, PIEZENSE_TOPIC, self._piezense_cb, 10
        )

        # ── Timers ───────────────────────────────────────────────────────────
        self.create_timer(1.0 / 30.0, self.update_observation)   # pack shared_obs
        self.create_timer(1.0 / 30.0, self.control_callback)     # send velocity commands

        self.reset_to_home()
        self.get_logger().info("PolicyNode ready. Press S to start, D to pause, R to reset.")

    # ── Kinova setup ──────────────────────────────────────────────────────────

    def _connect_kinova(self):
        self._transport = TCPTransport()
        self._router = RouterClient(
            self._transport,
            lambda ex: self.get_logger().error(f"Kortex error: {ex}"),
        )
        self._transport.connect(KINOVA_IP, 10000)

        session_info = Session_pb2.CreateSessionInfo()
        session_info.username = "admin"
        session_info.password = "admin"
        session_info.session_inactivity_timeout = 60000
        session_info.connection_inactivity_timeout = 2000

        self._session_manager = SessionManager(self._router)
        self._session_manager.CreateSession(session_info)

        self._base        = BaseClient(self._router)
        self._base_cyclic = BaseCyclicClient(self._router)
        self.get_logger().info("Connected to Kinova Gen3")

    def _setup_servoing(self):
        mode = Base_pb2.ServoingModeInformation()
        mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
        self._base.SetServoingMode(mode)
        time.sleep(0.2)

    def _send_zero_twist(self):
        try:
            cmd = Base_pb2.TwistCommand()
            cmd.reference_frame = Base_pb2.CARTESIAN_REFERENCE_FRAME_MIXED
            cmd.duration = TWIST_WATCHDOG_MS
            self._base.SendTwistCommand(cmd)
        except Exception:
            pass

    # ── Observation callbacks ─────────────────────────────────────────────────

    def synced_obs_callback(self, pose_msg, gripper_msg, wrist_msg):  # zed_msg removed
        now = time.monotonic() - self.start_time
        self.gripper_state = gripper_msg.data

        # Pose -> 10D
        pose_10d = pose_msg_to_10d(pose_msg, self.gripper_state)
        self.pose_buffer.append((pose_10d, now))
        if len(self.pose_buffer) > OBS_HORIZON:
            self.pose_buffer.pop(0)

        # Camera images
        # for cam_key, msg in [("zed_front_rgb", zed_msg), ("dji_wrist_rgb", wrist_msg)]:
        for cam_key, msg in [("dji_wrist_rgb", wrist_msg)]:
            img = self._bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
            # dji_camera_node already publishes IMG_SIZE x IMG_SIZE (INTER_AREA),
            # so no resize here — keeps inference latency identical to collection.
            # Guard only resizes if a frame ever arrives at an unexpected size.
            if img.shape[0] != IMG_SIZE or img.shape[1] != IMG_SIZE:
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
            img = img.astype(np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
            buf = self.cam_buffers[cam_key]
            buf.append((img, now))
            if len(buf) > OBS_HORIZON:
                buf.pop(0)

    def _enable_dji_camera(self):
        msg = Bool()
        msg.data = True
        self._dji_enable_pub.publish(msg)

    def _piezense_cb(self, msg: PiezenseSystemArray):
        for sys_msg in msg.system:
            if sys_msg.system_id == PIEZENSE_SYSTEM_ID:
                readings = list(sys_msg.pressure_pa)
                self._latest_piezense = np.array(
                    [float(readings[c]) if c < len(readings) else 0.0
                     for c in PIEZENSE_INPUT_CHAN_IDS],
                    dtype=np.float32,
                )
                break
        now = time.monotonic() - self.start_time
        self.piezense_buffer.append((self._latest_piezense.copy(), now))
        if len(self.piezense_buffer) > OBS_HORIZON:
            self.piezense_buffer.pop(0)

    def update_observation(self):
        """Pack observation buffers into shared_obs for the GPU process."""
        if len(self.pose_buffer) < OBS_HORIZON:
            return
        for k in CAMERA_KEYS:
            if len(self.cam_buffers[k]) < OBS_HORIZON:
                return

        pose_slice = self.pose_buffer[-OBS_HORIZON:]

        # One-time diagnostic: log the obs rot6d vs expected training value
        if not getattr(self, '_obs_logged', False):
            self._obs_logged = True
            p10d = pose_slice[-1][0]
            obs_euler = rot6d_to_euler_xyz(p10d[3:9])
            self.get_logger().info(
                f"OBS diagnostic — pose rot6d={np.round(p10d[3:9], 4)}  "
                f"→ decoded theta_xyz={np.round(obs_euler, 2)}  "
                f"(expected ≈ [-179.3, -0.4, 89.3])"
            )

        obs_dict = {
            "pose": torch.from_numpy(
                np.stack([p[0] for p in pose_slice])
            ).unsqueeze(0),  # (1, 2, 10)
            "pose_timestamps": np.array([p[1] for p in pose_slice]),
        }

        for cam_key in CAMERA_KEYS:
            cam_slice = self.cam_buffers[cam_key][-OBS_HORIZON:]
            obs_dict[cam_key] = torch.from_numpy(
                np.stack([c[0] for c in cam_slice])
            ).unsqueeze(0)  # (1, 2, 3, 224, 224)
            obs_dict[f"{cam_key}_timestamps"] = np.array([c[1] for c in cam_slice])

        # Piezense: pad with last known value if buffer not yet full
        if len(self.piezense_buffer) >= OBS_HORIZON:
            pz_slice = self.piezense_buffer[-OBS_HORIZON:]
        elif self.piezense_buffer:
            pz_slice = [self.piezense_buffer[0]] * (OBS_HORIZON - len(self.piezense_buffer)) \
                       + self.piezense_buffer
        else:
            pz_slice = [(np.zeros(PIEZENSE_INPUT_CHANNELS, dtype=np.float32), 0.0)] * OBS_HORIZON

        obs_dict["piezense_pressure"] = torch.from_numpy(
            np.stack([p[0] for p in pz_slice])
        ).unsqueeze(0)  # (1, 2, 2)

        self.shared_obs["obs"] = obs_dict

    # ── Velocity control ──────────────────────────────────────────────────────

    def control_callback(self):
        """30 Hz: consume next target from shared_obs and drive the arm toward it."""
        if self.is_resetting:
            # _do_home_reset owns all twist state during reset — sending zero here
            # would continuously renew the watchdog and prevent ExecuteAction.
            self._smoothed_vel[:] = 0.0
            return
        if self.paused:
            self._smoothed_vel[:] = 0.0
            self._send_zero_twist()
            return

        # Snapshot targets once to avoid race with action_executor / reset threads
        target_xyz   = self.current_target_xyz
        target_euler = self.current_target_euler
        if target_xyz is None:
            self._send_zero_twist()
            return

        try:
            feedback = self._base_cyclic.RefreshFeedback()
            current_xyz = np.array([
                feedback.base.tool_pose_x,
                feedback.base.tool_pose_y,
                feedback.base.tool_pose_z,
            ])
            current_tz = feedback.base.tool_pose_theta_z

            # Linear P-loop
            pos_err = target_xyz - current_xyz
            raw_vel = P_GAIN * pos_err
            speed = np.linalg.norm(raw_vel)
            if speed > MAX_LINEAR_SPEED:
                raw_vel *= MAX_LINEAR_SPEED / speed

            # Velocity smoothing
            self._smoothed_vel = VEL_ALPHA * raw_vel + (1.0 - VEL_ALPHA) * self._smoothed_vel
            send_vel = self._smoothed_vel.copy()
            smooth_speed = np.linalg.norm(send_vel)
            if smooth_speed > MAX_LINEAR_SPEED:
                send_vel *= MAX_LINEAR_SPEED / smooth_speed

            # Angular P-loop in world/base frame.
            # CARTESIAN_REFERENCE_FRAME_BASE puts angular_z in the world Z axis,
            # so the P-loop sign is correct regardless of tool orientation.
            # (CARTESIAN_REFERENCE_FRAME_MIXED uses tool-frame angular, which at
            # theta_x≈-180° maps world +Z → tool -Z, inverting the sign.)
            ang_vel_z = 0.0
            if target_euler is not None:
                tz_err = (target_euler[2] - current_tz + 180.0) % 360.0 - 180.0
                ang_vel_z = float(np.clip(P_GAIN * tz_err, -MAX_ANGULAR_SPEED, MAX_ANGULAR_SPEED))

            cmd = Base_pb2.TwistCommand()
            cmd.reference_frame = Base_pb2.CARTESIAN_REFERENCE_FRAME_BASE
            cmd.duration = TWIST_WATCHDOG_MS
            cmd.twist.linear_x  = float(send_vel[0])
            cmd.twist.linear_y  = float(send_vel[1])
            cmd.twist.linear_z  = float(send_vel[2])
            cmd.twist.angular_x = 0.0
            cmd.twist.angular_y = 0.0
            cmd.twist.angular_z = ang_vel_z
            self._base.SendTwistCommand(cmd)

        except Exception as e:
            self.get_logger().error(f"Control loop error: {e}")
            self._send_zero_twist()

    def _execute_action(self, action_10d: np.ndarray):
        """Accept a 10D policy action and update the tracked target."""
        pos = action_10d[:3].copy()
        pos[0] = np.clip(pos[0], WS_X[0], WS_X[1])
        pos[1] = np.clip(pos[1], WS_Y[0], WS_Y[1])
        pos[2] = np.clip(pos[2], WS_Z[0], WS_Z[1])

        euler_deg = rot6d_to_euler_xyz(action_10d[3:9])
        grip = float(np.clip(action_10d[9], 0.0, 1.0))

        self.current_target_xyz   = pos
        self.current_target_euler = euler_deg
        self.current_gripper_cmd  = grip

        if not getattr(self, '_action_logged', False):
            self._action_logged = True
            self.get_logger().info(
                f"First action target: xyz={np.round(pos, 4)}  "
                f"theta_xyz={np.round(euler_deg, 2)}  grip={grip:.3f}"
            )

        # Gripper command
        try:
            gc = Base_pb2.GripperCommand()
            gc.mode = Base_pb2.GRIPPER_POSITION
            f = gc.gripper.finger.add()
            f.finger_identifier = 1
            f.value = grip
            self._base.SendGripperCommand(gc)
        except Exception as e:
            self.get_logger().warn(f"Gripper command error: {e}")

    # ── Keyboard controls ─────────────────────────────────────────────────────

    def pause_policy(self):
        self.get_logger().info("Paused.")
        self.paused = True
        self.shared_obs["paused"] = True
        self.current_target_xyz = None
        self._smoothed_vel[:] = 0.0

    def resume_policy(self):
        self.get_logger().info("Resumed.")
        self.paused = False
        self.shared_obs["paused"] = False

    def reset_to_home(self):
        if self.is_resetting:
            return
        self.get_logger().info("Resetting to home...")
        self.paused = True
        self.shared_obs["paused"] = True
        self.current_target_xyz = None
        self._smoothed_vel[:] = 0.0
        self.is_resetting = True
        threading.Thread(target=self._do_home_reset, daemon=True).start()

    def _do_home_reset(self):
        try:
            self._send_zero_twist()
            # 1 s gives the watchdog time to expire AND lets any in-flight
            # ACTION_END/ABORT notifications drain before we register the new listener.
            time.sleep(1.0)

            action = Base_pb2.Action()
            action.name = "Home"
            action.application_data = ""
            speed = Base_pb2.CartesianSpeed()
            speed.translation = 0.08
            speed.orientation = 12.0
            action.reach_pose.constraint.speed.CopyFrom(speed)
            pose = action.reach_pose.target_pose
            pose.x, pose.y, pose.z = HOME_X, HOME_Y, HOME_Z
            pose.theta_x, pose.theta_y, pose.theta_z = HOME_TX, HOME_TY, HOME_TZ

            finished = threading.Event()

            def _on_action(notif, ev=finished):
                if notif.action_event in (Base_pb2.ACTION_END, Base_pb2.ACTION_ABORT):
                    ev.set()

            self._base.OnNotificationActionTopic(_on_action, Base_pb2.NotificationOptions())
            self._base.ExecuteAction(action)

            if not finished.wait(timeout=30.0):
                self.get_logger().warn("Home reset timed out")
                self._base.StopAction()

            # Open gripper
            gc = Base_pb2.GripperCommand()
            gc.mode = Base_pb2.GRIPPER_POSITION
            f = gc.gripper.finger.add()
            f.finger_identifier = 1
            f.value = 0.0
            self._base.SendGripperCommand(gc)
            time.sleep(1.0)
            self.get_logger().info("Reset complete. Press S to start.")
        except Exception as e:
            self.get_logger().error(f"Reset error: {e}")
        finally:
            try:
                self._setup_servoing()
            except Exception:
                pass
            self.is_resetting = False

    def cleanup(self):
        try:
            self._send_zero_twist()
            self._session_manager.CloseSession()
            self._transport.disconnect()
        except Exception:
            pass


# ── Inference process ──────────────────────────────────────────────────────────

def inference_loop(model_path, shared_obs, action_queue,
                   n_action_steps=8, device="cuda", start_time=0,
                   dt=0.033, num_inference_steps=16,
                   latency_offset_s=0.0):
    """GPU process: load model and run inference, posting targets to the main process."""
    import sys as _sys
    _sys.stdout.reconfigure(line_buffering=True)  # force line-flush in spawned subprocess

    latency_steps = max(0, round(latency_offset_s / dt))
    if latency_steps:
        print(f"Latency offset: {latency_offset_s*1000:.0f} ms = {latency_steps} steps at dt={dt}s", flush=True)

    policy = load_policy(model_path, num_inference_steps)
    model_obs_keys = load_obs_keys(model_path)
    print(f"Model expects obs keys: {model_obs_keys}", flush=True)

    # Wait for first observation
    while shared_obs.get("obs") is None:
        time.sleep(0.05)
        print("Waiting for first observation...", flush=True)

    # Track timestamp freshness
    prev_timestamps = {}
    obs_now = shared_obs["obs"]
    if "pose_timestamps" in obs_now:
        prev_timestamps["pose"] = obs_now["pose_timestamps"][-1]
    for cam_key in CAMERA_KEYS:
        ts_key = f"{cam_key}_timestamps"
        if ts_key in obs_now:
            prev_timestamps[cam_key] = obs_now[ts_key][-1]

    print("Inference loop started.", flush=True)
    _actions_logged = False

    while True:
        if shared_obs.get("paused", True):
            time.sleep(0.05)
            continue

        loop_start = time.time()

        # Wait until all sensors have fresh data
        wait_start = time.time()
        while True:
            obs_now = shared_obs["obs"]
            all_new = True
            if "pose_timestamps" in obs_now:
                if np.min(obs_now["pose_timestamps"]) <= prev_timestamps.get("pose", -1):
                    all_new = False
            for cam_key in CAMERA_KEYS:
                ts_key = f"{cam_key}_timestamps"
                if ts_key in obs_now:
                    if np.min(obs_now[ts_key]) <= prev_timestamps.get(cam_key, -1):
                        all_new = False
            if all_new:
                break
            elapsed = time.time() - wait_start
            if elapsed > 1.0 and int(elapsed) != int(elapsed - 0.001):
                print(f"Waiting for new sensor data ({elapsed:.1f}s)...", flush=True)
            time.sleep(0.001)

        wait_time = time.time() - wait_start

        # Update timestamps
        if "pose_timestamps" in obs_now:
            prev_timestamps["pose"] = obs_now["pose_timestamps"][-1]
        for cam_key in CAMERA_KEYS:
            ts_key = f"{cam_key}_timestamps"
            if ts_key in obs_now:
                prev_timestamps[cam_key] = obs_now[ts_key][-1]

        # Build model obs dict (filter to expected keys, move to GPU)
        model_obs = {}
        for k in model_obs_keys:
            if k in obs_now:
                v = obs_now[k]
                model_obs[k] = v.to(device) if isinstance(v, torch.Tensor) else v

        # Run inference
        t_infer = time.time()
        with torch.no_grad():
            actions = policy.predict_action(model_obs)["action"][0].detach().cpu().numpy()
        infer_time = time.time() - t_infer

        if not _actions_logged:
            _actions_logged = True
            print(f"predict_action output: {actions.shape[0]} steps × {actions.shape[1]} dims", flush=True)

        # Schedule n_action_steps steps with timestamps, skipping latency_steps
        # to account for system delay between observation capture and execution.
        t_start = time.monotonic()
        action_queue.put(("CLEAR_PENDING", t_start))
        start_idx = min(latency_steps, len(actions) - 1)
        for i, act in enumerate(actions[start_idx: start_idx + n_action_steps]):
            ts = t_start + i * dt
            action_queue.put((act, ts))

        # Debug: log first predicted action in physical Euler angles
        a0 = actions[start_idx]
        euler = rot6d_to_euler_xyz(a0[3:9])
        print(f"Inference: {infer_time*1000:.0f}ms | act[0] xyz={a0[:3]} theta_xyz={euler}", flush=True)

        total_time = time.time() - loop_start
        print(f"  Wait: {wait_time*1000:.0f}ms | Total: {total_time*1000:.0f}ms | Actions: {n_action_steps}", flush=True)

        time.sleep(dt)


# ── Pygame control window ──────────────────────────────────────────────────────

def monitor_keys(policy_node: PolicyNode, shared_obs: dict):
    """Background thread: pygame window for pause / resume / reset."""
    try:
        pygame.init()
        screen = pygame.display.set_mode((340, 210))
        pygame.display.set_caption("Kinova Policy Control")
        clock = pygame.time.Clock()
        font       = pygame.font.SysFont("monospace", 18)
        font_small = pygame.font.SysFont("monospace", 14)

        COLOR_PAUSED  = (50, 50, 60)
        COLOR_RUNNING = (20, 60, 30)

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    policy_node.pause_policy()
                    os._exit(0)
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_d:
                        policy_node.pause_policy()
                    elif event.key == pygame.K_s:
                        policy_node.resume_policy()
                    elif event.key == pygame.K_r:
                        policy_node.reset_to_home()
                    elif event.key == pygame.K_q:
                        policy_node.pause_policy()
                        time.sleep(0.2)
                        os._exit(0)

            paused = shared_obs.get("paused", True)
            screen.fill(COLOR_PAUSED if paused else COLOR_RUNNING)

            status_text  = "PAUSED" if paused else "RUNNING"
            status_color = (255, 200, 50) if paused else (50, 255, 80)
            screen.blit(font.render(status_text, True, status_color), (120, 20))

            keys_info = [
                ("S", "Start / Resume"),
                ("D", "Done  / Pause"),
                ("R", "Reset to home"),
                ("Q", "Quit"),
            ]
            for i, (key, desc) in enumerate(keys_info):
                line = font_small.render(f"  {key}  -  {desc}", True, (200, 200, 200))
                screen.blit(line, (30, 65 + i * 30))

            pygame.display.flip()
            clock.tick(10)
    except Exception as e:
        print(f"Pygame error: {e}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Diffusion Policy Inference — Kinova Gen3")
    parser.add_argument("--model",           type=str, required=True,  help="Path to .ckpt file")
    parser.add_argument("--dt",              type=float, default=0.1,   help="Action step period (s)")
    parser.add_argument("--n-action-steps",  type=int,   default=8,    help="Actions executed per inference cycle")
    parser.add_argument("--diffusion-steps", type=int,   default=16,   help="DDIM inference steps")
    parser.add_argument("--latency-offset-s", type=float, default=0.0,
                        help="System latency to compensate (seconds). Skips this many steps at the "
                             "start of each predicted action sequence. Measure with latency_calculation.py.")
    parser.add_argument("--no-pygame",       action="store_true",      help="Disable pygame window")
    args = parser.parse_args()

    multiprocessing.set_start_method("spawn", force=True)

    manager   = Manager()
    shared_obs = manager.dict(obs=None, paused=True)
    action_queue = Queue()
    start_time = time.monotonic()

    rclpy.init()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Model:           {args.model}")
    print(f"dt:              {args.dt}s  ({1/args.dt:.0f} Hz)")
    print(f"Obs horizon:     {OBS_HORIZON}")
    print(f"Num Action steps:  {args.n_action_steps}")
    print(f"Diffusion steps: {args.diffusion_steps}")
    if args.latency_offset_s:
        print(f"Latency offset:  {args.latency_offset_s*1000:.0f} ms "
              f"({round(args.latency_offset_s / args.dt)} steps)")

    # GPU inference process
    inf_proc = Process(
        target=inference_loop,
        args=(args.model, shared_obs, action_queue,
              args.n_action_steps, device, start_time,
              args.dt, args.diffusion_steps, args.latency_offset_s),
        daemon=True,
    )
    inf_proc.start()

    node = PolicyNode(shared_obs, start_time, args.model)

    # Action execution thread: drains action_queue and calls _execute_action at scheduled times
    def action_executor():
        pending = []
        while True:
            while not action_queue.empty():
                item = action_queue.get()
                if isinstance(item, tuple) and isinstance(item[0], str) and item[0] == "CLEAR_PENDING":
                    pending.clear()
                else:
                    pending.append(item)
            now = time.monotonic()
            remaining = []
            for act, ts in pending:
                if now >= ts and not node.paused:
                    node._execute_action(act)
                else:
                    remaining.append((act, ts))
            pending = remaining
            time.sleep(0.005)

    threading.Thread(target=action_executor, daemon=True).start()

    if not args.no_pygame:
        key_thread = threading.Thread(
            target=monitor_keys, args=(node, shared_obs), daemon=True
        )
        key_thread.start()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cleanup()
        node.destroy_node()
        rclpy.shutdown()
        inf_proc.terminate()


if __name__ == "__main__":
    main()
