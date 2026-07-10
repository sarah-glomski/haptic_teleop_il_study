#!/usr/bin/env python3
"""
Real-time diffusion policy inference (charm-lab UMI pipeline, real UMI repo).

Runs checkpoints trained by train.py on data from convert_data.py. Two-process
architecture (ROS2 obs collection + GPU inference) with Kinova P-loop control.

Observation/action schema (single robot):
  robot0_eef_pos(3) + robot0_eef_rot_axis_angle (rot6d ROWS) +
  robot0_gripper_width (meters) + robot0_eef_rot_axis_angle_wrt_start (rot6d),
  camera key 'camera0_rgb'. Encode/decode uses UMI's own pose_util
  (pose_to_mat, mat_to_pose10d, pose10d_to_mat). Obs frames are sampled dt
  apart (matching training down_sample_steps spacing).

pose_repr is 'abs' end-to-end (matching config/task/kinova_teleop.yaml), so
predicted poses are absolute — no relative-frame composition at decode time.

Gripper: model I/O is physical width in METERS (Robotiq 2F-85):
    width = GRIPPER_MAX_WIDTH_M * (1 - kortex_norm);  kortex_norm = 1 - width/GRIPPER_MAX_WIDTH_M
  The constant MUST match convert_data.py.

Usage:
    python inference.py --model /path/to/checkpoint.ckpt
    python inference.py --model ... --dt 0.1 --n-action-steps 8
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

# Path setup: real UMI repo (charm-lab/HapticTeleopIL)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_UMI_ROOT = os.path.join(_THIS_DIR, "..", "..", "HapticTeleopIL",
                         "Imitation Learning", "universal_manipulation_interface")

if _UMI_ROOT not in sys.path:
    sys.path.insert(0, _UMI_ROOT)

os.chdir(_UMI_ROOT)

from omegaconf import OmegaConf
OmegaConf.register_new_resolver("eval", eval, use_cache=True)

from diffusion_policy.workspace.base_workspace import BaseWorkspace  # noqa: F401
# UMI's own pose math (rows-convention rot6d) — the same functions UmiDataset
# used at training time, so encode/decode conventions cannot diverge.
from umi.common.pose_util import pose_to_mat, mat_to_pose10d, pose10d_to_mat
from diffusion_policy.common.pose_repr_util import convert_pose_mat_rep

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
DJI_WRIST_TOPIC = "/dji_wrist/dji_wrist/color/image_raw"
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
VEL_ALPHA         = 0.4
TWIST_WATCHDOG_MS = 200

IMG_SIZE     = 224
# Per-key obs horizons are read from the checkpoint's shape_meta at startup
# (img and low_dim horizons may differ, e.g. camera 2 / low_dim 8) — see
# load_obs_meta(). OBS_HORIZON_FALLBACK is only used if a key lacks one.
OBS_HORIZON_FALLBACK = 2

# MUST match convert_data.py (Robotiq 2F-85 stroke).
GRIPPER_MAX_WIDTH_M = 0.085

CAMERA_KEYS = ["camera0_rgb"]
CAMERA_TOPICS = {"camera0_rgb": DJI_WRIST_TOPIC}


# ── UMI-schema encode/decode helpers ───────────────────────────────────────────

def gripper_norm_to_width_m(g: float) -> float:
    return GRIPPER_MAX_WIDTH_M * (1.0 - float(np.clip(g, 0.0, 1.0)))


def width_m_to_gripper_norm(w: float) -> float:
    return float(np.clip(1.0 - w / GRIPPER_MAX_WIDTH_M, 0.0, 1.0))


def pose_msg_to_raw(pose_msg: PoseStamped, gripper_norm: float) -> np.ndarray:
    """PoseStamped + kortex gripper -> raw 7D [pos(3), rotvec(3), width_m(1)].

    Matches convert_data.py exactly: scipy from_quat([qx,qy,qz,qw]) -> rotvec
    (no wxyz/xyzw ambiguity possible), gripper as physical width in meters.
    rot6d is NEVER computed here — it's derived later via UMI's own pose_to_mat /
    mat_to_pose10d (rows convention), identical to what UmiDataset did in training.
    """
    pos = np.array([pose_msg.pose.position.x,
                    pose_msg.pose.position.y,
                    pose_msg.pose.position.z])
    q = np.array([pose_msg.pose.orientation.x,
                  pose_msg.pose.orientation.y,
                  pose_msg.pose.orientation.z,
                  pose_msg.pose.orientation.w])
    rotvec = R.from_quat(q).as_rotvec()
    return np.concatenate([pos, rotvec,
                           [gripper_norm_to_width_m(gripper_norm)]]).astype(np.float32)


def action10d_to_pos_euler_grip(action_10d: np.ndarray):
    """UMI 10D action [pos(3), rot6d ROWS (6), width_m(1)] -> (pos, euler_xyz_deg, grip_norm).

    pose10d_to_mat is UMI's own decoder (Gram-Schmidt on the rows-convention
    rot6d) — the exact inverse of the training-time mat_to_pose10d.
    pose_repr is 'abs', so the predicted pose is already absolute (no base-pose
    composition needed).
    """
    mat = pose10d_to_mat(action_10d[:9].astype(np.float64))
    pos = mat[:3, 3].copy()
    euler_deg = R.from_matrix(mat[:3, :3]).as_euler("xyz", degrees=True)
    grip_norm = width_m_to_gripper_norm(float(action_10d[9]))
    return pos, euler_deg, grip_norm


def pick_frames_by_time(buffer, n, spacing_s):
    """Pick n entries from [(value, t), ...] ending at the latest, spaced spacing_s
    apart (nearest-neighbor in time). Replicates training-time down_sample_steps
    spacing instead of taking consecutive ~30 Hz frames.
    """
    if not buffer:
        return None
    t_latest = buffer[-1][1]
    picked = []
    for i in range(n):
        t_target = t_latest - (n - 1 - i) * spacing_s
        best = min(buffer, key=lambda e: abs(e[1] - t_target))
        picked.append(best)
    return picked


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
    # Sanity: checkpoint must use the UMI per-signal schema (robot0_* obs keys)
    obs_keys = list(cfg.policy.shape_meta["obs"].keys())
    if not any(k.startswith("robot0_") for k in obs_keys):
        raise RuntimeError(
            f"Checkpoint obs keys {obs_keys} are not UMI-schema (expected robot0_* keys).")
    return policy


def load_obs_keys(model_path: str) -> list:
    path = pathlib.Path(model_path)
    payload = torch.load(path.open("rb"), pickle_module=dill, map_location="cpu")
    return list(payload["cfg"].policy.shape_meta["obs"].keys())


def load_obs_meta(model_path: str):
    """Read (obs_keys, per-key horizons) from the checkpoint's shape_meta.

    Inference must pack each obs key with the SAME horizon it was trained with
    (img and low_dim horizons can differ, e.g. camera0_rgb: 2, robot0_*: 8).
    """
    path = pathlib.Path(model_path)
    payload = torch.load(path.open("rb"), pickle_module=dill, map_location="cpu")
    obs_meta = payload["cfg"].policy.shape_meta["obs"]
    keys = list(obs_meta.keys())
    horizons = {k: int(obs_meta[k].get("horizon", OBS_HORIZON_FALLBACK)) for k in keys}
    return keys, horizons


# ── PolicyNode ─────────────────────────────────────────────────────────────────

class PolicyNode(Node):
    """ROS2 node: collects observations (UMI schema), tracks policy targets via P-loop."""

    def __init__(self, shared_obs: dict, start_time: float, model_path: str, dt: float):
        super().__init__("kinova_policy_node")
        np.set_printoptions(suppress=True, precision=4)

        self.shared_obs = shared_obs
        self.start_time = start_time
        self.dt = dt
        self.obs_keys, self.key_horizons = load_obs_meta(model_path)
        self.max_horizon = max(self.key_horizons.values())
        # Raw ~30 Hz buffer must span (max_horizon-1)*dt of history plus margin.
        self.raw_buffer_len = max(40, int((self.max_horizon + 2) * self.dt * 30) + 10)
        self.get_logger().info(
            f"Policy obs keys: {self.obs_keys} | horizons: {self.key_horizons} "
            f"| raw buffer: {self.raw_buffer_len}")

        self.get_logger().info(f"Connecting to Kinova Gen3 at {KINOVA_IP}...")
        self._connect_kinova()
        self._setup_servoing()

        # ── Controller state ─────────────────────────────────────────────────
        self.current_target_xyz = None
        self.current_target_euler = None
        self.current_gripper_cmd  = 0.0
        self._smoothed_vel = np.zeros(3)
        self.paused = True
        self.is_resetting = False

        # Episode start pose (4x4 mat) for robot0_eef_rot_axis_angle_wrt_start.
        # Captured from the first observation after each home reset — the same
        # anchor as training's robot0_demo_start_pose (episode's first frame,
        # robot at home). Cleared on every reset.
        self.episode_start_pose_mat = None

        sensor_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
        )

        self._bridge = CvBridge()
        self.gripper_state = 0.0

        self.pose_sub    = message_filters.Subscriber(self, PoseStamped, POSE_TOPIC,    qos_profile=sensor_qos)
        self.gripper_sub = message_filters.Subscriber(self, Float32,     GRIPPER_TOPIC, qos_profile=sensor_qos)
        self.wrist_sub   = message_filters.Subscriber(self, Image,       DJI_WRIST_TOPIC, qos_profile=sensor_qos)

        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.pose_sub, self.gripper_sub, self.wrist_sub],
            queue_size=100,
            slop=0.05,
            allow_headerless=True,
        )
        self.sync.registerCallback(self.synced_obs_callback)

        _dji_enable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=1,
        )
        self._dji_enable_pub = self.create_publisher(Bool, '/dji_camera/enable', _dji_enable_qos)
        self.create_timer(0.5, self._enable_dji_camera)

        # ── Observation buffers (raw ~30 Hz; dt-spaced frames picked at pack time)
        self.pose_buffer = []                                  # (raw7, t)
        self.cam_buffers = {k: [] for k in CAMERA_KEYS}        # (img CHW float, t)

        self._latest_piezense = np.array([111337.0, 110375.0], dtype=np.float32)
        self.piezense_buffer  = []
        self.create_subscription(
            PiezenseSystemArray, PIEZENSE_TOPIC, self._piezense_cb, 10
        )

        self.create_timer(1.0 / 30.0, self.update_observation)
        self.create_timer(1.0 / 30.0, self.control_callback)

        self.reset_to_home()
        self.get_logger().info("PolicyNode (UMI) ready. Press S to start, D to pause, R to reset.")

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

    def synced_obs_callback(self, pose_msg, gripper_msg, wrist_msg):
        now = time.monotonic() - self.start_time
        self.gripper_state = gripper_msg.data

        raw7 = pose_msg_to_raw(pose_msg, self.gripper_state)
        self.pose_buffer.append((raw7, now))
        if len(self.pose_buffer) > self.raw_buffer_len:
            self.pose_buffer.pop(0)

        # Capture episode start pose on the first observation after a reset
        # completes (mirrors training's demo_start_pose = episode's first frame).
        if self.episode_start_pose_mat is None and not self.is_resetting:
            self.episode_start_pose_mat = pose_to_mat(raw7[:6].astype(np.float64))
            self.get_logger().info(
                f"Episode start pose captured: xyz={np.round(raw7[:3], 4)}")

        for cam_key in CAMERA_KEYS:
            msg = wrist_msg
            img = self._bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
            if img.shape[0] != IMG_SIZE or img.shape[1] != IMG_SIZE:
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
            img = img.astype(np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
            buf = self.cam_buffers[cam_key]
            buf.append((img, now))
            if len(buf) > self.raw_buffer_len:
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
        if len(self.piezense_buffer) > self.raw_buffer_len:
            self.piezense_buffer.pop(0)

    def update_observation(self):
        """Pack UMI-schema observation dict into shared_obs for the GPU process.

        Obs frames are picked self.dt apart (training down_sample_steps spacing),
        NOT consecutive ~30 Hz frames. Each key uses ITS OWN horizon from the
        checkpoint's shape_meta (img and low_dim horizons may differ). If the
        buffer doesn't yet span the full horizon, nearest-neighbor picking
        repeats the earliest frame — the same start-of-episode padding the
        training sampler used.
        """
        if not self.pose_buffer:
            return
        for k in CAMERA_KEYS:
            if not self.cam_buffers[k]:
                return
        if self.episode_start_pose_mat is None:
            return

        H = self.key_horizons  # per-key horizons from the checkpoint

        # Pose-derived keys share one raw stack at the max low-dim horizon;
        # each key then takes its last H[key] rows (frames end at "now").
        h_pose = max(H.get("robot0_eef_pos", OBS_HORIZON_FALLBACK),
                     H.get("robot0_eef_rot_axis_angle", OBS_HORIZON_FALLBACK),
                     H.get("robot0_eef_rot_axis_angle_wrt_start", OBS_HORIZON_FALLBACK),
                     H.get("robot0_gripper_width", OBS_HORIZON_FALLBACK))
        pose_slice = pick_frames_by_time(self.pose_buffer, h_pose, self.dt)
        raw = np.stack([p[0] for p in pose_slice])              # (h_pose, 7)
        pose_mat = pose_to_mat(raw[:, :6].astype(np.float64))   # (h_pose, 4, 4)

        # abs mode: absolute pose -> [pos(3), rot6d ROWS(6)] via UMI's encoder
        p10 = mat_to_pose10d(pose_mat)                          # (h_pose, 9)
        # rotation relative to episode start (UMI's convert_pose_mat_rep, the
        # exact function UmiDataset used in training)
        rel_mat = convert_pose_mat_rep(
            pose_mat, base_pose_mat=self.episode_start_pose_mat,
            pose_rep='relative', backward=False)
        rot_wrt_start = mat_to_pose10d(rel_mat)[:, 3:]          # (h_pose, 6)

        if not getattr(self, '_obs_logged', False):
            self._obs_logged = True
            mat_last = pose_mat[-1]
            euler = R.from_matrix(mat_last[:3, :3]).as_euler("xyz", degrees=True)
            self.get_logger().info(
                f"OBS diagnostic (UMI) — pos={np.round(raw[-1, :3], 4)}  "
                f"theta_xyz={np.round(euler, 2)} (expected ≈ [{HOME_TX}, {HOME_TY}, {HOME_TZ}])  "
                f"width={raw[-1, 6]:.4f} m  obs spacing={self.dt}s  horizons={H}")

        def tail(arr, key):
            """Last H[key] rows of a stack whose frames end at 'now'."""
            return arr[-H.get(key, OBS_HORIZON_FALLBACK):]

        obs_dict = {
            "robot0_eef_pos": torch.from_numpy(
                tail(p10[:, :3], "robot0_eef_pos").astype(np.float32)).unsqueeze(0),
            "robot0_eef_rot_axis_angle": torch.from_numpy(
                tail(p10[:, 3:], "robot0_eef_rot_axis_angle").astype(np.float32)).unsqueeze(0),
            "robot0_eef_rot_axis_angle_wrt_start": torch.from_numpy(
                tail(rot_wrt_start, "robot0_eef_rot_axis_angle_wrt_start").astype(np.float32)).unsqueeze(0),
            "robot0_gripper_width": torch.from_numpy(
                tail(raw[:, 6:7], "robot0_gripper_width").astype(np.float32)).unsqueeze(0),
            "pose_timestamps": np.array([p[1] for p in pose_slice]),
        }

        for cam_key in CAMERA_KEYS:
            h_img = H.get(cam_key, OBS_HORIZON_FALLBACK)
            cam_slice = pick_frames_by_time(self.cam_buffers[cam_key], h_img, self.dt)
            obs_dict[cam_key] = torch.from_numpy(
                np.stack([c[0] for c in cam_slice])
            ).unsqueeze(0)                                      # (1, h_img, 3, 224, 224)
            obs_dict[f"{cam_key}_timestamps"] = np.array([c[1] for c in cam_slice])

        h_pz = H.get("piezense0_pressures", OBS_HORIZON_FALLBACK)
        if self.piezense_buffer:
            pz_slice = pick_frames_by_time(self.piezense_buffer, h_pz, self.dt)
        else:
            pz_slice = [(np.zeros(PIEZENSE_INPUT_CHANNELS, dtype=np.float32), 0.0)] * h_pz

        obs_dict["piezense0_pressures"] = torch.from_numpy(
            np.stack([p[0] for p in pz_slice])
        ).unsqueeze(0)                                          # (1, h_pz, 2)

        self.shared_obs["obs"] = obs_dict

    # ── Velocity control (identical to inference.py) ──────────────────────────

    def control_callback(self):
        if self.is_resetting:
            self._smoothed_vel[:] = 0.0
            return
        if self.paused:
            self._smoothed_vel[:] = 0.0
            self._send_zero_twist()
            return

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

            pos_err = target_xyz - current_xyz
            raw_vel = P_GAIN * pos_err
            speed = np.linalg.norm(raw_vel)
            if speed > MAX_LINEAR_SPEED:
                raw_vel *= MAX_LINEAR_SPEED / speed

            self._smoothed_vel = VEL_ALPHA * raw_vel + (1.0 - VEL_ALPHA) * self._smoothed_vel
            send_vel = self._smoothed_vel.copy()
            smooth_speed = np.linalg.norm(send_vel)
            if smooth_speed > MAX_LINEAR_SPEED:
                send_vel *= MAX_LINEAR_SPEED / smooth_speed

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
        """Accept a UMI 10D policy action [pos, rot6d ROWS, width_m] and update targets."""
        pos, euler_deg, grip = action10d_to_pos_euler_grip(action_10d)
        pos[0] = np.clip(pos[0], WS_X[0], WS_X[1])
        pos[1] = np.clip(pos[1], WS_Y[0], WS_Y[1])
        pos[2] = np.clip(pos[2], WS_Z[0], WS_Z[1])

        self.current_target_xyz   = pos
        self.current_target_euler = euler_deg
        self.current_gripper_cmd  = grip

        if not getattr(self, '_action_logged', False):
            self._action_logged = True
            self.get_logger().info(
                f"First action target: xyz={np.round(pos, 4)}  "
                f"theta_xyz={np.round(euler_deg, 2)}  "
                f"width={action_10d[9]:.4f} m -> grip={grip:.3f}")

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
        self.episode_start_pose_mat = None   # re-anchor _wrt_start on next obs
        self.is_resetting = True
        threading.Thread(target=self._do_home_reset, daemon=True).start()

    def _do_home_reset(self):
        try:
            self._send_zero_twist()
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
                   dt=0.1, num_inference_steps=16,
                   latency_offset_s=0.0):
    """GPU process: load model and run inference, posting targets to the main process."""
    import sys as _sys
    _sys.stdout.reconfigure(line_buffering=True)

    latency_steps = max(0, round(latency_offset_s / dt))
    if latency_steps:
        print(f"Latency offset: {latency_offset_s*1000:.0f} ms = {latency_steps} steps at dt={dt}s", flush=True)

    policy = load_policy(model_path, num_inference_steps)
    model_obs_keys = load_obs_keys(model_path)
    print(f"Model expects obs keys: {model_obs_keys}", flush=True)

    while shared_obs.get("obs") is None:
        time.sleep(0.05)
        print("Waiting for first observation...", flush=True)

    prev_timestamps = {}
    obs_now = shared_obs["obs"]
    if "pose_timestamps" in obs_now:
        prev_timestamps["pose"] = obs_now["pose_timestamps"][-1]
    for cam_key in CAMERA_KEYS:
        ts_key = f"{cam_key}_timestamps"
        if ts_key in obs_now:
            prev_timestamps[cam_key] = obs_now[ts_key][-1]

    print("Inference loop started (UMI pipeline).", flush=True)
    _actions_logged = False

    while True:
        if shared_obs.get("paused", True):
            time.sleep(0.05)
            continue

        loop_start = time.time()

        # Wait until the LATEST obs frame is fresh (obs frames are dt apart by
        # construction, so we compare the newest timestamp, not the window min).
        wait_start = time.time()
        while True:
            obs_now = shared_obs["obs"]
            all_new = True
            if "pose_timestamps" in obs_now:
                if obs_now["pose_timestamps"][-1] <= prev_timestamps.get("pose", -1):
                    all_new = False
            for cam_key in CAMERA_KEYS:
                ts_key = f"{cam_key}_timestamps"
                if ts_key in obs_now:
                    if obs_now[ts_key][-1] <= prev_timestamps.get(cam_key, -1):
                        all_new = False
            if all_new:
                break
            elapsed = time.time() - wait_start
            if elapsed > 1.0 and int(elapsed) != int(elapsed - 0.001):
                print(f"Waiting for new sensor data ({elapsed:.1f}s)...", flush=True)
            time.sleep(0.001)

        wait_time = time.time() - wait_start

        if "pose_timestamps" in obs_now:
            prev_timestamps["pose"] = obs_now["pose_timestamps"][-1]
        for cam_key in CAMERA_KEYS:
            ts_key = f"{cam_key}_timestamps"
            if ts_key in obs_now:
                prev_timestamps[cam_key] = obs_now[ts_key][-1]

        model_obs = {}
        for k in model_obs_keys:
            if k in obs_now:
                v = obs_now[k]
                model_obs[k] = v.to(device) if isinstance(v, torch.Tensor) else v

        t_infer = time.time()
        with torch.no_grad():
            actions = policy.predict_action(model_obs)["action"][0].detach().cpu().numpy()
        infer_time = time.time() - t_infer

        if not _actions_logged:
            _actions_logged = True
            print(f"predict_action output: {actions.shape[0]} steps × {actions.shape[1]} dims", flush=True)

        t_start = time.monotonic()
        action_queue.put(("CLEAR_PENDING", t_start))
        start_idx = min(latency_steps, len(actions) - 1)
        for i, act in enumerate(actions[start_idx: start_idx + n_action_steps]):
            ts = t_start + i * dt
            action_queue.put((act, ts))

        a0 = actions[start_idx]
        pos, euler, grip = action10d_to_pos_euler_grip(a0)
        print(f"Inference: {infer_time*1000:.0f}ms | act[0] xyz={np.round(pos,4)} "
              f"theta_xyz={np.round(euler,2)} width={a0[9]:.4f}m grip={grip:.3f}", flush=True)

        total_time = time.time() - loop_start
        print(f"  Wait: {wait_time*1000:.0f}ms | Total: {total_time*1000:.0f}ms | Actions: {n_action_steps}", flush=True)

        time.sleep(dt)


# ── Pygame control window ──────────────────────────────────────────────────────

def monitor_keys(policy_node: PolicyNode, shared_obs: dict):
    try:
        pygame.init()
        screen = pygame.display.set_mode((340, 210))
        pygame.display.set_caption("Kinova Policy Control (UMI)")
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
    parser = argparse.ArgumentParser(description="Diffusion Policy Inference (UMI pipeline) — Kinova Gen3")
    parser.add_argument("--model",           type=str, required=True,  help="Path to UMI-pipeline .ckpt")
    parser.add_argument("--dt",              type=float, default=0.1,  help="Action step period (s); must match "
                                                                            "training obs_down_sample_steps/30Hz (3/30 = 0.1)")
    parser.add_argument("--n-action-steps",  type=int,   default=8,    help="Actions executed per inference cycle")
    parser.add_argument("--diffusion-steps", type=int,   default=16,   help="DDIM inference steps")
    parser.add_argument("--latency-offset-s", type=float, default=0.0,
                        help="System latency to compensate (seconds)")
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
    print(f"Obs horizons:    read from checkpoint shape_meta (frames {args.dt}s apart)")
    print(f"Num Action steps:{args.n_action_steps}")
    print(f"Diffusion steps: {args.diffusion_steps}")
    print(f"Gripper:         width_m = {GRIPPER_MAX_WIDTH_M} * (1 - kortex_norm)")

    inf_proc = Process(
        target=inference_loop,
        args=(args.model, shared_obs, action_queue,
              args.n_action_steps, device, start_time,
              args.dt, args.diffusion_steps, args.latency_offset_s),
        daemon=True,
    )
    inf_proc.start()

    node = PolicyNode(shared_obs, start_time, args.model, args.dt)

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
