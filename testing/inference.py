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
import queue
import signal
import subprocess
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
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float32, Bool
from cv_bridge import CvBridge
from scipy.spatial.transform import Rotation as R
from piezense_interfaces.msg import PiezenseSystemArray

# Rollout recorder lives alongside this script (testing/). Ensure the dir is
# importable even after os.chdir(_UMI_ROOT) above.
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
from rollout_recorder import RolloutRecorder

from kortex_api.autogen.messages import Base_pb2

# Shared arm-control layer, used by data_collection/kinova_hand_controller.py
# too. Everything from the target pose down to the wire — session lifecycle,
# speed caps, workspace bounds, orientation clamps, fault latch, stall guard —
# lives there so teleop and rollout cannot drift apart again.
_DATA_COLLECTION_DIR = os.path.abspath(
    os.path.join(_THIS_DIR, "..", "data_collection"))
if _DATA_COLLECTION_DIR not in sys.path:
    sys.path.insert(0, _DATA_COLLECTION_DIR)
from kinova_arm import (ArmLimits, KinovaArm, HOME_JOINTS_DEG,
                        TWIST_WATCHDOG_MS)

# ── Constants ──────────────────────────────────────────────────────────────────

POSE_TOPIC      = "robot_obs/pose"
GRIPPER_TOPIC   = "robot_obs/gripper"
DJI_WRIST_TOPIC = "/dji_wrist/dji_wrist/color/image_raw"
PIEZENSE_TOPIC  = "piezense/data"

PIEZENSE_SYSTEM_ID      = 0
PIEZENSE_INPUT_CHANNELS = 2
PIEZENSE_INPUT_CHAN_IDS = [2, 3]

KINOVA_IP = "192.168.1.10"

# Speed caps, workspace bounds, orientation clamps and stall thresholds all come
# from kinova_arm.ArmLimits — the SAME defaults teleop runs with. They used to be
# duplicated here and had silently drifted (0.10 m/s vs 0.50, workspace x capped
# at 0.45 while collection had moved to 0.70, no soft boundary, no stall guard).
ARM_LIMITS = ArmLimits()

# Home orientation in Kortex Euler XYZ (degrees). The home POSITION is no longer
# defined here: reset drives to kinova_arm.HOME_JOINTS_DEG in joint space, the
# same posture data collection returns to, so rollouts start from exactly the
# configuration the demos started from.
HOME_TX, HOME_TY, HOME_TZ = -180.0, 0.0, 90.0
HOME_ROT = ARM_LIMITS.home_rotation(HOME_TX, HOME_TY, HOME_TZ)

IMG_SIZE     = 224
# Per-key obs horizons are read from the checkpoint's shape_meta at startup
# (img and low_dim horizons may differ, e.g. camera 2 / low_dim 8) — see
# load_obs_meta(). OBS_HORIZON_FALLBACK is only used if a key lacks one.
OBS_HORIZON_FALLBACK = 2

# MUST match convert_data.py (Robotiq 2F-85 stroke).
GRIPPER_MAX_WIDTH_M = 0.085

# Home reset: release, lift clear, THEN travel home. Going straight home while
# still holding drags the object across the table; opening at home (what data
# collection does) drops it wherever home happens to be. Lifting after the
# release also keeps the fingers from catching the object on the way out.
RESET_LIFT_M        = 0.03    # "a couple of cm" above wherever the release happened
RESET_LIFT_SPEED    = 0.05    # m/s — deliberately slow, this runs next to the objects
RESET_LIFT_TIMEOUT_S = 4.0

# Gripper opens by RAMPING the position setpoint rather than jumping to 0.
# GRIPPER_POSITION mode carries no speed field, so stepping the setpoint is the
# only way to slow the fingers down.
RESET_GRIPPER_OPEN_S     = 2.0    # total open time
RESET_GRIPPER_OPEN_STEPS = 8

# Speed cap for the home move. JOINT_CONSTRAINT_SPEED is rejected by this
# firmware (ACTION_ABORT/METHOD_FAILED, verified 2026-07-27 — see the note in
# kinova_arm.reach_home_joints), but JOINT_CONSTRAINT_DURATION is a separate
# constraint type: it sets how long the whole trajectory should take, so a
# LARGER number is SLOWER. Unconstrained, the firmware picks its own profile,
# which is what made inference's reset dash home.
RESET_HOME_DURATION_S = 5.0

CAMERA_KEYS = ["camera0_rgb"]
CAMERA_TOPICS = {"camera0_rgb": DJI_WRIST_TOPIC}

# Rollout recording (opt-in via --record); episode_N.hdf5 under testing/.
ROLLOUT_DIR_DEFAULT = os.path.join(_THIS_DIR, "rollout_data")

# Live monitor (wrist cam + force): own process, own window (L toggles it).
_LIVE_VIEWER = os.path.join(_THIS_DIR, "live_viewer.py")

# Set in main() from the checkpoint's horizon (n_action_steps * dt). Actions
# older than this are dropped by the executor rather than fired late.
STALE_ACTION_S = 0.8


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


# Rate the demos were recorded at. This is the ONE piece of timing the training
# config does not carry: task.dataset_frequeny is deliberately 0 (it zeroes the
# latency_steps terms, i.e. latency compensation off), so there is no usable
# source rate in the checkpoint. The real value lives in the HDF5 attrs as
# collection_rate_hz and is fixed by hdf5_data_collector.py.
SOURCE_RATE_HZ = 30.0


def load_run_config(model_path: str) -> dict:
    """Derive the run's timing/sampling settings from the checkpoint itself.

    dt, n_action_steps and diffusion_steps used to be CLI flags whose defaults
    happened to match the training config. Nothing enforced that: retrain with
    obs_down_sample_steps=1 and the correct dt becomes 0.033, but the flag still
    defaulted to 0.1 and inference silently fed the policy observations at the
    wrong spacing. No error, just a quietly wrong rollout. Reading them from the
    checkpoint makes that class of mismatch impossible.

      dt              = obs_down_sample_steps / source_rate
      n_action_steps  = cfg.n_action_steps
      diffusion_steps = cfg.policy.num_inference_steps
    """
    payload = torch.load(pathlib.Path(model_path).open("rb"),
                         pickle_module=dill, map_location="cpu")
    cfg = payload["cfg"]

    # down_sample_steps is carried per-key inside shape_meta; every key shares
    # the same value (task.obs_down_sample_steps), so take the first that has it.
    down_sample = None
    for spec in cfg.policy.shape_meta["obs"].values():
        if "down_sample_steps" in spec:
            down_sample = int(spec["down_sample_steps"])
            break
    if down_sample is None:
        down_sample = int(cfg.task.get("obs_down_sample_steps", 3))

    rate = float(cfg.task.get("dataset_frequeny", 0) or 0)
    if rate <= 0:
        rate = SOURCE_RATE_HZ

    return dict(
        dt=down_sample / rate,
        n_action_steps=int(cfg.n_action_steps),
        diffusion_steps=int(cfg.policy.num_inference_steps),
        down_sample_steps=down_sample,
        source_rate_hz=rate,
    )


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

    def __init__(self, shared_obs: dict, start_time: float, model_path: str, dt: float,
                 record: bool = False, record_dir: str = None, pred_queue=None,
                 enable_piezense: bool = True):
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

        self._node_start_time = time.monotonic()

        # Camera heartbeat on its OWN subscription, NOT inside
        # synced_obs_callback. That callback only fires when the
        # ApproximateTimeSynchronizer matches pose + gripper + image, so
        # stamping it there measured the synced stream: when the Kortex session
        # died on 2026-07-31 the pose/gripper topics stopped, the sync stopped,
        # and the camera dot went red while the camera was streaming fine. A dot
        # that blames the wrong subsystem is worse than no dot.
        self._cam_last_seen = None
        self.create_subscription(
            Image, DJI_WRIST_TOPIC, self._cam_heartbeat_cb, sensor_qos)

        # Resting baseline, used until the first real reading lands. The policy
        # and the recorder both read it, so they agree: before 2026-07-31 the
        # recorder logged this constant while update_observation fell back to
        # ZEROS, so a rollout with the driver down fed the policy values it had
        # never seen while the episode file showed a plausible-looking flat line.
        self._latest_piezense = np.array([111337.0, 110375.0], dtype=np.float32)
        self.piezense_buffer  = []
        self._enable_piezense    = enable_piezense
        self._piezense_last_seen = None
        self._piezense_warned    = False
        if self._enable_piezense:
            self.create_subscription(
                PiezenseSystemArray, PIEZENSE_TOPIC, self._piezense_cb, 10
            )
            self.create_timer(2.0, self._check_piezense_health)
        else:
            self.get_logger().warn(
                "Piezense DISABLED (enable_piezense=false) — piezense0_pressures is a "
                "trained obs key, so the policy will run on the constant baseline.")

        # ── Rollout recording (opt-in) ───────────────────────────────────────
        # joint_states is recorded for parity with collected demos; not used by
        # the policy, so it's a latest-value side channel like the collector's.
        self._latest_joint_states = np.zeros(7, dtype=np.float32)
        self.create_subscription(JointState, 'robot_obs/joint_states',
                                 self._joint_states_cb, sensor_qos)
        self._pred_queue = pred_queue
        self._recorder = RolloutRecorder(record_dir or ROLLOUT_DIR_DEFAULT) if record else None
        if self._recorder is not None:
            self.get_logger().info(
                f"Rollout recording ON → {record_dir or ROLLOUT_DIR_DEFAULT}  "
                f"(S start, D save, R/Q discard)")

        self.create_timer(1.0 / 30.0, self.update_observation)
        self.create_timer(1.0 / 30.0, self.control_callback)

        # NO automatic reset on startup. The arm stays exactly where it is (and
        # the gripper keeps its current opening) so a rollout can be started
        # from any configuration you have set up by hand. episode_start_pose_mat
        # is captured from the first observation, so 'wrt_start' anchors to
        # wherever you actually begin. Press R for a deliberate home reset.
        self.get_logger().info(
            "PolicyNode (UMI) ready — arm left AS-IS (no startup reset). "
            "Position the arm/gripper, then press S to start. R homes it, D pauses.")

    # ── Kinova setup ──────────────────────────────────────────────────────────

    def _connect_kinova(self):
        """Open the shared arm-control layer (session + safety live in there)."""
        self.arm = KinovaArm(
            robot_ip=KINOVA_IP,
            limits=ARM_LIMITS,
            log=self.get_logger(),
            on_stall=self._resync_after_stall,
            on_fault=self._halt_on_fault,
            recovery_hint='restart the inference stack',
        )
        self.arm.connect()
        # Kept as attributes because obs callbacks and the recorder read them.
        self._base = self.arm.base
        self._base_cyclic = self.arm.base_cyclic

    def _setup_servoing(self):
        self.arm.setup_servoing()

    def _send_zero_twist(self):
        self.arm.send_zero_twist()

    def _halt_on_fault(self):
        """Latching fault: stop the policy and drop every pending target.

        The action queue is the rollout's equivalent of teleop's clutch offset —
        if it keeps draining while the arm is stopped, the arm dashes through
        the backlog the moment it frees up.
        """
        self.paused = True
        self.shared_obs["paused"] = True
        self.current_target_xyz = None
        self.current_target_euler = None

    def _resync_after_stall(self, reason: str):
        """Drop queued targets and re-anchor on where the arm actually is.

        Teleop re-captures its clutch reference here; a rollout has no clutch,
        so the equivalent is to discard the stale target and let the next policy
        action be issued relative to the arm's real pose.
        """
        self.current_target_xyz = None
        self.current_target_euler = None
        self.arm.reset_velocity_state()
        self.get_logger().warn(f"Rollout target re-anchored — {reason}")

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

        # ── Rollout recording ────────────────────────────────────────────────
        # Record one frame per synced tick while an episode is active. Gated on a
        # live action target so the few frames between S and the first executed
        # action are skipped, keeping obs/action/image buffers equal length.
        rec = self._recorder
        if (rec is not None and rec.is_recording and not self.paused
                and not self.is_resetting and self.current_target_xyz is not None
                and self.current_target_euler is not None):
            obs_pose7 = np.array([
                pose_msg.pose.position.x, pose_msg.pose.position.y, pose_msg.pose.position.z,
                pose_msg.pose.orientation.x, pose_msg.pose.orientation.y,
                pose_msg.pose.orientation.z, pose_msg.pose.orientation.w,
            ], dtype=np.float32)
            q = R.from_euler("xyz", self.current_target_euler, degrees=True).as_quat()  # xyzw
            action_pose7 = np.concatenate(
                [np.asarray(self.current_target_xyz, dtype=np.float32), q.astype(np.float32)])
            # Re-decode the wrist image to uint8 CHW RGB (the loop above made a
            # float32/255 copy for the policy; the collector stores uint8).
            img_u8 = self._bridge.imgmsg_to_cv2(wrist_msg, desired_encoding="rgb8")
            if img_u8.shape[0] != IMG_SIZE or img_u8.shape[1] != IMG_SIZE:
                img_u8 = cv2.resize(img_u8, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
            img_u8 = np.ascontiguousarray(img_u8.transpose(2, 0, 1))
            rec.append(obs_pose7, float(self.gripper_state), action_pose7,
                       float(self.current_gripper_cmd), self._latest_joint_states.copy(),
                       self._latest_piezense.copy(), img_u8, now)

    def _joint_states_cb(self, msg: JointState):
        angles = list(msg.position)[:7]
        if len(angles) == 7:
            self._latest_joint_states = np.array(angles, dtype=np.float32)

    def _enable_dji_camera(self):
        msg = Bool()
        msg.data = True
        self._dji_enable_pub.publish(msg)

    def _cam_heartbeat_cb(self, msg: Image):
        """Liveness only — the synced callback does the decoding."""
        self._cam_last_seen = time.monotonic()

    def _piezense_cb(self, msg: PiezenseSystemArray):
        # Retract the warning explicitly. The banner is a snapshot ("nothing yet"),
        # not a verdict on the run — the driver's connect time varies from ~2 s to
        # ~20 s, and once it lands the policy reads live pressure from that moment.
        # Saying so out loud beats leaving a scary banner as the last word.
        if self._piezense_last_seen is None:
            self.get_logger().info(
                "Piezense: first reading received — pressure obs are LIVE"
                + (" (clears the warning above)" if self._piezense_warned else ""))
        elif self._piezense_warned:
            self.get_logger().info("Piezense: RECOVERED — pressure obs are LIVE again")
        self._piezense_last_seen = time.monotonic()
        self._piezense_warned    = False
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

    # ── Sensor health (same states/colours as the collector's pygame dots) ────
    #
    # The piezense driver takes a few seconds to come up, and nothing else in a
    # rollout fails visibly when it doesn't: the policy keeps running on the
    # baseline and the episode file keeps recording it. Data collection has had
    # these dots since the collector was written; inference needs them for the
    # same reason — you cannot tell from the console that the sensor is dead.

    # The piezense driver's own connect time is highly variable — measured at
    # 2.2 s on one run and 19.4 s on another ("[example] connecting..." retries
    # until the device answers). A 5 s grace period cried wolf on the slow-but-
    # healthy case, so the banner and the red dot both wait 30 s.
    PIEZENSE_STARTUP_GRACE_S = 30.0
    CAMERA_STARTUP_GRACE_S   = 10.0

    def _check_piezense_health(self):
        """Warn while pressure is missing — recoverable, not a verdict on the run."""
        if self._piezense_warned:
            return
        now    = time.monotonic()
        banner = '!' * 50
        if self._piezense_last_seen is None:
            if (now - self._node_start_time) > self.PIEZENSE_STARTUP_GRACE_S:
                self._piezense_warned = True
                self.get_logger().error(
                    f"\n{banner}\n  PIEZENSE: no data on {PIEZENSE_TOPIC} YET "
                    f"({now - self._node_start_time:.0f}s).\n"
                    f"  Is piezense_driver running? Until it connects the policy reads\n"
                    f"  the constant baseline instead of real pressure. This clears\n"
                    f"  itself if the driver comes up — watch the pygame dot go green,\n"
                    f"  and don't press S until it does.\n{banner}")
        elif (now - self._piezense_last_seen) > 3.0:
            # Dropping out MID-rollout is the dangerous case: the buffer is not
            # cleared, so the policy keeps reading stale pressure that looks
            # plausible. Nothing else surfaces this.
            self._piezense_warned = True
            self.get_logger().error(
                f"\n{banner}\n  PIEZENSE STOPPED "
                f"({now - self._piezense_last_seen:.1f}s since the last reading).\n"
                f"  The policy is now reading STALE pressure from the buffer.\n{banner}")

    def get_piezense_health(self) -> str:
        if not self._enable_piezense:
            return 'disabled'
        now = time.monotonic()
        if self._piezense_last_seen is None:
            return ('waiting'
                    if (now - self._node_start_time) < self.PIEZENSE_STARTUP_GRACE_S
                    else 'dead')
        return 'ready' if (now - self._piezense_last_seen) < 3.0 else 'dead'

    def get_camera_health(self) -> str:
        """DJI wrist camera — the policy's only image input."""
        now = time.monotonic()
        if self._cam_last_seen is None:
            return ('waiting'
                    if (now - self._node_start_time) < self.CAMERA_STARTUP_GRACE_S
                    else 'dead')
        return 'ready' if (now - self._cam_last_seen) < 6.0 else 'dead'

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
            # Baseline, NOT zeros. Training data is absolute pressure around
            # 111 kPa, so a zero-filled window is far outside anything the
            # policy saw; the baseline at least sits where an unloaded sensor
            # sits. It also matches what the recorder writes for these frames,
            # so a rollout episode reflects what the policy actually got.
            pz_slice = [(self._latest_piezense.copy(), 0.0)] * h_pz

        obs_dict["piezense0_pressures"] = torch.from_numpy(
            np.stack([p[0] for p in pz_slice])
        ).unsqueeze(0)                                          # (1, h_pz, 2)

        self.shared_obs["obs"] = obs_dict

    # ── Velocity control (identical to inference.py) ──────────────────────────

    def control_callback(self):
        # Drain predicted action horizons from the GPU process into the recorder.
        # append_prediction no-ops when not recording, so this also keeps the
        # queue from backing up between episodes.
        if self._recorder is not None and self._pred_queue is not None:
            while True:
                try:
                    horizon, t = self._pred_queue.get_nowait()
                except queue.Empty:
                    break
                self._recorder.append_prediction(horizon, t)

        # Fault is LATCHED — suppress every command and stay off the router.
        if self.arm.fault.latched:
            self.arm.fault.nag()
            return

        if self.is_resetting:
            self.arm.reset_velocity_state()
            return
        if self.paused:
            self.arm.reset_velocity_state()
            self._send_zero_twist()
            return

        target_xyz   = self.current_target_xyz
        target_euler = self.current_target_euler
        if target_xyz is None:
            self.arm.reset_velocity_state()
            self._send_zero_twist()
            return

        try:
            feedback = self.arm.refresh_feedback()
            current_xyz = self.arm.tcp_position(feedback)
            current_rot = self.arm.tcp_rotation(feedback)

            # Feed the stall guard every tick, in every mode — this estimate is
            # the only way to notice the arm has stopped without being told.
            self.arm.stall.update_velocity(current_xyz)

            # P-loop -> cap -> soft walls -> smoothing -> cap.
            send_vel = self.arm.linear_velocity(current_xyz, target_xyz)

            gap = float(np.linalg.norm(target_xyz - current_xyz))
            if self.arm.stall.check(send_vel, gap):
                self._send_zero_twist()
                return

            # Full rotation error, not yaw alone: the demos are orientation-rich
            # (orientation teleop is the collection default) and the policy
            # predicts a full 6D rotation, so driving only angular_z discarded
            # most of what it learned. Clamped against home on all three axes.
            target_rot = None
            if target_euler is not None:
                target_rot, _ = self.arm.clamp_orientation(
                    R.from_euler("xyz", target_euler, degrees=True), HOME_ROT)
            ang_send = self.arm.angular_velocity(current_rot, target_rot)

            self.arm.send_twist(send_vel, ang_send, base_frame=True)

        except Exception as e:
            kind = self.arm.fault.classify(e)
            if kind == "fault":
                return                      # latched; _halt_on_fault stopped us
            if kind == "transient":
                return                      # next tick retries
            self.get_logger().error(f"Control loop error: {e}")
            self._send_zero_twist()

    def _execute_action(self, action_10d: np.ndarray):
        """Accept a UMI 10D policy action [pos, rot6d ROWS, width_m] and update targets."""
        if self.arm.fault.latched:
            return
        pos, euler_deg, grip = action10d_to_pos_euler_grip(action_10d)
        # Same workspace box as teleop, including the hard margin.
        pos = self.arm.clip_to_workspace(pos)

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
            self.arm.send_gripper(grip)
        except Exception as e:
            if self.arm.fault.classify(e) == "other":
                self.get_logger().warn(f"Gripper command error: {e}")

    # ── Keyboard controls ─────────────────────────────────────────────────────

    def _reset_obs_buffers(self):
        """Drop all observation history so a rollout starts with a clean window.

        The buffers are rolling ~30 Hz windows and the policy reads roughly
        0.7 s of history from them (8 low-dim frames spaced dt apart). Without
        this, the window at the start of a rollout straddles the pause: the arm
        sitting still — possibly for a long time, somewhere else, with the scene
        changed — followed by fresh frames. That discontinuity is not something
        the policy ever saw in training, and it clears itself only once the
        window has refilled, which is why a wobble at the start of a rollout
        settles after roughly a second.

        No refill wait is needed. pick_frames_by_time pads a short buffer by
        repeating its earliest frame, which is exactly the start-of-episode
        padding the training sampler applied to the first frames of every demo —
        so acting immediately on a freshly-cleared buffer reproduces training's
        episode start, while waiting for a full window would give the policy a
        history no demo ever began with.
        """
        self.pose_buffer.clear()
        for k in CAMERA_KEYS:
            self.cam_buffers[k].clear()
        self.piezense_buffer.clear()

    def pause_policy(self):
        # Stops the arm ONLY. The recorder is deliberately untouched, so an
        # episode survives a pause and S resumes appending to it.
        self.get_logger().info("Paused.")
        self.paused = True
        self.shared_obs["paused"] = True
        self.current_target_xyz = None
        self.current_target_euler = None
        self.arm.reset_velocity_state()

    def resume_policy(self):
        # episode_start_pose_mat is NOT cleared here. It is the anchor for the
        # episode, re-captured only after a home reset, so pausing and resuming
        # mid-rollout continues to measure wrt_start from where this episode
        # actually began rather than from the pause point.
        pz = self.get_piezense_health()
        if pz != 'ready':
            self.get_logger().warn(
                f"Starting rollout with piezense {pz.upper()} — pressure obs will be "
                f"the constant baseline. Check the pygame dot before trusting this run.")
        if self.get_camera_health() != 'ready':
            self.get_logger().warn(
                f"Starting rollout with wrist camera {self.get_camera_health().upper()}.")
        self._reset_obs_buffers()
        self.get_logger().info("Resumed — observation history cleared.")
        self.paused = False
        self.shared_obs["paused"] = False
        if self._recorder is not None:
            self._recorder.start()   # begin a new rollout episode

    def save_recording(self):
        """Flush the in-progress rollout episode to disk (called on D)."""
        if self._recorder is None:
            return
        res = self._recorder.save()
        if res:
            path, n = res
            self.get_logger().info(f"Saved rollout {path}  ({n} frames)")
        else:
            self.get_logger().info("No rollout data to save")

    def discard_recording(self):
        """Drop the in-progress rollout episode (called on R and Q/close)."""
        if self._recorder is not None:
            self._recorder.discard()

    def cancel_recording(self):
        """C — end the episode WITHOUT saving. Same as D minus the write.

        Distinct from pause: this closes the episode, so the next S opens a new
        one. Pause leaves the episode open and S continues appending to it.
        """
        if self._recorder is None:
            self.get_logger().info("Cancelled (not recording).")
            return
        had = self._recorder.is_recording
        self._recorder.discard()
        self.get_logger().info("Episode CANCELLED — not saved." if had
                               else "Nothing recording to cancel.")

    def reset_to_home(self):
        if self.is_resetting:
            return
        self.get_logger().info("Resetting to home...")
        self.paused = True
        self.shared_obs["paused"] = True
        self.current_target_xyz = None
        self.current_target_euler = None
        self.arm.reset_velocity_state()
        self.episode_start_pose_mat = None   # re-anchor _wrt_start on next obs
        self.is_resetting = True
        self.discard_recording()             # abandon any partial rollout
        threading.Thread(target=self._do_home_reset, daemon=True).start()

    def _open_gripper_slowly(self):
        """Ramp the gripper open instead of commanding 0.0 in one jump."""
        start = float(np.clip(self.current_gripper_cmd, 0.0, 1.0))
        if start <= 0.01:
            self.arm.send_gripper(0.0)          # already open; make it explicit
            time.sleep(0.2)
            return
        step_dt = RESET_GRIPPER_OPEN_S / RESET_GRIPPER_OPEN_STEPS
        for i in range(1, RESET_GRIPPER_OPEN_STEPS + 1):
            self.arm.send_gripper(start * (1.0 - i / RESET_GRIPPER_OPEN_STEPS))
            time.sleep(step_dt)

    def _reach_home_limited(self) -> bool:
        """Home move with a DURATION constraint so it does not dash.

        inference-only: kinova_arm.reach_home_joints() sends the action with no
        constraint at all, and that is data collection's path too — this cap
        must not change it. If the firmware rejects the constraint (as it does
        JOINT_CONSTRAINT_SPEED) this falls straight back to the shared
        unconstrained reach, so R can never be broken by the attempt.
        """
        try:
            action = Base_pb2.Action()
            action.name = 'Home'
            action.application_data = ''
            for i, ang in enumerate(HOME_JOINTS_DEG):
                ja = action.reach_joint_angles.joint_angles.joint_angles.add()
                ja.joint_identifier = i
                ja.value = ang
            action.reach_joint_angles.constraint.type = Base_pb2.JOINT_CONSTRAINT_DURATION
            action.reach_joint_angles.constraint.value = RESET_HOME_DURATION_S
            self.arm.base.ExecuteAction(action)
        except Exception as e:
            self.get_logger().warn(
                f"Duration-constrained home rejected by the firmware ({e}) — "
                f"falling back to the unconstrained move.")
            return self.arm.reach_home_joints()

        # Wrap-aware arrival poll: joints 1/5 sit at ~359.6 deg, on the 0/360 seam.
        def joint_err_deg(fb):
            return max(min(d, 360.0 - d) for d in
                       (abs(fb.actuators[i].position - tgt) % 360.0
                        for i, tgt in enumerate(HOME_JOINTS_DEG)))

        deadline = time.monotonic() + max(30.0, RESET_HOME_DURATION_S * 3)
        while time.monotonic() < deadline:
            try:
                if joint_err_deg(self.arm.refresh_feedback()) < 2.0:
                    return True
            except Exception:
                pass
            time.sleep(0.1)
        return False

    def _joint_and_tcp(self):
        """(joint angles deg, TCP xyz) for reset instrumentation, or (None, None)."""
        try:
            fb = self.arm.refresh_feedback()
            return ([fb.actuators[i].position for i in range(7)],
                    self.arm.tcp_position(fb))
        except Exception:
            return None, None

    def _lift_clear(self):
        """Raise the TCP RESET_LIFT_M straight up, clamped to the workspace ceiling.

        Closed-loop on measured z rather than open-loop timing: the twist
        watchdog and the arm's own acceleration profile make a fixed sleep
        land somewhere different every time. Runs BEFORE twists_suppressed is
        set, since it is itself a twist.
        """
        try:
            z0     = float(self.arm.tcp_position(self.arm.refresh_feedback())[2])
            ceiling = self.arm.limits.z[1] - self.arm.limits.hard_margin_m
            z_target = min(z0 + RESET_LIFT_M, ceiling)
            if z_target - z0 < 0.002:
                self.get_logger().info(
                    f"Lift skipped — already at the workspace ceiling (z={z0:.3f} m)")
                return

            deadline = time.monotonic() + RESET_LIFT_TIMEOUT_S
            while time.monotonic() < deadline:
                z = float(self.arm.tcp_position(self.arm.refresh_feedback())[2])
                if z >= z_target:
                    break
                self.arm.send_twist(np.array([0.0, 0.0, RESET_LIFT_SPEED]),
                                    np.zeros(3), base_frame=True)
                time.sleep(1.0 / 30.0)
            else:
                self.get_logger().warn("Lift timed out — continuing to home anyway")

            self._send_zero_twist()
            time.sleep(0.3)          # let the watchdog settle before the reach action
            z_end = float(self.arm.tcp_position(self.arm.refresh_feedback())[2])
            self.get_logger().info(f"Lifted clear: z {z0:.3f} -> {z_end:.3f} m")
        except Exception as e:
            # A failed lift must not block the reset — going home matters more.
            if self.arm.fault.classify(e) != "fault":
                self.get_logger().warn(f"Lift failed ({e}) — continuing to home")

    def _do_home_reset(self):
        """Return to the SAME joint-space home data collection uses.

        This used to reach a Cartesian pose (0.35, 0.0, 0.12) that no longer
        matched collection's far-forward home at all, so rollouts started from a
        different posture than any demo. Cartesian reach is also IK-ambiguous on
        a 7-DOF arm — the elbow could settle differently each reset. Joint space
        pins the exact configuration every time.
        """
        try:
            self._send_zero_twist()
            time.sleep(1.0)

            # 1. Release FIRST, while still parked over wherever the rollout
            #    ended. Opening at home instead (data collection's order) drops
            #    whatever is held onto whatever is under home.
            self._open_gripper_slowly()      # ramps to 0 = fully open

            # 2. Lift clear of the object before any lateral travel.
            self._lift_clear()

            # 3. Only now go home. Own the router while the reach action runs:
            #    a twist here would cancel the in-flight action and race its RPCs.
            self.arm.twists_suppressed = True

            # Instrumented so the reach can be COMPARED against data collection's
            # rather than argued about. The ExecuteAction payload, joint targets,
            # tolerance (2 deg), timeout (30 s) and poll interval are identical in
            # both stacks, so any real difference in speed or path has to come
            # from somewhere else — most likely the pose the reach STARTS from,
            # which these numbers capture. Run R from a comparable pose in each
            # stack and compare the printed duration and start error.
            j0, xyz0 = self._joint_and_tcp()
            t0 = time.monotonic()
            reached  = self._reach_home_limited()
            dt_reach = time.monotonic() - t0
            j1, xyz1 = self._joint_and_tcp()

            if j0 is not None:
                err0 = max(min(d, 360.0 - d) for d in
                           (abs(a - b) % 360.0 for a, b in zip(j0, HOME_JOINTS_DEG)))
                self.get_logger().info(
                    f"HOME REACH: {dt_reach:.2f}s | start joint err {err0:.1f} deg | "
                    f"TCP {np.round(xyz0, 3)} -> {np.round(xyz1, 3)}")
                self.get_logger().info(
                    f"HOME REACH start joints: {[round(a, 1) for a in j0]}")
            else:
                self.get_logger().info(f"HOME REACH: {dt_reach:.2f}s (feedback unavailable)")

            if reached:
                self.get_logger().info("Reset complete. Press S to start.")
            else:
                # The shared reach_home_joints() returns False silently, so the
                # warning has to live here.
                self.get_logger().warn(
                    "Home reset did NOT reach home within the timeout — arm is not "
                    "at the demo start posture. Fix before pressing S.")
                try:
                    self.arm.base.StopAction()   # don't leave the action in flight
                except Exception:
                    pass
        except Exception as e:
            if self.arm.fault.classify(e) != "fault":
                self.get_logger().error(f"Reset error: {e}")
        finally:
            self.arm.twists_suppressed = False
            try:
                self._setup_servoing()
            except Exception:
                pass
            self.arm.reset_velocity_state()
            self.is_resetting = False

    def cleanup(self):
        # Step-by-step teardown lives in KinovaArm.disconnect(): a faulted arm
        # makes the zero twist raise, and sharing one try block meant the
        # session was then never closed — which latches NETWORK_ERROR into the
        # next run.
        self.arm.disconnect()


# ── Inference process ──────────────────────────────────────────────────────────

def inference_loop(model_path, shared_obs, action_queue,
                   n_action_steps=8, device="cuda", start_time=0,
                   dt=0.1, num_inference_steps=16,
                   latency_offset_s=0.0, pred_queue=None):
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
        # Publish the FULL predicted horizon (raw model output) for recording.
        if pred_queue is not None:
            pred_queue.put((actions.copy().astype(np.float32), t_start - start_time))
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
        PANEL_W, PANEL_H = 340, 275
        screen = pygame.display.set_mode((PANEL_W, PANEL_H))
        pygame.display.set_caption("Kinova Policy Control (UMI)")

        live_proc = None       # live_viewer.py subprocess, toggled by F
        clock = pygame.time.Clock()
        font       = pygame.font.SysFont("monospace", 18)
        font_small = pygame.font.SysFont("monospace", 14)

        COLOR_PAUSED  = (50, 50, 60)
        COLOR_RUNNING = (20, 60, 30)

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    policy_node.pause_policy()
                    policy_node.discard_recording()
                    if live_proc is not None and live_proc.poll() is None:
                        live_proc.terminate()
                    os._exit(0)
                if event.type == pygame.KEYDOWN:
                    # D / C both END the episode (next S opens a new one);
                    # P only stops the arm — RolloutRecorder.start() is
                    # idempotent, so resuming after P continues the SAME episode.
                    if event.key == pygame.K_d:
                        policy_node.pause_policy()
                        policy_node.save_recording()
                    elif event.key == pygame.K_c:
                        policy_node.pause_policy()
                        policy_node.cancel_recording()
                    elif event.key == pygame.K_p:
                        policy_node.pause_policy()
                    elif event.key == pygame.K_s:
                        policy_node.resume_policy()
                    elif event.key == pygame.K_r:
                        policy_node.reset_to_home()
                    elif event.key == pygame.K_q:
                        policy_node.pause_policy()
                        policy_node.discard_recording()
                        if live_proc is not None and live_proc.poll() is None:
                            live_proc.terminate()
                        time.sleep(0.2)
                        os._exit(0)
                    elif event.key == pygame.K_l:
                        # Own process, own window — see live_viewer.py's header
                        # for why it is not drawn in here.
                        if live_proc is not None and live_proc.poll() is None:
                            live_proc.terminate()
                            live_proc = None
                        else:
                            live_proc = subprocess.Popen(
                                [sys.executable, _LIVE_VIEWER])

            paused = shared_obs.get("paused", True)
            screen.fill(COLOR_PAUSED if paused else COLOR_RUNNING)

            status_text  = "PAUSED" if paused else "RUNNING"
            status_color = (255, 200, 50) if paused else (50, 255, 80)
            screen.blit(font.render(status_text, True, status_color), (120, 20))

            keys_info = [
                ("S", "Start / Resume"),
                ("P", "Pause (keep episode)"),
                ("D", "Done  -> SAVE episode"),
                ("C", "Cancel -> discard"),
                ("R", "Reset to home"),
                ("L", "Live view window"),
                ("Q", "Quit"),
            ]
            for i, (key, desc) in enumerate(keys_info):
                line = font_small.render(f"  {key}  -  {desc}", True, (200, 200, 200))
                screen.blit(line, (30, 62 + i * 22))

            # ── Sensor health dots ───────────────────────────────────────────
            # Same states and colours as the collector's GUI, so a green
            # piezense dot means the same thing on both sides of the pipeline.
            # Check it BEFORE pressing S: the driver takes a few seconds to
            # come up and nothing else in a rollout shows that it hasn't.
            health_colors = {
                'ready':    ( 80, 200,  80),
                'waiting':  (255, 200,  50),
                'dead':     (220,  50,  50),
                'disabled': ( 80,  80,  80),
            }
            health_items = [
                ('piezense', policy_node.get_piezense_health()),
                ('wrist_cam', policy_node.get_camera_health()),
            ]
            # Flow chips left-to-right on measured widths (as in the collector),
            # so a wider label can't overlap its neighbour.
            row_left, row_right, row_y = 20, PANEL_W - 20, 238
            chips = [(font_small.render(label, True, (200, 200, 200)), status)
                     for label, status in health_items]
            chips_w = sum(16 + surf.get_width() for surf, _ in chips)
            gap = max(6, (row_right - row_left - chips_w) // max(1, len(chips) - 1))
            cx = row_left
            for surf, status in chips:
                pygame.draw.circle(screen, health_colors[status], (cx + 6, row_y + 6), 6)
                screen.blit(surf, (cx + 16, row_y))
                cx += 16 + surf.get_width() + gap

            pygame.display.flip()
            clock.tick(10)
    except Exception as e:
        print(f"Pygame error: {e}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Diffusion Policy Inference (UMI pipeline) — Kinova Gen3")
    parser.add_argument("--model",           type=str, required=True,  help="Path to UMI-pipeline .ckpt")
    # dt / n-action-steps / diffusion-steps default to the CHECKPOINT (see
    # load_run_config). They used to be plain flags whose defaults merely
    # happened to match the training config, so retraining with a different
    # obs_down_sample_steps or n_action_steps silently desynced inference with
    # no error at all. Defaulting to None keeps that safety: you get the
    # trained value unless you say otherwise, and saying otherwise prints a
    # loud banner, because a timing mismatch is invisible in a rollout — the
    # arm just moves at the wrong speed and looks like a bad policy.
    timing = parser.add_mutually_exclusive_group()
    timing.add_argument("--dt",              type=float, default=None,
                        help="Override the action step period in SECONDS. "
                             "Default: obs_down_sample_steps / dataset_frequeny "
                             "from the checkpoint.")
    timing.add_argument("--hz",              type=float, default=None,
                        help="Same override expressed as a RATE (dt = 1/hz).")
    parser.add_argument("--n-action-steps",  type=int,   default=None,
                        help="Override actions executed per inference cycle")
    parser.add_argument("--diffusion-steps", type=int,   default=None,
                        help="Override DDIM inference steps (latency vs quality)")
    parser.add_argument("--latency-offset-s", type=float, default=0.0,
                        help="System latency to compensate (seconds)")
    parser.add_argument("--no-pygame",       action="store_true",      help="Disable pygame window")
    parser.add_argument("--no-piezense",     action="store_true",
                        help="Skip the piezense subscription (pressure obs stay at baseline)")
    parser.add_argument("--record",          action="store_true",
                        help="Record rollouts to episode_N.hdf5 (S start, D save, R/Q discard)")
    parser.add_argument("--record-dir",      type=str, default=ROLLOUT_DIR_DEFAULT,
                        help=f"Directory for rollout episodes (default: {ROLLOUT_DIR_DEFAULT})")
    args = parser.parse_args()

    # Timing/sampling come from the checkpoint that is about to be run, so by
    # default they cannot disagree with how the policy was trained.
    run_cfg = load_run_config(args.model)
    ckpt_dt = run_cfg["dt"]

    overrides = []
    if args.hz is not None:
        args.dt = 1.0 / args.hz
        overrides.append(("dt", f"{ckpt_dt:.4f}s ({1/ckpt_dt:.1f} Hz)",
                          f"{args.dt:.4f}s ({args.hz:.1f} Hz)  [--hz]"))
    elif args.dt is not None:
        overrides.append(("dt", f"{ckpt_dt:.4f}s ({1/ckpt_dt:.1f} Hz)",
                          f"{args.dt:.4f}s ({1/args.dt:.1f} Hz)  [--dt]"))
    else:
        args.dt = ckpt_dt

    if args.n_action_steps is None:
        args.n_action_steps = run_cfg["n_action_steps"]
    else:
        overrides.append(("n_action_steps", str(run_cfg["n_action_steps"]),
                          f"{args.n_action_steps}  [--n-action-steps]"))

    if args.diffusion_steps is None:
        args.diffusion_steps = run_cfg["diffusion_steps"]
    else:
        overrides.append(("diffusion_steps", str(run_cfg["diffusion_steps"]),
                          f"{args.diffusion_steps}  [--diffusion-steps]"))

    if overrides:
        bar = "!" * 68
        print(f"\n{bar}\n  OVERRIDING WHAT THIS CHECKPOINT WAS TRAINED WITH")
        for name, was, now in overrides:
            print(f"    {name:16s} ckpt {was:24s} ->  {now}")
        print("  The policy was trained at the ckpt values. A timing mismatch does")
        print("  not error — the arm simply moves at the wrong speed.")
        print(f"{bar}\n")

    # An action older than one full predicted horizon is obsolete — a newer
    # prediction has superseded it. Backstop against any source of backlog, not
    # just pause/resume. Generous on purpose: dropping actions that are merely
    # LATE would fight the executor during normal operation.
    global STALE_ACTION_S
    STALE_ACTION_S = args.n_action_steps * args.dt

    # launch_inference.py runs this under ros2 launch, which sends SIGINT and
    # then escalates to SIGTERM. Python's default SIGTERM disposition kills the
    # process WITHOUT running the finally block, leaving the spawned GPU child
    # and the Manager orphaned — that is how 67 strays accumulated. Route it
    # through the same orderly path as Ctrl-C.
    def _term(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, _term)

    multiprocessing.set_start_method("spawn", force=True)

    manager   = Manager()
    shared_obs = manager.dict(obs=None, paused=True)
    action_queue = Queue()
    pred_queue = Queue() if args.record else None
    start_time = time.monotonic()

    rclpy.init()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Model:           {args.model}")
    print(f"dt:              {args.dt:.4f}s  ({1/args.dt:.1f} Hz)   [from ckpt: "
          f"down_sample {run_cfg['down_sample_steps']} / {run_cfg['source_rate_hz']:.1f} Hz]")
    print(f"Obs horizons:    read from checkpoint shape_meta (frames {args.dt:.4f}s apart)")
    print(f"Num Action steps:{args.n_action_steps}   [from ckpt]")
    print(f"Diffusion steps: {args.diffusion_steps}   [from ckpt]")
    print(f"Max linear speed:{ARM_LIMITS.max_linear_speed_mps} m/s   "
          f"(shared with teleop via kinova_arm.ArmLimits)")
    print(f"Workspace:       x{ARM_LIMITS.x} y{ARM_LIMITS.y} z{ARM_LIMITS.z}")
    print(f"Gripper:         width_m = {GRIPPER_MAX_WIDTH_M} * (1 - kortex_norm)")

    inf_proc = Process(
        target=inference_loop,
        args=(args.model, shared_obs, action_queue,
              args.n_action_steps, device, start_time,
              args.dt, args.diffusion_steps, args.latency_offset_s, pred_queue),
        daemon=True,
    )
    inf_proc.start()

    node = PolicyNode(shared_obs, start_time, args.model, args.dt,
                      record=args.record, record_dir=args.record_dir, pred_queue=pred_queue,
                      enable_piezense=not args.no_piezense)

    def action_executor():
        pending = []
        # Nothing may execute until the FIRST CLEAR_PENDING of the current run
        # arrives. CLEAR_PENDING is emitted by the inference process immediately
        # before each fresh batch, so it is proof that what follows was computed
        # from observations taken AFTER the resume.
        #
        # Clearing `pending` while paused (below) is necessary but not
        # sufficient on its own: the inference loop is a separate process that
        # only tests shared_obs["paused"] at the top of its cycle, so a batch it
        # was already computing can land in action_queue right around the resume
        # boundary and be drained before the new CLEAR_PENDING. Gating on
        # CLEAR_PENDING makes "no pre-pause action can ever run" a structural
        # guarantee instead of a race the executor usually wins.
        armed = False
        while True:
            while not action_queue.empty():
                item = action_queue.get()
                if isinstance(item, tuple) and isinstance(item[0], str) and item[0] == "CLEAR_PENDING":
                    pending.clear()
                    armed = True
                else:
                    pending.append(item)
            now = time.monotonic()

            # While paused, DISCARD the backlog instead of holding it.
            #
            # Actions are stamped for a specific moment. The old code skipped
            # them when paused but re-queued them, so every action whose moment
            # passed during the pause stayed pending with a timestamp now in the
            # past. On resume the `now >= ts` test was instantly true for all of
            # them and the whole backlog fired in one 5 ms tick — the arm lurched
            # through stale targets from the previous run before the first fresh
            # prediction landed (~100-300 ms later, since the inference loop has
            # to see paused=False, wait for fresh sensor data, then denoise).
            # That is the fast out-and-back seen on every run after the first;
            # the first was clean only because `pending` starts empty, and a
            # relaunch "fixed" it for the same reason.
            if node.paused:
                pending.clear()
                armed = False        # re-arm only on the next run's CLEAR_PENDING
                time.sleep(0.005)
                continue

            if not armed:
                time.sleep(0.005)
                continue

            remaining = []
            for act, ts in pending:
                if now - ts > STALE_ACTION_S:
                    continue            # too old to be meaningful — drop it
                if now >= ts:
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
        # Every step in its own try/except: a failure early on must not skip the
        # ones after it (same reasoning as KinovaArm.disconnect).
        for label, step in (
            ("arm cleanup",   node.cleanup),
            ("destroy node",  node.destroy_node),
            ("rclpy shutdown", rclpy.shutdown),
            # terminate() only signals. Without the join() the parent can exit
            # while the spawned GPU child and the Manager are still alive; they
            # get reparented to systemd and linger forever. 67 such strays had
            # accumulated by 2026-07-29, the oldest 23 days old.
            ("stop inference proc", lambda: _stop_proc(inf_proc)),
            ("shutdown manager",    manager.shutdown),
        ):
            try:
                step()
            except Exception as e:
                print(f'Shutdown step "{label}" failed: {e}', flush=True)


def _stop_proc(proc, timeout=5.0):
    """Terminate a child and REAP it, escalating to kill if it ignores SIGTERM."""
    if proc is None or not proc.is_alive():
        return
    proc.terminate()
    proc.join(timeout)
    if proc.is_alive():
        proc.kill()
        proc.join(timeout)


if __name__ == "__main__":
    main()
