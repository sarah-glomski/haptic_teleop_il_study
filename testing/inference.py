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
                        TWIST_WATCHDOG_MS, home_pose_vec6)

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

CAMERA_KEYS = ["camera0_rgb"]
CAMERA_TOPICS = {"camera0_rgb": DJI_WRIST_TOPIC}

# Rollout recording (opt-in via --record); episode_N.hdf5 under testing/.
ROLLOUT_DIR_DEFAULT = os.path.join(_THIS_DIR, "rollout_data")

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
                 record: bool = False, record_dir: str = None, pred_queue=None):
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

        # Anchor for robot0_eef_rot_axis_angle_wrt_start: the FIXED home pose,
        # matching what convert_data.py writes as robot0_demo_start_pose.
        #
        # It is deliberately constant and never re-anchored. wrt_start therefore
        # means "how far the wrist is from home", which is the same quantity at
        # the same point in the task regardless of where the rollout was
        # started — that is what makes starting halfway through the task valid.
        # Anchoring to wherever the episode began would instead report identity
        # at the start of every rollout, telling the policy it is at the
        # beginning of the task when it is not.
        self.episode_start_pose_mat = pose_to_mat(
            home_pose_vec6()[:6].astype(np.float64))

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

        # No per-episode start-pose capture: episode_start_pose_mat is the FIXED
        # home pose, set once in __init__ (see the comment there).

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
            # (recorded with --orientation) and the policy predicts a full 6D
            # rotation, so driving only angular_z discarded most of what it
            # learned. Clamped against home on all three axes.
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
        self.get_logger().info("Paused.")
        self.paused = True
        self.shared_obs["paused"] = True
        self.current_target_xyz = None
        self.current_target_euler = None
        self.arm.reset_velocity_state()

    def resume_policy(self):
        # episode_start_pose_mat is intentionally NOT touched: wrt_start is
        # anchored to the fixed home pose, so a rollout begun partway through
        # the task reports its true offset from home rather than resetting to
        # identity.
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

    def reset_to_home(self):
        if self.is_resetting:
            return
        self.get_logger().info("Resetting to home...")
        self.paused = True
        self.shared_obs["paused"] = True
        self.current_target_xyz = None
        self.current_target_euler = None
        self.arm.reset_velocity_state()
        self.is_resetting = True
        self.discard_recording()             # abandon any partial rollout
        threading.Thread(target=self._do_home_reset, daemon=True).start()

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

            # Own the router while the reach action runs: a twist here would
            # cancel the in-flight action and race its RPCs.
            self.arm.twists_suppressed = True
            if not self.arm.reach_home_joints():
                self.get_logger().warn("Home reset timed out")

            self.arm.send_gripper(0.0)      # open
            time.sleep(1.0)
            self.get_logger().info("Reset complete. Press S to start.")
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
                    policy_node.discard_recording()
                    os._exit(0)
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_d:
                        policy_node.pause_policy()
                        policy_node.save_recording()
                    elif event.key == pygame.K_s:
                        policy_node.resume_policy()
                    elif event.key == pygame.K_r:
                        policy_node.reset_to_home()
                    elif event.key == pygame.K_q:
                        policy_node.pause_policy()
                        policy_node.discard_recording()
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
    # dt / n-action-steps / diffusion-steps are now READ FROM THE CHECKPOINT
    # (see load_run_config). They were flags whose defaults merely happened to
    # match the training config, so retraining with a different
    # obs_down_sample_steps or n_action_steps silently desynced inference with
    # no error. Uncomment to override a trained value deliberately — and print
    # loudly when you do, because a mismatch here is invisible in the rollout.
    # parser.add_argument("--dt",              type=float, default=None,
    #                     help="Override action step period (s); default = "
    #                          "obs_down_sample_steps / source_rate from the ckpt")
    # parser.add_argument("--n-action-steps",  type=int,   default=None,
    #                     help="Override actions executed per inference cycle")
    # parser.add_argument("--diffusion-steps", type=int,   default=None,
    #                     help="Override DDIM inference steps")
    parser.add_argument("--latency-offset-s", type=float, default=0.0,
                        help="System latency to compensate (seconds)")
    parser.add_argument("--no-pygame",       action="store_true",      help="Disable pygame window")
    parser.add_argument("--record",          action="store_true",
                        help="Record rollouts to episode_N.hdf5 (S start, D save, R/Q discard)")
    parser.add_argument("--record-dir",      type=str, default=ROLLOUT_DIR_DEFAULT,
                        help=f"Directory for rollout episodes (default: {ROLLOUT_DIR_DEFAULT})")
    args = parser.parse_args()

    # Timing/sampling come from the checkpoint that is about to be run, so they
    # cannot disagree with how the policy was trained.
    run_cfg = load_run_config(args.model)
    args.dt = run_cfg["dt"]
    args.n_action_steps = run_cfg["n_action_steps"]
    args.diffusion_steps = run_cfg["diffusion_steps"]

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
    print(f"dt:              {args.dt}s  ({1/args.dt:.0f} Hz)   [from ckpt: "
          f"down_sample {run_cfg['down_sample_steps']} / {run_cfg['source_rate_hz']:.0f} Hz]")
    print(f"Obs horizons:    read from checkpoint shape_meta (frames {args.dt}s apart)")
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
                      record=args.record, record_dir=args.record_dir, pred_queue=pred_queue)

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
