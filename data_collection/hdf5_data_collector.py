#!/usr/bin/env python3
"""
HDF5 Data Collector — Haptic Teleop IL Study (HoloLens + Kinova Gen3 + Piezense)

Adapted from Robomimic/data_collection/hdf5_data_collector.py.
Collects time-synchronized data from:
  - Kinova Gen3: commanded pose/gripper (robot_action/*) + observed pose/gripper (robot_obs/*)
  - HoloLens hand: palm pose in robot frame (hand/pose)
  - ZED M camera: front view (images/zed_isometric)
  - DJI Osmo Action 4: wrist-mounted camera (images/dji_wrist)
  - Piezense: 2-channel pressure sensor input (piezense/data)

Additional data captured as latest-value at each sync tick (not in sync filter):
  - robot_obs/joint_states
  - hand/gripper_cmd, hand/hand_width, hand/finger_tips
  - raw HoloLens: /hololens/palm/right, /hololens/thumb/right, /hololens/index/right, /hololens/gaze
  - piezense/data (pressure_pa, 2 input channels)

HDF5 schema (per episode):
  episode_N.hdf5
  ├── action/
  │   ├── pose:          (T, 7)  float32   [x, y, z, qx, qy, qz, qw]  robot-frame target
  │   └── gripper:       (T,)    float32   0=open, 1=closed
  ├── observation/
  │   ├── pose:          (T, 7)  float32   current TCP pose
  │   ├── gripper:       (T,)    float32   current gripper
  │   └── joint_states:  (T, 7)  float32   joint angles (rad)
  ├── hololens/
  │   ├── palm_pose:     (T, 7)  float32   [xyz, qxyzw]  (Unity/ROS frame)
  │   ├── thumb_pose:    (T, 7)  float32
  │   ├── index_pose:    (T, 7)  float32
  │   ├── gaze_pose:     (T, 7)  float32
  │   ├── finger_tips:   (T, 15) float32   [thumb(3), index(3), middle(3), ring(3), pinky(3)]
  │   └── hand_width:    (T,)    float32   thumb-index distance (m)
  ├── piezense/
  │   └── pressure_input: (T, 2) float32   input channel pressures (Pa)
  └── images/
      ├── zed_isometric:     (T, 3, H, W) uint8  LZF-compressed CHW
      └── dji_wrist:      (T, 3, H, W) uint8  LZF-compressed CHW

Pygame keyboard controls:
  R — Reset robot to home position
  S — Start recording episode
  D — Done / end recording and SAVE (increments episode counter)
  C — Done / CANCEL recording (discard buffer, do not save or increment)
  P — Pause recording and robot motion
  U — Unpause / resume
  Q — Quit

Usage:
  python3 hdf5_data_collector.py
"""

import glob
import os
import threading
import time

import h5py
import numpy as np
import pygame
import rclpy
from geometry_msgs.msg import PoseStamped
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Bool, Float32, Float32MultiArray, String
from piezense_interfaces.msg import PiezenseSystemArray


# ── Camera topic configuration ─────────────────────────────────────────────────
CAMERA_STREAMS = {
    'zed_isometric': '/zed_isometric/zed_node/left/image_rect_color',
    'dji_wrist':  '/dji_wrist/dji_wrist/color/image_raw',
}

# Number of joint angles to record
NUM_JOINTS = 7

# Piezense: system 0, channels 2 and 3 are the two input sensors
PIEZENSE_SYSTEM_ID      = 0
PIEZENSE_INPUT_CHANNELS = 2
PIEZENSE_INPUT_CHAN_IDS = [2, 3]   # channel indices within the system

# ── Helpers ────────────────────────────────────────────────────────────────────

def _pose_to_vec7(msg: PoseStamped) -> list:
    """Extract [x, y, z, qx, qy, qz, qw] from a PoseStamped."""
    p = msg.pose.position
    o = msg.pose.orientation
    return [p.x, p.y, p.z, o.x, o.y, o.z, o.w]


def _pose_to_vec7_raw(pose_msg) -> list:
    """Same as above but accepts either PoseStamped or None (returns zeros)."""
    if pose_msg is None:
        return [0.0] * 7
    return _pose_to_vec7(pose_msg)


class HDF5DataCollector(Node):
    """
    ROS2 node for synchronized data collection.
    Runs a 7-stream ApproximateTimeSynchronizer for the core streams and
    captures HoloLens/joint data as latest-value side channels.
    """

    def __init__(self):
        super().__init__('hdf5_data_collector')
        self.get_logger().info('Initializing HDF5 Data Collector …')

        self._enable_zed = self.declare_parameter('enable_zed', True).value
        self._enable_dji = self.declare_parameter('enable_dji', True).value
        self._dji_cam_active = False  # tracks whether we have told the DJI node to stream

        if self._enable_dji:
            enable_qos = QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                history=HistoryPolicy.KEEP_LAST,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
                depth=1,
            )
            self._dji_enable_pub = self.create_publisher(Bool, '/dji_camera/enable', enable_qos)
            self._dji_enable_pub.publish(Bool(data=False))

        sensor_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
        )

        # ── Core synchronized subscribers ──────────────────────────────────────
        self._sub_action_pose    = Subscriber(self, PoseStamped, 'robot_action/pose',    qos_profile=sensor_qos)
        self._sub_action_gripper = Subscriber(self, Float32,     'robot_action/gripper', qos_profile=sensor_qos)
        self._sub_obs_pose       = Subscriber(self, PoseStamped, 'robot_obs/pose',       qos_profile=sensor_qos)
        self._sub_obs_gripper    = Subscriber(self, Float32,     'robot_obs/gripper',    qos_profile=sensor_qos)
        self._sub_hand_pose      = Subscriber(self, PoseStamped, 'hand/pose',            qos_profile=sensor_qos)

        # Cameras are latest-value side channels — NOT in the sync — so a camera
        # dropout never stalls the synchronizer or blocks data collection.
        self._latest_zed_frame = None
        self._latest_dji_frame = None
        if self._enable_zed:
            self.create_subscription(
                Image, CAMERA_STREAMS['zed_isometric'],
                self._zed_cb, qos_profile=sensor_qos,
            )
        if self._enable_dji:
            self.create_subscription(
                Image, CAMERA_STREAMS['dji_wrist'],
                self._dji_cb, qos_profile=sensor_qos,
            )

        self._sync = ApproximateTimeSynchronizer(
            [
                self._sub_action_pose,
                self._sub_action_gripper,
                self._sub_obs_pose,
                self._sub_obs_gripper,
                self._sub_hand_pose,
            ],
            queue_size=100,
            slop=0.12,
            allow_headerless=True,
        )
        self._sync.registerCallback(self._synced_callback)

        # ── Side-channel subscriptions (latest-value at each sync tick) ──────
        # Robot joint states
        self._latest_joint_states = np.zeros(NUM_JOINTS, dtype=np.float32)
        self.create_subscription(JointState, 'robot_obs/joint_states',
                                 self._joint_states_cb, qos_profile=sensor_qos)

        # Processed HoloLens data
        self._latest_hand_gripper   = 0.0
        self._latest_hand_width     = 0.0
        self._latest_finger_tips    = np.zeros(15, dtype=np.float32)
        self.create_subscription(Float32,          'hand/gripper_cmd',  self._hand_gripper_cb,  qos_profile=sensor_qos)
        self.create_subscription(Float32,          'hand/hand_width',   self._hand_width_cb,    qos_profile=sensor_qos)
        self.create_subscription(Float32MultiArray,'hand/finger_tips',  self._finger_tips_cb,   qos_profile=sensor_qos)

        # Raw HoloLens PoseStamped topics
        self._latest_holo_palm  = None
        self._latest_holo_thumb = None
        self._latest_holo_index = None
        self._latest_holo_gaze  = None
        self._holo_last_seen    = None   # palm pose → hands actively tracked
        self._holo_link_seen    = None   # any /hololens/* topic → app connected
        self.create_subscription(PoseStamped, '/hololens/palm/right',  self._holo_palm_cb,                    10)
        self.create_subscription(PoseStamped, '/hololens/thumb/right', self._holo_latest_cb('_latest_holo_thumb'), 10)
        self.create_subscription(PoseStamped, '/hololens/index/right', self._holo_latest_cb('_latest_holo_index'), 10)
        self.create_subscription(PoseStamped, '/hololens/gaze',        self._holo_latest_cb('_latest_holo_gaze'),  10)

        # QR-anchor status from the HoloLens app (health dot: teleop frame is
        # only valid once the wall-QR calibration has locked → "initialized").
        self._qr_last_seen = None
        self._qr_value     = ''
        self.create_subscription(String, '/hololens/qr_status', self._qr_cb, 10)

        # Per-episode first-person video via HoloLens Mixed Reality Capture.
        # Start on S, stop+download to demo_data/episode_N.mp4 on D, discard on
        # C. All portal calls run in daemon threads and failures only warn —
        # data collection never blocks on the headset.
        self._enable_mrc = self.declare_parameter('enable_mrc', True).value
        self._mrc = None
        if self._enable_mrc:
            try:
                from hololens_mrc import MRCClient
                self._mrc = MRCClient()
                self.get_logger().info('MRC per-episode video enabled (HoloLens Device Portal)')
            except Exception as e:
                self.get_logger().warn(f'MRC video disabled: {e}')

        # Piezense pressure input (latest-value side channel)
        self._enable_piezense    = self.declare_parameter('enable_piezense', True).value
        self._latest_piezense_input = np.zeros(PIEZENSE_INPUT_CHANNELS, dtype=np.float32)
        self._piezense_last_seen    = None
        self._piezense_warned       = False
        if self._enable_piezense:
            # Piezense driver publishes RELIABLE (default QoS); BEST_EFFORT is incompatible.
            self.create_subscription(PiezenseSystemArray, 'piezense/data',
                                     self._piezense_cb, 10)
            self.create_timer(2.0, self._check_piezense_health)
        else:
            self.get_logger().info('Piezense disabled (enable_piezense=false)')

        # ── Camera health monitoring ──────────────────────────────────────────
        self._node_start_time = time.monotonic()
        active_cams = {
            k: v for k, v in CAMERA_STREAMS.items()
            if (k == 'zed_isometric' and self._enable_zed) or
               (k == 'dji_wrist' and self._enable_dji)
        }
        self._cam_last_seen   = {k: None  for k in active_cams}
        self._cam_drop_warned = {k: False for k in active_cams}
        for cam_name, topic in active_cams.items():
            self.create_subscription(
                Image, topic,
                lambda _msg, n=cam_name: self._cam_heartbeat(n),
                qos_profile=sensor_qos,
            )
        if active_cams:
            self.create_timer(2.0, self._check_camera_health)
        else:
            self.get_logger().info('All cameras disabled')

        # ── Control publishers ─────────────────────────────────────────────────
        self.reset_pub = self.create_publisher(Bool, '/reset_kinova', 10)
        self.pause_pub = self.create_publisher(Bool, '/pause_kinova', 10)

        # ── Collection state ──────────────────────────────────────────────────
        self.is_collecting = False
        self.is_paused     = False
        self.episode_start = None
        self._lock         = threading.Lock()
        self._reset_buffers()

        self._save_dir   = os.path.join(os.getcwd(), 'demo_data')
        self.demo_count  = self._scan_existing_episodes()

        self.get_logger().info('HDF5 Data Collector initialized')

    # ── Buffer management ─────────────────────────────────────────────────────
    def _reset_buffers(self):
        self._buf_action_pose      = []
        self._buf_action_gripper   = []
        self._buf_obs_pose         = []
        self._buf_obs_gripper      = []
        self._buf_joint_states     = []
        self._buf_hand_pose        = []   # robot-frame palm (from hand/pose)
        self._buf_holo_palm_pose   = []   # raw Unity-frame palm
        self._buf_holo_thumb_pose  = []
        self._buf_holo_index_pose  = []
        self._buf_holo_gaze_pose   = []
        self._buf_finger_tips      = []
        self._buf_hand_width       = []
        self._buf_piezense_input   = []
        self._buf_zed_isometric        = []
        self._buf_dji_wrist         = []

    # ── Side-channel callbacks ────────────────────────────────────────────────
    def _joint_states_cb(self, msg: JointState):
        angles = list(msg.position[:NUM_JOINTS])
        angles += [0.0] * (NUM_JOINTS - len(angles))
        self._latest_joint_states = np.array(angles, dtype=np.float32)

    def _hand_gripper_cb(self, msg: Float32):   self._latest_hand_gripper = float(msg.data)
    def _hand_width_cb(self,   msg: Float32):   self._latest_hand_width   = float(msg.data)

    def _finger_tips_cb(self, msg: Float32MultiArray):
        data = list(msg.data)
        data += [0.0] * (15 - len(data))
        self._latest_finger_tips = np.array(data[:15], dtype=np.float32)

    def _piezense_cb(self, msg: PiezenseSystemArray):
        self._piezense_last_seen = time.monotonic()
        self._piezense_warned    = False
        for sys_msg in msg.system:
            if sys_msg.system_id == PIEZENSE_SYSTEM_ID:
                readings = list(sys_msg.pressure_pa)
                self._latest_piezense_input = np.array(
                    [float(readings[c]) if c < len(readings) else 0.0
                     for c in PIEZENSE_INPUT_CHAN_IDS],
                    dtype=np.float32,
                )
                break

    def _check_piezense_health(self):
        if self._piezense_last_seen is None and self.is_collecting and not self._piezense_warned:
            self._piezense_warned = True
            self.get_logger().warn(
                'Piezense: no data on piezense/data — is piezense_driver running?'
            )

    def _holo_latest_cb(self, attr: str):
        """Latest-value setter that also marks the HoloLens link alive."""
        def cb(msg):
            setattr(self, attr, msg)
            self._holo_link_seen = time.monotonic()
        return cb

    def _holo_palm_cb(self, msg: PoseStamped):
        self._latest_holo_palm = msg
        self._holo_last_seen   = time.monotonic()
        self._holo_link_seen   = self._holo_last_seen

    def get_hololens_health(self) -> str:
        """ready = hands tracked; waiting = app connected but hands not visible;
        dead = no HoloLens traffic at all.

        Hand tracking drops out routinely (hands leave the FOV whenever the
        operator is not actively teleoperating), which is normal and should not
        look like a failure. Only a dead link earns a red dot, so the two are
        tracked on separate timestamps: palm pose for tracking, ANY /hololens/*
        topic for the link — gaze and qr_status keep publishing while the app
        runs regardless of hands.
        """
        now = time.monotonic()
        if self._holo_link_seen is None:
            return 'waiting' if (now - self._node_start_time) < 5.0 else 'dead'
        if (now - self._holo_link_seen) > 3.0:
            return 'dead'
        if self._holo_last_seen is None or (now - self._holo_last_seen) > 3.0:
            return 'waiting'
        return 'ready'

    def _qr_cb(self, msg: String):
        self._qr_last_seen = time.monotonic()
        self._qr_value = msg.data
        self._holo_link_seen = self._qr_last_seen

    def get_qr_health(self) -> str:
        """Green only once the headset reports the QR anchor has locked."""
        now = time.monotonic()
        if self._qr_last_seen is None:
            return 'waiting' if (now - self._node_start_time) < 5.0 else 'dead'
        if (now - self._qr_last_seen) > 3.0:
            return 'dead'
        return 'ready' if self._qr_value == 'initialized' else 'waiting'

    # ── Core synced callback ──────────────────────────────────────────────────
    def _zed_cb(self, msg: Image):
        self._latest_zed_frame = self._decode_image(msg)

    def _dji_cb(self, msg: Image):
        self._latest_dji_frame = self._decode_image(msg)

    def _synced_callback(
        self,
        action_pose_msg: PoseStamped,
        action_gripper_msg: Float32,
        obs_pose_msg: PoseStamped,
        obs_gripper_msg: Float32,
        hand_pose_msg: PoseStamped,
    ):
        if not self.is_collecting or self.is_paused:
            return

        with self._lock:
            self._buf_action_pose.append(_pose_to_vec7(action_pose_msg))
            self._buf_action_gripper.append(float(action_gripper_msg.data))
            self._buf_obs_pose.append(_pose_to_vec7(obs_pose_msg))
            self._buf_obs_gripper.append(float(obs_gripper_msg.data))
            self._buf_joint_states.append(self._latest_joint_states.copy())
            self._buf_hand_pose.append(_pose_to_vec7(hand_pose_msg))

            # Raw HoloLens data (latest-value)
            self._buf_holo_palm_pose.append(_pose_to_vec7_raw(self._latest_holo_palm))
            self._buf_holo_thumb_pose.append(_pose_to_vec7_raw(self._latest_holo_thumb))
            self._buf_holo_index_pose.append(_pose_to_vec7_raw(self._latest_holo_index))
            self._buf_holo_gaze_pose.append(_pose_to_vec7_raw(self._latest_holo_gaze))
            self._buf_finger_tips.append(self._latest_finger_tips.copy())
            self._buf_hand_width.append(self._latest_hand_width)

            # Piezense pressure input (latest-value)
            if self._enable_piezense:
                self._buf_piezense_input.append(self._latest_piezense_input.copy())

            # Images (latest-value — decoupled from sync so dropouts don't stall)
            if self._enable_zed and self._latest_zed_frame is not None:
                self._buf_zed_isometric.append(self._latest_zed_frame.copy())
            if self._enable_dji and self._latest_dji_frame is not None:
                self._buf_dji_wrist.append(self._latest_dji_frame.copy())

        count = len(self._buf_action_pose)
        if count % 30 == 0:
            self.get_logger().info(f'Collected {count} frames')

    def _decode_image(self, msg: Image) -> np.ndarray:
        """Convert sensor_msgs/Image to CHW uint8 numpy array without cv_bridge."""
        frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        if msg.encoding in ('bgr8', 'bgr24'):
            frame = frame[:, :, ::-1]  # BGR → RGB
        return np.ascontiguousarray(frame.transpose(2, 0, 1))  # HWC → CHW

    # ── Camera health ─────────────────────────────────────────────────────────
    def _cam_heartbeat(self, name: str):
        now = time.monotonic()
        first = self._cam_last_seen[name] is None
        self._cam_last_seen[name] = now
        if first:
            self.get_logger().info(f'Camera {name}: first frame received')
        if self._cam_drop_warned[name]:
            self._cam_drop_warned[name] = False
            self.get_logger().info(f'Camera {name}: RECOVERED')

    def _check_camera_health(self):
        now    = time.monotonic()
        uptime = now - self._node_start_time
        banner = '!' * 50

        for name in self._cam_last_seen:
            last = self._cam_last_seen[name]
            if last is None:
                if name == 'dji_wrist' and not self._dji_cam_active:
                    pass  # idle by design — only streams during recording
                elif uptime > 5.0 and not self._cam_drop_warned[name]:
                    self._cam_drop_warned[name] = True
                    self.get_logger().error(
                        f'\n{banner}\n  CAMERA {name} has NEVER published!\n{banner}'
                    )
            elif (now - last) > 6.0 and not self._cam_drop_warned[name]:
                if name == 'dji_wrist' and not self._dji_cam_active:
                    pass  # idle by design — disabled after end of episode
                else:
                    self._cam_drop_warned[name] = True
                    self.get_logger().warn(
                        f'\n{banner}\n  CAMERA {name} STOPPED (last frame '
                        f'{now - last:.1f}s ago) — recording continues\n{banner}'
                    )

    def get_camera_health(self) -> dict:
        now = time.monotonic()
        result = {}
        for name in CAMERA_STREAMS:
            if name not in self._cam_last_seen:
                result[name] = 'disabled'
            elif name == 'dji_wrist' and not self._dji_cam_active:
                result[name] = 'idle'
            else:
                last = self._cam_last_seen[name]
                result[name] = ('ready' if last is not None and (now - last) < 6.0
                                else ('waiting' if last is None else 'dead'))
        return result

    def get_dji_preview(self):
        """Latest DJI wrist frame as HWC RGB uint8, or None when not streaming.

        These are the exact pixels buffered into images/dji_wrist, so the GUI
        preview doubles as a check that the camera has not rotated its output
        mid-episode. Returns None whenever the DJI node is not streaming (it is
        disabled between episodes) so the GUI shows a placeholder rather than a
        frozen last frame.

        Reading _latest_dji_frame without the lock is safe: the callback rebinds
        the attribute to a fresh array rather than mutating in place, so we get
        either the previous or the next frame, never a torn one. Same pattern as
        _synced_callback.
        """
        if not (self._enable_dji and self._dji_cam_active):
            return None
        frame = self._latest_dji_frame
        if frame is None:
            return None
        return frame.transpose(1, 2, 0)  # CHW → HWC

    def get_piezense_health(self) -> str:
        if not self._enable_piezense:
            return 'disabled'
        now = time.monotonic()
        if self._piezense_last_seen is None:
            return 'waiting' if (now - self._node_start_time) < 5.0 else 'dead'
        return 'ready' if (now - self._piezense_last_seen) < 3.0 else 'dead'

    # ── HoloLens MRC video (all calls threaded; failures warn, never block) ──
    def _mrc_start(self):
        if self._mrc is None:
            return
        def run():
            try:
                self._mrc.start()
                self.get_logger().info('MRC first-person video: recording')
            except Exception as e:
                self.get_logger().warn(f'MRC start failed: {e} — episode continues without video')
        threading.Thread(target=run, daemon=True).start()

    def _mrc_finish(self, episode_idx: int):
        if self._mrc is None:
            return
        dest = os.path.join(self._save_dir, f'episode_{episode_idx}.mp4')
        def run():
            try:
                path, size = self._mrc.stop_and_fetch(dest)
                self.get_logger().info(f'Saved {path} ({size / 1e6:.1f} MB)')
            except Exception as e:
                self.get_logger().warn(f'MRC video fetch failed: {e}')
        threading.Thread(target=run, daemon=True).start()

    def _mrc_discard(self):
        if self._mrc is None:
            return
        def run():
            try:
                self._mrc.stop_and_discard()
            except Exception as e:
                self.get_logger().warn(f'MRC discard failed: {e}')
        threading.Thread(target=run, daemon=True).start()

    # ── Collection controls ───────────────────────────────────────────────────
    def start_collection(self):
        if not self.is_collecting:
            with self._lock:
                self._reset_buffers()
            self.is_collecting = True
            self.is_paused     = False
            self.episode_start = self.get_clock().now()
            if self._enable_dji:
                self._dji_cam_active = True
                self._cam_drop_warned['dji_wrist'] = False  # re-arm warning for this episode
                # Drop the last episode's frame: the DJI node takes a moment to
                # reopen the device, and the GUI preview would otherwise show a
                # stale image that looks live.
                self._latest_dji_frame = None
                self._dji_enable_pub.publish(Bool(data=True))
            self._mrc_start()
            self.get_logger().info(f'Started recording episode {self.demo_count}')

    def end_collection(self):
        if self.is_collecting:
            self.is_collecting = False
            if self._enable_dji:
                self._dji_cam_active = False
                self._dji_enable_pub.publish(Bool(data=False))
            self._save_episode()
            dur = (self.get_clock().now() - self.episode_start).nanoseconds / 1e9
            n   = len(self._buf_action_pose)
            self.get_logger().info(
                f'Episode {self.demo_count} | {n} frames | {dur:.1f}s | {n/dur:.1f} Hz'
            )
            self._mrc_finish(self.demo_count)
            self.demo_count += 1

    def cancel_collection(self):
        """End recording WITHOUT saving. Discards the buffer and does not
        increment demo_count, so the next Start reuses the same episode index."""
        if self.is_collecting:
            self.is_collecting = False
            if self._enable_dji:
                self._dji_cam_active = False
                self._dji_enable_pub.publish(Bool(data=False))
            n = len(self._buf_action_pose)
            with self._lock:
                self._reset_buffers()
            self._mrc_discard()
            self.get_logger().info(
                f'Episode {self.demo_count} CANCELLED — discarded {n} frames (not saved)'
            )

    def pause_collection(self):
        if not self.is_paused:
            self.is_paused = True
            self.pause_pub.publish(Bool(data=True))
            self.get_logger().info('Paused')

    def unpause_collection(self):
        if self.is_paused:
            self.is_paused = False
            self.pause_pub.publish(Bool(data=False))
            self.get_logger().info('Resumed')

    def reset_robot(self):
        self.get_logger().info('Sending reset command to Kinova')
        self.reset_pub.publish(Bool(data=True))

    # ── Episode persistence ───────────────────────────────────────────────────
    def _scan_existing_episodes(self) -> int:
        if not os.path.isdir(self._save_dir):
            return 0
        existing = glob.glob(os.path.join(self._save_dir, 'episode_*.hdf5'))
        indices = []
        for path in existing:
            try:
                idx = int(os.path.basename(path).replace('episode_', '').replace('.hdf5', ''))
                indices.append(idx)
            except ValueError:
                pass
        if not indices:
            return 0
        next_idx = max(indices) + 1
        self.get_logger().info(
            f'Found {len(indices)} existing episode(s). Resuming at episode {next_idx}.'
        )
        return next_idx

    def _save_episode(self):
        with self._lock:
            if not self._buf_action_pose:
                self.get_logger().warn('No data to save')
                return

            action_pose     = np.array(self._buf_action_pose,     dtype=np.float32)
            action_gripper  = np.array(self._buf_action_gripper,  dtype=np.float32)
            obs_pose        = np.array(self._buf_obs_pose,        dtype=np.float32)
            obs_gripper     = np.array(self._buf_obs_gripper,     dtype=np.float32)
            joint_states    = np.array(self._buf_joint_states,    dtype=np.float32)
            hand_pose       = np.array(self._buf_hand_pose,       dtype=np.float32)
            holo_palm       = np.array(self._buf_holo_palm_pose,  dtype=np.float32)
            holo_thumb      = np.array(self._buf_holo_thumb_pose, dtype=np.float32)
            holo_index      = np.array(self._buf_holo_index_pose, dtype=np.float32)
            holo_gaze       = np.array(self._buf_holo_gaze_pose,  dtype=np.float32)
            finger_tips      = np.array(self._buf_finger_tips,      dtype=np.float32)
            hand_width       = np.array(self._buf_hand_width,       dtype=np.float32)
            if self._enable_piezense:
                piezense_input = np.array(self._buf_piezense_input, dtype=np.float32)
            T = len(action_pose)
            if self._enable_zed and self._buf_zed_isometric:
                zed_isometric = np.array(self._buf_zed_isometric[:T], dtype=np.uint8)
            if self._enable_dji and self._buf_dji_wrist:
                dji_wrist = np.array(self._buf_dji_wrist[:T], dtype=np.uint8)

        os.makedirs(self._save_dir, exist_ok=True)
        filename = os.path.join(self._save_dir, f'episode_{self.demo_count}.hdf5')

        with h5py.File(filename, 'w') as f:
            act = f.create_group('action')
            act.create_dataset('pose',    data=action_pose)
            act.create_dataset('gripper', data=action_gripper)

            obs = f.create_group('observation')
            obs.create_dataset('pose',         data=obs_pose)
            obs.create_dataset('gripper',      data=obs_gripper)
            obs.create_dataset('joint_states', data=joint_states)

            hl = f.create_group('hololens')
            hl.create_dataset('palm_pose',   data=holo_palm)
            hl.create_dataset('thumb_pose',  data=holo_thumb)
            hl.create_dataset('index_pose',  data=holo_index)
            hl.create_dataset('gaze_pose',   data=holo_gaze)
            hl.create_dataset('finger_tips', data=finger_tips)
            hl.create_dataset('hand_width',  data=hand_width)
            # hand/pose (robot-frame palm) lives here too for easy access
            hl.create_dataset('hand_pose_robot_frame', data=hand_pose)

            if self._enable_piezense:
                pz = f.create_group('piezense')
                pz.create_dataset('pressure_input', data=piezense_input)
                pz.attrs['channel_ids'] = PIEZENSE_INPUT_CHAN_IDS
                pz.attrs['units'] = 'Pa'

            if (self._enable_zed and self._buf_zed_isometric) or \
               (self._enable_dji and self._buf_dji_wrist):
                imgs = f.create_group('images')
                if self._enable_zed and self._buf_zed_isometric:
                    imgs.create_dataset('zed_isometric', data=zed_isometric, compression='lzf')
                if self._enable_dji and self._buf_dji_wrist:
                    imgs.create_dataset('dji_wrist', data=dji_wrist, compression='lzf')

            f.attrs['num_frames']         = len(action_pose)
            f.attrs['collection_rate_hz'] = 30
            f.attrs['episode_index']      = self.demo_count

        self.get_logger().info(f'Saved {filename}  ({len(action_pose)} frames)')


# ── Pygame UI ─────────────────────────────────────────────────────────────────

PANEL_W    = 620   # left control panel — the original window width
PREVIEW_PX = 224   # DJI preview edge, shown 1:1 with the policy input
GUI_W      = PANEL_W + PREVIEW_PX + 40
GUI_H      = 340


def _draw_dji_preview(screen, node, small_font):
    """Draw the live DJI wrist feed at native 224x224, or a placeholder.

    Rendered 1:1 with no scaling so it is exactly what gets written to
    images/dji_wrist and fed to the policy — which makes a mid-session camera
    rotation visible while there is still time to redo the demo.
    """
    x, y = PANEL_W + 20, 46
    screen.blit(small_font.render('DJI wrist — policy input', True, (200, 200, 200)), (x, 20))

    frame = node.get_dji_preview()
    if frame is None:
        pygame.draw.rect(screen, (30, 33, 39), (x, y, PREVIEW_PX, PREVIEW_PX))
        # Two short lines: as one line this text is wider than the 224 px box
        # and spills past the border on both sides.
        lines = ['idle', 'streams while recording']
        ly = y + PREVIEW_PX // 2 - 10 * len(lines)
        for line in lines:
            label = small_font.render(line, True, (110, 118, 128))
            screen.blit(label, (x + max(0, (PREVIEW_PX - label.get_width()) // 2), ly))
            ly += 20
    else:
        # make_surface wants (W, H, 3); the frame arrives HWC.
        surf = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
        if surf.get_size() != (PREVIEW_PX, PREVIEW_PX):
            surf = pygame.transform.smoothscale(surf, (PREVIEW_PX, PREVIEW_PX))
        screen.blit(surf, (x, y))
        h, w = frame.shape[:2]
        screen.blit(small_font.render(f'{w}x{h}', True, (120, 130, 140)),
                    (x, y + PREVIEW_PX + 6))

    pygame.draw.rect(screen, (90, 96, 105), (x, y, PREVIEW_PX, PREVIEW_PX), 1)


def run_pygame(node: HDF5DataCollector):
    """Pygame keyboard control loop. Runs in the main thread."""
    pygame.init()
    screen     = pygame.display.set_mode((GUI_W, GUI_H))
    pygame.display.set_caption('Haptic Teleop IL — Data Collection')
    font       = pygame.font.Font(None, 32)
    small_font = pygame.font.Font(None, 24)
    clock      = pygame.time.Clock()

    while rclpy.ok():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
            elif event.type == pygame.KEYDOWN:
                if   event.key == pygame.K_r:      node.reset_robot()
                elif event.key == pygame.K_s:      node.start_collection()
                elif event.key == pygame.K_d:      node.end_collection()
                elif event.key == pygame.K_c:      node.cancel_collection()
                elif event.key == pygame.K_p:      node.pause_collection()
                elif event.key == pygame.K_u:      node.unpause_collection()
                elif event.key in (pygame.K_q, pygame.K_ESCAPE):
                    return

        screen.fill((40, 44, 52))

        # Color key (top right)
        key = [
            ('ready',       ( 80, 200,  80)),
            ('waiting',  (255, 200,  50)),
            ('dead',     (220,  50,  50)),
            ('idle',     ( 60, 100, 160)),
            ('disabled', ( 80,  80,  80)),
        ]
        kx = PANEL_W - 90
        for row, (label, kcolor) in enumerate(key):
            ky = 12 + row * 18
            pygame.draw.circle(screen, kcolor, (kx, ky + 5), 4)
            screen.blit(small_font.render(label, True, (140, 140, 140)), (kx + 10, ky))

        # Status
        if not node.is_collecting:
            status, color = 'IDLE',      (150, 150, 150)
        elif node.is_paused:
            status, color = 'PAUSED',    (255, 200,  50)
        else:
            status, color = 'RECORDING', ( 80, 200,  80)

        screen.blit(font.render(f'Status: {status}', True, color), (20, 20))

        n_frames = len(node._buf_action_pose) if node.is_collecting else 0
        screen.blit(font.render(f'Frames: {n_frames}',          True, (200, 200, 200)), (20, 55))
        screen.blit(font.render(f'Next episode: {node.demo_count}', True, (200, 200, 200)), (20, 90))

        # Sensor health dots (cameras + piezense)
        health_colors = {
            'ready':       ( 80, 200,  80),
            'waiting':  (255, 200,  50),
            'dead':     (220,  50,  50),
            'idle':     ( 60, 100, 160),  # dim blue — intentionally off between episodes
            'disabled': ( 80,  80,  80),
        }
        health_items = list(node.get_camera_health().items())
        health_items.append(('piezense', node.get_piezense_health()))
        health_items.append(('hololens', node.get_hololens_health()))
        health_items.append(('qr', node.get_qr_health()))
        # Flow the chips left-to-right using their MEASURED widths: a fixed
        # step overlaps as soon as one label is wider than its share of the row
        # ('zed_isometric' alone exceeds a fifth of it).
        row_left  = 20
        row_right = PANEL_W - 20
        chips = [(small_font.render(label, True, (200, 200, 200)), status)
                 for label, status in health_items]
        chips_w = sum(16 + surf.get_width() for surf, _ in chips)   # 16 = dot + pad
        gap = max(6, (row_right - row_left - chips_w) // max(1, len(chips) - 1))
        cx = row_left
        for surf, status in chips:
            pygame.draw.circle(screen, health_colors[status], (cx + 6, 132), 6)
            screen.blit(surf, (cx + 16, 126))
            cx += 16 + surf.get_width() + gap

        # Controls
        controls = [
            'Controls:',
            '  R - Reset robot to home',
            '  S - Start recording',
            '  D - Done / SAVE episode',
            '  C - Done / CANCEL (discard, no save)',
            '  P - Pause',
            '  U - Unpause',
            '  Q - Quit',
        ]
        y = 160
        for line in controls:
            screen.blit(small_font.render(line, True, (120, 130, 140)), (20, y))
            y += 20

        _draw_dji_preview(screen, node, small_font)

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


# ── Entry point ───────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = HDF5DataCollector()

    executor = MultiThreadedExecutor()
    executor.add_node(node)
    ros_thread = threading.Thread(target=executor.spin, daemon=True)
    ros_thread.start()

    try:
        run_pygame(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info('Shutting down …')
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
