#!/usr/bin/env python3
"""
HDF5 Data Collector — Haptic Teleop IL Study (HoloLens + Kinova Gen3)

Adapted from Robomimic/data_collection/hdf5_data_collector.py.
Collects time-synchronized data from:
  - Kinova Gen3: commanded pose/gripper (robot_action/*) + observed pose/gripper (robot_obs/*)
  - HoloLens hand: palm pose in robot frame (hand/pose)
  - ZED M camera: front view (images/zed_front)
  - RealSense D435i: wrist-mounted camera (images/rs_wrist)

Additional data captured as latest-value at each sync tick (not in sync filter):
  - robot_obs/joint_states
  - hand/gripper_cmd, hand/hand_width, hand/finger_tips
  - raw HoloLens: /hololens/palm/right, /hololens/thumb/right, /hololens/index/right, /hololens/gaze

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
  └── images/
      ├── zed_front:     (T, 3, H, W) uint8  LZF-compressed CHW
      └── rs_wrist:      (T, 3, H, W) uint8  LZF-compressed CHW

Pygame keyboard controls:
  R — Reset robot to home position
  S — Start recording episode
  D — Done / end recording and save
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

import cv2
import h5py
import numpy as np
import pygame
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Bool, Float32, Float32MultiArray


# ── Camera topic configuration ─────────────────────────────────────────────────
CAMERA_STREAMS = {
    'zed_front': '/zed_front/zed_node/left/image_rect_color',
    'rs_wrist':  '/rs_wrist/rs_wrist/color/image_raw',
}

# Number of joint angles to record
NUM_JOINTS = 7

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

        self._bridge = CvBridge()

        sensor_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
        )

        # ── Core synchronized subscribers ──────────────────────────────────────
        # 7 streams: robot action/obs (pose + gripper) + hand/pose + 2 cameras
        self._sub_action_pose    = Subscriber(self, PoseStamped, 'robot_action/pose',    qos_profile=sensor_qos)
        self._sub_action_gripper = Subscriber(self, Float32,     'robot_action/gripper', qos_profile=sensor_qos)
        self._sub_obs_pose       = Subscriber(self, PoseStamped, 'robot_obs/pose',       qos_profile=sensor_qos)
        self._sub_obs_gripper    = Subscriber(self, Float32,     'robot_obs/gripper',    qos_profile=sensor_qos)
        self._sub_hand_pose      = Subscriber(self, PoseStamped, 'hand/pose',            qos_profile=sensor_qos)
        self._sub_zed_front      = Subscriber(self, Image, CAMERA_STREAMS['zed_front'],  qos_profile=sensor_qos)
        self._sub_rs_wrist       = Subscriber(self, Image, CAMERA_STREAMS['rs_wrist'],   qos_profile=sensor_qos)

        self._sync = ApproximateTimeSynchronizer(
            [
                self._sub_action_pose,
                self._sub_action_gripper,
                self._sub_obs_pose,
                self._sub_obs_gripper,
                self._sub_hand_pose,
                self._sub_zed_front,
                self._sub_rs_wrist,
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
        self.create_subscription(PoseStamped, '/hololens/palm/right',  lambda m: setattr(self, '_latest_holo_palm',  m), 10)
        self.create_subscription(PoseStamped, '/hololens/thumb/right', lambda m: setattr(self, '_latest_holo_thumb', m), 10)
        self.create_subscription(PoseStamped, '/hololens/index/right', lambda m: setattr(self, '_latest_holo_index', m), 10)
        self.create_subscription(PoseStamped, '/hololens/gaze',        lambda m: setattr(self, '_latest_holo_gaze',  m), 10)

        # ── Camera health monitoring ──────────────────────────────────────────
        self._cam_last_seen  = {k: None for k in CAMERA_STREAMS}
        self._cam_drop_warned = {k: False for k in CAMERA_STREAMS}
        self._node_start_time = time.monotonic()
        for cam_name, topic in CAMERA_STREAMS.items():
            self.create_subscription(
                Image, topic,
                lambda _msg, n=cam_name: self._cam_heartbeat(n),
                qos_profile=sensor_qos,
            )
        self.create_timer(2.0, self._check_camera_health)

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
        self._buf_action_pose     = []
        self._buf_action_gripper  = []
        self._buf_obs_pose        = []
        self._buf_obs_gripper     = []
        self._buf_joint_states    = []
        self._buf_hand_pose       = []   # robot-frame palm (from hand/pose)
        self._buf_holo_palm_pose  = []   # raw Unity-frame palm
        self._buf_holo_thumb_pose = []
        self._buf_holo_index_pose = []
        self._buf_holo_gaze_pose  = []
        self._buf_finger_tips     = []
        self._buf_hand_width      = []
        self._buf_zed_front       = []
        self._buf_rs_wrist        = []

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

    # ── Core synced callback ──────────────────────────────────────────────────
    def _synced_callback(
        self,
        action_pose_msg: PoseStamped,
        action_gripper_msg: Float32,
        obs_pose_msg: PoseStamped,
        obs_gripper_msg: Float32,
        hand_pose_msg: PoseStamped,
        zed_front_msg: Image,
        rs_wrist_msg: Image,
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

            # Images
            self._buf_zed_front.append(self._decode_image(zed_front_msg))
            self._buf_rs_wrist.append(self._decode_image(rs_wrist_msg))

        count = len(self._buf_action_pose)
        if count % 30 == 0:
            self.get_logger().info(f'Collected {count} frames')

    def _decode_image(self, msg: Image) -> np.ndarray:
        """Convert sensor_msgs/Image to CHW uint8 numpy array."""
        rgb = self._bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        return rgb.transpose(2, 0, 1)  # HWC → CHW

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

        for name in CAMERA_STREAMS:
            last = self._cam_last_seen[name]
            if last is None:
                if uptime > 5.0 and not self._cam_drop_warned[name]:
                    self._cam_drop_warned[name] = True
                    self.get_logger().error(
                        f'\n{banner}\n  CAMERA {name} has NEVER published!\n{banner}'
                    )
            elif (now - last) > 3.0 and not self._cam_drop_warned[name]:
                self._cam_drop_warned[name] = True
                msg = (
                    f'\n{banner}\n  CAMERA {name} STOPPED (last frame '
                    f'{now - last:.1f}s ago)\n'
                )
                if self.is_collecting:
                    n = len(self._buf_action_pose)
                    self.end_collection()
                    msg += f'  Trial AUTO-STOPPED after {n} frames.\n'
                self.get_logger().error(msg + banner)

    def get_camera_health(self) -> dict:
        now = time.monotonic()
        return {
            name: ('ok' if last is not None and (now - last) < 3.0
                   else ('waiting' if last is None else 'dead'))
            for name, last in self._cam_last_seen.items()
        }

    # ── Collection controls ───────────────────────────────────────────────────
    def start_collection(self):
        if not self.is_collecting:
            with self._lock:
                self._reset_buffers()
            self.is_collecting = True
            self.is_paused     = False
            self.episode_start = self.get_clock().now()
            self.get_logger().info(f'Started recording episode {self.demo_count}')

    def end_collection(self):
        if self.is_collecting:
            self.is_collecting = False
            self._save_episode()
            dur = (self.get_clock().now() - self.episode_start).nanoseconds / 1e9
            n   = len(self._buf_action_pose)
            self.get_logger().info(
                f'Episode {self.demo_count} | {n} frames | {dur:.1f}s | {n/dur:.1f} Hz'
            )
            self.demo_count += 1

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
            finger_tips     = np.array(self._buf_finger_tips,     dtype=np.float32)
            hand_width      = np.array(self._buf_hand_width,      dtype=np.float32)
            zed_front       = np.array(self._buf_zed_front,       dtype=np.uint8)
            rs_wrist        = np.array(self._buf_rs_wrist,        dtype=np.uint8)

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

            imgs = f.create_group('images')
            imgs.create_dataset('zed_front', data=zed_front, compression='lzf')
            imgs.create_dataset('rs_wrist',  data=rs_wrist,  compression='lzf')

            f.attrs['num_frames']         = len(action_pose)
            f.attrs['collection_rate_hz'] = 30
            f.attrs['episode_index']      = self.demo_count

        self.get_logger().info(f'Saved {filename}  ({len(action_pose)} frames)')


# ── Pygame UI ─────────────────────────────────────────────────────────────────

def run_pygame(node: HDF5DataCollector):
    """Pygame keyboard control loop. Runs in the main thread."""
    pygame.init()
    screen     = pygame.display.set_mode((520, 300))
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
                elif event.key == pygame.K_p:      node.pause_collection()
                elif event.key == pygame.K_u:      node.unpause_collection()
                elif event.key in (pygame.K_q, pygame.K_ESCAPE):
                    return

        screen.fill((40, 44, 52))

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

        # Camera health dots
        health_colors = {'ok': (80, 200, 80), 'waiting': (255, 200, 50), 'dead': (220, 50, 50)}
        x_pos = 20
        for cam_name, cam_status in node.get_camera_health().items():
            color = health_colors[cam_status]
            pygame.draw.circle(screen, color, (x_pos + 6, 132), 6)
            screen.blit(small_font.render(cam_name, True, (200, 200, 200)), (x_pos + 16, 126))
            x_pos += 160

        # Controls
        controls = [
            'Controls:',
            '  R - Reset robot to home',
            '  S - Start recording',
            '  D - Done / save episode',
            '  P - Pause',
            '  U - Unpause',
            '  Q - Quit',
        ]
        y = 160
        for line in controls:
            screen.blit(small_font.render(line, True, (120, 130, 140)), (20, y))
            y += 20

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
