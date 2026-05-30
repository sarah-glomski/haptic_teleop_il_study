#!/usr/bin/env python3
"""
Replay a recorded episode on the Kinova Gen3.

Usage:
  python3.12 replay_episode.py demo_data/episode_9.hdf5
  python3.12 replay_episode.py demo_data/episode_9.hdf5 --robot-ip 192.168.1.10

Controls during replay (terminal):
  q + ENTER  — abort replay early
"""

import argparse
import threading
import time

import h5py
import numpy as np

from kortex_api.TCPTransport import TCPTransport
from kortex_api.RouterClient import RouterClient
from kortex_api.SessionManager import SessionManager
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.messages import Session_pb2, Base_pb2

CONTROL_HZ  = 30.0
WATCHDOG_MS = 200
HOME_POS    = (0.474, 0.02, 0.107)
HOME_ANGLES = (-179.3, -0.4, 89.3)
WS_BOUNDS   = dict(x=(0.30, 0.60), y=(-0.37, 0.37), z=(0.025, 0.30))
MARGIN      = 0.005


class KinovaConn:
    def __init__(self, ip: str, username: str = 'admin', password: str = 'admin'):
        self._transport = TCPTransport()
        self._router = RouterClient(self._transport, lambda ex: print(f'[WARN] kortex: {ex}'))
        self._transport.connect(ip, 10000)

        info = Session_pb2.CreateSessionInfo()
        info.username = username
        info.password = password
        info.session_inactivity_timeout    = 60000
        info.connection_inactivity_timeout = 2000
        self._session = SessionManager(self._router)
        self._session.CreateSession(info)

        self.base   = BaseClient(self._router)
        self.cyclic = BaseCyclicClient(self._router)
        print(f'Connected to Kinova Gen3 at {ip}')

    def disconnect(self):
        try:
            self._session.CloseSession()
            self._transport.disconnect()
        except Exception:
            pass

    def __enter__(self):  return self
    def __exit__(self, *_): self.disconnect()


def _get_tcp(conn: KinovaConn) -> np.ndarray:
    fb = conn.cyclic.RefreshFeedback()
    return np.array([fb.base.tool_pose_x, fb.base.tool_pose_y, fb.base.tool_pose_z])


def _send_twist(conn: KinovaConn, vel: np.ndarray):
    cmd = Base_pb2.TwistCommand()
    cmd.reference_frame = Base_pb2.CARTESIAN_REFERENCE_FRAME_MIXED
    cmd.duration        = WATCHDOG_MS
    cmd.twist.linear_x  = float(vel[0])
    cmd.twist.linear_y  = float(vel[1])
    cmd.twist.linear_z  = float(vel[2])
    conn.base.SendTwistCommand(cmd)


def _send_gripper(conn: KinovaConn, value: float):
    gc = Base_pb2.GripperCommand()
    gc.mode = Base_pb2.GRIPPER_POSITION
    f = gc.gripper.finger.add()
    f.finger_identifier = 1
    f.value = float(np.clip(value, 0.0, 1.0))
    conn.base.SendGripperCommand(gc)


def _clip_workspace(pos: np.ndarray) -> np.ndarray:
    return np.array([
        np.clip(pos[0], WS_BOUNDS['x'][0] + MARGIN, WS_BOUNDS['x'][1] - MARGIN),
        np.clip(pos[1], WS_BOUNDS['y'][0] + MARGIN, WS_BOUNDS['y'][1] - MARGIN),
        np.clip(pos[2], WS_BOUNDS['z'][0] + MARGIN, WS_BOUNDS['z'][1] - MARGIN),
    ])


def reset_to_home(conn: KinovaConn):
    print('Moving to home position …')
    action = Base_pb2.Action()
    action.name = 'ReplayHome'
    spd = Base_pb2.CartesianSpeed()
    spd.translation = 0.08
    spd.orientation = 12.0
    action.reach_pose.constraint.speed.CopyFrom(spd)
    pose = action.reach_pose.target_pose
    pose.x, pose.y, pose.z                   = HOME_POS
    pose.theta_x, pose.theta_y, pose.theta_z = HOME_ANGLES

    done = threading.Event()
    def _cb(notif):
        if notif.action_event in (Base_pb2.ACTION_END, Base_pb2.ACTION_ABORT):
            done.set()

    conn.base.OnNotificationActionTopic(_cb, Base_pb2.NotificationOptions())
    conn.base.ExecuteAction(action)
    if not done.wait(timeout=30.0):
        conn.base.StopAction()
        raise RuntimeError('Home reset timed out')

    _send_gripper(conn, 0.0)
    time.sleep(0.5)
    print('At home, gripper open.')


def replay(
    conn: KinovaConn,
    action_pos: np.ndarray,
    action_gripper: np.ndarray,
    p_gain: float = 2.0,
    vel_alpha: float = 0.4,
    max_speed: float = 0.50,
):
    mode = Base_pb2.ServoingModeInformation()
    mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
    conn.base.SetServoingMode(mode)
    time.sleep(0.2)

    dt = 1.0 / CONTROL_HZ
    smoothed_vel = np.zeros(3)
    aborted = False

    abort_event = threading.Event()
    def _listen():
        while not abort_event.is_set():
            try:
                if input().strip().lower() == 'q':
                    nonlocal aborted
                    aborted = True
                    abort_event.set()
                    print('Replay aborted.')
            except EOFError:
                break
    threading.Thread(target=_listen, daemon=True).start()

    print(f'Replaying {len(action_pos)} frames @ {CONTROL_HZ:.0f} Hz.')
    print("  'q' + ENTER = abort\n")

    for tgt, grip in zip(action_pos, action_gripper):
        if aborted:
            break

        tick_start = time.monotonic()
        obs = _get_tcp(conn)

        tgt_clipped = _clip_workspace(tgt)
        err         = tgt_clipped - obs
        raw_vel     = p_gain * err
        spd         = float(np.linalg.norm(raw_vel))
        if spd > max_speed:
            raw_vel *= max_speed / spd
        smoothed_vel = vel_alpha * raw_vel + (1.0 - vel_alpha) * smoothed_vel

        _send_twist(conn, smoothed_vel)
        _send_gripper(conn, grip)

        elapsed = time.monotonic() - tick_start
        if elapsed < dt:
            time.sleep(dt - elapsed)

    abort_event.set()
    _send_twist(conn, np.zeros(3))
    print('Replay complete.')


def main():
    parser = argparse.ArgumentParser(description='Replay a recorded episode on the Kinova Gen3')
    parser.add_argument('episode',      help='Path to episode_N.hdf5')
    parser.add_argument('--robot-ip',   default='192.168.1.10')
    parser.add_argument('--p-gain',     type=float, default=2.0)
    parser.add_argument('--vel-alpha',  type=float, default=0.4)
    parser.add_argument('--max-speed',  type=float, default=0.50, metavar='M/S')
    parser.add_argument('--skip-reset', action='store_true', help='Skip home reset')
    args = parser.parse_args()

    with h5py.File(args.episode, 'r') as f:
        T   = int(f.attrs.get('num_frames', 0) or f['action/pose'].shape[0])
        hz  = float(f.attrs.get('collection_rate_hz', 30))
        action_pos     = f['action/pose'][:, :3].astype(np.float64)
        action_gripper = f['action/gripper'][()].astype(np.float64)

    print(f'Loaded {T} frames @ {hz:.0f} Hz  ({T/hz:.1f} s)  ←  {args.episode}')

    with KinovaConn(args.robot_ip) as conn:
        if not args.skip_reset:
            reset_to_home(conn)

        print('Press ENTER to start replay.')
        input()

        replay(conn, action_pos, action_gripper,
               p_gain=args.p_gain,
               vel_alpha=args.vel_alpha,
               max_speed=args.max_speed)


if __name__ == '__main__':
    main()
