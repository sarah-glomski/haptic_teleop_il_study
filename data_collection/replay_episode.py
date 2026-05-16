#!/usr/bin/env python3
"""
Episode replay with end-to-end latency tuning — Kinova Gen3.

Replays a recorded episode with a forward time offset applied to the action
sequence to pre-compensate for system latency.  Iterates until the robot
successfully completes the task (ball in bin).

Latency measurements:
  Offline  — cross-correlate hololens/hand_pose_robot_frame vs observation/pose
             from the HDF5 file (full pipeline latency at record time)
  Live     — cross-correlate commanded vs observed TCP during each replay attempt
             (control-loop latency at replay time)

Time-offset tuning:
  At step t the robot is commanded action[t + offset] instead of action[t],
  so the robot reaches the "correct" position earlier, compensating for lag.
  The first offset is seeded from the offline latency measurement.
  After each failed attempt the measured live lag is added to the current
  offset and offered as the suggested next value.

Usage:
  python3.12 replay_episode.py demo_data/episode_9.hdf5
  python3.12 replay_episode.py demo_data/episode_9.hdf5 --offset 8
  python3.12 replay_episode.py demo_data/episode_9.hdf5 --robot-ip 192.168.1.10

Controls during replay (terminal):
  s + ENTER  — mark success (ball in bin)
  q + ENTER  — abort replay early
"""

import argparse
import os
import threading
import time

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import correlate

from kortex_api.TCPTransport import TCPTransport
from kortex_api.RouterClient import RouterClient
from kortex_api.SessionManager import SessionManager
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.messages import Session_pb2, Base_pb2

CONTROL_HZ  = 30.0
WATCHDOG_MS = 200
HOME_POS    = (0.474, 0.02, 0.107)   # m
HOME_ANGLES = (-179.3, -0.4, 89.3)   # deg
WS_BOUNDS   = dict(x=(0.30, 0.60), y=(-0.37, 0.37), z=(0.025, 0.30))
MARGIN      = 0.005  # m


# ── Kortex connection ──────────────────────────────────────────────────────────

class KinovaConn:
    def __init__(self, ip: str, username: str = 'admin', password: str = 'admin'):
        self._transport = TCPTransport()
        self._router = RouterClient(self._transport, lambda ex: print(f'[WARN] kortex: {ex}'))
        self._transport.connect(ip, 10000)

        info = Session_pb2.CreateSessionInfo()
        info.username = self.username = username
        info.password = self.password = password
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


# ── Low-level commands ─────────────────────────────────────────────────────────

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


# ── Reset to home ──────────────────────────────────────────────────────────────

def reset_to_home(conn: KinovaConn):
    print('Moving to home position …')
    action = Base_pb2.Action()
    action.name = 'ReplayHome'
    spd = Base_pb2.CartesianSpeed()
    spd.translation = 0.08
    spd.orientation = 12.0
    action.reach_pose.constraint.speed.CopyFrom(spd)
    pose = action.reach_pose.target_pose
    pose.x, pose.y, pose.z             = HOME_POS
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


# ── Time-offset application ────────────────────────────────────────────────────

def apply_offset(
    action_pos: np.ndarray,
    action_gripper: np.ndarray,
    offset: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Shift the entire action sequence (position + gripper) forward by `offset`
    samples.  At step t the robot receives action[t + offset], so it reaches
    each waypoint offset frames earlier, compensating for system latency.
    """
    if offset == 0:
        return action_pos, action_gripper
    idx = np.clip(np.arange(len(action_pos)) + offset, 0, len(action_pos) - 1)
    return action_pos[idx], action_gripper[idx]


# ── Replay ─────────────────────────────────────────────────────────────────────

def replay(
    conn: KinovaConn,
    action_pos: np.ndarray,
    action_gripper: np.ndarray,
    p_gain: float = 2.0,
    vel_alpha: float = 0.4,
    max_speed: float = 0.50,
) -> dict:
    mode = Base_pb2.ServoingModeInformation()
    mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
    conn.base.SetServoingMode(mode)
    time.sleep(0.2)

    dt = 1.0 / CONTROL_HZ
    smoothed_vel = np.zeros(3)
    t_cmd, cmd_pos, obs_pos, cmd_grip = [], [], [], []

    result   = {'success_t': None, 'aborted': False}
    inp_done = threading.Event()

    def _listen():
        while not inp_done.is_set():
            try:
                key = input().strip().lower()
                if key == 's':
                    result['success_t'] = time.monotonic()
                    print('[✓] Success marked!')
                elif key == 'q':
                    result['aborted'] = True
                    inp_done.set()
            except EOFError:
                break
    threading.Thread(target=_listen, daemon=True).start()

    print(f'Replaying {len(action_pos)} frames @ {CONTROL_HZ:.0f} Hz.')
    print("  's' + ENTER = mark success    'q' + ENTER = abort\n")

    t0 = time.monotonic()
    for tgt, grip in zip(action_pos, action_gripper):
        if result['aborted']:
            print('Replay aborted.')
            break

        tick_start = time.monotonic()
        obs        = _get_tcp(conn)

        tgt_clipped = _clip_workspace(tgt)
        err         = tgt_clipped - obs
        raw_vel     = p_gain * err
        spd         = float(np.linalg.norm(raw_vel))
        if spd > max_speed:
            raw_vel *= max_speed / spd
        smoothed_vel = vel_alpha * raw_vel + (1.0 - vel_alpha) * smoothed_vel

        _send_twist(conn, smoothed_vel)
        _send_gripper(conn, grip)

        t_cmd.append(time.monotonic() - t0)
        cmd_pos.append(tgt_clipped.copy())
        obs_pos.append(obs.copy())
        cmd_grip.append(float(grip))

        elapsed = time.monotonic() - tick_start
        if elapsed < dt:
            time.sleep(dt - elapsed)

    inp_done.set()
    _send_twist(conn, np.zeros(3))

    if result['success_t']:
        result['success_t'] -= t0

    return {
        't':           np.array(t_cmd),
        'cmd':         np.array(cmd_pos),
        'obs':         np.array(obs_pos),
        'cmd_gripper': np.array(cmd_grip),
        **result,
    }


# ── Latency calculation ────────────────────────────────────────────────────────

def _xcorr_lag(signal_a: np.ndarray, signal_b: np.ndarray) -> int | None:
    """
    Return lag (samples) such that signal_b[t] ≈ signal_a[t - lag].
    Positive → b lags behind a.  None if either signal has near-zero variance.
    """
    a = signal_a - signal_a.mean()
    b = signal_b - signal_b.mean()
    if a.std() < 1e-6 or b.std() < 1e-6:
        return None
    corr = correlate(b, a, mode='full')
    return int(np.argmax(corr)) - (len(a) - 1)


def compute_latency(cmd: np.ndarray, obs: np.ndarray, hz: float, label: str) -> dict:
    axis_names = ['x', 'y', 'z']
    lags, valid_axes = [], []
    for i, ax in enumerate(axis_names):
        lag = _xcorr_lag(cmd[:, i], obs[:, i])
        if lag is not None:
            lags.append(lag)
            valid_axes.append(ax)

    if not lags:
        print(f'[{label}] Insufficient motion to compute latency.')
        return {}

    mean_lag_s   = np.mean(lags) / hz
    median_lag_s = np.median(lags) / hz

    print(f'\n── {label} ─────────────────────────────')
    for ax, lag in zip(valid_axes, lags):
        print(f'  {ax}: {lag:+d} samples  ({lag/hz*1000:+.1f} ms)')
    print(f'  Mean:   {np.mean(lags):.1f} samples = {mean_lag_s*1000:.1f} ms')
    print(f'  Median: {np.median(lags):.1f} samples = {median_lag_s*1000:.1f} ms')

    return {
        'label':        label,
        'lags_samples': lags,
        'axes':         valid_axes,
        'mean_lag_s':   mean_lag_s,
        'median_lag_s': median_lag_s,
        'hz':           hz,
    }


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_results(
    episode_path: str,
    offline: dict,
    attempts: list[dict],   # each entry: {offset, replay_log, live_lat}
):
    n_attempts = len(attempts)
    fig = plt.figure(figsize=(16, 4 + 3 * n_attempts))
    n_rows = 1 + n_attempts          # 1 offline row + 1 row per attempt
    gs = fig.add_gridspec(n_rows, 4, hspace=0.5, wspace=0.35)

    # ── Offline row ────────────────────────────────────────────────────────────
    hand_pos = offline['hand_pos']
    rec_obs  = offline['obs_pos']
    rec_t    = offline['t']
    for i, lbl in enumerate(['x (m)', 'y (m)', 'z (m)']):
        ax = fig.add_subplot(gs[0, i])
        ax.plot(rec_t, hand_pos[:, i], label='hand')
        ax.plot(rec_t, rec_obs[:, i],  label='robot', linestyle='--')
        ax.set_title(f'[Offline] {lbl}', fontsize=8)
        ax.set_xlabel('time (s)', fontsize=7)
        ax.legend(fontsize=6)
        ax.tick_params(labelsize=7)

    ax_lat = fig.add_subplot(gs[0, 3])
    if offline['latency']:
        lat = offline['latency']
        ax_lat.bar(lat['axes'], [s / lat['hz'] * 1000 for s in lat['lags_samples']],
                   color='steelblue')
        ax_lat.axhline(lat['mean_lag_s'] * 1000, color='red', linestyle='--',
                       label=f"mean {lat['mean_lag_s']*1000:.0f} ms")
        ax_lat.set_ylabel('lag (ms)')
        ax_lat.set_title('Offline latency\n(HoloLens→robot)', fontsize=8)
        ax_lat.legend(fontsize=7)
    else:
        ax_lat.text(0.5, 0.5, 'N/A', transform=ax_lat.transAxes, ha='center')
        ax_lat.set_title('Offline latency', fontsize=8)
    ax_lat.tick_params(labelsize=7)

    # ── One row per attempt ────────────────────────────────────────────────────
    for row, attempt in enumerate(attempts, start=1):
        offset    = attempt['offset']
        log       = attempt['replay_log']
        live      = attempt['live_lat']
        succeeded = bool(log.get('success_t'))
        tag       = f'Attempt {row}  offset={offset} ({offset/CONTROL_HZ*1000:.0f} ms)' \
                    + (' ✓' if succeeded else '')

        for i, lbl in enumerate(['x (m)', 'y (m)', 'z (m)']):
            ax = fig.add_subplot(gs[row, i])
            ax.plot(log['t'], log['cmd'][:, i], label='cmd')
            ax.plot(log['t'], log['obs'][:, i], label='obs', linestyle='--')
            if log.get('success_t'):
                ax.axvline(log['success_t'], color='green', linestyle='--', alpha=0.7)
            ax.set_title(f'[{tag}] {lbl}', fontsize=7)
            ax.set_xlabel('time (s)', fontsize=7)
            ax.legend(fontsize=6)
            ax.tick_params(labelsize=7)

        ax_s = fig.add_subplot(gs[row, 3])
        if live:
            lag_ms = [s / live['hz'] * 1000 for s in live['lags_samples']]
            ax_s.bar(live['axes'], lag_ms, color='darkorange')
            ax_s.axhline(live['mean_lag_s'] * 1000, color='red', linestyle='--',
                         label=f"mean {live['mean_lag_s']*1000:.0f} ms")
            ax_s.set_ylabel('lag (ms)')
            ax_s.legend(fontsize=7)
        ax_s.set_title(f'[{tag}]\nLive lag', fontsize=7)
        ax_s.tick_params(labelsize=7)

    fig.suptitle(f'Latency tuning — {os.path.basename(episode_path)}', fontsize=11)
    plt.tight_layout()

    out_path = episode_path.replace('.hdf5', '_latency.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'\nFigure saved → {out_path}')
    plt.show()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Replay episode with time-offset tuning to compensate latency'
    )
    parser.add_argument('episode',       help='Path to episode_N.hdf5')
    parser.add_argument('--robot-ip',    default='192.168.1.10')
    parser.add_argument('--p-gain',      type=float, default=2.0)
    parser.add_argument('--vel-alpha',   type=float, default=0.4)
    parser.add_argument('--max-speed',   type=float, default=0.50, metavar='M/S')
    parser.add_argument('--offset',      type=int,   default=None,
                        help='Starting time offset in samples (default: auto from offline lag)')
    parser.add_argument('--skip-reset',  action='store_true',
                        help='Skip home reset on the first attempt')
    args = parser.parse_args()

    # ── Load episode ───────────────────────────────────────────────────────────
    with h5py.File(args.episode, 'r') as f:
        T   = int(f.attrs.get('num_frames', 0) or f['action/pose'].shape[0])
        hz  = float(f.attrs.get('collection_rate_hz', 30))
        action_pos     = f['action/pose'][:, :3].astype(np.float64)
        action_gripper = f['action/gripper'][()].astype(np.float64)
        obs_pos_rec    = f['observation/pose'][:, :3].astype(np.float64)
        hand_pos_rec   = f['hololens/hand_pose_robot_frame'][:, :3].astype(np.float64)

    print(f'Loaded {T} frames @ {hz:.0f} Hz  ({T/hz:.1f} s)  ←  {args.episode}')

    # ── Offline latency ────────────────────────────────────────────────────────
    print('\nComputing offline latency from recorded data …')
    rec_t = np.arange(T) / hz
    offline_lat = compute_latency(hand_pos_rec, obs_pos_rec, hz,
                                  label='Offline (HoloLens hand → robot obs)')
    offline = {
        'hand_pos': hand_pos_rec,
        'obs_pos':  obs_pos_rec,
        't':        rec_t,
        'latency':  offline_lat,
    }

    # Seed offset from offline lag if not given explicitly
    if args.offset is not None:
        offset = args.offset
    elif offline_lat:
        offset = round(offline_lat['mean_lag_s'] * CONTROL_HZ)
        print(f'\nAuto-seeding offset from offline lag: {offset} samples '
              f'({offset/CONTROL_HZ*1000:.0f} ms)')
    else:
        offset = 0

    # ── Tuning loop ────────────────────────────────────────────────────────────
    print('\n' + '─' * 50)
    print('Connecting to robot …')
    attempts = []

    with KinovaConn(args.robot_ip) as conn:
        attempt_num = 0
        first       = True

        while True:
            attempt_num += 1
            print(f'\n══ Attempt {attempt_num}  |  offset = {offset} samples '
                  f'({offset/CONTROL_HZ*1000:.0f} ms) ══')

            if not (first and args.skip_reset):
                reset_to_home(conn)
            first = False

            print('Place the ball in the gripper, '
                  'then press ENTER to start replay.')
            input()

            pos_shifted, grip_shifted = apply_offset(action_pos, action_gripper, offset)

            replay_log = replay(
                conn, pos_shifted, grip_shifted,
                p_gain=args.p_gain,
                vel_alpha=args.vel_alpha,
                max_speed=args.max_speed,
            )

            live_lat = compute_latency(
                replay_log['cmd'], replay_log['obs'],
                CONTROL_HZ, label=f'Live (attempt {attempt_num})'
            )

            attempts.append({
                'offset':     offset,
                'replay_log': replay_log,
                'live_lat':   live_lat,
            })

            if replay_log.get('success_t'):
                print(f'\n✓ Success!  offset = {offset} samples '
                      f'({offset/CONTROL_HZ*1000:.0f} ms)  '
                      f'ball in bin at t = {replay_log["success_t"]:.2f} s')
                break

            if replay_log['aborted']:
                print('Aborted — exiting tuning loop.')
                break

            # Suggest next offset
            if live_lat:
                suggested = offset + round(live_lat['mean_lag_s'] * CONTROL_HZ)
                print(f'\nResidual live lag: {live_lat["mean_lag_s"]*1000:.0f} ms  '
                      f'→ suggested next offset: {suggested} samples '
                      f'({suggested/CONTROL_HZ*1000:.0f} ms)')
            else:
                suggested = offset

            print(f'\nOptions:  r = retry with suggested offset ({suggested})'
                  f'  |  <N> = custom offset  |  q = quit')
            resp = input('> ').strip().lower()
            if resp == 'q':
                print('Exiting tuning loop.')
                break
            elif resp == 'r' or resp == '':
                offset = suggested
            elif resp.lstrip('-').isdigit():
                offset = int(resp)
            # else: keep current offset and retry

    # ── Plot all attempts ──────────────────────────────────────────────────────
    if attempts:
        plot_results(args.episode, offline, attempts)


if __name__ == '__main__':
    main()
