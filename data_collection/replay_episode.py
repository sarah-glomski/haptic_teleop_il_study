#!/usr/bin/env python3
"""
Episode replay with end-to-end latency measurement — Kinova Gen3.

Two latency estimates are computed:

  1. Offline (from the recording):
       Cross-correlate hololens/hand_pose_robot_frame vs observation/pose
       → full pipeline latency at record time (HoloLens sensing → robot state)

  2. Live replay:
       Replay action/pose targets through the same P-loop controller used
       during collection, cross-correlate commanded vs observed TCP positions
       → control-loop latency at replay time (command → robot response)

Usage:
  python3.12 replay_episode.py demo_data/episode_9.hdf5
  python3.12 replay_episode.py demo_data/episode_9.hdf5 --robot-ip 192.168.1.10

Controls during replay (terminal):
  s + ENTER  — mark success (ball in bin)
  q + ENTER  — abort replay early
"""

import argparse
import os
import sys
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

CONTROL_HZ    = 30.0
WATCHDOG_MS   = 200
HOME_POS      = (0.474, 0.02, 0.107)     # m  — matches kinova_hand_controller defaults
HOME_ANGLES   = (-179.3, -0.4, 89.3)     # deg
WS_BOUNDS     = dict(x=(0.30, 0.60), y=(-0.37, 0.37), z=(0.025, 0.30))
MARGIN        = 0.005  # m — hard clip inside workspace walls


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

        self.base    = BaseClient(self._router)
        self.cyclic  = BaseCyclicClient(self._router)
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


# ── Replay ─────────────────────────────────────────────────────────────────────

def replay(
    conn: KinovaConn,
    action_pos: np.ndarray,     # (T, 3)
    action_gripper: np.ndarray, # (T,)
    p_gain: float = 2.0,
    vel_alpha: float = 0.4,
    max_speed: float = 0.50,
) -> dict:
    """
    Run the P-loop velocity controller against the recorded action trajectory.

    Returns a log dict with per-tick timestamps, commanded positions,
    and observed positions.
    """
    mode = Base_pb2.ServoingModeInformation()
    mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
    conn.base.SetServoingMode(mode)
    time.sleep(0.2)

    dt = 1.0 / CONTROL_HZ
    smoothed_vel = np.zeros(3)
    t_cmd, cmd_pos, obs_pos, cmd_grip = [], [], [], []

    # Keyboard listener for success/abort
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
    for i, (tgt, grip) in enumerate(zip(action_pos, action_gripper)):
        if result['aborted']:
            print('Replay aborted.')
            break

        tick_start = time.monotonic()

        # Observe
        obs = _get_tcp(conn)

        # P-loop
        tgt_clipped = _clip_workspace(tgt)
        err         = tgt_clipped - obs
        raw_vel     = p_gain * err
        spd         = float(np.linalg.norm(raw_vel))
        if spd > max_speed:
            raw_vel *= max_speed / spd
        smoothed_vel = vel_alpha * raw_vel + (1.0 - vel_alpha) * smoothed_vel

        _send_twist(conn, smoothed_vel)
        _send_gripper(conn, grip)

        # Log
        t_cmd.append(time.monotonic() - t0)
        cmd_pos.append(tgt_clipped.copy())
        obs_pos.append(obs.copy())
        cmd_grip.append(float(grip))

        # Pace to CONTROL_HZ
        elapsed = time.monotonic() - tick_start
        if elapsed < dt:
            time.sleep(dt - elapsed)

    inp_done.set()
    _send_twist(conn, np.zeros(3))   # safe stop

    if result['success_t']:
        result['success_t'] -= t0    # relative to replay start

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
    Cross-correlate two 1-D signals and return the lag (in samples) such that
    signal_b[t] ≈ signal_a[t - lag].  Positive lag → b lags behind a.
    Returns None if either signal has near-zero variance.
    """
    a = signal_a - signal_a.mean()
    b = signal_b - signal_b.mean()
    if a.std() < 1e-6 or b.std() < 1e-6:
        return None
    corr = correlate(b, a, mode='full')
    return int(np.argmax(corr)) - (len(a) - 1)


def compute_latency(cmd: np.ndarray, obs: np.ndarray, hz: float, label: str) -> dict:
    """
    Compute per-axis cross-correlation lag between cmd (T, 3) and obs (T, 3).

    Reports and returns a summary dict.
    """
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

    print(f'\n── {label} latency ─────────────────────────────')
    for ax, lag in zip(valid_axes, lags):
        print(f'  {ax}: {lag:+d} samples  ({lag/hz*1000:+.1f} ms)')
    print(f'  Mean:   {np.mean(lags):.1f} samples = {mean_lag_s*1000:.1f} ms')
    print(f'  Median: {np.median(lags):.1f} samples = {median_lag_s*1000:.1f} ms')

    return {
        'label':          label,
        'lags_samples':   lags,
        'axes':           valid_axes,
        'mean_lag_s':     mean_lag_s,
        'median_lag_s':   median_lag_s,
        'hz':             hz,
    }


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_results(
    episode_path: str,
    offline: dict,
    replay_log: dict,
    live: dict,
):
    fig = plt.figure(figsize=(16, 10))
    gs  = fig.add_gridspec(3, 4, hspace=0.45, wspace=0.35)
    axes_labels = ['x (m)', 'y (m)', 'z (m)']

    # ── Top row: offline latency (recorded hand vs recorded robot) ─────────────
    hand_pos = offline['hand_pos']   # (T, 3)
    rec_obs  = offline['obs_pos']    # (T, 3)
    rec_t    = offline['t']

    for i, lbl in enumerate(axes_labels):
        ax = fig.add_subplot(gs[0, i])
        ax.plot(rec_t, hand_pos[:, i], label='hand (HoloLens)')
        ax.plot(rec_t, rec_obs[:, i],  label='robot obs', linestyle='--')
        ax.set_title(f'[Offline] {lbl}', fontsize=8)
        ax.set_xlabel('timestep', fontsize=7)
        ax.legend(fontsize=6)
        ax.tick_params(labelsize=7)

    ax_lat = fig.add_subplot(gs[0, 3])
    if offline['latency']:
        lat = offline['latency']
        ax_lat.bar(lat['axes'], [s / lat['hz'] * 1000 for s in lat['lags_samples']],
                   color='steelblue')
        ax_lat.axhline(lat['mean_lag_s'] * 1000, color='red', linestyle='--',
                       label=f'mean {lat["mean_lag_s"]*1000:.1f} ms')
        ax_lat.set_ylabel('lag (ms)')
        ax_lat.set_title('Offline latency\n(HoloLens→robot)', fontsize=8)
        ax_lat.legend(fontsize=7)
    else:
        ax_lat.text(0.5, 0.5, 'N/A', transform=ax_lat.transAxes, ha='center')
        ax_lat.set_title('Offline latency', fontsize=8)
    ax_lat.tick_params(labelsize=7)

    # ── Middle row: replay trajectories ───────────────────────────────────────
    t_rep = replay_log['t']
    cmd   = replay_log['cmd']
    obs   = replay_log['obs']

    for i, lbl in enumerate(axes_labels):
        ax = fig.add_subplot(gs[1, i])
        ax.plot(t_rep, cmd[:, i], label='commanded')
        ax.plot(t_rep, obs[:, i], label='observed', linestyle='--')
        if live and live.get('mean_lag_s'):
            lag_s = live['mean_lag_s']
            ax.plot(t_rep, np.interp(t_rep - lag_s, t_rep, obs[:, i]),
                    label=f'obs −{lag_s*1000:.0f} ms', linestyle=':', alpha=0.7)
        if replay_log.get('success_t'):
            ax.axvline(replay_log['success_t'], color='green', linestyle='--', alpha=0.6,
                       label='success')
        ax.set_title(f'[Replay] {lbl}', fontsize=8)
        ax.set_xlabel('time (s)', fontsize=7)
        ax.legend(fontsize=6)
        ax.tick_params(labelsize=7)

    ax_grip = fig.add_subplot(gs[1, 3])
    ax_grip.plot(t_rep, replay_log['cmd_gripper'], label='commanded')
    ax_grip.set_title('[Replay] Gripper', fontsize=8)
    ax_grip.set_xlabel('time (s)', fontsize=7)
    ax_grip.set_ylim(-0.05, 1.05)
    ax_grip.legend(fontsize=7)
    ax_grip.tick_params(labelsize=7)

    # ── Bottom row: live latency per axis + summary ────────────────────────────
    ax_live_lat = fig.add_subplot(gs[2, :2])
    if live:
        lag_ms = [s / live['hz'] * 1000 for s in live['lags_samples']]
        ax_live_lat.bar(live['axes'], lag_ms, color='darkorange')
        ax_live_lat.axhline(live['mean_lag_s'] * 1000, color='red', linestyle='--',
                            label=f'mean {live["mean_lag_s"]*1000:.1f} ms')
        ax_live_lat.set_ylabel('lag (ms)')
        ax_live_lat.set_title('Live replay latency\n(command→robot)', fontsize=8)
        ax_live_lat.legend(fontsize=7)
    ax_live_lat.tick_params(labelsize=7)

    ax_summary = fig.add_subplot(gs[2, 2:])
    ax_summary.axis('off')
    lines = [
        f'Episode:  {os.path.basename(episode_path)}',
        '',
    ]
    if offline['latency']:
        lat = offline['latency']
        lines += [
            f'Offline (HoloLens→robot):',
            f'  {lat["mean_lag_s"]*1000:.1f} ms  ({lat["mean_lag_s"]*lat["hz"]:.1f} frames)',
        ]
    if live:
        lines += [
            '',
            f'Live replay (cmd→robot):',
            f'  {live["mean_lag_s"]*1000:.1f} ms  ({live["mean_lag_s"]*live["hz"]:.1f} frames)',
        ]
    if replay_log.get('success_t'):
        lines += ['', f'Ball in bin at t = {replay_log["success_t"]:.2f} s']

    ax_summary.text(0.05, 0.95, '\n'.join(lines), transform=ax_summary.transAxes,
                    fontsize=9, verticalalignment='top', fontfamily='monospace')

    fig.suptitle(f'End-to-end latency — {os.path.basename(episode_path)}', fontsize=11)
    plt.tight_layout()

    out_path = episode_path.replace('.hdf5', '_latency.png')
    plt.savefig(out_path, dpi=150)
    print(f'\nFigure saved → {out_path}')
    plt.show()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Replay a recorded episode and measure end-to-end latency'
    )
    parser.add_argument('episode', help='Path to episode_N.hdf5')
    parser.add_argument('--robot-ip',   default='192.168.1.10')
    parser.add_argument('--p-gain',     type=float, default=2.0)
    parser.add_argument('--vel-alpha',  type=float, default=0.4)
    parser.add_argument('--max-speed',  type=float, default=0.50, metavar='M/S')
    parser.add_argument('--skip-reset', action='store_true',
                        help='Skip home reset (robot must already be at home)')
    args = parser.parse_args()

    # ── Load episode ───────────────────────────────────────────────────────────
    with h5py.File(args.episode, 'r') as f:
        T   = int(f.attrs.get('num_frames', 0) or f['action/pose'].shape[0])
        hz  = float(f.attrs.get('collection_rate_hz', 30))
        action_pos     = f['action/pose'][:, :3].astype(np.float64)      # (T, 3)
        action_gripper = f['action/gripper'][()].astype(np.float64)      # (T,)
        obs_pos_rec    = f['observation/pose'][:, :3].astype(np.float64) # (T, 3)
        hand_pos_rec   = f['hololens/hand_pose_robot_frame'][:, :3].astype(np.float64)

    print(f'Loaded {T} frames @ {hz:.0f} Hz  ({T/hz:.1f} s)  ←  {args.episode}')

    # ── Offline latency (recorded hand → recorded robot) ──────────────────────
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

    # ── Live replay ────────────────────────────────────────────────────────────
    print('\n' + '─' * 50)
    print('Connecting to robot for live replay …')
    with KinovaConn(args.robot_ip) as conn:
        if not args.skip_reset:
            reset_to_home(conn)
            print('\nPlace the ball on the table at its starting position, then press ENTER to start replay.')
            input()

        replay_log = replay(
            conn, action_pos, action_gripper,
            p_gain=args.p_gain,
            vel_alpha=args.vel_alpha,
            max_speed=args.max_speed,
        )

    # ── Live latency ───────────────────────────────────────────────────────────
    live_lat = compute_latency(replay_log['cmd'], replay_log['obs'],
                               CONTROL_HZ, label='Live replay (cmd → robot obs)')

    # ── Plot ───────────────────────────────────────────────────────────────────
    plot_results(args.episode, offline, replay_log, live_lat)


if __name__ == '__main__':
    main()
