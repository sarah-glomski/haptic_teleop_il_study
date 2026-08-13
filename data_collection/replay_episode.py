#!/usr/bin/env python3
"""
Replay a recorded episode on the Kinova Gen3.

Drives the arm through `action/pose` + `action/gripper` from an episode HDF5,
using the SAME control layer teleop and policy rollout use — kinova_arm — so
the home pose, workspace box, speed caps, orientation clamps, fault latch and
stall guard are whatever those two are running with today.

WHAT THIS USED TO DO, AND WHY IT MATTERED
    This script carried its own private copies of all of that, and they had
    gone stale:

        home          Cartesian (0.350, 0.000, 0.120)   ~22 cm from the real
                                                         joint-space home, and
                                                         IK-ambiguous on a 7-DOF
                                                         arm so the elbow could
                                                         settle anywhere
        workspace     x 0.25-0.45                        collection had long
                                                         since moved to 0.70
        fault latch   none                               a fault would leave it
                                                         chasing a target
        stall guard   none                               a stall would build an
                                                         offset, then dash
        orientation   ignored                            demos are orientation-
                                                         rich; replay flattened
                                                         them to position only

    Measured on episode_0 (2026-08-13): 227 of 227 commanded frames sat outside
    that x ceiling, so a replay would have jammed against a wall for the whole
    run while appearing to work.

REACHABILITY IS CHECKED BEFORE ANYTHING MOVES
    An episode recorded in a workspace wider than the current one cannot be
    replayed faithfully — the clip is silent and the arm simply does something
    else. Frames outside the box are counted up front and the replay refuses to
    start unless you pass --allow-clipping.

Usage:
    python3.12 replay_episode.py demo_data/episode_0.hdf5
    python3.12 replay_episode.py demo_data/episode_0.hdf5 --rate 10
    python3.12 replay_episode.py demo_data/episode_0.hdf5 --no-orientation
    python3.12 replay_episode.py demo_data/episode_0.hdf5 --dry-run

Controls during replay (terminal):
    q + ENTER   abort and stop the arm
"""

import argparse
import math
import sys
import threading
import time

import h5py
import numpy as np
from scipy.spatial.transform import Rotation as R

from kinova_arm import ArmLimits, KinovaArm

# Matches teleop and testing/inference.py. The home POSITION is not defined
# here — reset drives to kinova_arm.HOME_JOINTS_DEG in joint space, the same
# posture data collection returns to.
HOME_TX, HOME_TY, HOME_TZ = -180.0, 0.0, 90.0


# ── Episode ────────────────────────────────────────────────────────────────────

def load_episode(path: str):
    """(pos (T,3), quat (T,4) xyzw, grip (T,), recorded_hz)."""
    with h5py.File(path, 'r') as f:
        pose = f['action/pose'][()].astype(np.float64)      # [xyz, qxyzw]
        grip = f['action/gripper'][()].astype(np.float64)
        hz = float(f.attrs.get('collection_rate_hz', 30) or 30)
    return pose[:, :3], pose[:, 3:7], grip, hz


def report_reachability(pos: np.ndarray, limits: ArmLimits) -> int:
    """Print how much of the episode falls outside the workspace. Returns the
    number of frames that would be clipped."""
    m = limits.hard_margin_m
    bounds = {'x': limits.x, 'y': limits.y, 'z': limits.z}
    outside = np.zeros(len(pos), dtype=bool)

    print('\nReachability against the current workspace:')
    for i, ax in enumerate('xyz'):
        lo, hi = bounds[ax][0] + m, bounds[ax][1] - m
        bad = (pos[:, i] < lo) | (pos[:, i] > hi)
        outside |= bad
        flag = '' if not bad.any() else f'   <-- {bad.sum()} frame(s) clipped'
        print(f'  {ax}  episode {pos[:, i].min():+.3f} .. {pos[:, i].max():+.3f}'
              f'   box [{lo:+.3f}, {hi:+.3f}]{flag}')
    return int(outside.sum())


# ── Replay ─────────────────────────────────────────────────────────────────────

def replay(arm: KinovaArm, pos, quat, grip, hz: float, orientation: bool):
    home_rot = arm.limits.home_rotation(HOME_TX, HOME_TY, HOME_TZ)
    dt = 1.0 / hz
    aborted = threading.Event()

    def _listen():
        while not aborted.is_set():
            try:
                if input().strip().lower() == 'q':
                    aborted.set()
                    print('Aborting …')
            except EOFError:
                return
    threading.Thread(target=_listen, daemon=True).start()

    print(f'\nReplaying {len(pos)} frames @ {hz:.1f} Hz '
          f'({len(pos) / hz:.1f} s).   q + ENTER aborts.\n')

    arm.reset_velocity_state()
    for i, (p, q, g) in enumerate(zip(pos, quat, grip)):
        if aborted.is_set():
            break
        if arm.fault.latched:
            print('Robot faulted — replay stopped. Clear it in the Kinova web '
                  'app and restart.')
            break

        tick = time.monotonic()
        try:
            fb = arm.refresh_feedback()
            cur_pos = arm.tcp_position(fb)
            cur_rot = arm.tcp_rotation(fb)

            # Feed the stall guard every tick — it is the only way to notice
            # the arm has stopped without being told.
            arm.stall.update_velocity(cur_pos)

            target = arm.clip_to_workspace(p)
            vel = arm.linear_velocity(cur_pos, target)

            gap = float(np.linalg.norm(target - cur_pos))
            if arm.stall.check(vel, gap):
                arm.send_zero_twist()
                print('Stalled — replay stopped rather than chasing the backlog.')
                break

            ang = np.zeros(3)
            if orientation:
                tgt_rot, _ = arm.clamp_orientation(R.from_quat(q), home_rot)
                ang = arm.angular_velocity(cur_rot, tgt_rot)

            arm.send_twist(vel, ang, base_frame=True)
            arm.send_gripper(g)

        except Exception as e:
            kind = arm.fault.classify(e)
            if kind == 'fault':
                continue                 # latched; the check above ends the run
            if kind == 'transient':
                continue                 # next tick retries
            print(f'Replay error: {e}')
            arm.send_zero_twist()
            break

        if i and i % int(max(hz, 1)) == 0:
            print(f'  {i}/{len(pos)}  ({i / hz:.1f} s)')

        slack = dt - (time.monotonic() - tick)
        if slack > 0:
            time.sleep(slack)

    aborted.set()
    arm.send_zero_twist()
    print('\nReplay finished.')


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description='Replay a recorded episode on the Kinova Gen3')
    ap.add_argument('episode', help='path to episode_N.hdf5')
    ap.add_argument('--robot-ip', default='192.168.1.10')
    ap.add_argument('--rate', type=float, default=None,
                    help='replay rate in Hz (default: the file\'s '
                         'collection_rate_hz attribute)')
    ap.add_argument('--no-orientation', dest='orientation', action='store_false',
                    help='position + gripper only, leaving the wrist where it '
                         'is. Default tracks the recorded orientation, which is '
                         'what the demos contain.')
    ap.add_argument('--skip-home', action='store_true',
                    help='start from the current pose instead of homing first')
    ap.add_argument('--allow-clipping', action='store_true',
                    help='replay even though some frames fall outside the '
                         'workspace and will be silently clipped')
    ap.add_argument('--dry-run', action='store_true',
                    help='report the episode and its reachability, touch nothing')
    args = ap.parse_args()

    pos, quat, grip, rec_hz = load_episode(args.episode)
    hz = args.rate or rec_hz
    limits = ArmLimits()

    print(f'{args.episode}')
    print(f'  {len(pos)} frames, recorded at {rec_hz:.0f} Hz '
          f'({len(pos) / rec_hz:.1f} s)')
    if args.rate:
        print(f'  replaying at {hz:.1f} Hz ({len(pos) / hz:.1f} s) — overridden')
    print(f'  gripper {grip.min():.2f} .. {grip.max():.2f}')
    print(f'  orientation tracking: {"ON" if args.orientation else "OFF"}')

    clipped = report_reachability(pos, limits)
    if clipped:
        pct = 100.0 * clipped / len(pos)
        print(f'\n  {clipped}/{len(pos)} frames ({pct:.0f}%) fall outside the '
              f'workspace and would be clipped.')
        print('  The arm would not follow the recorded path. This usually means '
              'the episode\n  was recorded under different workspace bounds than '
              'the ones in kinova_arm.py.')
        if not args.allow_clipping and not args.dry_run:
            print('\nRefusing to replay. Pass --allow-clipping to override.')
            sys.exit(1)
    else:
        print('\n  Entire episode is inside the workspace.')

    if args.dry_run:
        print('\n--dry-run: nothing was moved.')
        return

    arm = KinovaArm(robot_ip=args.robot_ip,
                    recovery_hint='restart the replay')
    arm.connect()
    arm.setup_servoing()
    try:
        if not args.skip_home:
            print('\nHoming (joint space, the posture collection returns to) …')
            arm.twists_suppressed = True      # the reach action owns the router
            try:
                reached = arm.reach_home_joints()
            finally:
                arm.twists_suppressed = False
            if not reached:
                print('Did not reach home within the timeout — not replaying.')
                return
            arm.send_gripper(0.0)
            time.sleep(1.0)
            arm.setup_servoing()
            print('At home, gripper open.')

        input('\nPress ENTER to start the replay (Ctrl-C to bail) … ')
        replay(arm, pos, quat, grip, hz, args.orientation)
    except KeyboardInterrupt:
        print('\nInterrupted.')
    finally:
        arm.disconnect()


if __name__ == '__main__':
    main()
