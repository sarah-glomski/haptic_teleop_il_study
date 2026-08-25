#!/usr/bin/env python3
"""Park the arm over the grape-task bowl position so the bowl can be placed.

WHERE THIS NUMBER COMES FROM (2026-08-25). The bowl was never marked on the
table, but every grape episode ends by opening the gripper over it, so the
release points ARE the bowl. Taking the last closed->open gripper transition
of each episode in the two most recent grape collections:

    Task1Collection5   n=14   x 0.542+-0.007   y -0.252+-0.005   z 0.086+-0.010
    Task1Collection6   n=25   x 0.548+-0.008   y -0.242+-0.010   z 0.081+-0.012
    combined           n=39   median (0.547, -0.247, 0.081), mean distance from
                              median 16 mm, worst 37 mm

Older collections agree on x,y to about a centimetre (C1-C4: x 0.553-0.576,
y -0.229 to -0.263) but released ~7 cm higher (z 0.148-0.164), so the drop
height changed between C4 and C5 while the bowl stayed put. C5+C6 describe the
CURRENT rig, so they are what BOWL_XY uses. If the bowl moves, re-run
    python3.12 goto_bowl.py --recompute <collection> [<collection> ...]
and paste the printed values here.

The release point is where the gripper was when the grape dropped, i.e. ABOVE
the bowl's rim, so it is the bowl's centre in x,y but NOT a height to drive to
blind. By default the arm parks at PLACE_Z (well clear of the table) so the
bowl slides underneath; --at-drop-height goes to the recorded release z
instead, which is only useful for checking the number against the real bowl.

    python3.12 goto_bowl.py                 # home, then park above the bowl spot
    python3.12 goto_bowl.py --dry-run       # print target vs current, move nothing
    python3.12 goto_bowl.py --no-home       # skip homing (arm already clear)
    python3.12 goto_bowl.py --at-drop-height

q + ENTER aborts at any point; the workspace box, stall guard and fault latch
in kinova_arm apply exactly as they do during teleop and replay.
"""

import argparse
import glob
import os
import re
import sys
import threading
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kinova_arm import ArmLimits, KinovaArm, HOME_JOINTS_DEG   # noqa: E402

# Median of the 39 final-release points in Task1Collection5 + Task1Collection6.
BOWL_XY = (0.547, -0.247)
DROP_Z = 0.081            # recorded release height (gripper open, grape falls)
PLACE_Z = 0.200           # park height: clear of the table so the bowl fits under
SOURCE = 'Task1Collection5 + Task1Collection6, 39 final-release points, 2026-08-25'

REACH_TOL_M = 0.005       # "arrived" when this close
REACH_TIMEOUT_S = 45.0
HZ = 40.0


def recompute(collections):
    """Re-derive the bowl point from grape collections. Prints, changes nothing."""
    import h5py
    pts = []
    for coll in collections:
        d = coll if os.path.isdir(coll) else os.path.join('demo_data', coll)
        found = 0
        for p in sorted(glob.glob(os.path.join(d, 'episode_*.hdf5')),
                        key=lambda p: int(re.search(r'_(\d+)', p).group(1))):
            with h5py.File(p) as f:
                pos = f['observation/pose'][:, :3]
                g = f['observation/gripper'][:]
            closed = g > 0.5
            opens = [i for i in range(1, len(g)) if closed[i - 1] and not closed[i]]
            if opens:
                pts.append(pos[opens[-1]]); found += 1
        print(f'  {d}: {found} releases')
    if not pts:
        print('No gripper releases found — is this a grape collection?'); return 1
    P = np.array(pts)
    med = np.median(P, axis=0)
    dist = np.linalg.norm(P - med, axis=1)
    print(f'\nn={len(P)}')
    print(f'  mean   {P.mean(axis=0).round(4)}')
    print(f'  median {med.round(4)}      <- use this')
    print(f'  std    {P.std(axis=0).round(4)}')
    print(f'  spread from median: mean {dist.mean()*1000:.0f} mm, worst {dist.max()*1000:.0f} mm')
    print(f'\n    BOWL_XY = ({med[0]:.3f}, {med[1]:.3f})')
    print(f'    DROP_Z = {med[2]:.3f}')
    return 0


def servo_to(arm, target, aborted, label):
    """Velocity-servo the TCP to `target`, holding orientation at home."""
    from scipy.spatial.transform import Rotation as R
    home_rot = arm.limits.home_rotation()
    target = arm.clip_to_workspace(np.asarray(target, dtype=float))
    print(f'{label}: -> [{target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f}]  '
          f'(q + ENTER aborts)')
    arm.reset_velocity_state()
    arm.stall.reset()
    t0 = time.monotonic()
    dt = 1.0 / HZ
    while not aborted.is_set():
        tick = time.monotonic()
        if arm.fault.latched:
            print('Robot faulted — stopping. Clear it in the Kinova web app.')
            return False
        if tick - t0 > REACH_TIMEOUT_S:
            arm.send_zero_twist()
            print(f'Timed out after {REACH_TIMEOUT_S:.0f}s short of the target.')
            return False
        try:
            fb = arm.refresh_feedback()
            cur = arm.tcp_position(fb)
            cur_rot = arm.tcp_rotation(fb)
            arm.stall.update_velocity(cur)
            gap = float(np.linalg.norm(target - cur))
            if gap <= REACH_TOL_M:
                arm.send_zero_twist()
                print(f'  arrived: [{cur[0]:.3f}, {cur[1]:.3f}, {cur[2]:.3f}] '
                      f'({gap*1000:.0f} mm from target)')
                return True
            vel = arm.linear_velocity(cur, target)
            if arm.stall.check(vel, gap):
                arm.send_zero_twist()
                print('Stalled — stopping rather than pushing into something.')
                return False
            tgt_rot, _ = arm.clamp_orientation(home_rot, home_rot)
            ang = arm.angular_velocity(cur_rot, tgt_rot)
            arm.send_twist(vel, ang, base_frame=True)
        except Exception as e:
            kind = arm.fault.classify(e)
            if kind in ('fault', 'transient'):
                continue
            arm.send_zero_twist()
            print(f'Error: {e}')
            return False
        slack = dt - (time.monotonic() - tick)
        if slack > 0:
            time.sleep(slack)
    arm.send_zero_twist()
    print('Aborted.')
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--robot-ip', default='192.168.1.10')
    ap.add_argument('--dry-run', action='store_true',
                    help='print the target and the current TCP, move nothing')
    ap.add_argument('--no-home', action='store_true',
                    help='skip the joint-space home move before travelling')
    ap.add_argument('--at-drop-height', action='store_true',
                    help=f'park at the recorded release height ({DROP_Z} m) '
                         f'instead of the {PLACE_Z} m placing height')
    ap.add_argument('--recompute', nargs='+', metavar='COLLECTION',
                    help='re-derive the bowl point from grape collections and exit')
    a = ap.parse_args()

    if a.recompute:
        return recompute(a.recompute)

    z = DROP_Z if a.at_drop_height else PLACE_Z
    target = np.array([BOWL_XY[0], BOWL_XY[1], z])
    print('=' * 62)
    print('Grape-task bowl position')
    print('=' * 62)
    print(f'  source:  {SOURCE}')
    print(f'  bowl xy: ({BOWL_XY[0]:.3f}, {BOWL_XY[1]:.3f}) m   base frame')
    print(f'  park z:  {z:.3f} m' + ('  (recorded drop height)' if a.at_drop_height
                                     else '  (clear of the table; --at-drop-height for 0.081)'))
    print('=' * 62)

    limits = ArmLimits()
    arm = KinovaArm(robot_ip=a.robot_ip, limits=limits)
    arm.connect()
    try:
        fb = arm.refresh_feedback()
        cur = arm.tcp_position(fb)
        print(f'  current TCP: [{cur[0]:.3f}, {cur[1]:.3f}, {cur[2]:.3f}]  '
              f'-> {np.linalg.norm(target - cur)*1000:.0f} mm to travel')
        clipped = arm.clip_to_workspace(target)
        if not np.allclose(clipped, target):
            print(f'  NOTE: target clipped into the workspace box -> {clipped.round(3)}')
        if a.dry_run:
            print('\n--dry-run: nothing moved.')
            return 0

        aborted = threading.Event()

        def listen():
            while not aborted.is_set():
                try:
                    if input().strip().lower() == 'q':
                        aborted.set(); print('Aborting …')
                except EOFError:
                    return
        threading.Thread(target=listen, daemon=True).start()

        arm.setup_servoing()
        if not a.no_home:
            print('\nHoming first (joint space) so the travel starts from a known pose …')
            arm.twists_suppressed = True
            try:
                if not arm.reach_home_joints():
                    print('Home move did not complete — stopping.')
                    return 3
            finally:
                arm.twists_suppressed = False

        print()
        if not servo_to(arm, target, aborted, 'Travelling to the bowl spot'):
            return 4
        arm.send_zero_twist()
        print('\nParked. The gripper is centred over where the bowl was — '
              'place the bowl directly under it.')
        print('Press ENTER when the bowl is in place (the arm holds position).')
        try:
            input()
        except EOFError:
            pass
        return 0
    finally:
        try:
            arm.send_zero_twist()
        except Exception:
            pass
        arm.disconnect()
        print('Disconnected.')


if __name__ == '__main__':
    sys.exit(main())
