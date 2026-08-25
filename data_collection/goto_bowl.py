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

In these collections the operator LOWERED the grape into the bowl before
opening the gripper (confirmed by the operator 2026-08-25), so the release
point is inside the bowl, not above it — which is also why z fell ~7 cm
between C4 and C5 while x,y held. That makes DROP_Z the height to park at
when positioning the bowl: put the gripper there and centre the bowl around
it. It is the default here.

For scale: the first-grasp height (gripper closing on a grape) is
0.035 +- 0.009 and the workspace floor is 0.025, so the table sits near
z ~ 0.03 and DROP_Z is about 5 cm above it. --clearance parks at PLACE_Z
instead, high enough to slide the bowl underneath without touching the
gripper; that number is a chosen clearance, not a measurement.

    python3.12 goto_bowl.py                 # home, park at the release height, then
                                            #   after ENTER show the view comparison
    python3.12 goto_bowl.py --compare-only  # just the view comparison, no motion
    python3.12 goto_bowl.py --clearance     # park high instead, to slide the bowl under
    python3.12 goto_bowl.py --dry-run       # print target vs current, move nothing
    python3.12 goto_bowl.py --no-home       # skip homing (arm already clear)
    python3.12 goto_bowl.py --no-compare    # park only

The last stage answers "is the bowl actually back where it was?" by putting
the median wrist-camera frame at release across all the grape demos next to
the live wrist view, plus a 50/50 blend — one bowl in the blend means the
placement matches, two means it moved.

q + ENTER aborts at any point; the workspace box, stall guard and fault latch
in kinova_arm apply exactly as they do during teleop and replay.
"""

import argparse
import glob
import os
import re
import subprocess
import sys
import threading
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kinova_arm import ArmLimits, KinovaArm, HOME_JOINTS_DEG   # noqa: E402

# Median of the 39 final-release points in Task1Collection5 + Task1Collection6.
BOWL_XY = (0.547, -0.247)
DROP_Z = 0.081            # measured: gripper height at release, INSIDE the bowl
PLACE_Z = 0.200           # chosen, not measured: clearance to slide the bowl under
                          # (demos never went above 0.190; ceiling is 0.250)
SOURCE = 'Task1Collection5 + Task1Collection6, 39 final-release points, 2026-08-25'

GRAPE_COLLECTIONS = ('Task1Collection5', 'Task1Collection6')
WRIST_TOPIC = '/dji_wrist/dji_wrist/color/image_raw'
ENABLE_TOPIC = '/dji_camera/enable'
CHECK_DIR = 'grape_bowl_checks'   # timestamped placement checks land here

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


def reference_release_view(collections=GRAPE_COLLECTIONS):
    """Median wrist-camera image over every episode's release frame.

    One frame per episode, taken at the same event the bowl position came from
    (the last closed->open gripper transition). The pixelwise median across
    episodes keeps what was the same every time — bowl, table, background —
    and washes out what was not (the grape, the operator's hand). That is the
    view the wrist camera SHOULD have now if the bowl is back where it was.
    """
    import h5py
    frames = []
    for coll in collections:
        d = coll if os.path.isdir(coll) else os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'demo_data', coll)
        for p in sorted(glob.glob(os.path.join(d, 'episode_*.hdf5')),
                        key=lambda p: int(re.search(r'_(\d+)', p).group(1))):
            with h5py.File(p) as f:
                if 'images' not in f or 'dji_wrist' not in f['images']:
                    continue
                g = f['observation/gripper'][:]
                closed = g > 0.5
                opens = [i for i in range(1, len(g)) if closed[i - 1] and not closed[i]]
                if not opens:
                    continue
                i = min(opens[-1], f['images/dji_wrist'].shape[0] - 1)
                frames.append(f['images/dji_wrist'][i])        # CHW uint8, RGB
    if not frames:
        return None, 0
    stack = np.stack(frames).astype(np.uint8)                  # (N, 3, H, W)
    med = np.median(stack, axis=0).astype(np.uint8)
    return np.ascontiguousarray(med.transpose(1, 2, 0)), len(frames)   # HWC


def live_wrist_frame(timeout_s=25.0):
    """Grab one frame from the wrist camera, starting its node if needed.

    The DJI node boots idle and only opens the device when something publishes
    True on /dji_camera/enable (that is how the collector arms it per episode),
    so this both starts the node when absent and sends the enable.
    Returns (HWC RGB uint8, note) or (None, reason).
    """
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
    from sensor_msgs.msg import Image
    from std_msgs.msg import Bool

    here = os.path.dirname(os.path.abspath(__file__))
    started = None
    if subprocess.run(['pgrep', '-f', 'dji_camera_node.py'],
                      capture_output=True).returncode != 0:
        print('  starting the DJI camera node …')
        started = subprocess.Popen(
            ['/usr/bin/python3.12', os.path.join(here, 'dji_camera_node.py'),
             '--ros-args', '-r', f'/wrist_cam/image_raw:={WRIST_TOPIC}'],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True)
        time.sleep(4.0)

    rclpy.init()
    node = Node('goto_bowl_wrist_peek')
    got = {}

    sensor_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                            history=HistoryPolicy.KEEP_LAST, depth=1)

    def cb(msg):
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        if msg.encoding in ('bgr8', 'bgr24'):
            arr = arr[:, :, ::-1]
        got['img'] = np.ascontiguousarray(arr)                 # HWC RGB

    node.create_subscription(Image, WRIST_TOPIC, cb, sensor_qos)
    enable_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                            history=HistoryPolicy.KEEP_LAST, depth=1,
                            durability=DurabilityPolicy.TRANSIENT_LOCAL)
    pub = node.create_publisher(Bool, ENABLE_TOPIC, enable_qos)

    t0 = time.monotonic()
    last_enable = 0.0
    try:
        while time.monotonic() - t0 < timeout_s and 'img' not in got:
            if time.monotonic() - last_enable > 1.0:
                pub.publish(Bool(data=True)); last_enable = time.monotonic()
            rclpy.spin_once(node, timeout_sec=0.2)
        img = got.get('img')
    finally:
        # Leave the camera as we found it: the collector expects to own the enable.
        try:
            pub.publish(Bool(data=False)); rclpy.spin_once(node, timeout_sec=0.2)
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        if started is not None:
            started.terminate()
            try:
                started.wait(timeout=5)
            except Exception:
                started.kill()
    if img is None:
        return None, (f'no frame on {WRIST_TOPIC} within {timeout_s:.0f}s — is the '
                      f'DJI powered on? (it auto-powers-off when idle)')
    return img, 'live'


def compare_views(out_path=None, timeout_s=25.0):
    """Side-by-side: the demos' release view vs the wrist camera right now.

    Each run writes a timestamped png into data_collection/grape_bowl_checks/
    so successive checks (nudge the bowl, look again) can be compared instead
    of overwriting one another.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    print('\nBuilding the reference view from the grape demos …')
    ref, n = reference_release_view()
    if ref is None:
        print('  no wrist footage found in the grape collections — cannot compare.')
        return 1
    print(f'  median of {n} release frames')
    print('Grabbing the live wrist view …')
    live, note = live_wrist_frame(timeout_s=timeout_s)
    if live is None:
        print(f'  {note}')
        return 1

    # Match the live frame to the reference geometry for an honest overlay.
    live_r = live
    if live.shape[:2] != ref.shape[:2]:
        import cv2
        live_r = cv2.resize(live, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_AREA)

    fig, axes = plt.subplots(1, 3, figsize=(13, 5.3), dpi=120)
    fig.patch.set_facecolor('white')
    for ax, img, title in (
            (axes[0], ref, f'DEMOS — median of {n} release frames\n(Collections 5+6)'),
            (axes[1], live_r, f'NOW — live wrist camera\n({live.shape[1]}x{live.shape[0]} '
                              f'shown at {ref.shape[1]}x{ref.shape[0]})'),
            (axes[2], (0.5 * ref.astype(float) + 0.5 * live_r.astype(float)).astype(np.uint8),
             'BLEND 50/50\nghosting = the bowl has moved')):
        ax.imshow(img); ax.set_title(title, fontsize=9); ax.axis('off')
    fig.suptitle('Wrist view at grape release: demos vs now — the bowl should sit in the same place',
                 fontsize=11, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    if out_path is None:
        d = os.path.join(os.path.dirname(os.path.abspath(__file__)), CHECK_DIR)
        os.makedirs(d, exist_ok=True)
        out_path = os.path.join(
            d, f'bowl_view_check_{time.strftime("%Y%m%d_%H%M%S")}.png')
    else:
        parent = os.path.dirname(os.path.abspath(out_path))
        if parent:
            os.makedirs(parent, exist_ok=True)
    fig.savefig(out_path, bbox_inches='tight')
    print(f'\nWrote {out_path}')
    subprocess.run(['xdg-open', out_path],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print('If the bowl sits in the same spot in both panels (and the blend shows '
          'one bowl, not two), the placement matches the demos.')
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
    ap.add_argument('--clearance', action='store_true',
                    help=f'park at {PLACE_Z} m (chosen clearance) instead of the '
                         f'measured {DROP_Z} m release height, to slide the bowl under')
    ap.add_argument('--at-drop-height', action='store_true',
                    help='(default behaviour; kept so older notes still work)')
    ap.add_argument('--recompute', nargs='+', metavar='COLLECTION',
                    help='re-derive the bowl point from grape collections and exit')
    ap.add_argument('--compare-only', action='store_true',
                    help='skip all robot motion; just show the wrist-view comparison')
    ap.add_argument('--no-compare', action='store_true',
                    help='park only, skip the wrist-view comparison at the end')
    ap.add_argument('--out', metavar='PNG', default=None,
                    help=f'write the comparison here instead of a timestamped '
                         f'file in {CHECK_DIR}/')
    a = ap.parse_args()

    if a.recompute:
        return recompute(a.recompute)
    if a.compare_only:
        return compare_views(out_path=a.out)

    z = PLACE_Z if a.clearance else DROP_Z
    target = np.array([BOWL_XY[0], BOWL_XY[1], z])
    print('=' * 62)
    print('Grape-task bowl position')
    print('=' * 62)
    print(f'  source:  {SOURCE}')
    print(f'  bowl xy: ({BOWL_XY[0]:.3f}, {BOWL_XY[1]:.3f}) m   base frame')
    print(f'  park z:  {z:.3f} m' + ('  (chosen clearance, ~17 cm above the table)'
                                     if a.clearance else
                                     '  (MEASURED release height — gripper sat here '
                                     'inside the bowl; --clearance for 0.200)'))
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
        print('\nParked. The gripper is where it sat when grapes were released — '
              'centre the bowl around it.' if not a.clearance else
              '\nParked clear of the table — slide the bowl underneath the gripper.')
        if a.no_compare:
            print('Press ENTER when done (the arm holds position).')
            try:
                input()
            except EOFError:
                pass
            return 0
        print('Press ENTER once the bowl is placed — then the wrist view is '
              'compared against the demos.')
        try:
            input()
        except EOFError:
            print('  (no terminal input; comparing now)')
        return compare_views(out_path=a.out)
    finally:
        try:
            arm.send_zero_twist()
        except Exception:
            pass
        arm.disconnect()
        print('Disconnected.')


if __name__ == '__main__':
    sys.exit(main())
