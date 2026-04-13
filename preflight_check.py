#!/usr/bin/python3.12
"""
Preflight check — HoloLens → Kinova teleop system.

Runs two phases:
  Phase 1 (no ROS needed): USB device presence, robot TCP ping
  Phase 2 (ROS required):  Topic Hz + live values for every key stream

Usage:
  # USB + ping only (before launching anything):
  source /opt/ros/jazzy/setup.bash
  /usr/bin/python3.12 preflight_check.py --usb-only

  # Full check (run while launch_teleop.py is running in another terminal):
  source /opt/ros/jazzy/setup.bash
  /usr/bin/python3.12 preflight_check.py

  # Change robot IP or listen duration:
  /usr/bin/python3.12 preflight_check.py --robot-ip 192.168.1.10 --duration 8
"""

import argparse
import socket
import subprocess
import sys
import threading
import time
from collections import defaultdict

# ── ANSI colours ──────────────────────────────────────────────────────────────
GREEN  = '\033[92m'
RED    = '\033[91m'
YELLOW = '\033[93m'
CYAN   = '\033[96m'
BOLD   = '\033[1m'
RESET  = '\033[0m'

def ok(msg):   return f'{GREEN}✓{RESET} {msg}'
def fail(msg): return f'{RED}✗{RESET} {msg}'
def warn(msg): return f'{YELLOW}⚠{RESET} {msg}'
def hdr(msg):  return f'\n{BOLD}{msg}{RESET}'


# ── Phase 1: USB ───────────────────────────────────────────────────────────────

def check_usb():
    print(hdr('USB Devices'))
    try:
        out = subprocess.check_output(['lsusb'], text=True)
    except FileNotFoundError:
        print(f'  {warn("lsusb not found — skipping USB check")}')
        return

    # ZED M has two USB interfaces; check for either
    zed_ids = {'2b03:f681': 'ZED-M HID', '2b03:f682': 'ZED-M camera',
               '2b03:f880': 'ZED-M', '2b03:f881': 'ZED2', '2b03:f885': 'ZED2i'}
    zed_found = [(vid, name) for vid, name in zed_ids.items() if vid in out]
    if zed_found:
        for vid, name in zed_found:
            print(f'  {ok(f"ZED M detected ({vid} {name})")}')
    else:
        print(f'  {fail("ZED M not detected — check USB cable / power")}')

    # RealSense (Intel vendor 8086, product 0b3a/0b07 etc.)
    if '8086:0b' in out or 'RealSense' in out:
        print(f'  {ok("RealSense detected")}')
    else:
        print(f'  {warn("RealSense not detected (optional for teleop-only)")}')


# ── Phase 1: Robot TCP ping ────────────────────────────────────────────────────

def check_robot_tcp(ip: str, port: int = 10000, timeout: float = 1.0):
    print(hdr('Robot Network'))
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(timeout)
        s.connect((ip, port))
        s.close()
        print(f'  {ok(f"Kinova Gen3 reachable at {ip}:{port}")}')
        return True
    except (socket.timeout, ConnectionRefusedError, OSError) as e:
        print(f'  {fail(f"Kinova not reachable at {ip}:{port} — {e}")}')
        print(f'    Check: robot powered on, e-stop released, same subnet')
        return False


# ── Phase 2: ROS2 topic monitor ───────────────────────────────────────────────

class TopicProbe:
    """Subscribe to one topic, track Hz and last message summary."""
    def __init__(self, node, topic, msg_type, summarise_fn=None):
        self.count  = 0
        self.t0     = None
        self.last   = None
        self._lock  = threading.Lock()
        self._summarise = summarise_fn or (lambda _: '')

        node.create_subscription(msg_type, topic,
                                 self._cb, 10)

    def _cb(self, msg):
        now = time.monotonic()
        with self._lock:
            if self.t0 is None:
                self.t0 = now
            self.count += 1
            self.last  = msg

    def snapshot(self):
        with self._lock:
            n  = self.count
            t0 = self.t0
            last = self.last
        if n == 0 or t0 is None:
            return 0.0, None
        elapsed = time.monotonic() - t0
        hz = (n - 1) / elapsed if elapsed > 0 else 0.0
        return hz, last


def _pose_summary(msg):
    p = msg.pose.position
    return f'x={p.x:+.3f}  y={p.y:+.3f}  z={p.z:+.3f}'

def _float_summary(msg):
    return f'{msg.data:.3f}'

def _bool_summary(msg):
    return f'{BOLD}{GREEN if msg.data else RED}{msg.data}{RESET}'


def check_topics(robot_ip: str, duration: float):
    import rclpy
    from rclpy.node import Node
    from rclpy.executors import SingleThreadedExecutor
    from geometry_msgs.msg import PoseStamped
    from std_msgs.msg import Bool, Float32

    print(hdr(f'ROS2 Topics  (listening {duration:.0f} s — make sure launch_teleop.py is running)'))

    rclpy.init()
    node = Node('preflight_checker')

    # topic name → (probe, label, min_expected_hz)
    specs = [
        ('/hololens/palm/right',   'HoloLens palm/right (raw)',       PoseStamped, _pose_summary,  10.0),
        ('/hololens/thumb/right',  'HoloLens thumb/right (raw)',      PoseStamped, _pose_summary,   5.0),
        ('/hololens/index/right',  'HoloLens index/right (raw)',      PoseStamped, _pose_summary,   5.0),
        ('hand/pose',              'hand/pose (robot-frame)',          PoseStamped, _pose_summary,  10.0),
        ('hand/gripper_cmd',       'hand/gripper_cmd  (0–1)',         Float32,     _float_summary,  5.0),
        ('hand/tracking_active',   'hand/tracking_active',            Bool,        _bool_summary,   1.0),
        ('robot_obs/pose',         'robot_obs/pose (kinova state)',   PoseStamped, _pose_summary,  25.0),
    ]

    probes = {}
    for topic, label, mtype, sumfn, min_hz in specs:
        probes[topic] = (TopicProbe(node, topic, mtype, sumfn), label, min_hz)

    executor = SingleThreadedExecutor()
    executor.add_node(node)

    # Spin in background
    stop_evt = threading.Event()
    def _spin():
        while not stop_evt.is_set():
            executor.spin_once(timeout_sec=0.05)
    spin_thread = threading.Thread(target=_spin, daemon=True)
    spin_thread.start()

    # Progress bar
    print(f'  {CYAN}Collecting…{RESET}', end='', flush=True)
    for _ in range(int(duration * 4)):
        time.sleep(0.25)
        print('.', end='', flush=True)
    print()

    stop_evt.set()
    spin_thread.join(timeout=1.0)

    # Report
    all_ok = True
    tracking_active = False
    hand_pose_hz    = 0.0

    for topic, (probe, label, min_hz) in probes.items():
        hz, last = probe.snapshot()
        if hz >= min_hz * 0.5:   # allow 50 % slack
            summary = probe._summarise(last) if last else ''
            print(f'  {ok(f"{label:<42} {hz:5.1f} Hz  {summary}")}')
            if topic == 'hand/tracking_active' and last:
                tracking_active = last.data
            if topic == 'hand/pose':
                hand_pose_hz = hz
        else:
            hint = _no_data_hint(topic)
            print(f'  {fail(f"{label:<42} no data{hint}")}')
            all_ok = False

    # Diagnostic hints
    print(hdr('Diagnostics'))
    if hand_pose_hz < 5:
        print(f'  {warn("hand/pose not flowing — HoloLens app may not be connected")}')
        print( '    □ Open HoloLens app')
        print( '    □ RosConnector URL must be  ws://<THIS_MACHINE_IP>:9090')
        print( '    □ Press "Arm" button in the app')
    elif not tracking_active:
        print(f'  {warn("hand/pose flowing but tracking_active is False")}')
        print( '    □ Press the "Arm" button in the HoloLens app to enable tracking')
    else:
        print(f'  {ok("HoloLens arm tracking is active")}')

    # Robot pose sanity
    _, robot_last = probes['robot_obs/pose'][0].snapshot()
    if robot_last:
        p = robot_last.pose.position
        print(f'\n  Current robot TCP:  x={p.x:+.3f}  y={p.y:+.3f}  z={p.z:+.3f}')
        _, hand_last = probes['hand/pose'][0].snapshot()
        if hand_last:
            hp = hand_last.pose.position
            print(f'  Current hand pose:  x={hp.x:+.3f}  y={hp.y:+.3f}  z={hp.z:+.3f}')
            print( '  (hand/pose should be close to workspace centre for arm to move)')
    else:
        print(f'  {warn("No robot state — kinova_state_publisher may not be connected")}')

    node.destroy_node()
    try:
        rclpy.shutdown()
    except Exception:
        pass

    return all_ok


def _no_data_hint(topic):
    hints = {
        '/hololens/palm/right':  ' → HoloLens not sending; check rosbridge + app IP',
        'hand/pose':             ' → hololens_hand_node not running or no palm data',
        'hand/tracking_active':  ' → hololens_hand_node not running',
        'robot_obs/pose':        ' → kinova_state_publisher not connected to robot',
    }
    h = hints.get(topic, '')
    return f'  {YELLOW}{h}{RESET}' if h else ''


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Haptic Teleop preflight check')
    parser.add_argument('--robot-ip',  default='192.168.1.10')
    parser.add_argument('--duration',  type=float, default=6.0,
                        help='Seconds to listen for ROS topics (default 6)')
    parser.add_argument('--usb-only',  action='store_true',
                        help='Only check USB + robot ping; skip ROS topic check')
    args = parser.parse_args()

    print()
    print(BOLD + '═' * 60 + RESET)
    print(BOLD + '  Haptic Teleop Preflight Check' + RESET)
    print(BOLD + '═' * 60 + RESET)

    check_usb()
    check_robot_tcp(args.robot_ip)

    if not args.usb_only:
        check_topics(args.robot_ip, args.duration)

    print()


if __name__ == '__main__':
    main()
