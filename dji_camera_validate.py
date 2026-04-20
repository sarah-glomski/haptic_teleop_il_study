#!/usr/bin/env python3
"""
DJI Osmo Action 4 Camera Validation Script

Detects the DJI Osmo Action 4 connected via USB-C in webcam (UVC) mode,
opens a live preview window, and reports resolution and frame-rate statistics.
Optionally publishes frames to a ROS2 topic so you can verify the full
pipeline before a data collection session.

────────────────────────────────────────────────────────────────────────────────
Camera setup (do this on the DJI Osmo Action 4 before connecting USB):
  Menu → Settings → Control Method → UVC Camera

Then connect the camera to the computer with a USB-C cable.
On Linux it will appear as /dev/videoN (usually /dev/video0 or /dev/video2).
────────────────────────────────────────────────────────────────────────────────

Usage:
  # Auto-detect DJI camera and open preview:
  python3 dji_camera_validate.py

  # Specify V4L2 device index manually:
  python3 dji_camera_validate.py --device 2

  # Set capture resolution (default: 1920x1080):
  python3 dji_camera_validate.py --width 1920 --height 1080

  # Also publish to a ROS2 topic (requires sourced ROS2):
  source /opt/ros/jazzy/setup.bash
  python3 dji_camera_validate.py --ros

  # Change the published topic (default: /wrist_cam/image_raw):
  python3 dji_camera_validate.py --ros --topic /rs_wrist/rs_wrist/color/image_raw

Press Q in the preview window to quit.
"""

import argparse
import subprocess
import sys
import time
from collections import deque

import cv2

# ── DJI USB vendor ID ─────────────────────────────────────────────────────────
DJI_VENDOR_ID = '2ca3'

GREEN  = '\033[92m'
RED    = '\033[91m'
YELLOW = '\033[93m'
BOLD   = '\033[1m'
RESET  = '\033[0m'

def ok(msg):   return f'{GREEN}✓{RESET} {msg}'
def fail(msg): return f'{RED}✗{RESET} {msg}'
def warn(msg): return f'{YELLOW}⚠{RESET} {msg}'
def hdr(msg):  return f'\n{BOLD}{msg}{RESET}'


# ── Device detection ──────────────────────────────────────────────────────────

def find_dji_usb() -> str | None:
    """Return the lsusb line for the DJI camera, or None if not found."""
    try:
        out = subprocess.check_output(['lsusb'], text=True)
    except FileNotFoundError:
        return None
    for line in out.splitlines():
        if DJI_VENDOR_ID in line.lower():
            return line.strip()
    return None


def list_v4l2_devices() -> dict[int, str]:
    """Return {device_index: description} for V4L2 video capture devices."""
    devices = {}
    try:
        out = subprocess.check_output(
            ['v4l2-ctl', '--list-devices'], text=True, stderr=subprocess.DEVNULL
        )
        current_name = ''
        for line in out.splitlines():
            if not line.startswith('\t'):
                current_name = line.strip().rstrip(':')
            elif '/dev/video' in line:
                dev = line.strip()
                try:
                    idx = int(dev.replace('/dev/video', ''))
                    devices[idx] = current_name
                except ValueError:
                    pass
    except (FileNotFoundError, subprocess.CalledProcessError):
        # v4l2-ctl not installed — fall back to scanning /dev/video*
        import glob
        for path in sorted(glob.glob('/dev/video*')):
            try:
                idx = int(path.replace('/dev/video', ''))
                devices[idx] = path
            except ValueError:
                pass
    return devices


def find_dji_device_index(devices: dict[int, str]) -> int | None:
    """Try to match a V4L2 device to the DJI camera by name."""
    for idx, name in devices.items():
        if 'dji' in name.lower() or 'action' in name.lower() or 'osmo' in name.lower():
            return idx
    return None


def probe_device(index: int) -> tuple[bool, int, int, float]:
    """
    Try to open /dev/videoN and read one frame.
    Returns (success, width, height, fps).
    """
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not cap.isOpened():
        return False, 0, 0, 0.0
    ok_frame, _ = cap.read()
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return ok_frame, w, h, fps


# ── Phase 1: hardware check ────────────────────────────────────────────────────

def check_hardware(args) -> int | None:
    """Print hardware detection results and return the V4L2 device index to use."""
    print(hdr('USB Detection'))
    dji_usb = find_dji_usb()
    if dji_usb:
        print(f'  {ok(f"DJI device found:  {dji_usb}")}')
    else:
        print(f'  {fail("No DJI USB device found")}')
        print( '    □ Make sure the camera is in UVC mode:')
        print( '      Menu → Settings → Control Method → UVC Camera')
        print( '    □ Connect USB-C cable and wait ~5 s for enumeration')
        print( '    □ Check with:  lsusb | grep 2ca3')

    print(hdr('V4L2 Video Devices'))
    devices = list_v4l2_devices()
    if not devices:
        print(f'  {fail("No /dev/video* devices found")}')
        return None

    for idx, name in sorted(devices.items()):
        success, w, h, fps = probe_device(idx)
        status = ok if success else warn
        cap_str = f'{w}×{h} @ {fps:.0f} fps' if success else 'not readable (metadata/non-capture node)'
        print(f'  {status(f"/dev/video{idx}  —  {name}  ({cap_str})")}')

    # Decide which device to use
    if args.device is not None:
        chosen = args.device
        if chosen not in devices:
            print(f'\n  {warn(f"Device index {chosen} not in device list — will try anyway")}')
        else:
            print(f'\n  Using --device {chosen} as specified.')
        return chosen

    dji_idx = find_dji_device_index(devices)
    if dji_idx is not None:
        print(f'\n  {ok(f"Auto-detected DJI camera at /dev/video{dji_idx}")}')
        return dji_idx

    # Fall back to first readable device
    for idx in sorted(devices.keys()):
        success, _, _, _ = probe_device(idx)
        if success:
            print(f'\n  {warn(f"DJI not auto-detected by name — falling back to /dev/video{idx}")}')
            print( '    If this is the wrong camera, rerun with  --device N')
            return idx

    print(f'\n  {fail("Could not find a readable capture device")}')
    return None


# ── Phase 2: live preview ─────────────────────────────────────────────────────

def run_preview(device_index: int, width: int, height: int, use_ros: bool, ros_topic: str):
    """Open the camera and stream a live preview window. Optionally publish ROS2."""

    # ── ROS2 setup ───────────────────────────────────────────────────────────
    ros_pub  = None
    ros_node = None
    bridge   = None
    if use_ros:
        try:
            import rclpy
            from rclpy.node import Node
            from sensor_msgs.msg import Image
            from cv_bridge import CvBridge

            rclpy.init()
            ros_node = rclpy.create_node('dji_camera_validate')
            ros_pub  = ros_node.create_publisher(Image, ros_topic, 10)
            bridge   = CvBridge()
            print(f'\n  {ok(f"ROS2 publisher ready on  {ros_topic}")}')
        except ImportError as e:
            print(f'\n  {warn(f"ROS2 unavailable ({e}) — running without publishing")}')
            use_ros = False

    # ── Open capture ─────────────────────────────────────────────────────────
    print(hdr(f'Opening /dev/video{device_index}  ({width}×{height})'))
    cap = cv2.VideoCapture(device_index, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f'  {fail(f"Failed to open /dev/video{device_index}")}')
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # minimise latency

    actual_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f'  {ok(f"Capture opened — actual resolution: {actual_w}×{actual_h}")}')
    print( '  Press  Q  in the preview window to quit.\n')

    # ── Streaming loop ────────────────────────────────────────────────────────
    frame_times: deque = deque(maxlen=30)
    frame_count = 0
    t_start     = time.monotonic()
    window_name = f'DJI Osmo Action 4 — /dev/video{device_index}'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print(f'  {warn("Frame grab failed — camera disconnected?")}')
                break

            now = time.monotonic()
            frame_times.append(now)
            frame_count += 1

            # Compute rolling FPS over last 30 frames
            if len(frame_times) >= 2:
                rolling_fps = (len(frame_times) - 1) / (frame_times[-1] - frame_times[0])
            else:
                rolling_fps = 0.0

            # Overlay stats
            elapsed = now - t_start
            overlay = frame.copy()
            cv2.putText(overlay, f'{actual_w}x{actual_h}  {rolling_fps:.1f} fps',
                        (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(overlay, f'frame {frame_count}  elapsed {elapsed:.1f}s',
                        (12, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow(window_name, overlay)

            # Publish to ROS2 if requested
            if use_ros and ros_pub is not None and bridge is not None:
                img_msg = bridge.cv2_to_imgmsg(frame, encoding='bgr8')
                img_msg.header.stamp = ros_node.get_clock().now().to_msg()
                img_msg.header.frame_id = 'wrist_cam'
                ros_pub.publish(img_msg)
                rclpy.spin_once(ros_node, timeout_sec=0)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if use_ros and ros_node is not None:
            ros_node.destroy_node()
            try:
                import rclpy
                rclpy.shutdown()
            except Exception:
                pass

    # ── Final stats ──────────────────────────────────────────────────────────
    elapsed = time.monotonic() - t_start
    avg_fps = frame_count / elapsed if elapsed > 0 else 0.0
    print(hdr('Session Summary'))
    print(f'  Frames captured : {frame_count}')
    print(f'  Total time      : {elapsed:.1f} s')
    print(f'  Average FPS     : {avg_fps:.1f}')
    if avg_fps < 15:
        print(f'  {warn("FPS is low — check USB bandwidth (use USB 3 port) and camera settings")}')
    else:
        print(f'  {ok("FPS looks healthy")}')


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Validate DJI Osmo Action 4 webcam feed',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--device', type=int, default=None,
                        help='V4L2 device index (e.g. 0 for /dev/video0). '
                             'Auto-detected if omitted.')
    parser.add_argument('--width',  type=int, default=1920,
                        help='Requested capture width  (default: 1920)')
    parser.add_argument('--height', type=int, default=1080,
                        help='Requested capture height (default: 1080)')
    parser.add_argument('--ros', action='store_true',
                        help='Also publish frames to a ROS2 topic (requires '
                             'sourced ROS2 environment)')
    parser.add_argument('--topic', default='/wrist_cam/image_raw',
                        help='ROS2 topic to publish on (default: /wrist_cam/image_raw). '
                             'Use /rs_wrist/rs_wrist/color/image_raw to slot directly '
                             'into the existing data collector pipeline.')
    args = parser.parse_args()

    print()
    print(BOLD + '═' * 60 + RESET)
    print(BOLD + '  DJI Osmo Action 4 — Camera Validation' + RESET)
    print(BOLD + '═' * 60 + RESET)

    device_index = check_hardware(args)
    if device_index is None:
        print(f'\n{fail("Aborting — no usable camera device found.")}\n')
        sys.exit(1)

    run_preview(device_index, args.width, args.height, args.ros, args.topic)
    print()


if __name__ == '__main__':
    main()
