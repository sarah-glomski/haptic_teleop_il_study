#!/usr/bin/env python3
"""
DJI Osmo Action 4 Camera Validation Script

Detects the DJI Osmo Action 4 connected via USB-C in webcam (UVC) mode,
opens a live preview window, and reports resolution and frame-rate statistics.
Optionally publishes frames to a ROS2 topic so you can verify the full
pipeline before a data collection session.

Works on Linux and macOS. On macOS cameras are accessed via AVFoundation
(no /dev/video* or v4l2-ctl); USB presence is checked with system_profiler.

────────────────────────────────────────────────────────────────────────────────
Camera setup (do this on the DJI Osmo Action 4 before connecting USB):
  Menu → Settings → Control Method → UVC Camera

Then connect with a USB-C cable and wait ~5 s for enumeration.
────────────────────────────────────────────────────────────────────────────────

Usage:
  # Auto-detect DJI camera and open preview:
  python3 dji_camera_validate.py

  # Specify device index manually (Linux: /dev/videoN, macOS: AVFoundation index):
  python3 dji_camera_validate.py --device 1

  # Set capture resolution (default: 1920x1080):
  python3 dji_camera_validate.py --width 1920 --height 1080

  # Also publish to a ROS2 topic (Linux only, requires sourced ROS2):
  source /opt/ros/jazzy/setup.bash
  python3 dji_camera_validate.py --ros

  # Publish on the data-collector topic instead of the default:
  python3 dji_camera_validate.py --ros --topic /dji_wrist/dji_wrist/color/image_raw

Press Q in the preview window to quit.
"""

import argparse
import subprocess
import sys
import time
from collections import deque

import cv2

# ── Platform ──────────────────────────────────────────────────────────────────
IS_MACOS = sys.platform == 'darwin'
IS_LINUX = sys.platform.startswith('linux')

# OpenCV capture backend — V4L2 on Linux, AVFoundation on macOS
_CV_BACKEND = cv2.CAP_AVFOUNDATION if IS_MACOS else cv2.CAP_V4L2

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


# ── USB detection ─────────────────────────────────────────────────────────────

def find_dji_usb() -> str | None:
    """Return a description string if a DJI USB device is found, else None."""
    if IS_MACOS:
        try:
            out = subprocess.check_output(
                ['system_profiler', 'SPUSBDataType'], text=True, stderr=subprocess.DEVNULL
            )
            lines = out.splitlines()
            for i, line in enumerate(lines):
                if DJI_VENDOR_ID.lower() in line.lower() or 'dji' in line.lower():
                    return line.strip()
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass
        return None
    else:
        try:
            out = subprocess.check_output(['lsusb'], text=True)
        except FileNotFoundError:
            return None
        for line in out.splitlines():
            if DJI_VENDOR_ID in line.lower():
                return line.strip()
        return None


# ── Camera enumeration ────────────────────────────────────────────────────────

def _macos_camera_names() -> dict[int, str]:
    """
    Return {index: name} for cameras listed by system_profiler WITHOUT opening
    any camera device. Index order matches AVFoundation / OpenCV ordering.
    """
    names = {}
    try:
        out = subprocess.check_output(
            ['system_profiler', 'SPCameraDataType'], text=True, stderr=subprocess.DEVNULL
        )
        idx = 0
        for line in out.splitlines():
            line = line.strip()
            if line.endswith(':') and line not in ('Video Cameras:', 'Cameras:'):
                names[idx] = line.rstrip(':')
                idx += 1
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    return names


def list_camera_devices() -> dict[int, str]:
    """
    Return {device_index: description} for all capture devices.
    On macOS this uses system_profiler only — no camera is opened.
    """
    if IS_MACOS:
        # Read names from system_profiler without touching any camera device
        return _macos_camera_names()

    else:
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
            import glob
            for path in sorted(glob.glob('/dev/video*')):
                try:
                    idx = int(path.replace('/dev/video', ''))
                    devices[idx] = path
                except ValueError:
                    pass
        return devices


def find_dji_device_index(devices: dict[int, str]) -> int | None:
    for idx, name in devices.items():
        if any(k in name.lower() for k in ('dji', 'action', 'osmo')):
            return idx
    return None


def probe_device(index: int) -> tuple[bool, int, int, float]:
    """
    Open a single specific device and read one frame. Returns (success, w, h, fps).
    Only call this on the device the user has chosen to open — never during enumeration.
    """
    cap = cv2.VideoCapture(index, _CV_BACKEND)
    if not cap.isOpened():
        return False, 0, 0, 0.0
    ret, _ = cap.read()
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return ret, w, h, fps


# ── Phase 1: hardware check ───────────────────────────────────────────────────

def check_hardware(args) -> int | None:
    platform_label = 'macOS (AVFoundation)' if IS_MACOS else 'Linux (V4L2)'
    print(f'  Platform: {platform_label}')

    print(hdr('USB Detection'))
    dji_usb = find_dji_usb()
    if dji_usb:
        print(f'  {ok(f"DJI device found:  {dji_usb}")}')
    else:
        print(f'  {fail("No DJI USB device found")}')
        print( '    □ Enable UVC mode on the camera:')
        print( '      Menu → Settings → Control Method → UVC Camera')
        print( '    □ Connect USB-C cable and wait ~5 s')
        if IS_MACOS:
            print( '    □ Check with:  system_profiler SPUSBDataType | grep -i dji')
        else:
            print( '    □ Check with:  lsusb | grep 2ca3')

    print(hdr('Camera Devices'))
    devices = list_camera_devices()
    if not devices:
        noun = 'AVFoundation cameras' if IS_MACOS else '/dev/video* devices'
        print(f'  {fail(f"No {noun} found")}')
        if IS_MACOS:
            print( '    □ Grant Terminal camera permission:')
            print( '      System Settings → Privacy & Security → Camera → Terminal ✓')
        return None

    # Print device list — on macOS names come from system_profiler (no camera opened)
    for idx, name in sorted(devices.items()):
        label = f'/dev/video{idx}' if IS_LINUX else f'Camera {idx}'
        if IS_LINUX:
            success, w, h, fps = probe_device(idx)
            cap_str = f'{w}×{h} @ {fps:.0f} fps' if success else 'not readable (metadata node)'
            status = ok if success else warn
        else:
            cap_str = 'listed by system_profiler'
            status = ok
        print(f'  {status(f"{label}  —  {name}  ({cap_str})")}')

    # Choose device — on macOS we never open a camera until run_preview()
    if args.device is not None:
        chosen = args.device
        if chosen not in devices:
            print(f'\n  {warn(f"Device {chosen} not in list — will try anyway")}')
        else:
            print(f'\n  Using --device {chosen} as specified.')
        return chosen

    dji_idx = find_dji_device_index(devices)
    if dji_idx is not None:
        label = f'/dev/video{dji_idx}' if IS_LINUX else f'Camera {dji_idx}'
        print(f'\n  {ok(f"Auto-detected DJI camera at {label}")}')
        return dji_idx

    if IS_LINUX:
        # On Linux we can safely probe to find the first readable device
        for idx in sorted(devices.keys()):
            success, _, _, _ = probe_device(idx)
            if success:
                print(f'\n  {warn(f"DJI not auto-detected by name — falling back to /dev/video{idx}")}')
                print( '    If this is the wrong camera, rerun with  --device N')
                return idx
    elif devices:
        # On macOS, never open a camera blindly — require explicit --device
        print(f'\n  {warn("DJI camera not found by name in the list above.")}')
        print( '    Rerun with  --device N  to specify which camera to open.')
        return None

    print(f'\n  {fail("Could not find a DJI camera device")}')
    return None


# ── Camera open helper ───────────────────────────────────────────────────────

def _open_camera(index: int):
    """
    Open a camera by index, returning an opened cv2.VideoCapture or None.

    On macOS, system_profiler and OpenCV AVFoundation use different index
    numbering (system_profiler may list the same physical camera under two
    entries, e.g. 'Camera' and 'FaceTime HD Camera'). If CAP_AVFOUNDATION
    fails, we fall back to CAP_ANY so OpenCV can negotiate a working backend.
    """
    cap = cv2.VideoCapture(index, _CV_BACKEND)
    if cap.isOpened():
        return cap
    cap.release()

    if IS_MACOS:
        # CAP_AVFOUNDATION failed — try letting OpenCV pick the backend
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            return cap
        cap.release()

    return None


# ── Phase 2: live preview ─────────────────────────────────────────────────────

def run_preview(device_index: int, width: int, height: int, use_ros: bool, ros_topic: str):
    ros_pub  = None
    ros_node = None
    bridge   = None
    if use_ros:
        if IS_MACOS:
            print(f'\n  {warn("--ros is not supported on macOS (no ROS2 install). Skipping.")}')
            use_ros = False
        else:
            try:
                import rclpy
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

    label = f'/dev/video{device_index}' if IS_LINUX else f'Camera {device_index}'
    print(hdr(f'Opening {label}  ({width}×{height})'))

    cap = _open_camera(device_index)
    if cap is None:
        print(f'  {fail(f"Failed to open {label}")}')
        if IS_MACOS:
            print()
            print('  macOS index mismatch note:')
            print('  system_profiler and OpenCV AVFoundation number cameras differently.')
            print('  Your DJI is Camera 2 in system_profiler but OpenCV only sees 0–1.')
            print('  This usually means the built-in webcam occupies two entries in')
            print('  system_profiler (e.g. "Camera" + "FaceTime HD Camera") but only')
            print('  one in OpenCV, shifting all indices after it by 1.')
            print()
            print('  Try:  python3 dji_camera_validate.py --device 1')
            print('  If that opens the wrong camera, try --device 0')
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f'  {ok(f"Capture opened — actual resolution: {actual_w}×{actual_h}")}')
    print( '  Press  Q  in the preview window to quit.\n')

    frame_times: deque = deque(maxlen=30)
    frame_count = 0
    t_start     = time.monotonic()
    window_name = f'DJI Osmo Action 4 — {label}'
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

            rolling_fps = (
                (len(frame_times) - 1) / (frame_times[-1] - frame_times[0])
                if len(frame_times) >= 2 else 0.0
            )

            overlay = frame.copy()
            cv2.putText(overlay, f'{actual_w}x{actual_h}  {rolling_fps:.1f} fps',
                        (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(overlay, f'frame {frame_count}  elapsed {now - t_start:.1f}s',
                        (12, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow(window_name, overlay)

            if use_ros and ros_pub is not None and bridge is not None:
                import rclpy
                img_msg = bridge.cv2_to_imgmsg(frame, encoding='bgr8')
                img_msg.header.stamp    = ros_node.get_clock().now().to_msg()
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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Validate DJI Osmo Action 4 webcam feed (Linux + macOS)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--device', type=int, default=None,
                        help='Camera index to open. Auto-detected if omitted.')
    parser.add_argument('--width',  type=int, default=1920,
                        help='Requested capture width  (default: 1920)')
    parser.add_argument('--height', type=int, default=1080,
                        help='Requested capture height (default: 1080)')
    parser.add_argument('--ros', action='store_true',
                        help='Also publish frames to a ROS2 topic (Linux only)')
    parser.add_argument('--topic', default='/wrist_cam/image_raw',
                        help='ROS2 topic (default: /wrist_cam/image_raw)')
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
