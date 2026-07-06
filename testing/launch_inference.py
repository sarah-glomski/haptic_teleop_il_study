#!/usr/bin/env python3
"""
Launch file for diffusion policy inference on Kinova Gen3.

Adapted from Robomimic/testing/launch_inference.py for the HoloLens + Kinova Gen3
setup (ZED front camera + DJI wrist camera, Kortex velocity control).

Starts all required nodes:
  1. ZED M camera            — /zed_front/zed_node/left/image_rect_color
  2. DJI wrist camera        — /dji_wrist/dji_wrist/color/image_raw
  3. kinova_state_publisher  — robot_obs/pose, robot_obs/gripper
  4. piezense_driver         — piezense/data
  5. inference.py            — loads policy, runs predict_action(), controls robot

Usage:
    python launch_inference.py --model /path/to/checkpoint.ckpt
    python launch_inference.py --model data/outputs/.../last.ckpt --action-horizon 4
    python launch_inference.py --model last.ckpt --latest   # find latest checkpoint automatically
"""

import argparse
import glob
import os
import sys

from launch import LaunchDescription, LaunchService
from launch.actions import ExecuteProcess
from launch_ros.actions import Node


_PYTHON = sys.executable

_THIS_DIR         = os.path.dirname(os.path.abspath(__file__))
_DATA_COLLECT_DIR = os.path.join(_THIS_DIR, "..", "data_collection")
_KINOVA_STATE_PUB = os.path.join(_DATA_COLLECT_DIR, "kinova_state_publisher.py")
_DJI_CAMERA_NODE  = os.path.join(_DATA_COLLECT_DIR, "dji_camera_node.py")
_INFERENCE_SCRIPT = os.path.join(_THIS_DIR, "inference.py")


def find_latest_checkpoint(search_dir: str) -> str:
    """Find latest.ckpt if it exists, otherwise the most recently modified .ckpt."""
    ckpts = glob.glob(os.path.join(search_dir, "**", "*.ckpt"), recursive=True)
    if not ckpts:
        raise FileNotFoundError(f"No .ckpt files found under {search_dir}")
    # Prefer the most-recently-modified latest.ckpt (newest training run);
    # fall back to the newest .ckpt of any name.
    latest = [c for c in ckpts if os.path.basename(c) == "latest.ckpt"]
    if latest:
        return max(latest, key=os.path.getmtime)
    return max(ckpts, key=os.path.getmtime)


ZED_SERIAL = "17875187"


def generate_launch_description(
    model_path: str,
    robot_ip: str = "192.168.1.10",
    zed_serial: str = ZED_SERIAL,
    dji_device: int = 0,
    dt: float = 0.033,
    n_action_steps: int = 8,
    diffusion_steps: int = 16,
    latency_offset_s: float = 0.0,
    no_pygame: bool = False,
) -> LaunchDescription:

    inference_cmd = [
        _PYTHON, _INFERENCE_SCRIPT,
        "--model", model_path,
        "--dt", str(dt),
        "--n-action-steps", str(n_action_steps),
        "--diffusion-steps", str(diffusion_steps),
        "--latency-offset-s", str(latency_offset_s),
    ]
    if no_pygame:
        inference_cmd.append("--no-pygame")

    return LaunchDescription([

        # ── 1. ZED M camera — front view ──────────────────────────────────────
        # Node(
        #     package="zed_wrapper",
        #     executable="zed_wrapper",
        #     name="zed_node",
        #     namespace="zed_front",
        #     output="screen",
        #     parameters=[{
        #         "camera_model":    "zedm",
        #         "camera_name":     "zed_front",
        #         "serial_number":   int(zed_serial) if zed_serial else 0,
        #         "grab_resolution": "HD720",
        #         "grab_frame_rate": 30,
        #         "pub_frame_rate":  30.0,
        #         "general.grab_frame_rate": 30,
        #         "depth.depth_mode": 1,
        #         "video.extrinsic_in_camera_frame": False,
        #     }],
        # ),

        # ── 2. DJI Osmo Action 4 — wrist camera ───────────────────────────────
        # Remapped to /dji_wrist/dji_wrist/color/image_raw to match inference.py
        ExecuteProcess(
            cmd=[
                '/usr/bin/python3.12', _DJI_CAMERA_NODE,
                "--ros-args",
                "-p", f"device_index:={dji_device}",
                "-r", "/wrist_cam/image_raw:=/dji_wrist/dji_wrist/color/image_raw",
            ],
            name="dji_wrist_camera",
            output="screen",
        ),

        # ── 3. Kinova Gen3 state publisher ────────────────────────────────────
        ExecuteProcess(
            cmd=[
                _PYTHON, _KINOVA_STATE_PUB,
                "--ros-args", "-p", f"robot_ip:={robot_ip}",
            ],
            name="kinova_state_publisher",
            output="screen",
        ),

        # # ── 4. Piezense pressure sensor driver ────────────────────────────────
        # ExecuteProcess(
        #     cmd=["ros2", "launch", "piezense_ros", "ar_teleop_piezense_launch.py"],
        #     name="piezense_driver",
        #     output="screen",
        # ),

        # ── 5. Diffusion policy inference (controls robot via Kortex API) ──────
        ExecuteProcess(
            cmd=inference_cmd,
            name="policy_inference",
            output="screen",
        ),
    ])


def main(argv=sys.argv[1:]):
    parser = argparse.ArgumentParser(
        description="Launch diffusion policy inference for Kinova Gen3"
    )
    parser.add_argument("--model",           type=str,   default=None,
                        help="Path to .ckpt checkpoint file")
    parser.add_argument("--latest",          action="store_true",
                        help="Find and use the latest checkpoint in training/data/outputs/")
    parser.add_argument("--robot-ip",        type=str,   default="192.168.1.10",
                        help="Kinova Gen3 IP address (default: 192.168.1.10)")
    parser.add_argument("--zed-serial",      type=str,   default=ZED_SERIAL,
                        help=f"ZED M serial number (default: {ZED_SERIAL})")
    parser.add_argument("--dji-device",      type=int,   default=0,
                        help="V4L2 device index for DJI wrist camera (default: 0)")
    parser.add_argument("--dt",              type=float, default=0.033,
                        help="Action step period in seconds (default: 0.033 = 30 Hz)")
    parser.add_argument("--n-action-steps",  type=int,   default=8,
                        help="Actions to execute per inference cycle (default: 8)")
    parser.add_argument("--diffusion-steps", type=int,   default=16,
                        help="DDIM inference steps (default: 16)")
    parser.add_argument("--latency-offset-s", type=float, default=0.0,
                        help="System latency to compensate in seconds (default: 0). "
                             "Skips this many steps at the start of each predicted action sequence. "
                             "Measure with latency_calculation.py.")
    parser.add_argument("--no-pygame",       action="store_true",
                        help="Disable pygame keyboard control window")
    args, launch_argv = parser.parse_known_args(argv)

    # Resolve checkpoint path
    if args.latest:
        outputs_dir = os.path.join(_THIS_DIR, "..", "training", "data", "outputs")
        args.model = find_latest_checkpoint(outputs_dir)
        print(f"Using latest checkpoint: {args.model}")
    elif args.model is None:
        parser.error("Provide --model /path/to/checkpoint.ckpt or use --latest")

    if not os.path.isfile(args.model):
        parser.error(f"Checkpoint not found: {args.model}")

    print("=" * 60)
    print("Diffusion Policy Inference — Kinova Gen3")
    print("=" * 60)
    print(f"  Model:           {args.model}")
    print(f"  Robot IP:        {args.robot_ip}")
    print(f"  ZED serial:      {args.zed_serial or '(auto-detect)'}")
    print(f"  DJI device:      /dev/video{args.dji_device}")
    print(f"  dt:              {args.dt}s  ({1/args.dt:.0f} Hz)")
    print(f"  Num action steps:  {args.n_action_steps}")
    print(f"  Diffusion steps: {args.diffusion_steps}")
    if args.latency_offset_s:
        print(f"  Latency offset:  {args.latency_offset_s*1000:.0f} ms "
              f"({round(args.latency_offset_s / args.dt)} steps)")
    print()
    print("Keyboard controls (focus pygame window):")
    print("  S - Start / Resume | D - Done / Pause | R - Reset home | Q - Quit")
    print("=" * 60)

    ld = generate_launch_description(
        model_path=args.model,
        robot_ip=args.robot_ip,
        zed_serial=args.zed_serial,
        dji_device=args.dji_device,
        dt=args.dt,
        n_action_steps=args.n_action_steps,
        diffusion_steps=args.diffusion_steps,
        latency_offset_s=args.latency_offset_s,
        no_pygame=args.no_pygame,
    )
    ls = LaunchService(argv=launch_argv)
    ls.include_launch_description(ld)
    return ls.run()


if __name__ == "__main__":
    sys.exit(main())
