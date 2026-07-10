#!/usr/bin/env python3
"""
Launch file for diffusion policy inference (charm-lab UMI pipeline).

Starts kinova_state_publisher, dji_camera_node, and inference.py, which expects
checkpoints trained by train.py (UMI obs/action schema — robot0_*/camera0_* keys).

Usage:
    python launch_inference.py --model /path/to/checkpoint.ckpt
    python launch_inference.py --latest   # newest training run's latest.ckpt
"""

import argparse
import glob
import os
import sys

from launch import LaunchDescription, LaunchService
from launch.actions import ExecuteProcess


_PYTHON = sys.executable

_THIS_DIR         = os.path.dirname(os.path.abspath(__file__))
_DATA_COLLECT_DIR = os.path.join(_THIS_DIR, "..", "data_collection")
_KINOVA_STATE_PUB = os.path.join(_DATA_COLLECT_DIR, "kinova_state_publisher.py")
_DJI_CAMERA_NODE  = os.path.join(_DATA_COLLECT_DIR, "dji_camera_node.py")
_INFERENCE_SCRIPT = os.path.join(_THIS_DIR, "inference.py")


def find_latest_checkpoint(search_dir: str) -> str:
    """Most recently modified latest.ckpt across all training runs.

    inference.py hard-rejects non-UMI-schema checkpoints at load time.
    """
    ckpts = glob.glob(os.path.join(search_dir, "**", "*.ckpt"), recursive=True)
    if not ckpts:
        raise FileNotFoundError(
            f"No checkpoints found under {search_dir} "
            f"(train one with train.py first, or pass --model explicitly)")
    latest = [c for c in ckpts if os.path.basename(c) == "latest.ckpt"]
    if latest:
        return max(latest, key=os.path.getmtime)
    return max(ckpts, key=os.path.getmtime)


def generate_launch_description(
    model_path: str,
    robot_ip: str = "192.168.1.10",
    dji_device: int = 0,
    dt: float = 0.1,
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

        # ── DJI Osmo Action 4 — wrist camera ───────────────────────────────
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

        # ── Kinova Gen3 state publisher ────────────────────────────────────
        ExecuteProcess(
            cmd=[
                _PYTHON, _KINOVA_STATE_PUB,
                "--ros-args", "-p", f"robot_ip:={robot_ip}",
            ],
            name="kinova_state_publisher",
            output="screen",
        ),

        # ── Diffusion policy inference (UMI pipeline) ──────────────────────
        ExecuteProcess(
            cmd=inference_cmd,
            name="policy_inference",
            output="screen",
        ),
    ])


def main(argv=sys.argv[1:]):
    parser = argparse.ArgumentParser(
        description="Launch diffusion policy inference (UMI pipeline) for Kinova Gen3"
    )
    parser.add_argument("--model",           type=str,   default=None,
                        help="Path to UMI-pipeline .ckpt checkpoint file")
    parser.add_argument("--latest",          action="store_true",
                        help="Use the newest run's checkpoint in training/data/outputs/")
    parser.add_argument("--robot-ip",        type=str,   default="192.168.1.10")
    parser.add_argument("--dji-device",      type=int,   default=0,
                        help="V4L2 device index for DJI wrist camera (default: 0)")
    parser.add_argument("--dt",              type=float, default=0.1,
                        help="Action step period in seconds (default: 0.1 = 10 Hz, "
                             "matching training obs_down_sample_steps 3 @ 30 Hz)")
    parser.add_argument("--n-action-steps",  type=int,   default=8,
                        help="Actions to execute per inference cycle (default: 8)")
    parser.add_argument("--diffusion-steps", type=int,   default=16,
                        help="DDIM inference steps (default: 16)")
    parser.add_argument("--latency-offset-s", type=float, default=0.0,
                        help="System latency to compensate in seconds (default: 0)")
    parser.add_argument("--no-pygame",       action="store_true",
                        help="Disable pygame keyboard control window")
    args, launch_argv = parser.parse_known_args(argv)

    if args.latest:
        outputs_dir = os.path.join(_THIS_DIR, "..", "training", "data", "outputs")
        args.model = find_latest_checkpoint(outputs_dir)
        print(f"Using latest UMI checkpoint: {args.model}")
    elif args.model is None:
        parser.error("Provide --model /path/to/checkpoint.ckpt or use --latest")

    if not os.path.isfile(args.model):
        parser.error(f"Checkpoint not found: {args.model}")

    print("=" * 60)
    print("Diffusion Policy Inference (UMI pipeline) — Kinova Gen3")
    print("=" * 60)
    print(f"  Model:           {args.model}")
    print(f"  Robot IP:        {args.robot_ip}")
    print(f"  DJI device:      /dev/video{args.dji_device}")
    print(f"  dt:              {args.dt}s  ({1/args.dt:.0f} Hz)")
    print(f"  Num action steps:{args.n_action_steps}")
    print(f"  Diffusion steps: {args.diffusion_steps}")
    if args.latency_offset_s:
        print(f"  Latency offset:  {args.latency_offset_s*1000:.0f} ms "
              f"({round(args.latency_offset_s / args.dt)} steps)")
    print()
    print("Keyboard controls (focus pygame window):")
    print("  S - Start / Resume | D - Done / Pause | R - Reset home | Q - Quit")
    print("=" * 60)

    ls = LaunchService(argv=launch_argv)
    ls.include_launch_description(generate_launch_description(
        model_path=args.model,
        robot_ip=args.robot_ip,
        dji_device=args.dji_device,
        dt=args.dt,
        n_action_steps=args.n_action_steps,
        diffusion_steps=args.diffusion_steps,
        latency_offset_s=args.latency_offset_s,
        no_pygame=args.no_pygame,
    ))
    return ls.run()


if __name__ == "__main__":
    sys.exit(main())
