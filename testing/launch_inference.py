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

# Shared with the collection launchers — same driver, same leaked BLE link.
# inference.py already reaches into data_collection/ for kinova_arm.
if os.path.abspath(_DATA_COLLECT_DIR) not in sys.path:
    sys.path.insert(0, os.path.abspath(_DATA_COLLECT_DIR))
from piezense_ble import release_stale_piezense_ble  # noqa: E402
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
    latency_offset_s: float = 0.0,
    dt: float = None,
    hz: float = None,
    n_action_steps: int = None,
    diffusion_steps: int = None,
    no_pygame: bool = False,
    record: bool = False,
    record_dir: str = None,
    no_piezense: bool = False,
) -> LaunchDescription:

    inference_cmd = [
        _PYTHON, _INFERENCE_SCRIPT,
        "--model", model_path,
        # dt / n-action-steps / diffusion-steps default to the checkpoint in
        # inference.py (load_run_config); only forwarded below when overridden.
        "--latency-offset-s", str(latency_offset_s),
    ]
    # Passed through only when set, so the checkpoint value stays the default
    # all the way down; inference.py prints a banner if any of these land.
    for flag, val in (("--dt", dt), ("--hz", hz),
                      ("--n-action-steps", n_action_steps),
                      ("--diffusion-steps", diffusion_steps)):
        if val is not None:
            inference_cmd += [flag, str(val)]
    if no_pygame:
        inference_cmd.append("--no-pygame")
    if record:
        inference_cmd.append("--record")
        if record_dir:
            inference_cmd += ["--record-dir", record_dir]
    if no_piezense:
        inference_cmd.append("--no-piezense")

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

        # ── Piezense pressure sensor controller ────────────────────────────
        # piezense0_pressures is a trained obs key, so the policy needs this
        # driver for the same reason data collection does. It was missing here
        # until 2026-07-31: every rollout ran with the pressure channel frozen
        # at its baseline, which the episode files recorded as a flat line.
        *([ExecuteProcess(
            cmd=['ros2', 'launch', 'piezense_ros', 'ar_teleop_piezense_launch.py'],
            name='piezense_driver',
            output='screen',
        )] if not no_piezense else []),

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
    parser.add_argument("--dji-device",      type=int,   default=-1,
                        help="V4L2 device index for DJI wrist camera (default: -1 = auto-detect by USB id)")
    # Default to the checkpoint's own values — see inference.load_run_config.
    # Set one of these only to deliberately run a trained policy at different
    # timing; inference.py prints a loud banner when you do.
    timing = parser.add_mutually_exclusive_group()
    timing.add_argument("--dt",              type=float, default=None,
                        help="Override action step period in SECONDS")
    timing.add_argument("--hz",              type=float, default=None,
                        help="Override the rate instead (dt = 1/hz)")
    parser.add_argument("--n-action-steps",  type=int,   default=None,
                        help="Override actions executed per inference cycle")
    parser.add_argument("--diffusion-steps", type=int,   default=None,
                        help="Override DDIM steps (latency vs quality)")
    parser.add_argument("--latency-offset-s", type=float, default=0.0,
                        help="System latency to compensate in seconds (default: 0)")
    parser.add_argument("--no-pygame",       action="store_true",
                        help="Disable pygame keyboard control window")
    parser.add_argument("--record",          action="store_true",
                        help="Record rollouts to testing/rollout_data/episode_N.hdf5 "
                             "(S start, D save, R/Q discard)")
    parser.add_argument("--record-dir",      type=str, default=None,
                        help="Directory for rollout episodes (default: testing/rollout_data)")
    parser.add_argument("--no-piezense",     action="store_true",
                        help="Skip the piezense driver and disable piezense obs.")
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
    # -1 means "auto-detect by USB id" — printing /dev/video-1 for it looks like
    # a resolved device that does not exist.
    print("  DJI device:      auto-detect by USB id" if args.dji_device < 0
          else f"  DJI device:      /dev/video{args.dji_device}")
    if args.hz is not None:
        print(f"  dt / action steps:  OVERRIDDEN -> {args.hz} Hz")
    elif args.dt is not None:
        print(f"  dt / action steps:  OVERRIDDEN -> dt={args.dt}s")
    else:
        print("  dt / action steps / diffusion steps: from checkpoint config")
    print(f"  Piezense:        {'DISABLED (baseline obs)' if args.no_piezense else 'driver launched'}")
    if args.latency_offset_s:
        print(f"  Latency offset:  {args.latency_offset_s*1000:.0f} ms")
    print()
    if args.record:
        print(f"  Recording:       ON → {args.record_dir or 'testing/rollout_data'}")
    print()
    print("Keyboard controls (focus pygame window):")
    print("  S - Start / Resume | D - Done / Pause | R - Reset home | Q - Quit")
    if args.record:
        print("  (recording: S starts an episode, D saves it, R/Q discard it)")
    print("=" * 60)

    # The collector and teleop start the same BLE driver, and a Ctrl-C in any
    # of them leaves the link held with no owner. Inference then finds the
    # sensor not advertising and sits on "[example] connecting..." — which is
    # what happened on 2026-08-14, inheriting the link data collection left
    # behind the day before.
    if not args.no_piezense:
        release_stale_piezense_ble()

    ls = LaunchService(argv=launch_argv)
    ls.include_launch_description(generate_launch_description(
        model_path=args.model,
        robot_ip=args.robot_ip,
        dji_device=args.dji_device,
        latency_offset_s=args.latency_offset_s,
        dt=args.dt,
        hz=args.hz,
        n_action_steps=args.n_action_steps,
        diffusion_steps=args.diffusion_steps,
        no_pygame=args.no_pygame,
        record=args.record,
        record_dir=args.record_dir,
        no_piezense=args.no_piezense,
    ))
    return ls.run()


if __name__ == "__main__":
    sys.exit(main())
