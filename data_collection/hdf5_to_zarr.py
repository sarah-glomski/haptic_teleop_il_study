#!/usr/bin/env python3
"""
Convert HDF5 episodes (from hdf5_data_collector.py) to UMI-style flat zarr
format for diffusion policy training.

Adapted from Robomimic/data_collection/hdf5_to_zarr.py — same rot6d conversion
and ReplayBuffer-compatible episode_ends pattern.

Source HDF5 schema (per episode_N.hdf5):
  action/pose          (T, 7)   float32  [x,y,z,qx,qy,qz,qw]
  action/gripper       (T,)     float32
  observation/pose     (T, 7)   float32
  observation/gripper  (T,)     float32
  observation/joint_states (T, 7) float32
  hololens/palm_pose   (T, 7)   float32
  hololens/finger_tips    (T, 15)  float32
  hololens/hand_width     (T,)     float32
  piezense/pressure_input (T, 2)   float32  Pa — input channels 2 and 3
  images/zed_isometric        (T, 3, H, W) uint8  CHW
  images/dji_wrist         (T, 3, H, W) uint8  CHW

Target zarr schema (UMI-style flat concatenation):
  output.zarr/
  ├── data/
  │   ├── zed_isometric_rgb:        (N, 224, 224, 3)  uint8    HWC
  │   ├── dji_wrist_rgb:         (N, 224, 224, 3)  uint8    HWC
  │   ├── pose:                 (N, 10)        float32  [x,y,z,rot6d(6),gripper]  obs
  │   ├── action:               (N, 10)        float32  [x,y,z,rot6d(6),gripper]  act
  │   └── piezense_pressure:    (N, 2)         float32  Pa — sensor input channels
  └── meta/
      └── episode_ends:         (num_episodes,) int64  cumulative end indices

Usage:
  python3 hdf5_to_zarr.py <input_dir> <output.zarr> [--max-episodes N]
"""

import argparse
import glob
import os
import sys

import cv2
import h5py
import numpy as np
import zarr
from natsort import natsorted

from rotation_transformer import RotationTransformer

rot_tf = RotationTransformer(from_rep='quaternion', to_rep='rotation_6d')


# ── Pose conversion ────────────────────────────────────────────────────────────

def quat_pose_gripper_to_10d(pose_7d: np.ndarray, gripper: np.ndarray) -> np.ndarray:
    """Convert [x,y,z,qx,qy,qz,qw] + scalar gripper → [x,y,z,rot6d(6),gripper].

    Matches the existing pipeline (hdf5_to_zarr.py in Robomimic/data_collection).
    """
    xyz    = pose_7d[:, :3]          # (T, 3)
    quats  = pose_7d[:, 3:]          # (T, 4)
    rot6d  = rot_tf.forward(quats)   # (T, 6)
    grip   = gripper.reshape(-1, 1)  # (T, 1)
    return np.concatenate([xyz, rot6d, grip], axis=1).astype(np.float32)  # (T, 10)


# ── Episode loading ────────────────────────────────────────────────────────────

def load_episode(h5_path: str) -> dict:
    """Load a single HDF5 episode.  Returns dict of numpy arrays.

    Keys:
      pose              (T, 10)  float32  — observation TCP
      action            (T, 10)  float32  — commanded TCP
      piezense_pressure    (T, 2)   float32  — input channel pressures (Pa)
      zed_isometric_rgb        (T, H, W, 3) uint8  — CHW→HWC
      dji_wrist_rgb         (T, H, W, 3) uint8  — CHW→HWC
    """
    data = {}
    with h5py.File(h5_path, 'r') as f:
        # ── Robot pose / action (quaternion → 10D) ───────────────────────────
        obs_pose    = f['observation/pose'][()]
        obs_grip    = f['observation/gripper'][()]
        act_pose    = f['action/pose'][()]
        act_grip    = f['action/gripper'][()]

        data['pose']   = quat_pose_gripper_to_10d(obs_pose, obs_grip)
        data['action'] = quat_pose_gripper_to_10d(act_pose, act_grip)

        # ── Piezense pressure passthrough ─────────────────────────────────────
        if 'piezense/pressure_input' in f:
            data['piezense_pressure'] = f['piezense/pressure_input'][()].astype(np.float32)

        # ── Images: CHW (T,3,H,W) → HWC (T,H,W,3) ───────────────────────────
        img_map = {
            'images/zed_isometric': 'zed_isometric_rgb',
            'images/dji_wrist':  'dji_wrist_rgb',
        }
        for h5_key, zarr_key in img_map.items():
            if h5_key in f:
                chw = f[h5_key][()]              # (T, 3, H, W)
                hwc = np.moveaxis(chw, 1, -1)   # (T, H, W, 3)
                resized = np.stack([
                    cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
                    for frame in hwc
                ])                               # (T, 224, 224, 3)
                data[zarr_key] = resized

    return data


# ── Zarr structure management ─────────────────────────────────────────────────

def create_zarr_structure(root: zarr.Group, first_ep: dict):
    """Create empty data/ and meta/ groups matching the first episode's dtypes."""
    data_grp = root.require_group('data', overwrite=False)
    meta_grp = root.require_group('meta', overwrite=False)

    for key, value in first_ep.items():
        shape  = (0,) + value.shape[1:]
        chunks = (1,) + value.shape[1:] if value.ndim >= 3 else (1000,) + value.shape[1:]
        data_grp.zeros(key, shape=shape, chunks=chunks, dtype=value.dtype)

    meta_grp.zeros('episode_ends', shape=(0,), dtype=np.int64, compressor=None)


def append_episode(root: zarr.Group, ep_data: dict):
    """Append one episode to the flat zarr store (mirrors ReplayBuffer pattern)."""
    data_grp      = root['data']
    meta_grp      = root['meta']
    episode_ends  = meta_grp['episode_ends']

    curr_len = int(episode_ends[-1]) if episode_ends.shape[0] > 0 else 0
    T = next(iter(ep_data.values())).shape[0]
    new_len = curr_len + T

    for key, value in ep_data.items():
        arr = data_grp[key]
        assert value.shape[1:] == arr.shape[1:], (
            f'Shape mismatch for "{key}": expected {arr.shape[1:]}, got {value.shape[1:]}'
        )
        arr.resize((new_len,) + value.shape[1:])
        arr[curr_len:new_len] = value

    episode_ends.resize(episode_ends.shape[0] + 1)
    episode_ends[-1] = new_len


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Convert HDF5 episodes to UMI-style flat zarr (haptic_teleop_il_study)',
    )
    parser.add_argument('input_dir',    help='Directory containing episode_*.hdf5 files')
    parser.add_argument('output_zarr',  help='Output zarr path')
    parser.add_argument('--max-episodes', type=int, default=None,
                        help='Limit number of episodes to convert')
    args = parser.parse_args()

    h5_files = natsorted(glob.glob(os.path.join(args.input_dir, 'episode_*.hdf5')))
    if not h5_files:
        print(f'Error: No episode_*.hdf5 files in {args.input_dir}')
        sys.exit(1)

    # Skip episodes listed in exclude.txt if present
    exclude_file = os.path.join(args.input_dir, 'exclude.txt')
    if os.path.exists(exclude_file):
        with open(exclude_file) as ef:
            excluded = {line.strip() for line in ef if line.strip()}
        before = len(h5_files)
        h5_files = [p for p in h5_files
                    if os.path.basename(p).replace('.hdf5', '') not in excluded]
        print(f'Skipping {before - len(h5_files)} excluded episode(s): {", ".join(sorted(excluded))}')

    if args.max_episodes is not None:
        h5_files = h5_files[:args.max_episodes]

    print(f'Found {len(h5_files)} episode(s) to convert in {args.input_dir}')

    root = zarr.open(args.output_zarr, mode='w')

    total_frames = 0
    for idx, h5_path in enumerate(h5_files):
        print(f'  [{idx + 1}/{len(h5_files)}] {os.path.basename(h5_path)} … ', end='')
        ep_data = load_episode(h5_path)
        T = next(iter(ep_data.values())).shape[0]

        if idx == 0:
            create_zarr_structure(root, ep_data)

        append_episode(root, ep_data)
        total_frames += T
        print(f'({T} frames)')

    print(f'\nDone! Wrote {len(h5_files)} episode(s) → {args.output_zarr}')
    print(f'  Total frames : {total_frames}')

    # Summary
    root = zarr.open(args.output_zarr, mode='r')
    ends = root['meta/episode_ends'][:]
    print(f'  episode_ends : {ends}')
    for key in sorted(root['data'].array_keys()):
        arr = root['data'][key]
        print(f'  data/{key:25s}  shape={arr.shape}  dtype={arr.dtype}')


if __name__ == '__main__':
    main()
