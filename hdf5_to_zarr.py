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
  hololens/finger_tips (T, 15)  float32
  hololens/hand_width  (T,)     float32
  images/zed_front     (T, 3, H, W) uint8  CHW
  images/rs_wrist      (T, 3, H, W) uint8  CHW

Target zarr schema (UMI-style flat concatenation):
  output.zarr/
  ├── data/
  │   ├── zed_front_rgb:      (N, H, W, 3)  uint8    HWC
  │   ├── rs_wrist_rgb:       (N, H, W, 3)  uint8    HWC
  │   ├── pose:               (N, 10)        float32  [x,y,z,rot6d(6),gripper]  obs
  │   ├── action:             (N, 10)        float32  [x,y,z,rot6d(6),gripper]  act
  │   ├── joint_states:       (N, 7)         float32  joint angles (rad)
  │   ├── holo_palm_pose:     (N, 7)         float32  [xyz, qxyzw]
  │   ├── holo_hand_width:    (N,)           float32
  │   └── holo_finger_tips:   (N, 15)        float32
  └── meta/
      └── episode_ends:       (num_episodes,) int64  cumulative end indices

Usage:
  python3 hdf5_to_zarr.py <input_dir> <output.zarr> [--max-episodes N]
"""

import argparse
import glob
import os
import sys

import h5py
import numpy as np
import zarr
from natsort import natsorted

# Add dt_ag-main to path for RotationTransformer (same as original script)
sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), '..', 'Robomimic', 'dt_ag-main', 'dt_ag'),
)
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
      joint_states      (T, 7)   float32
      holo_palm_pose    (T, 7)   float32
      holo_hand_width   (T,)     float32
      holo_finger_tips  (T, 15)  float32
      zed_front_rgb     (T, H, W, 3) uint8  — CHW→HWC
      rs_wrist_rgb      (T, H, W, 3) uint8  — CHW→HWC
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

        # ── Kinematics passthrough ────────────────────────────────────────────
        if 'observation/joint_states' in f:
            data['joint_states'] = f['observation/joint_states'][()].astype(np.float32)

        # ── HoloLens passthrough ──────────────────────────────────────────────
        if 'hololens/palm_pose' in f:
            data['holo_palm_pose']   = f['hololens/palm_pose'][()].astype(np.float32)
        if 'hololens/hand_width' in f:
            data['holo_hand_width']  = f['hololens/hand_width'][()].astype(np.float32)
        if 'hololens/finger_tips' in f:
            data['holo_finger_tips'] = f['hololens/finger_tips'][()].astype(np.float32)

        # ── Images: CHW (T,3,H,W) → HWC (T,H,W,3) ───────────────────────────
        img_map = {
            'images/zed_front': 'zed_front_rgb',
            'images/rs_wrist':  'rs_wrist_rgb',
        }
        for h5_key, zarr_key in img_map.items():
            if h5_key in f:
                chw = f[h5_key][()]              # (T, 3, H, W)
                data[zarr_key] = np.moveaxis(chw, 1, -1)  # (T, H, W, 3)

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

    if args.max_episodes is not None:
        h5_files = h5_files[:args.max_episodes]

    print(f'Found {len(h5_files)} episode(s) in {args.input_dir}')

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
