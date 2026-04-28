#!/usr/bin/env python3
"""
Convert Kinova Gen3 HDF5 teleop episodes to flat UMI-style zarr for KinovaImageDataset.

Adapted from Robomimic/training/convert_data.py for the HoloLens + Kinova Gen3
setup (ZED front + DJI wrist cameras, piezense pressure).

Reads episode_*.hdf5 files produced by data_collection/hdf5_data_collector.py,
converts quaternion poses to 10D [x,y,z,rot6d(6),gripper] using scipy,
resizes images to 224x224, and writes a flat zarr store compatible with
KinovaImageDataset.

Output zarr schema:
  output.zarr/
  ├── data/
  │   ├── zed_front_rgb:      (N, 224, 224, 3)  uint8   HWC
  │   ├── dji_wrist_rgb:      (N, 224, 224, 3)  uint8   HWC
  │   ├── pose:               (N, 10)            float32 [xyz, rot6d, gripper_obs]
  │   ├── action:             (N, 10)            float32 [xyz, rot6d, gripper_cmd]
  │   └── piezense_pressure:  (N, 2)             float32 input channel pressures (Pa)
  └── meta/
      └── episode_ends:       (E,)               int64   cumulative end indices

Usage:
    python convert_data.py --input ../data_collection/demo_data --output kinova_teleop.zarr
    python convert_data.py --input /path/to/demos --output out.zarr --max-episodes 50
"""

import argparse
import os
from pathlib import Path

import cv2
import h5py
import numpy as np
import zarr
from natsort import natsorted
from scipy.spatial.transform import Rotation

IMG_SIZE = 224

CAMERA_KEYS = {
    "images/zed_front": "zed_front_rgb",
    "images/dji_wrist":  "dji_wrist_rgb",
}


def quat_xyzw_to_10d(pose_7: np.ndarray, gripper: np.ndarray) -> np.ndarray:
    """Convert (T,7) [x,y,z,qx,qy,qz,qw] + (T,) gripper to (T,10) [xyz,rot6d,grip].

    Rotation 6D = first two columns of the rotation matrix concatenated.
    """
    xyz = pose_7[:, :3]
    quat_xyzw = pose_7[:, 3:]
    rot_mats = Rotation.from_quat(quat_xyzw).as_matrix()  # (T, 3, 3)
    rot6d = np.concatenate([rot_mats[:, :, 0], rot_mats[:, :, 1]], axis=1)  # (T, 6)
    grip = gripper.reshape(-1, 1)
    return np.concatenate([xyz, rot6d, grip], axis=1).astype(np.float32)  # (T, 10)


def resize_images(imgs_chw: np.ndarray, size: int = IMG_SIZE) -> np.ndarray:
    """Resize (T, 3, H, W) CHW uint8 images to (T, size, size, 3) HWC uint8."""
    T = imgs_chw.shape[0]
    out = np.empty((T, size, size, 3), dtype=np.uint8)
    for t in range(T):
        frame = np.transpose(imgs_chw[t], (1, 2, 0))  # CHW -> HWC
        out[t] = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
    return out


def convert(input_dir: str, output_path: str, max_episodes: int = None):
    h5_files = natsorted(Path(input_dir).glob("episode_*.hdf5"))
    if not h5_files:
        print(f"No episode_*.hdf5 files found in {input_dir}")
        return

    if max_episodes is not None:
        h5_files = h5_files[:max_episodes]

    print(f"Found {len(h5_files)} episode(s) in {input_dir}")

    root     = zarr.open(output_path, mode="w")
    data_grp = root.require_group("data")
    meta_grp = root.require_group("meta")

    initialized = False
    total_frames = 0

    for ep_idx, h5_path in enumerate(h5_files):
        print(f"  [{ep_idx + 1}/{len(h5_files)}] {h5_path.name} ", end="")

        with h5py.File(h5_path, "r") as f:
            obs_pose = f["observation/pose"][:]
            obs_grip = f["observation/gripper"][:]
            act_pose = f["action/pose"][:]
            act_grip = f["action/gripper"][:]

            pose_10d   = quat_xyzw_to_10d(obs_pose, obs_grip)
            action_10d = quat_xyzw_to_10d(act_pose, act_grip)
            T = pose_10d.shape[0]

            cam_data = {}
            for h5_key, zarr_key in CAMERA_KEYS.items():
                if h5_key in f:
                    cam_data[zarr_key] = resize_images(f[h5_key][:], IMG_SIZE)
                else:
                    print(f"\n    Warning: {h5_key} not found, skipping")

            if "piezense/pressure_input" in f:
                piezense = f["piezense/pressure_input"][:].astype(np.float32)
            else:
                print("\n    Warning: piezense/pressure_input not found, writing zeros")
                piezense = np.zeros((T, 2), dtype=np.float32)

        ep_data = {
            "pose":              pose_10d,
            "action":            action_10d,
            "piezense_pressure": piezense,
            **cam_data,
        }

        if not initialized:
            for key, val in ep_data.items():
                shape  = (0,) + val.shape[1:]
                chunks = (1,) + val.shape[1:] if val.ndim >= 3 else (min(T, 1000),) + val.shape[1:]
                data_grp.zeros(key, shape=shape, chunks=chunks, dtype=val.dtype)
            meta_grp.zeros("episode_ends", shape=(0,), dtype=np.int64, compressor=None)
            initialized = True

        # Append this episode to each flat array
        curr = total_frames
        new  = curr + T
        for key, val in ep_data.items():
            arr = data_grp[key]
            arr.resize((new,) + val.shape[1:])
            arr[curr:new] = val

        ends = meta_grp["episode_ends"]
        ends.resize(ends.shape[0] + 1)
        ends[-1] = new
        total_frames = new

        print(f"({T} frames, cumulative={new})")

    print(f"\nDone. Zarr written to {output_path}")
    print(f"Total frames : {total_frames}")
    print(f"episode_ends : {meta_grp['episode_ends'][:]}")
    for key in sorted(data_grp.array_keys()):
        arr = data_grp[key]
        print(f"  data/{key:25s}  {arr.shape}  {arr.dtype}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert Kinova Gen3 HDF5 episodes to flat UMI-style zarr"
    )
    parser.add_argument(
        "--input", type=str,
        default=os.path.join(os.path.dirname(__file__), "..", "data_collection", "demo_data"),
        help="Directory containing episode_*.hdf5 files",
    )
    parser.add_argument(
        "--output", type=str, default="kinova_teleop.zarr",
        help="Output zarr path",
    )
    parser.add_argument(
        "--max-episodes", type=int, default=None,
        help="Limit number of episodes to convert",
    )
    args = parser.parse_args()
    convert(args.input, args.output, args.max_episodes)
