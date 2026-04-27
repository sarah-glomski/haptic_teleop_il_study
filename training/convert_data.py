#!/usr/bin/env python3
"""
Convert Kinova Gen3 HDF5 teleop episodes to per-episode zarr for KinovaImageDataset.

Adapted from Robomimic/training/convert_data.py for the HoloLens + Kinova Gen3
setup (ZED front + DJI wrist cameras, piezense pressure).

Reads episode_*.hdf5 files produced by data_collection/hdf5_data_collector.py,
converts quaternion poses to 10D [x,y,z,rot6d(6),gripper] using scipy,
resizes images to 224x224, and writes a zarr store with per-episode groups
compatible with KinovaImageDataset.

Output zarr schema:
  output.zarr/
  ├── episode_0/
  │   ├── zed_front_rgb:      (T, 224, 224, 3)  uint8   HWC
  │   ├── rs_wrist_rgb:       (T, 224, 224, 3)  uint8   HWC
  │   ├── pose:               (T, 10)            float32 [xyz, rot6d, gripper_obs]
  │   ├── action:             (T, 10)            float32 [xyz, rot6d, gripper_cmd]
  │   └── piezense_pressure:  (T, 2)             float32 input channel pressures (Pa)
  ├── episode_1/
  │   └── ...

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
    "images/rs_wrist":  "rs_wrist_rgb",
}


def quat_xyzw_to_10d(pose_7: np.ndarray, gripper: np.ndarray) -> np.ndarray:
    """Convert (T,7) [x,y,z,qx,qy,qz,qw] + (T,) gripper to (T,10) [xyz,rot6d,grip].

    HDF5 stores pose in ROS/scipy convention: quaternion [qx, qy, qz, qw].
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
    store = zarr.open(output_path, mode="w")

    for ep_idx, h5_path in enumerate(h5_files):
        ep_name = f"episode_{ep_idx}"
        print(f"  [{ep_idx + 1}/{len(h5_files)}] {h5_path.name} -> {ep_name}")

        with h5py.File(h5_path, "r") as f:
            # Robot pose and action (quaternion -> 10D)
            obs_pose = f["observation/pose"][:]     # (T, 7)
            obs_grip = f["observation/gripper"][:]  # (T,)
            act_pose = f["action/pose"][:]          # (T, 7)
            act_grip = f["action/gripper"][:]       # (T,)

            pose_10d   = quat_xyzw_to_10d(obs_pose, obs_grip)
            action_10d = quat_xyzw_to_10d(act_pose, act_grip)
            T = pose_10d.shape[0]

            grp = store.create_group(ep_name)

            # Camera images
            for h5_key, zarr_key in CAMERA_KEYS.items():
                if h5_key not in f:
                    print(f"    Warning: {h5_key} not found, skipping")
                    continue
                raw = f[h5_key][:]  # (T, 3, H, W) CHW uint8
                resized = resize_images(raw, IMG_SIZE)  # (T, 224, 224, 3) HWC
                grp.create_dataset(
                    zarr_key, data=resized,
                    chunks=(1, IMG_SIZE, IMG_SIZE, 3), dtype="uint8",
                )
                print(f"    {zarr_key}: {resized.shape}")

            # Robot state
            grp.create_dataset("pose",   data=pose_10d,   chunks=(T, 10), dtype="float32")
            grp.create_dataset("action", data=action_10d, chunks=(T, 10), dtype="float32")
            print(f"    pose: {pose_10d.shape}  action: {action_10d.shape}")

            # Piezense pressure (2 input channels)
            if "piezense/pressure_input" in f:
                piezense = f["piezense/pressure_input"][:].astype(np.float32)  # (T, 2)
            else:
                print("    Warning: piezense/pressure_input not found, writing zeros")
                piezense = np.zeros((T, 2), dtype=np.float32)
            grp.create_dataset("piezense_pressure", data=piezense,
                               chunks=(min(T, 1000), 2), dtype="float32")
            print(f"    piezense_pressure: {piezense.shape}")

    print(f"\nDone. Zarr store written to {output_path}")
    print(f"Total episodes: {len(h5_files)}")

    z = zarr.open(output_path, mode="r")
    for ep in sorted(z.group_keys()):
        g = z[ep]
        shapes = {k: g[k].shape for k in g.array_keys()}
        print(f"  {ep}: {shapes}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert Kinova Gen3 HDF5 episodes to per-episode zarr"
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
