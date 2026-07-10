#!/usr/bin/env python3
"""
Convert Kinova Gen3 HDF5 teleop episodes to the REAL UMI repo's native format:
a ReplayBuffer zarr written into a .zarr.zip, with UMI's per-signal key schema.

This is the UMI-migration counterpart of convert_data.py (which writes the flat
combined-vector zarr for the Aiden/Robomimic pipeline). Both read the SAME raw
HDF5 episodes — data collection is unchanged and upstream of this choice.

Target consumer: UmiDataset in
  HapticTeleopIL/Imitation Learning/universal_manipulation_interface
which opens the file with zarr.ZipStore and expects (single robot):

  data/
    camera0_rgb              (N, 224, 224, 3) uint8   HWC (dataset converts to CHW /255 at read)
    robot0_eef_pos           (N, 3)  float32   absolute TCP position (observed)
    robot0_eef_rot_axis_angle(N, 3)  float32   absolute TCP rotation, AXIS-ANGLE (rotvec)
    robot0_gripper_width     (N, 1)  float32   physical width in METERS (see GRIPPER_MAX_WIDTH_M)
    robot0_demo_start_pose   (N, 6)  float32   episode start [pos, rotvec], repeated per frame
    robot0_demo_end_pose     (N, 6)  float32   episode end   [pos, rotvec], repeated per frame
    piezense0_pressures      (N, 2)  float32
    action                   (N, 7)  float32   commanded [pos(3), rotvec(3), gripper_width(1)]
  meta/
    episode_ends             (E,)    int64

Schema facts verified against umi_dataset.py / sampler.py:
  - Rotations are STORED as axis-angle (scipy rotvec). rot6d is computed live at
    read time via pose_to_mat -> mat_to_pose10d (first-two-ROWS convention).
    We never store rot6d, so the rows-vs-columns question cannot bite here.
  - robot0_demo_start_pose is REQUIRED: umi_dataset.__getitem__ reads it to build
    the robot0_eef_rot_axis_angle_wrt_start observation (start pose + noise).
  - action is stored 7D per robot ([...,:6] -> pose_to_mat, [...,6:7] -> gripper).
  - sampler.py hard-reads replay_buffer['robot0_gripper_width'][:, 0] -> must be (N,1).
  - Normalizers are range/identity per key, so units self-calibrate for training;
    they only need to be CONSISTENT between this converter and inference_umi.py.

Gripper conversion (Kinova Gen3 + Robotiq 2F-85, 85 mm stroke):
  HDF5 stores the normalized Kortex command/reading g in [0,1], 0=open 1=closed.
  UMI's gripper_width is a physical opening width in meters, larger=more open.
    width_m = GRIPPER_MAX_WIDTH_M * (1.0 - g)
  inference_umi.py MUST invert with the same constant: g = 1 - width/GRIPPER_MAX_WIDTH_M.

Usage (from training/):
    python convert_data_umi.py --input ../data_collection/demo_data/Collection5 \
        --output ../data_collection/demo_data/Collection5/kinova_teleop_umi.zarr.zip
    python convert_data_umi.py --input ... --output ... --max-episodes 3

Honors <input>/exclude.txt (drops + end-crops), same format as convert_data.py.
Originals are never modified.
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np
import zarr
from natsort import natsorted
from scipy.spatial.transform import Rotation

# Real UMI repo (NOT Robomimic/dt_ag-main) for ReplayBuffer
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_UMI_REAL = os.path.join(_THIS_DIR, "..", "..", "HapticTeleopIL",
                         "Imitation Learning", "universal_manipulation_interface")
if _UMI_REAL not in sys.path:
    sys.path.insert(0, _UMI_REAL)

from diffusion_policy.common.replay_buffer import ReplayBuffer  # noqa: E402

IMG_SIZE = 224
# Robotiq 2F-85 stroke. Only consistency with inference_umi.py matters (range
# normalizer self-calibrates), but meters keeps UMI semantic conventions.
GRIPPER_MAX_WIDTH_M = 0.085

WRIST_CAM_H5_KEY = "images/dji_wrist"


def parse_exclude(exclude_path: str):
    """Same exclude.txt format as convert_data.py: drops + `crop S E` lines."""
    full, crops = set(), {}
    if not os.path.exists(exclude_path):
        return full, crops
    with open(exclude_path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            name = parts[0]
            if len(parts) >= 4 and parts[1] == "crop":
                crops[name] = (int(parts[2]), int(parts[3]))
            else:
                full.add(name)
    return full, crops


def quat_pose_to_pos_rotvec(pose_7: np.ndarray):
    """(T,7) [x,y,z,qx,qy,qz,qw] -> ((T,3) pos, (T,3) rotvec).

    scipy from_quat takes xyzw directly — the physical quaternion, no reorder.
    rotvec has no wxyz/xyzw ambiguity, matching umi/common/pose_util.py exactly.
    """
    pos = pose_7[:, :3].astype(np.float32)
    rotvec = Rotation.from_quat(pose_7[:, 3:]).as_rotvec().astype(np.float32)
    return pos, rotvec


def gripper_norm_to_width_m(g: np.ndarray) -> np.ndarray:
    """Kortex normalized (0=open, 1=closed) -> physical width in meters, (T,1)."""
    g = np.clip(g.astype(np.float32), 0.0, 1.0)
    return (GRIPPER_MAX_WIDTH_M * (1.0 - g)).reshape(-1, 1)


def resize_images_chw_to_hwc(imgs_chw: np.ndarray, size: int = IMG_SIZE) -> np.ndarray:
    """(T,3,H,W) uint8 CHW -> (T,size,size,3) uint8 HWC."""
    T = imgs_chw.shape[0]
    out = np.empty((T, size, size, 3), dtype=np.uint8)
    for t in range(T):
        frame = np.transpose(imgs_chw[t], (1, 2, 0))
        if frame.shape[:2] != (size, size):
            frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
        out[t] = frame
    return out


def load_episode_umi(h5_path: Path, crop) -> dict:
    """Read one HDF5 episode -> dict of UMI-schema arrays. crop=(s,e) or (0,None)."""
    cs, ce = crop
    with h5py.File(h5_path, "r") as f:
        obs_pose = f["observation/pose"][cs:ce]
        obs_grip = f["observation/gripper"][cs:ce]
        act_pose = f["action/pose"][cs:ce]
        act_grip = f["action/gripper"][cs:ce]
        imgs = f[WRIST_CAM_H5_KEY][cs:ce]
        if "piezense/pressure_input" in f:
            piezense = f["piezense/pressure_input"][cs:ce].astype(np.float32)
        else:
            piezense = np.zeros((len(obs_pose), 2), dtype=np.float32)

    obs_pos, obs_rotvec = quat_pose_to_pos_rotvec(obs_pose)
    act_pos, act_rotvec = quat_pose_to_pos_rotvec(act_pose)
    T = len(obs_pos)

    # Episode start/end pose [pos, rotvec], repeated per frame (required by
    # umi_dataset for the _wrt_start observation; end pose kept for parity with
    # UMI's own pipeline output).
    start_pose = np.concatenate([obs_pos[0], obs_rotvec[0]]).astype(np.float32)
    end_pose = np.concatenate([obs_pos[-1], obs_rotvec[-1]]).astype(np.float32)

    action = np.concatenate(
        [act_pos, act_rotvec, gripper_norm_to_width_m(act_grip)], axis=1
    ).astype(np.float32)  # (T, 7)

    return {
        "camera0_rgb": resize_images_chw_to_hwc(imgs),
        "robot0_eef_pos": obs_pos,
        "robot0_eef_rot_axis_angle": obs_rotvec,
        "robot0_gripper_width": gripper_norm_to_width_m(obs_grip),
        "robot0_demo_start_pose": np.tile(start_pose, (T, 1)),
        "robot0_demo_end_pose": np.tile(end_pose, (T, 1)),
        "piezense0_pressures": piezense,
        "action": action,
    }


def convert(input_dir: str, output_path: str, max_episodes=None):
    h5_files = natsorted(Path(input_dir).glob("episode_*.hdf5"))
    if not h5_files:
        print(f"No episode_*.hdf5 files found in {input_dir}")
        return

    full_drops, crops = parse_exclude(os.path.join(input_dir, "exclude.txt"))
    if full_drops:
        h5_files = [p for p in h5_files if p.stem not in full_drops]
        print(f"exclude.txt: dropping {len(full_drops)} episode(s): "
              f"{', '.join(sorted(full_drops))}")
    if crops:
        print(f"exclude.txt: end-cropping {len(crops)} episode(s): "
              f"{', '.join(f'{n}[{s}:{e}]' for n, (s, e) in sorted(crops.items()))}")
    if max_episodes is not None:
        h5_files = h5_files[:max_episodes]

    print(f"Converting {len(h5_files)} episode(s) from {input_dir}")
    print(f"  gripper: width_m = {GRIPPER_MAX_WIDTH_M} * (1 - g_norm)  "
          f"(inference_umi.py must invert with the same constant)")

    buffer = ReplayBuffer.create_empty_numpy()
    for i, h5_path in enumerate(h5_files):
        crop = crops.get(h5_path.stem, (0, None))
        ep = load_episode_umi(h5_path, crop)
        T = ep["robot0_eef_pos"].shape[0]
        tag = f"[crop {crop[0]}:{crop[1]}] " if h5_path.stem in crops else ""
        buffer.add_episode(ep)
        print(f"  [{i + 1}/{len(h5_files)}] {h5_path.name} {tag}({T} frames)")

    # Per-frame chunking for images; generous chunks for low-dim signals.
    chunks = {
        "camera0_rgb": (1, IMG_SIZE, IMG_SIZE, 3),
        "robot0_eef_pos": (1000, 3),
        "robot0_eef_rot_axis_angle": (1000, 3),
        "robot0_gripper_width": (1000, 1),
        "robot0_demo_start_pose": (1000, 6),
        "robot0_demo_end_pose": (1000, 6),
        "piezense0_pressures": (1000, 2),
        "action": (1000, 7),
    }
    # Clamp chunk lengths to actual array length (zarr requires chunk <= shape is
    # fine, but keep tidy for tiny test conversions)
    n_total = buffer.n_steps
    chunks = {k: (min(c[0], n_total),) + c[1:] for k, c in chunks.items()}

    out = os.path.expanduser(output_path)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    if os.path.exists(out):
        os.remove(out)
    print(f"Writing ZipStore -> {out}")
    with zarr.ZipStore(out, mode="w") as zip_store:
        buffer.save_to_store(store=zip_store, chunks=chunks)

    print(f"\nDone. {buffer.n_episodes} episode(s), {n_total} frames total.")
    print(f"episode_ends: {buffer.episode_ends[:]}")

    # Read-back sanity check
    with zarr.ZipStore(out, mode="r") as zip_store:
        root = zarr.group(zip_store)
        for key in sorted(root["data"].array_keys()):
            arr = root["data"][key]
            print(f"  data/{key:28s} {arr.shape}  {arr.dtype}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert Kinova HDF5 episodes to the real-UMI ReplayBuffer .zarr.zip"
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Directory containing episode_*.hdf5 files")
    parser.add_argument("--output", type=str, required=True,
                        help="Output .zarr.zip path")
    parser.add_argument("--max-episodes", type=int, default=None,
                        help="Limit number of episodes (for testing)")
    args = parser.parse_args()
    convert(args.input, args.output, args.max_episodes)
