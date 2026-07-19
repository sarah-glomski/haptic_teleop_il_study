#!/usr/bin/env python3
"""
Rollout recorder — saves inference rollouts to HDF5 episode files.

Produces `episode_N.hdf5` files in (almost) the same schema as the teleop
data-collection episodes written by `data_collection/hdf5_data_collector.py`,
so they can be inspected/replayed with the same tooling. Differences from a
collected episode:
  - No `hololens/*` datasets (there is no hand tracking during inference).
  - Extra `observation/timestamp` (T,) and a `predictions/` group holding the
    FULL predicted action horizon from every inference call — rollout-only
    extras that any episode-format reader safely ignores.

This module is intentionally ROS-free and hardware-free (unlike the collector,
which is an rclpy Node) so `inference.py` can import it without side effects.

Schema written by `write_episode_hdf5`:
  action/pose              (T,7)  float32  [x,y,z,qx,qy,qz,qw]  commanded target
  action/gripper           (T,)   float32  0=open..1=closed
  observation/pose         (T,7)  float32  measured TCP pose (quat)
  observation/gripper      (T,)   float32
  observation/joint_states (T,7)  float32  radians
  observation/timestamp    (T,)   float32  seconds since rollout start
  piezense/pressure_input  (T,2)  float32  (group attrs channel_ids, units='Pa')
  images/dji_wrist         (T,3,H,W) uint8 RGB CHW, lzf-compressed
  predictions/action_horizon (P,H,10) float32  raw model output per inference
  predictions/timestamp    (P,)   float32  seconds since rollout start
  root attrs: num_frames, collection_rate_hz=30, episode_index
"""

import glob
import os
import threading

import h5py
import numpy as np

# Matches PIEZENSE_INPUT_CHAN_IDS in inference.py / the collector.
PIEZENSE_CHAN_IDS = [2, 3]
COLLECTION_RATE_HZ = 30


def scan_existing_episodes(save_dir: str) -> int:
    """Next episode index = max(existing episode_N.hdf5) + 1, or 0 if none."""
    if not os.path.isdir(save_dir):
        return 0
    indices = []
    for path in glob.glob(os.path.join(save_dir, 'episode_*.hdf5')):
        try:
            indices.append(int(os.path.basename(path)
                                .replace('episode_', '').replace('.hdf5', '')))
        except ValueError:
            pass
    return max(indices) + 1 if indices else 0


def write_episode_hdf5(path: str, buffers: dict, episode_index: int) -> None:
    """Write one rollout episode. `buffers` holds already-stacked numpy arrays."""
    T = int(buffers['action_pose'].shape[0])
    with h5py.File(path, 'w') as f:
        act = f.create_group('action')
        act.create_dataset('pose',    data=buffers['action_pose'])
        act.create_dataset('gripper', data=buffers['action_gripper'])

        obs = f.create_group('observation')
        obs.create_dataset('pose',         data=buffers['obs_pose'])
        obs.create_dataset('gripper',      data=buffers['obs_gripper'])
        obs.create_dataset('joint_states', data=buffers['joint_states'])
        obs.create_dataset('timestamp',    data=buffers['obs_time'])

        pz = f.create_group('piezense')
        pz.create_dataset('pressure_input', data=buffers['piezense'])
        pz.attrs['channel_ids'] = PIEZENSE_CHAN_IDS
        pz.attrs['units'] = 'Pa'

        dji = buffers.get('dji')
        if dji is not None and len(dji):
            imgs = f.create_group('images')
            imgs.create_dataset('dji_wrist', data=dji, compression='lzf')

        pred = buffers.get('pred_horizon')
        if pred is not None and len(pred):
            pg = f.create_group('predictions')
            pg.create_dataset('action_horizon', data=pred)
            pg.create_dataset('timestamp',      data=buffers['pred_time'])
            pg.attrs['format'] = 'umi_10d_pos3_rot6d_width'

        f.attrs['num_frames']         = T
        f.attrs['collection_rate_hz'] = COLLECTION_RATE_HZ
        f.attrs['episode_index']      = int(episode_index)


class RolloutRecorder:
    """Buffers rollout data and flushes one episode HDF5 per rollout.

    Thread-safe: `append`/`append_prediction` run in the ROS executor thread,
    while `start`/`save`/`discard` run in the pygame key-handler thread.
    """

    def __init__(self, save_dir: str):
        self._save_dir = save_dir
        self._lock = threading.Lock()
        self._recording = False
        self._episode_index = scan_existing_episodes(save_dir)
        self._reset_buffers()

    def _reset_buffers(self):
        self._buf_action_pose = []
        self._buf_action_gripper = []
        self._buf_obs_pose = []
        self._buf_obs_gripper = []
        self._buf_joint_states = []
        self._buf_obs_time = []
        self._buf_piezense = []
        self._buf_dji = []
        self._buf_pred_horizon = []
        self._buf_pred_time = []

    @property
    def is_recording(self) -> bool:
        return self._recording

    def start(self):
        with self._lock:
            if self._recording:
                return
            self._reset_buffers()
            self._recording = True

    def append(self, obs_pose7, obs_gripper, action_pose7, action_gripper,
               joint7, piezense2, img_u8_chw, obs_time):
        with self._lock:
            if not self._recording:
                return
            self._buf_obs_pose.append(np.asarray(obs_pose7, dtype=np.float32))
            self._buf_obs_gripper.append(np.float32(obs_gripper))
            self._buf_action_pose.append(np.asarray(action_pose7, dtype=np.float32))
            self._buf_action_gripper.append(np.float32(action_gripper))
            self._buf_joint_states.append(np.asarray(joint7, dtype=np.float32))
            self._buf_piezense.append(np.asarray(piezense2, dtype=np.float32))
            self._buf_dji.append(np.ascontiguousarray(img_u8_chw, dtype=np.uint8))
            self._buf_obs_time.append(np.float32(obs_time))

    def append_prediction(self, horizon_arr, pred_time):
        with self._lock:
            if not self._recording:
                return
            self._buf_pred_horizon.append(np.asarray(horizon_arr, dtype=np.float32))
            self._buf_pred_time.append(np.float32(pred_time))

    def discard(self):
        with self._lock:
            self._reset_buffers()
            self._recording = False

    def save(self):
        """Flush the current episode to disk. Returns (path, num_frames) or None."""
        with self._lock:
            if not self._recording:
                return None
            if not self._buf_action_pose:
                self._recording = False
                return None
            T = len(self._buf_action_pose)
            preds = None
            pred_time = None
            if self._buf_pred_horizon:
                preds = np.stack(self._buf_pred_horizon).astype(np.float32)
                pred_time = np.array(self._buf_pred_time, dtype=np.float32)
            buffers = dict(
                action_pose=np.array(self._buf_action_pose, dtype=np.float32),
                action_gripper=np.array(self._buf_action_gripper, dtype=np.float32),
                obs_pose=np.array(self._buf_obs_pose, dtype=np.float32),
                obs_gripper=np.array(self._buf_obs_gripper, dtype=np.float32),
                joint_states=np.array(self._buf_joint_states, dtype=np.float32),
                obs_time=np.array(self._buf_obs_time, dtype=np.float32),
                piezense=np.array(self._buf_piezense, dtype=np.float32),
                dji=(np.array(self._buf_dji[:T], dtype=np.uint8) if self._buf_dji else None),
                pred_horizon=preds,
                pred_time=pred_time,
            )
            idx = self._episode_index
            self._episode_index += 1
            self._recording = False

        os.makedirs(self._save_dir, exist_ok=True)
        path = os.path.join(self._save_dir, f'episode_{idx}.hdf5')
        write_episode_hdf5(path, buffers, idx)
        return path, T
