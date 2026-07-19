#!/usr/bin/env python3
"""
Visualize a single HDF5 rollout episode recorded during inference
(testing/rollout_recorder.py).

A trimmed copy of data_collection/visualize_episode.py: rollouts have no
HoloLens/hand data, so those reads and plots are dropped. Everything else
(action vs observed TCP pose, gripper, piezense, DJI wrist images) is kept.

Displays:
  - Top rows:    Camera images (dji_wrist) at N evenly-spaced timesteps
  - Bottom plots: Action/obs TCP pose XYZ, gripper, piezense pressure (if any)

Usage:
    python3.12 visualize_rollout.py <episode.hdf5> [--num-steps 10]
    python3.12 visualize_rollout.py rollout_data/episode_0.hdf5
    python3.12 visualize_rollout.py rollout_data/   # all episodes in a folder
"""

import argparse
import glob
import os
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np

CAMERA_KEYS = ['zed_front', 'dji_wrist']


# ── Data loading ───────────────────────────────────────────────────────────────

def load_episode(path: str) -> tuple[dict, dict]:
    data, attrs = {}, {}
    with h5py.File(path, 'r') as f:
        attrs['num_frames']         = f.attrs.get('num_frames', None)
        attrs['collection_rate_hz'] = f.attrs.get('collection_rate_hz', None)
        attrs['episode_index']      = f.attrs.get('episode_index', None)

        data['action_pose']      = f['action/pose'][()]
        data['action_gripper']   = f['action/gripper'][()]
        data['obs_pose']         = f['observation/pose'][()]
        data['obs_gripper']      = f['observation/gripper'][()]
        data['obs_joint_states'] = f['observation/joint_states'][()]

        if 'piezense/pressure_input' in f:
            data['piezense'] = f['piezense/pressure_input'][()]

        if 'predictions/action_horizon' in f:
            data['pred_horizon'] = f['predictions/action_horizon'][()]

        for key in CAMERA_KEYS:
            ds = f'images/{key}'
            if ds in f:
                data[key] = f[ds][()]

    return data, attrs


# ── Summary ────────────────────────────────────────────────────────────────────

def print_summary(path: str, data: dict, attrs: dict):
    T = data['action_pose'].shape[0]
    hz = attrs['collection_rate_hz'] or 30
    dur = T / hz
    print(f"\n{'─'*60}")
    print(f"  {os.path.basename(path)}")
    print(f"  {T} frames  @  {hz} Hz  ({dur:.1f} s)")
    print(f"{'─'*60}")

    def rng(arr): return f"[{arr.min():+.3f}, {arr.max():+.3f}]"
    print(f"  action/pose          {data['action_pose'].shape}   {rng(data['action_pose'])}")
    print(f"  action/gripper       {data['action_gripper'].shape}   {rng(data['action_gripper'])}")
    print(f"  observation/pose     {data['obs_pose'].shape}   {rng(data['obs_pose'])}")
    print(f"  observation/gripper  {data['obs_gripper'].shape}   {rng(data['obs_gripper'])}")
    if 'piezense' in data:
        print(f"  piezense             {data['piezense'].shape}   {rng(data['piezense'])}")
    else:
        print(f"  piezense             (not recorded)")
    if 'pred_horizon' in data:
        print(f"  predictions/horizon  {data['pred_horizon'].shape}")
    for key in CAMERA_KEYS:
        if key in data:
            print(f"  images/{key:<12} {data[key].shape}   dtype={data[key].dtype}")
        else:
            print(f"  images/{key:<12} (not recorded)")


# ── Figure ─────────────────────────────────────────────────────────────────────

def plot_episode(path: str, data: dict, attrs: dict, num_steps: int):
    T = data['action_pose'].shape[0]
    num_steps = min(num_steps, T)
    step_indices = np.linspace(0, T - 1, num_steps, dtype=int)
    timesteps = np.arange(T)

    available_cams = [k for k in CAMERA_KEYS if k in data]
    num_cams = len(available_cams)
    has_piezense = 'piezense' in data

    # Layout: camera rows + 1 plot row of 3 groups
    num_plot_cols = 3
    num_rows = num_cams + 1
    height_ratios = [1.2] * num_cams + [1.8]
    fig_w = max(16, 2.0 * num_steps)
    fig_h = 3.0 * num_cams + 4.0

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(
        num_rows, num_steps,
        height_ratios=height_ratios,
        hspace=0.35, wspace=0.05,
    )

    # ── Camera image rows ──────────────────────────────────────────────────────
    for row_idx, cam_key in enumerate(available_cams):
        imgs = data[cam_key]   # (T, 3, H, W) CHW RGB uint8
        for col_idx, t in enumerate(step_indices):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            ax.imshow(np.moveaxis(imgs[t], 0, -1))   # CHW → HWC
            ax.set_xticks([])
            ax.set_yticks([])
            if col_idx == 0:
                ax.set_ylabel(cam_key, fontsize=8)
            if row_idx == 0:
                ax.set_title(f't={t}', fontsize=7)

    # ── Time-series plots ──────────────────────────────────────────────────────
    col_breaks = np.array_split(np.arange(num_steps), num_plot_cols)

    def plot_span(group_idx):
        cols = col_breaks[group_idx]
        return gs[num_cams, cols[0]:cols[-1]+1]

    # Plot 1: TCP pose XYZ — action target vs observed
    ax1 = fig.add_subplot(plot_span(0))
    for i, lbl in enumerate('xyz'):
        ax1.plot(timesteps, data['action_pose'][:, i], label=f'act {lbl}')
        ax1.plot(timesteps, data['obs_pose'][:, i], label=f'obs {lbl}',
                 linestyle='--', alpha=0.7)
    ax1.set_title('TCP Pose XYZ (action vs obs)', fontsize=8)
    ax1.set_xlabel('timestep', fontsize=7)
    ax1.legend(fontsize=6, ncol=2)
    ax1.tick_params(labelsize=7)

    # Plot 2: Gripper — commanded vs measured
    ax2 = fig.add_subplot(plot_span(1))
    ax2.plot(timesteps, data['action_gripper'], label='commanded')
    ax2.plot(timesteps, data['obs_gripper'],    label='measured', linestyle='--')
    ax2.set_title('Gripper (0=open, 1=closed)', fontsize=8)
    ax2.set_xlabel('timestep', fontsize=7)
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(fontsize=7)
    ax2.tick_params(labelsize=7)

    # Plot 3: Piezense pressure (if recorded)
    ax3 = fig.add_subplot(plot_span(2))
    if has_piezense:
        pz = data['piezense']
        ax3.plot(timesteps, pz[:, 0], label='ch2 (Pa)', color='tomato',     alpha=0.8)
        ax3.plot(timesteps, pz[:, 1], label='ch3 (Pa)', color='darkorange', alpha=0.8)
        ax3.set_ylabel('pressure (Pa)', fontsize=7)
        ax3.legend(fontsize=6, loc='upper right')
        ax3.set_title('Piezense', fontsize=8)
    else:
        ax3.text(0.5, 0.5, 'piezense not recorded', ha='center', va='center',
                 transform=ax3.transAxes, fontsize=8, color='gray')
        ax3.set_title('Piezense', fontsize=8)
    ax3.set_xlabel('timestep', fontsize=7)
    ax3.tick_params(labelsize=7)

    hz = attrs['collection_rate_hz'] or 30
    dur = T / hz
    fig.suptitle(
        f"{os.path.basename(path)}  —  {T} frames  @  {hz} Hz  ({dur:.1f} s)  [rollout]",
        fontsize=10,
    )
    plt.tight_layout()
    plt.show()


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Visualize an HDF5 inference rollout')
    parser.add_argument('episode',
                        help='Path to episode_N.hdf5 or a directory of episodes')
    parser.add_argument('--num-steps', type=int, default=10,
                        help='Number of timesteps to show in the image grid (default: 10)')
    args = parser.parse_args()

    if os.path.isdir(args.episode):
        paths = sorted(glob.glob(os.path.join(args.episode, 'episode_*.hdf5')))
        if not paths:
            print(f'No episode_*.hdf5 files found in {args.episode}')
            sys.exit(1)
    else:
        paths = [args.episode]

    for path in paths:
        try:
            data, attrs = load_episode(path)
        except Exception as e:
            print(f'[ERROR] Could not load {path}: {e}')
            continue

        print_summary(path, data, attrs)
        plot_episode(path, data, attrs, args.num_steps)


if __name__ == '__main__':
    main()
