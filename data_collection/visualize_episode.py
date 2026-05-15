#!/usr/bin/env python3
"""
Visualize a single HDF5 episode collected by hdf5_data_collector.py.

Adapted from Robomimic/data_collection/visualize_episode.py.

Displays:
  - Top rows:    Camera images (zed_front, dji_wrist) at N evenly-spaced timesteps
  - Bottom plots: Action/obs TCP pose XYZ, hand pose XYZ, gripper, hand width +
                  piezense pressure (if recorded)

Usage:
    python3.12 visualize_episode.py <episode.hdf5> [--num-steps 10]
    python3.12 visualize_episode.py demo_data/episode_3.hdf5
    python3.12 visualize_episode.py demo_data/  # visualise all episodes in a folder
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
        attrs['num_frames']        = f.attrs.get('num_frames', None)
        attrs['collection_rate_hz'] = f.attrs.get('collection_rate_hz', None)
        attrs['episode_index']     = f.attrs.get('episode_index', None)

        data['action_pose']       = f['action/pose'][()]
        data['action_gripper']    = f['action/gripper'][()]
        data['obs_pose']          = f['observation/pose'][()]
        data['obs_gripper']       = f['observation/gripper'][()]
        data['obs_joint_states']  = f['observation/joint_states'][()]

        data['hand_pose']         = f['hololens/hand_pose_robot_frame'][()]
        data['hand_width']        = f['hololens/hand_width'][()]
        data['palm_pose']         = f['hololens/palm_pose'][()]

        if 'piezense/pressure_input' in f:
            data['piezense'] = f['piezense/pressure_input'][()]

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
    print(f"  hand_pose (robot)    {data['hand_pose'].shape}   {rng(data['hand_pose'])}")
    print(f"  hand_width           {data['hand_width'].shape}   {rng(data['hand_width'])}")
    if 'piezense' in data:
        print(f"  piezense             {data['piezense'].shape}   {rng(data['piezense'])}")
    else:
        print(f"  piezense             (not recorded)")
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

    # Layout: camera rows + 1 plot row
    num_plot_cols = 4
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
    # Divide the bottom gridspec row into 4 equal groups of columns
    col_breaks = np.array_split(np.arange(num_steps), num_plot_cols)

    def plot_span(group_idx):
        cols = col_breaks[group_idx]
        return gs[num_cams, cols[0]:cols[-1]+1]

    # Plot 1: TCP pose XYZ — action target vs observed
    ax1 = fig.add_subplot(plot_span(0))
    for i, (lbl, ls) in enumerate(zip('xyz', ['-', '-', '-'])):
        ax1.plot(timesteps, data['action_pose'][:, i],
                 label=f'act {lbl}', linestyle=ls)
        ax1.plot(timesteps, data['obs_pose'][:, i],
                 label=f'obs {lbl}', linestyle='--', alpha=0.7)
    ax1.set_title('TCP Pose XYZ (action vs obs)', fontsize=8)
    ax1.set_xlabel('timestep', fontsize=7)
    ax1.legend(fontsize=6, ncol=2)
    ax1.tick_params(labelsize=7)

    # Plot 2: Hand pose XYZ (robot frame)
    ax2 = fig.add_subplot(plot_span(1))
    for i, lbl in enumerate('xyz'):
        ax2.plot(timesteps, data['hand_pose'][:, i], label=lbl)
    ax2.set_title('Hand Pose XYZ (robot frame)', fontsize=8)
    ax2.set_xlabel('timestep', fontsize=7)
    ax2.legend(fontsize=7)
    ax2.tick_params(labelsize=7)

    # Plot 3: Gripper — commanded vs measured
    ax3 = fig.add_subplot(plot_span(2))
    ax3.plot(timesteps, data['action_gripper'], label='commanded')
    ax3.plot(timesteps, data['obs_gripper'],    label='measured', linestyle='--')
    ax3.set_title('Gripper (0=open, 1=closed)', fontsize=8)
    ax3.set_xlabel('timestep', fontsize=7)
    ax3.set_ylim(-0.05, 1.05)
    ax3.legend(fontsize=7)
    ax3.tick_params(labelsize=7)

    # Plot 4: Hand width + piezense (if available)
    ax4 = fig.add_subplot(plot_span(3))
    ax4.plot(timesteps, data['hand_width'] * 100, label='hand width (cm)', color='steelblue')
    ax4.set_ylabel('hand width (cm)', fontsize=7, color='steelblue')
    ax4.tick_params(axis='y', labelcolor='steelblue', labelsize=7)
    ax4.set_xlabel('timestep', fontsize=7)
    if has_piezense:
        ax4b = ax4.twinx()
        pz = data['piezense']
        ax4b.plot(timesteps, pz[:, 0], label='ch2 (Pa)', color='tomato',   alpha=0.8)
        ax4b.plot(timesteps, pz[:, 1], label='ch3 (Pa)', color='darkorange', alpha=0.8)
        ax4b.set_ylabel('pressure (Pa)', fontsize=7, color='tomato')
        ax4b.tick_params(axis='y', labelcolor='tomato', labelsize=7)
        ax4b.legend(fontsize=6, loc='upper right')
    ax4.set_title('Hand Width' + (' + Piezense' if has_piezense else ''), fontsize=8)
    ax4.legend(fontsize=6, loc='upper left')
    ax4.tick_params(axis='x', labelsize=7)

    hz = attrs['collection_rate_hz'] or 30
    dur = T / hz
    fig.suptitle(
        f"{os.path.basename(path)}  —  {T} frames  @  {hz} Hz  ({dur:.1f} s)",
        fontsize=10,
    )
    plt.tight_layout()
    plt.show()


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Visualize an HDF5 episode')
    parser.add_argument('episode',
                        help='Path to episode_N.hdf5 or a directory of episodes')
    parser.add_argument('--num-steps', type=int, default=10,
                        help='Number of timesteps to show in the image grid (default: 10)')
    args = parser.parse_args()

    # Accept either a single file or a directory
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
