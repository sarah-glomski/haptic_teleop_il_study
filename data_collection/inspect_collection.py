#!/usr/bin/env python3
"""
Batch collection inspector — outlier detection before zarr conversion.

Loads every episode_*.hdf5 in a folder, computes per-episode stats, prints
a ranked summary table, and shows a dashboard figure so you can decide which
episodes to exclude.

Outlier signals flagged (★):
  duration    — more than 2 σ from the mean
  path length — TCP total distance traveled; too short = robot barely moved
  act-obs err — RMS error between commanded and observed TCP; high = bad tracking
  gripper     — gripper never closed (max < 0.3) or never opened (min > 0.7)

Usage:
    python3.12 inspect_collection.py demo_data/Collection3/
    python3.12 inspect_collection.py demo_data/Collection3/ --num-steps 6
"""

import argparse
import datetime as _dt
import glob
import os
import sys

import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

try:
    import natsort
    _sort = natsort.natsorted
except ImportError:
    _sort = sorted


# ── exclude.txt: full drops + end-crops ─────────────────────────────────────────
#
# Format (shared with inspect_transitions.py — keep the two in sync):
#     episode_N              drop the whole episode
#     episode_N crop S E     keep frames [S:E) at conversion (end-crop only)
# Both readers and writers here are ADDITIVE: an exclude written by this script
# must never discard crops recorded by inspect_transitions.py, nor drops made in
# an earlier run. To remove an entry, edit exclude.txt by hand.

def _epnum(name):
    digits = ''.join(c for c in name.replace('episode_', '') if c.isdigit())
    return int(digits) if digits else 0


def parse_exclude(path):
    """Return (full_drops:set[str], crops:dict[str,(start,end)])."""
    full, crops = set(), {}
    if os.path.exists(path):
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                name = parts[0]
                if len(parts) >= 4 and parts[1] == 'crop':
                    crops[name] = (int(parts[2]), int(parts[3]))
                else:
                    full.add(name)
    return full, crops


def write_exclude(path, full, crops):
    lines = ['# episode_N            → drop whole episode',
             '# episode_N crop S E   → keep frames [S:E) at conversion (end-crop)']
    for n in sorted(full, key=_epnum):
        lines.append(n)
    for n in sorted(crops, key=_epnum):
        if n in full:
            continue                       # a full drop supersedes a crop
        s, e = crops[n]
        lines.append(f'{n} crop {s} {e}')
    with open(path, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')


# ── Per-episode stats ──────────────────────────────────────────────────────────

def compute_stats(path: str, crop=None) -> dict | None:
    """crop=(start,end) truncates to the same window conversion will use, so the
    table and dashboard reflect the to-be-converted data. Originals are never
    modified — the crop lives only in exclude.txt."""
    try:
        with h5py.File(path, 'r') as f:
            hz  = float(f.attrs.get('collection_rate_hz', 30))
            act_pos = f['action/pose'][:, :3].astype(np.float64)    # (T,3)
            obs_pos = f['observation/pose'][:, :3].astype(np.float64)
            act_grip = f['action/gripper'][()].astype(np.float64)   # (T,)
            # Piezense force is optional — collections recorded before the
            # sensor existed, or with enable_piezense:=false, have no group.
            force = (f['piezense/pressure_input'][()].astype(np.float64)
                     if 'piezense/pressure_input' in f else None)   # (T,2) Pa

        if crop is not None:
            s, e = crop
            act_pos, obs_pos, act_grip = act_pos[s:e], obs_pos[s:e], act_grip[s:e]
            if force is not None:
                force = force[s:e]

        T = len(act_pos)
        steps = np.diff(act_pos, axis=0)
        path_len = float(np.sum(np.linalg.norm(steps, axis=1)))
        act_obs_rmse = float(np.sqrt(np.mean((act_pos - obs_pos) ** 2)))

        return dict(
            path       = path,
            name       = os.path.basename(path),
            T          = T,
            hz         = hz,
            dur_s      = T / hz,
            path_len   = path_len,
            rmse       = act_obs_rmse,
            grip_min   = float(act_grip.min()),
            grip_max   = float(act_grip.max()),
            tcp_xmin   = float(act_pos[:, 0].min()),
            tcp_xmax   = float(act_pos[:, 0].max()),
            tcp_ymin   = float(act_pos[:, 1].min()),
            tcp_ymax   = float(act_pos[:, 1].max()),
            tcp_zmin   = float(act_pos[:, 2].min()),
            tcp_zmax   = float(act_pos[:, 2].max()),
            act_pos    = act_pos,
            obs_pos    = obs_pos,
            act_grip   = act_grip,
            force      = force,
            crop       = crop,
        )
    except Exception as e:
        print(f'[WARN] Could not load {path}: {e}')
        return None


# ── Outlier detection ──────────────────────────────────────────────────────────

def flag_outliers(rows: list[dict]) -> list[dict]:
    for key in ('dur_s', 'path_len', 'rmse'):
        vals = np.array([r[key] for r in rows])
        mu, sigma = vals.mean(), vals.std()
        for r in rows:
            r.setdefault('flags', [])
            if abs(r[key] - mu) > 2 * sigma:
                r['flags'].append(key)
    for r in rows:
        r.setdefault('flags', [])
        if r['grip_max'] < 0.3:
            r['flags'].append('grip_never_closed')
        if r['grip_min'] > 0.7:
            r['flags'].append('grip_never_opened')
    return rows


# ── Terminal table ─────────────────────────────────────────────────────────────

def print_table(rows: list[dict]):
    GREEN, RED, YELLOW, RESET, BOLD = (
        '\033[92m', '\033[91m', '\033[93m', '\033[0m', '\033[1m')

    hdr = (f"{'Episode':<12} {'Frames':>6} {'Dur(s)':>7} "
           f"{'PathLen(m)':>10} {'RMSE(m)':>8} "
           f"{'Grip min':>8} {'Grip max':>8}  Flags")
    print(f'\n{BOLD}{hdr}{RESET}')
    print('─' * len(hdr))

    for r in rows:
        flags = r.get('flags', [])
        flag_str = ', '.join(flags) if flags else ''
        color = RED if flags else GREEN
        ep = r['name'].replace('episode_', '').replace('.hdf5', '')
        if r.get('crop') is not None:
            # Mark it: the row shows post-crop numbers, so a short duration here
            # is the crop doing its job rather than a truncated demo.
            ep += ' (crop)'
        print(
            f"{color}{ep:<12}{RESET} "
            f"{r['T']:>6d} "
            f"{r['dur_s']:>7.1f} "
            f"{r['path_len']:>10.3f} "
            f"{r['rmse']:>8.4f} "
            f"{r['grip_min']:>8.3f} "
            f"{r['grip_max']:>8.3f}  "
            f"{YELLOW}{flag_str}{RESET}"
        )

    flagged = [r['name'].replace('.hdf5', '') for r in rows if r.get('flags')]
    print(f'\n{len(flagged)} / {len(rows)} episodes flagged as potential outliers.')
    if flagged:
        print(f'  Flagged: {", ".join(flagged)}')


# ── Dashboard figure ───────────────────────────────────────────────────────────

def plot_dashboard(rows: list[dict], collection_dir: str, num_steps: int,
                   out_path: str):
    n   = len(rows)
    eps = [r['name'].replace('episode_', 'ep').replace('.hdf5', '') for r in rows]
    x   = np.arange(n)

    dur       = np.array([r['dur_s']   for r in rows])
    path_lens = np.array([r['path_len'] for r in rows])
    rmses     = np.array([r['rmse']     for r in rows])
    grip_min  = np.array([r['grip_min'] for r in rows])
    grip_max  = np.array([r['grip_max'] for r in rows])
    flagged   = [bool(r.get('flags')) for r in rows]

    colors = ['tomato' if f else 'steelblue' for f in flagged]

    fig = plt.figure(figsize=(18, 16))
    # Last row holds only the text summary, so it needs less height than the plots.
    gs  = fig.add_gridspec(4, 3, hspace=0.45, wspace=0.35,
                           height_ratios=[1, 1, 1, 0.4])

    # ── Duration ──────────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    ax.bar(x, dur, color=colors)
    ax.axhline(dur.mean(), color='k', linestyle='--', linewidth=0.8, label=f'mean {dur.mean():.1f}s')
    ax.axhspan(dur.mean() - 2*dur.std(), dur.mean() + 2*dur.std(), alpha=0.12, color='grey', label='±2σ')
    ax.set_xticks(x[::max(1, n//16)])
    ax.set_xticklabels(eps[::max(1, n//16)], rotation=45, fontsize=6)
    ax.set_ylabel('Duration (s)', fontsize=8)
    ax.set_title('Episode Duration', fontsize=9)
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=7)

    # ── Path length ───────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    ax.bar(x, path_lens, color=colors)
    ax.axhline(path_lens.mean(), color='k', linestyle='--', linewidth=0.8,
               label=f'mean {path_lens.mean():.2f}m')
    ax.axhspan(path_lens.mean() - 2*path_lens.std(),
               path_lens.mean() + 2*path_lens.std(), alpha=0.12, color='grey')
    ax.set_xticks(x[::max(1, n//16)])
    ax.set_xticklabels(eps[::max(1, n//16)], rotation=45, fontsize=6)
    ax.set_ylabel('Path length (m)', fontsize=8)
    ax.set_title('TCP Total Path Length', fontsize=9)
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=7)

    # ── Action-obs RMSE ───────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 2])
    ax.bar(x, rmses * 1000, color=colors)
    ax.axhline(rmses.mean() * 1000, color='k', linestyle='--', linewidth=0.8,
               label=f'mean {rmses.mean()*1000:.1f}mm')
    ax.axhspan((rmses.mean() - 2*rmses.std()) * 1000,
               (rmses.mean() + 2*rmses.std()) * 1000, alpha=0.12, color='grey')
    ax.set_xticks(x[::max(1, n//16)])
    ax.set_xticklabels(eps[::max(1, n//16)], rotation=45, fontsize=6)
    ax.set_ylabel('RMSE (mm)', fontsize=8)
    ax.set_title('Action-Obs TCP Error', fontsize=9)
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=7)

    # ── Gripper range ─────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    ax.fill_between(x, grip_min, grip_max, alpha=0.4, color='steelblue', label='min→max range')
    ax.plot(x, grip_min, 'o-', markersize=3, linewidth=0.8, color='steelblue', label='min')
    ax.plot(x, grip_max, 's-', markersize=3, linewidth=0.8, color='darkorange', label='max')
    ax.axhline(0.3, color='red', linestyle=':', linewidth=0.8, alpha=0.6)
    ax.axhline(0.7, color='green', linestyle=':', linewidth=0.8, alpha=0.6)
    ax.set_xticks(x[::max(1, n//16)])
    ax.set_xticklabels(eps[::max(1, n//16)], rotation=45, fontsize=6)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel('Gripper (0=open, 1=closed)', fontsize=8)
    ax.set_title('Gripper Range per Episode', fontsize=9)
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=7)

    # ── TCP XY trajectory overlay ─────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    cmap = matplotlib.colormaps['tab20']
    for i, r in enumerate(rows):
        pos = r['act_pos']
        lw  = 1.5 if flagged[i] else 0.6
        col = 'red' if flagged[i] else cmap(i % 20)
        ax.plot(pos[:, 0], pos[:, 1], linewidth=lw, color=col, alpha=0.7)
        ax.plot(pos[0, 0], pos[0, 1], 'o', markersize=3, color=col)
    ax.set_xlabel('X (m)', fontsize=8)
    ax.set_ylabel('Y (m)', fontsize=8)
    ax.set_title('TCP XY Trajectories (red = flagged)', fontsize=9)
    ax.tick_params(labelsize=7)
    ax.set_aspect('equal', adjustable='datalim')

    # ── TCP Z over time (normalised) ──────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 2])
    for i, r in enumerate(rows):
        pos = r['act_pos']
        t   = np.linspace(0, 1, len(pos))
        lw  = 1.5 if flagged[i] else 0.4
        col = 'red' if flagged[i] else cmap(i % 20)
        ax.plot(t, pos[:, 2], linewidth=lw, color=col, alpha=0.7)
    ax.set_xlabel('Normalised time', fontsize=8)
    ax.set_ylabel('Z (m)', fontsize=8)
    ax.set_title('TCP Z Profile (red = flagged)', fontsize=9)
    ax.tick_params(labelsize=7)

    # ── Gripper time profiles (normalised) ───────────────────────────────────
    ax = fig.add_subplot(gs[2, 0])
    for i, r in enumerate(rows):
        t  = np.linspace(0, 1, len(r['act_grip']))
        lw = 1.5 if flagged[i] else 0.4
        col = 'red' if flagged[i] else cmap(i % 20)
        ax.plot(t, r['act_grip'], linewidth=lw, color=col, alpha=0.7)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel('Normalised time', fontsize=8)
    ax.set_ylabel('Gripper cmd', fontsize=8)
    ax.set_title('Gripper Command Profiles', fontsize=9)
    ax.tick_params(labelsize=7)

    # ── Piezense force profiles (normalised time) ─────────────────────────────
    # One line per episode, same overlay style as the gripper panel, so a demo
    # with an atypical contact profile stands out against the others. The two
    # input channels get their own axes — a per-channel offset or a dead channel
    # is invisible when they share one. ch2/ch3 naming matches
    # visualize_episode.py (array columns 0 and 1).
    with_force = [r for r in rows if r.get('force') is not None]
    for ch, col_idx in ((0, 1), (1, 2)):
        ax = fig.add_subplot(gs[2, col_idx])
        plotted = 0
        for i, r in enumerate(rows):
            fr = r.get('force')
            if fr is None or fr.ndim != 2 or fr.shape[1] <= ch:
                continue
            t   = np.linspace(0, 1, len(fr))
            lw  = 1.5 if flagged[i] else 0.4
            col = 'red' if flagged[i] else cmap(i % 20)
            ax.plot(t, fr[:, ch], linewidth=lw, color=col, alpha=0.7)
            plotted += 1
        if plotted:
            ax.set_xlabel('Normalised time', fontsize=8)
            ax.set_ylabel('Pressure (Pa)', fontsize=8)
            ax.set_title(f'Force Profile — ch{ch + 2}  '
                         f'({plotted}/{n} episodes, red = flagged)', fontsize=9)
        else:
            ax.text(0.5, 0.5, 'no piezense data recorded', ha='center', va='center',
                    transform=ax.transAxes, fontsize=8, color='grey')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f'Force Profile — ch{ch + 2}', fontsize=9)
        ax.tick_params(labelsize=7)

    # ── Flagged episodes summary ───────────────────────────────────────────────
    ax = fig.add_subplot(gs[3, :])
    ax.axis('off')
    lines = [f'Collection: {os.path.basename(collection_dir.rstrip("/"))}',
             f'{n} episodes   {dur.mean():.1f}s avg   {dur.std():.1f}s σ',
             '']
    flagged_rows = [r for r in rows if r.get('flags')]
    if flagged_rows:
        lines.append(f'Flagged ({len(flagged_rows)}):')
        for r in flagged_rows:
            ep = r['name'].replace('.hdf5', '')
            lines.append(f'  {ep:<18}  {", ".join(r["flags"])}')
    else:
        lines.append('No outliers detected.')
    ax.text(0.02, 0.95, '\n'.join(lines), transform=ax.transAxes,
            fontsize=8, verticalalignment='top', fontfamily='monospace')

    fig.suptitle(f'Collection inspection — {os.path.basename(collection_dir.rstrip("/"))}',
                 fontsize=11)

    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    print(f'\nDashboard saved → {out_path}')
    plt.show()


# ── Drill-down: show one episode with wrist camera frames ─────────────────────

def plot_episode_detail(row: dict, num_steps: int):
    T   = row['T']
    hz  = row['hz']
    t   = np.arange(T) / hz
    idx = np.linspace(0, T - 1, num_steps, dtype=int)

    has_cam = 'dji_wrist' in row
    n_rows  = (1 if has_cam else 0) + 1
    h_ratios = ([1.5] if has_cam else []) + [2.0]

    fig = plt.figure(figsize=(max(14, 2*num_steps), 3*n_rows + 1))
    gs  = fig.add_gridspec(n_rows, num_steps, height_ratios=h_ratios,
                           hspace=0.35, wspace=0.05)

    if has_cam:
        imgs = row['dji_wrist']
        for col, t_i in enumerate(idx):
            ax = fig.add_subplot(gs[0, col])
            frame = np.ascontiguousarray(np.moveaxis(imgs[t_i], 0, -1))
            ax.imshow(frame, aspect='auto')
            ax.set_xticks([]); ax.set_yticks([])
            if col == 0: ax.set_ylabel('dji_wrist', fontsize=8)
            ax.set_title(f't={t_i}', fontsize=7)

    col_breaks = np.array_split(np.arange(num_steps), 3)
    def span(g):
        c = col_breaks[g]; return gs[n_rows-1, c[0]:c[-1]+1]

    ax1 = fig.add_subplot(span(0))
    for i, lbl in enumerate('xyz'):
        ax1.plot(t, row['act_pos'][:, i], label=f'act {lbl}')
        ax1.plot(t, row['obs_pos'][:, i], linestyle='--', alpha=0.6, label=f'obs {lbl}')
    ax1.set_title('TCP XYZ', fontsize=8); ax1.legend(fontsize=6, ncol=2)
    ax1.tick_params(labelsize=7)

    ax2 = fig.add_subplot(span(1))
    ax2.plot(t, row['act_grip'], label='cmd')
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_title('Gripper', fontsize=8); ax2.legend(fontsize=7)
    ax2.tick_params(labelsize=7)

    ax3 = fig.add_subplot(span(2))
    err = np.linalg.norm(row['act_pos'] - row['obs_pos'], axis=1) * 1000
    ax3.plot(t, err, color='tomato')
    ax3.set_title('Act-obs error (mm)', fontsize=8)
    ax3.tick_params(labelsize=7)

    if has_cam:
        del row['dji_wrist']   # free memory after plotting

    fig.suptitle(row['name'], fontsize=10, y=1.01)
    fig.tight_layout()
    plt.show(block=True)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Inspect a collection folder for outliers')
    parser.add_argument('collection', help='Path to collection directory')
    parser.add_argument('--num-steps', type=int, default=8,
                        help='Camera frames shown in detail view (default 8)')
    parser.add_argument('--detail', nargs='*', metavar='N',
                        help='Episode numbers to open in detail view (e.g. --detail 3 7)')
    parser.add_argument('--exclude', nargs='*', metavar='N',
                        help='Add to the exclude list (e.g. --exclude 0 4 6). '
                             'Omit numbers to be prompted interactively. '
                             'Additive — existing drops and crops are preserved.')
    parser.add_argument('--out', default=None,
                        help='Dashboard path (default: timestamped, never overwrites)')
    args = parser.parse_args()

    paths = _sort(glob.glob(os.path.join(args.collection, 'episode_*.hdf5')))
    if not paths:
        print(f'No episode_*.hdf5 files found in {args.collection}')
        sys.exit(1)

    # Skip episodes fully dropped in exclude.txt. Crops are read too — not to
    # apply them here, but so they survive an --exclude write later on.
    exclude_file = os.path.join(args.collection, 'exclude.txt')
    excluded, existing_crops = parse_exclude(exclude_file)
    if excluded:
        print(f'Skipping {len(excluded)} excluded episodes: '
              f'{", ".join(sorted(excluded, key=_epnum))}')
    if existing_crops:
        print(f'{len(existing_crops)} end-crop(s) recorded (applied at conversion, '
              f'not here): '
              f'{", ".join(f"{n}[:{e}]" for n, (s, e) in sorted(existing_crops.items(), key=lambda kv: _epnum(kv[0])))}')
    paths = [p for p in paths
             if os.path.basename(p).replace('.hdf5', '') not in excluded]

    print(f'Loading {len(paths)} episodes from {args.collection} …')

    # Load stats (no images yet — keep memory low). Cropped episodes are
    # truncated here so every stat, flag and plot describes the data that will
    # actually be converted, not the discarded tail.
    rows = []
    for p in paths:
        s = compute_stats(p, existing_crops.get(
            os.path.basename(p).replace('.hdf5', '')))
        if s:
            rows.append(s)
    print(f'Loaded {len(rows)} episodes.')

    rows = flag_outliers(rows)
    print_table(rows)

    # Timestamped so successive runs never overwrite each other — same
    # convention as inspect_transitions.py's transition_inspection_<ts>.png.
    out = args.out or os.path.join(
        args.collection,
        f'collection_inspection_{_dt.datetime.now():%Y%m%d_%H%M%S}.png')
    plot_dashboard(rows, args.collection, args.num_steps, out)

    # Exclude list
    if args.exclude is not None:
        if args.exclude:
            to_exclude = args.exclude
        else:
            print('\nEnter episode numbers to exclude (space-separated), then ENTER:')
            to_exclude = input('> ').split()

        # `rows` holds only the episodes still in play, so an already-excluded
        # number must count as valid too — otherwise re-running --exclude 3
        # reports episode 3 as "unknown".
        valid_names = {r['name'].replace('episode_', '').replace('.hdf5', '') for r in rows}
        valid_names |= {str(_epnum(n)) for n in excluded}
        bad = [e for e in to_exclude if e not in valid_names]
        if bad:
            print(f'[WARN] Unknown episode numbers (ignored): {bad}')
        to_exclude = [e for e in to_exclude if e in valid_names]

        # ADDITIVE: union with what is already in the file, and carry the crop
        # lines through untouched. This used to open the file with 'w' and write
        # only the newly chosen numbers, which silently destroyed both earlier
        # drops and every end-crop recorded by inspect_transitions.py.
        merged = set(excluded) | {f'episode_{e}' for e in to_exclude}
        added  = sorted(merged - set(excluded), key=_epnum)
        write_exclude(exclude_file, merged, existing_crops)

        print(f'\nExclude list written → {exclude_file}')
        print('  dropped: ' + '  '.join(sorted(merged, key=_epnum)))
        if added:
            print('  added this run: ' + '  '.join(added))
        else:
            print('  (nothing new — all selections were already listed)')
        if existing_crops:
            print('  crops preserved: ' + '  '.join(
                f'{n}[{s}:{e}]' for n, (s, e) in
                sorted(existing_crops.items(), key=lambda kv: _epnum(kv[0]))))

    plt.close('all')   # clear dashboard before opening detail windows

    # Detail view — load images only for requested episodes
    if args.detail is not None:
        wanted = set(args.detail) or {r['name'].replace('episode_', '').replace('.hdf5', '')
                                       for r in rows if r.get('flags')}
        for r in rows:
            ep_num = r['name'].replace('episode_', '').replace('.hdf5', '')
            if ep_num in wanted:
                print(f'\nLoading detail for {r["name"]} …')
                with h5py.File(r['path'], 'r') as f:
                    if 'images/dji_wrist' in f:
                        # Slice to the same window as the signals — otherwise the
                        # camera strip would show frames from the cropped-away
                        # tail while the plots below it stop at the crop.
                        if r.get('crop') is not None:
                            cs, ce = r['crop']
                            r['dji_wrist'] = f['images/dji_wrist'][cs:ce]
                        else:
                            r['dji_wrist'] = f['images/dji_wrist'][()]
                plot_episode_detail(r, args.num_steps)


if __name__ == '__main__':
    main()
