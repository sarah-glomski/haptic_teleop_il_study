#!/usr/bin/env python3
"""
Transition / state-machine inspector for a pick-and-place collection.

Where inspect_collection.py ranks episodes by size + tracking quality, this
script curates for a *diffusion-policy state machine*. It segments every episode
into the 7 task STATES, then separates two things:

    STATE diversity      (want HIGH) — the range of situations spanned WITHIN
        each state across the dataset (e.g. many object/transport positions).
    TRANSITION diversity (want LOW)  — the spread, across demos, of the robot
        CONDITION at each state→state switch (the "condition to move on").

The 7 states (segmented from gripper events + vertical/lateral motion):
    1 servo      visual-servo laterally until gripper is above the object
    2 descend    lower straight down until gripper is around the object
    3 grasp      close the gripper decisively (TCP roughly still)
    4 lift       raise straight up with the object
    5 transport  move laterally over to the light-blue target tape
    6 place      lower straight down onto the tape
    7 release    open the gripper decisively (TCP roughly still)

The 6 transitions are the switches between consecutive states. A clean state
machine fires each switch under a CONSISTENT condition, so per boundary we read
the robot's (Z height, TCP speed, gripper, XY) at the switch frame and measure
how tightly those cluster across demos.

A 7th "start→servo" entry condition checks the *initial* state: the arm should
begin at the home pose (--home-dev-max) with the gripper open (--home-grip-max).
Episodes that start off-home or not-open are flagged `bad_home` and recommended
for exclusion regardless of state novelty.

Per-episode signals
    transition messiness  (rank for exclusion) — deviation of this demo's switch
        conditions from the dataset median: inconsistent switch height / speed,
        place-over-tape scatter (the tape is FIXED), non-crisp gripper switches
        (TCP still moving while grasping/releasing), gripper indecisiveness,
        plus hard faults (re-grasp, broken state sequence).
    state novelty         (protect from exclusion) — how rare this episode's
        grasp location is; a messy trial that is the only one covering a region
        still adds STATE diversity, so it is not auto-dropped.

Outputs a timestamped dashboard PNG (never overwrites inspect_collection's
inspection.png) whose headline panels show the state-vs-transition split.

Usage:
    python3.12 inspect_transitions.py demo_data/Collection4/
    python3.12 inspect_transitions.py demo_data/Collection4/ --detail 28 34
    python3.12 inspect_transitions.py demo_data/Collection4/ --exclude 28
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

PHASES = ['servo', 'descend', 'grasp', 'lift', 'transport', 'place', 'release']
# The 6 transitions are the switches between consecutive states; each is named
# by the state it enters. The switch frame is that state's first frame.
TRANSITIONS = [
    ('servo→descend',    'descend'),
    ('descend→grasp',    'grasp'),
    ('grasp→lift',       'lift'),
    ('lift→transport',   'transport'),
    ('transport→place',  'place'),
    ('place→release',    'release'),
]
EPS = 1e-9


# ── Small helpers ───────────────────────────────────────────────────────────────

def _runs(mask):
    """Return list of [start, end) index runs where boolean `mask` is True."""
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return []
    splits = np.where(np.diff(idx) > 1)[0] + 1
    return [(g[0], g[-1] + 1) for g in np.split(idx, splits)]


# ── Gripper events ──────────────────────────────────────────────────────────────

def _ramp_bounds(grip, e, rising):
    """Frame span over which the gripper crosses 0.2↔0.8 around event index e."""
    lo_lvl, hi_lvl = (0.2, 0.8)
    if rising:
        lo = np.where(grip[:e + 1] < lo_lvl)[0]
        hi = np.where(grip[e:] > hi_lvl)[0]
    else:
        lo = np.where(grip[:e + 1] > hi_lvl)[0]
        hi = np.where(grip[e:] < lo_lvl)[0]
    a = lo[-1] if len(lo) else e
    b = (hi[0] + e) if len(hi) else e
    return int(min(a, b)), int(max(a, b))


def _grip_events(grip):
    closes = [i for i in range(1, len(grip)) if grip[i - 1] < 0.5 <= grip[i]]
    opens = [i for i in range(1, len(grip)) if grip[i - 1] >= 0.5 > grip[i]]
    return closes, opens


# ── Phase segmentation ──────────────────────────────────────────────────────────

def segment(pos, grip, hz):
    """Return (phase_bounds dict, flags list). phase_bounds maps name→[start,end)."""
    T = len(pos)
    flags = []
    closes, opens = _grip_events(grip)
    n_toggles = len(closes) + len(opens)

    # Decisive grasp = first close; release = last open after the grasp.
    valid_opens = [o for o in opens if closes and o > closes[0]]
    if not closes or not valid_opens:
        flags.append('broken_state_machine')
        return None, flags
    if n_toggles > 2:
        flags.append(f're_grasp(x{n_toggles - 2})')

    gc_lo, gc_hi = _ramp_bounds(grip, closes[0], rising=True)
    go_lo, go_hi = _ramp_bounds(grip, valid_opens[-1], rising=False)
    gc_lo = max(1, gc_lo)
    go_hi = min(T, go_hi + 1)
    if not (0 < gc_lo <= gc_hi < go_lo <= go_hi <= T):
        flags.append('segmentation_failed')
        return None, flags

    # Velocity signals (per frame, length T).
    vel = np.gradient(pos, 1.0 / hz, axis=0)
    horiz = np.linalg.norm(vel[:, :2], axis=1)
    vz = vel[:, 2]
    vertness = np.abs(vz) / (horiz + np.abs(vz) + EPS)   # 1 = purely vertical

    def _last_run_start(a, b, mask, default):
        rr = [r for r in _runs(mask[a:b]) if r[1] - r[0] >= 2]
        return a + rr[-1][0] if rr else default

    def _first_run_end(a, b, mask, default):
        rr = [r for r in _runs(mask[a:b]) if r[1] - r[0] >= 2]
        return a + rr[0][1] if rr else default

    # Pre-grasp [0, gc_lo): servo (lateral) then descend (vertical, sinking).
    descend_start = _last_run_start(
        0, gc_lo, (vertness > 0.5) & (vz < 0), default=int(0.6 * gc_lo))
    descend_start = int(np.clip(descend_start, 1, gc_lo - 1)) if gc_lo > 1 else gc_lo

    # Between grasp and release [gc_hi, go_lo): lift (up) → transport → place (down).
    lift_end = _first_run_end(
        gc_hi, go_lo, (vertness > 0.4) & (vz > 0), default=gc_hi + max(1, (go_lo - gc_hi) // 3))
    place_start = _last_run_start(
        gc_hi, go_lo, (vertness > 0.4) & (vz < 0), default=go_lo - max(1, (go_lo - gc_hi) // 3))
    lift_end = int(np.clip(lift_end, gc_hi + 1, go_lo - 1))
    place_start = int(np.clip(place_start, lift_end, go_lo - 1))

    bounds = {
        'servo':     (0, descend_start),
        'descend':   (descend_start, gc_lo),
        'grasp':     (gc_lo, gc_hi + 1),
        'lift':      (gc_hi + 1, lift_end),
        'transport': (lift_end, place_start),
        'place':     (place_start, go_lo),
        'release':   (go_lo, go_hi),
    }
    # Any empty phase → note partial and pad by one frame so profiles exist.
    for k, (a, b) in bounds.items():
        if b <= a:
            flags.append('seg_partial')
            bounds[k] = (a, min(T, a + 1))
    return bounds, flags


# ── Per-episode stats ────────────────────────────────────────────────────────────

def compute_stats(path, crop=None):
    """crop=(start,end) truncates to the same window conversion will use, so the
    dashboard reflects the to-be-converted data (originals are never modified)."""
    try:
        with h5py.File(path, 'r') as f:
            hz = float(f.attrs.get('collection_rate_hz', 30))
            pos = f['action/pose'][:, :3].astype(np.float64)
            grip = f['action/gripper'][()].astype(np.float64)
    except Exception as e:
        print(f'[WARN] Could not load {path}: {e}')
        return None

    if crop is not None:
        s, e = crop
        pos, grip = pos[s:e], grip[s:e]

    bounds, flags = segment(pos, grip, hz)
    speed = np.linalg.norm(np.gradient(pos, 1.0 / hz, axis=0), axis=1)
    pause_frac = float(np.mean(speed < 0.02))          # <2 cm/s ≈ idle

    # start→servo entry condition: the arm should begin at the home pose with the
    # gripper open. Position is averaged over the first few frames (stable home,
    # shrug off noise); the gripper uses the PEAK of the first frames — the task
    # always starts open, so any early spike (even one frame) is out of
    # distribution, and the start cannot be cropped without breaking timing.
    k = min(3, len(pos))
    home_pos = pos[:k].mean(axis=0)
    home_grip = float(np.max(grip[:k]))

    row = dict(
        path=path, name=os.path.basename(path), T=len(pos), hz=hz,
        pos=pos, grip=grip, bounds=bounds, flags=flags,
        pause_frac=pause_frac, home_pos=home_pos, home_grip=home_grip,
        n_toggles=len(_grip_events(grip)[0]) + len(_grip_events(grip)[1]),
    )
    if bounds is None:
        return row

    row['dur'] = {p: (b - a) / hz for p, (a, b) in bounds.items()}

    # STATE coverage: centroid of the TCP during each state. The spread of these
    # centroids across demos is the STATE diversity we WANT high.
    row['centroid'] = {p: pos[a:b].mean(axis=0) for p, (a, b) in bounds.items()}

    # TRANSITION conditions: the robot state at each state→state switch frame.
    # The spread of these across demos is the TRANSITION diversity we want LOW.
    row['switch'] = {}
    for name, nxt in TRANSITIONS:
        f = int(np.clip(bounds[nxt][0], 0, len(pos) - 1))
        row['switch'][name] = dict(
            z=float(pos[f, 2]), speed=float(speed[f]),
            grip=float(grip[f]), xy=pos[f, :2].copy())

    row['grasp_xy'] = pos[bounds['grasp'][0]][:2]        # object location (STATE diversity)
    row['place_xy'] = pos[bounds['release'][0]][:2]      # place→release switch over the tape
    gb, rb = bounds['grasp'], bounds['release']
    # Gripper decisiveness = ramp seconds (short = crisp switch).
    row['grasp_decis'] = (gb[1] - gb[0]) / hz
    row['release_decis'] = (rb[1] - rb[0]) / hz
    # Switch crispness: TCP should be still while the gripper switches (small = good).
    row['grasp_still'] = float(np.linalg.norm(
        pos[gb[0]:gb[1]] - pos[gb[0]], axis=1).max()) if gb[1] > gb[0] else 0.0
    row['release_still'] = float(np.linalg.norm(
        pos[rb[0]:rb[1]] - pos[rb[0]], axis=1).max()) if rb[1] > rb[0] else 0.0
    return row


# ── Scoring: transition messiness + state novelty ────────────────────────────────

def score(rows, home_grip_max=0.5, home_dev_max=0.08):
    ok = [r for r in rows if r.get('bounds') is not None]
    if not ok:
        return rows, {'state': {}, 'transition': {}}, np.zeros(2), np.zeros(3)

    trans_names = [n for n, _ in TRANSITIONS]

    # start→servo home reference (median across demos; the reset target).
    home_med = np.median([r['home_pos'] for r in ok], axis=0)

    # Median + std of each switch feature across demos (the canonical condition).
    med, sd = {}, {}
    for name in trans_names:
        for feat in ('z', 'speed'):
            vals = np.array([r['switch'][name][feat] for r in ok])
            med[(name, feat)] = float(np.median(vals))
            sd[(name, feat)] = float(vals.std()) + EPS
    place_med = np.median([r['place_xy'] for r in ok], axis=0)
    grasp_xy = np.array([r['grasp_xy'] for r in ok])

    # Per-episode transition-messiness components — how far this demo's switch
    # conditions sit from the dataset consensus.
    comps = {k: [] for k in
             ('switch_z', 'switch_speed', 'place_dev', 'crisp', 'indecisive',
              'home_dev', 'home_grip')}
    for r in ok:
        comps['switch_z'].append(sum(
            abs((r['switch'][n]['z'] - med[(n, 'z')]) / sd[(n, 'z')]) for n in trans_names))
        comps['switch_speed'].append(sum(
            abs((r['switch'][n]['speed'] - med[(n, 'speed')]) / sd[(n, 'speed')]) for n in trans_names))
        comps['place_dev'].append(float(np.linalg.norm(r['place_xy'] - place_med)))
        comps['crisp'].append(r['grasp_still'] + r['release_still'])
        comps['indecisive'].append(r['grasp_decis'] + r['release_decis'])
        r['home_dev'] = float(np.linalg.norm(r['home_pos'] - home_med))
        comps['home_dev'].append(r['home_dev'])        # arm off home (m)
        comps['home_grip'].append(r['home_grip'])      # gripper not-open (0 = open)

    def _z(v):
        v = np.array(v, float)
        s = v.std()
        return (v - v.mean()) / s if s > EPS else np.zeros_like(v)

    zsum = sum(np.clip(_z(comps[k]), 0, None) for k in comps)   # only worse-than-avg counts
    for r, m in zip(ok, zsum):
        r['messy'] = float(m)

    # State novelty = mean distance to 3 nearest OTHER grasp locations.
    for i, r in enumerate(ok):
        d = np.linalg.norm(grasp_xy - r['grasp_xy'], axis=1)
        d[i] = np.inf
        k = min(3, len(d) - 1)
        r['novelty'] = float(np.mean(np.sort(d)[:k])) if k > 0 else 0.0

    # Diversity indices for the summary plots.
    #   state[p]      = cross-demo spread of the state centroid  (want HIGH)
    #   transition[n] = cross-demo spread of the switch condition (want LOW)
    div = {'state': {}, 'transition': {}}
    for p in PHASES:
        cents = np.array([r['centroid'][p][:2] for r in ok])
        div['state'][p] = float(np.mean(np.std(cents, axis=0)))
    for name in trans_names:
        z = np.array([r['switch'][name]['z'] for r in ok])
        s = np.array([r['switch'][name]['speed'] for r in ok])
        div['transition'][name] = float(np.mean([z.std(), s.std()]))

    # Recommendation (tuned for a low-data, few-modes dataset):
    #   • broken state machine            → always drop
    #   • a distinct extra transition mode → drop even if the grasp is novel,
    #       because for hyperparameter/latency tuning we want the policy to
    #       learn a simple, near-unimodal state machine. A fumbled multi-second
    #       gripper close/open or a genuine re-grasp (≥2 extra toggles, not the
    #       single-crossing signal noise) is exactly such a mode.
    #   • otherwise messy AND not novel   → drop (keep messy-but-novel to
    #       protect STATE coverage, which we still want broad).
    msy = np.array([r['messy'] for r in ok])
    nov = np.array([r['novelty'] for r in ok])
    m_thr = msy.mean() + msy.std()
    n_thr = np.median(nov)
    for r in ok:
        indecisive = r['grasp_decis'] > 0.4 or r['release_decis'] > 0.4
        multimode = (r['n_toggles'] - 2) >= 2
        # start→servo entry outlier: arm not at home, or gripper not open.
        bad_home = r['home_dev'] > home_dev_max or r['home_grip'] > home_grip_max
        if bad_home:
            r['flags'].append('bad_home')
        if indecisive or multimode:
            r['flags'].append('transition_mode')
        rec = (indecisive or multimode or bad_home
               or (r['messy'] > m_thr and r['novelty'] <= n_thr))
        r['recommend_exclude'] = rec
    for r in rows:
        if r.get('bounds') is None:
            r['recommend_exclude'] = True         # broken state machine
            r['messy'] = float('nan')
            r['novelty'] = float('nan')

    return rows, div, place_med, home_med


# ── Terminal table ───────────────────────────────────────────────────────────────

def print_table(rows):
    GREEN, RED, YELLOW, RESET, BOLD = (
        '\033[92m', '\033[91m', '\033[93m', '\033[0m', '\033[1m')
    hdr = (f"{'Episode':<10} {'Messy':>6} {'Novelty':>8} {'Grasp dec':>9} "
           f"{'Rel dec':>8} {'PlaceDev':>9} {'HomeDev':>8} {'Grip0':>6} {'Pause%':>7}  Flags")
    print(f'\n{BOLD}{hdr}{RESET}')
    print('─' * len(hdr))
    for r in rows:
        ep = r['name'].replace('episode_', '').replace('.hdf5', '')
        flags = list(r.get('flags', []))
        if r.get('recommend_exclude'):
            flags = ['EXCLUDE'] + flags
        color = RED if r.get('recommend_exclude') else GREEN
        if r.get('bounds') is None:
            hd = f"{r['home_dev']*1000:>6.0f}mm" if 'home_dev' in r else f"{'—':>8}"
            print(f"{color}{ep:<10}{RESET} {'—':>6} {'—':>8} {'—':>9} "
                  f"{'—':>8} {'—':>9} {hd} {r['home_grip']:>6.2f} "
                  f"{r['pause_frac']*100:>6.0f}%  {YELLOW}{', '.join(flags)}{RESET}")
            continue
        pd = np.linalg.norm(r['place_xy'] - r.get('_place_med', r['place_xy']))
        print(f"{color}{ep:<10}{RESET} {r['messy']:>6.2f} {r['novelty']*100:>7.1f}c "
              f"{r['grasp_decis']:>8.2f}s {r['release_decis']:>7.2f}s "
              f"{pd*1000:>7.0f}mm {r['home_dev']*1000:>6.0f}mm {r['home_grip']:>6.2f} "
              f"{r['pause_frac']*100:>6.0f}%  {YELLOW}{', '.join(flags)}{RESET}")
    rec = [r['name'].replace('.hdf5', '') for r in rows if r.get('recommend_exclude')]
    print(f'\n{len(rec)} / {len(rows)} episodes recommended for exclusion.')
    if rec:
        print('  ' + ', '.join(rec))
    print('  (Messy = deviation of this demo\'s state→state switch conditions '
          '(height/speed/place/crispness) from the dataset median; Novelty = '
          'grasp-location rarity in cm; a messy but novel trial is kept to protect '
          'state diversity.)')


# ── Diversity dashboard ──────────────────────────────────────────────────────────

def plot_dashboard(rows, div, place_med, home_med, collection_dir, out_path,
                   home_grip_max=0.5, home_dev_max=0.08):
    ok = [r for r in rows if r.get('bounds') is not None]
    excl = {r['name'] for r in rows if r.get('recommend_exclude')}
    cmap = matplotlib.colormaps['viridis']
    trans_names = [n for n, _ in TRANSITIONS]
    tlabels = [n.replace('→', '→\n') for n in trans_names]

    fig = plt.figure(figsize=(19, 17))
    gs = fig.add_gridspec(4, 3, hspace=0.55, wspace=0.32)

    # ── Headline 1: STATE diversity — grasp (object) locations ─────────────────
    ax = fig.add_subplot(gs[0, 0])
    gxy = np.array([r['grasp_xy'] for r in ok])
    ax.scatter(gxy[:, 0], gxy[:, 1], c=[r['novelty'] for r in ok],
               cmap='plasma', s=45, edgecolor='k', linewidth=0.3)
    try:
        from scipy.spatial import ConvexHull
        h = ConvexHull(gxy)
        for s in h.simplices:
            ax.plot(gxy[s, 0], gxy[s, 1], 'k-', lw=0.6, alpha=0.5)
        cover = h.volume
    except Exception:
        cover = float(np.prod(gxy.max(0) - gxy.min(0)))
    ax.set_title(f'STATE diversity — grasp locations\n'
                 f'spread {gxy.std(0).mean()*100:.1f} cm, area {cover*1e4:.0f} cm²  '
                 f'(want BROAD)', fontsize=9)
    ax.set_xlabel('X (m)', fontsize=8); ax.set_ylabel('Y (m)', fontsize=8)
    ax.set_aspect('equal', adjustable='datalim'); ax.tick_params(labelsize=7)

    # ── Headline 2: switch HEIGHT (Z) per transition — should cluster tightly ──
    ax = fig.add_subplot(gs[0, 1])
    zdata = [[r['switch'][n]['z'] for r in ok] for n in trans_names]
    ax.boxplot(zdata, labels=tlabels, showfliers=True)
    ax.set_xticklabels(tlabels, rotation=0, fontsize=6)
    ax.set_ylabel('Z at switch (m)', fontsize=8)
    ax.set_title('TRANSITION switch height (tight cols = consistent trigger)', fontsize=9)
    ax.tick_params(labelsize=7)

    # ── Headline 3: switch SPEED per transition — crisp triggers = low + tight ─
    ax = fig.add_subplot(gs[0, 2])
    sdata = [[r['switch'][n]['speed'] for r in ok] for n in trans_names]
    ax.boxplot(sdata, labels=tlabels, showfliers=True)
    ax.set_xticklabels(tlabels, rotation=0, fontsize=6)
    ax.set_ylabel('speed at switch (m/s)', fontsize=8)
    ax.set_title('TRANSITION switch speed (tight = consistent trigger)', fontsize=9)
    ax.tick_params(labelsize=7)

    # ── Row 2a: place→release switch over the FIXED tape — should cluster ──────
    ax = fig.add_subplot(gs[1, 0])
    pxy = np.array([r['place_xy'] for r in ok])
    cols = ['tomato' if r['name'] in excl else 'steelblue' for r in ok]
    ax.scatter(pxy[:, 0], pxy[:, 1], c=cols, s=45, edgecolor='k', linewidth=0.3)
    ax.scatter(*place_med, marker='*', s=260, color='gold',
               edgecolor='k', zorder=5, label='median (tape)')
    ax.set_title(f'place→release switch — tape is FIXED\n'
                 f'scatter {pxy.std(0).mean()*1000:.0f} mm  (want TIGHT)', fontsize=9)
    ax.set_xlabel('X (m)', fontsize=8); ax.set_ylabel('Y (m)', fontsize=8)
    ax.legend(fontsize=7); ax.set_aspect('equal', adjustable='datalim')
    ax.tick_params(labelsize=7)

    # ── Row 2b: STATE coverage per state — want HIGH ───────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    sc = np.array([div['state'][p] for p in PHASES]) * 100
    ax.bar(np.arange(len(PHASES)), sc, color='seagreen')
    ax.set_xticks(np.arange(len(PHASES)))
    ax.set_xticklabels(PHASES, rotation=30, fontsize=7)
    ax.set_ylabel('centroid spread (cm)', fontsize=8)
    ax.set_title('STATE coverage per state (want HIGH)', fontsize=9)
    ax.tick_params(labelsize=7)

    # ── Row 2c: TRANSITION-condition spread per switch — want LOW ──────────────
    ax = fig.add_subplot(gs[1, 2])
    tc = np.array([div['transition'][n] for n in trans_names]) * 100
    ax.bar(np.arange(len(trans_names)), tc, color='indianred')
    ax.set_xticks(np.arange(len(trans_names)))
    ax.set_xticklabels(tlabels, rotation=0, fontsize=6)
    ax.set_ylabel('switch-condition spread', fontsize=8)
    ax.set_title('TRANSITION diversity per switch (want LOW)', fontsize=9)
    ax.tick_params(labelsize=7)

    # ── Row 3a: TCP XY colored by state (with switch points marked) ────────────
    ax = fig.add_subplot(gs[2, 0])
    phase_col = {p: cmap(i / (len(PHASES) - 1)) for i, p in enumerate(PHASES)}
    for r in ok:
        for p, (a, b) in r['bounds'].items():
            seg = r['pos'][a:b]
            ax.plot(seg[:, 0], seg[:, 1], color=phase_col[p], lw=0.6, alpha=0.6)
    for p in PHASES:
        ax.plot([], [], color=phase_col[p], label=p)
    ax.set_title('TCP XY by state', fontsize=9)
    ax.set_xlabel('X (m)', fontsize=8); ax.set_ylabel('Y (m)', fontsize=8)
    ax.legend(fontsize=6, ncol=2); ax.set_aspect('equal', adjustable='datalim')
    ax.tick_params(labelsize=7)

    # ── Row 3b: state durations (secondary timing consistency) ─────────────────
    ax = fig.add_subplot(gs[2, 1])
    data = [[r['dur'][p] for r in ok] for p in PHASES]
    ax.boxplot(data, labels=PHASES, showfliers=True)
    ax.set_xticklabels(PHASES, rotation=30, fontsize=7)
    ax.set_ylabel('duration (s)', fontsize=8)
    ax.set_title('State durations (tight = consistent tempo)', fontsize=9)
    ax.tick_params(labelsize=7)

    # ── Row 3c: transition messiness vs state novelty — keep/drop call ─────────
    ax = fig.add_subplot(gs[2, 2])
    for r in ok:
        c = 'tomato' if r['name'] in excl else 'steelblue'
        ax.scatter(r['novelty'] * 100, r['messy'], color=c, s=35,
                   edgecolor='k', linewidth=0.3)
        ax.annotate(r['name'].replace('episode_', '').replace('.hdf5', ''),
                    (r['novelty'] * 100, r['messy']), fontsize=5,
                    xytext=(2, 2), textcoords='offset points')
    ax.set_xlabel('state novelty (cm)  →  protect', fontsize=8)
    ax.set_ylabel('transition messiness  →  drop', fontsize=8)
    ax.set_title('Keep bottom-right, drop top-left', fontsize=9)
    ax.tick_params(labelsize=7)

    # ── Row 4: start→servo entry condition (home pose + gripper open) ──────────
    epn = [r['name'].replace('episode_', '').replace('.hdf5', '') for r in ok]
    hxy = np.array([r['home_pos'][:2] for r in ok])
    hg = np.array([r['home_grip'] for r in ok])
    hd = np.array([r['home_dev'] for r in ok]) * 1000          # mm
    is_excl = np.array([r['name'] in excl for r in ok])

    ax = fig.add_subplot(gs[3, 0])
    scat = ax.scatter(hxy[:, 0], hxy[:, 1], c=hg, cmap='coolwarm', vmin=0, vmax=1,
                      s=45, edgecolor='k', linewidth=0.3)
    ax.scatter(*home_med[:2], marker='*', s=260, color='gold',
               edgecolor='k', zorder=5, label='median home')
    if is_excl.any():
        ax.scatter(hxy[is_excl, 0], hxy[is_excl, 1], s=150, facecolors='none',
                   edgecolors='red', linewidths=1.5, zorder=6)
    fig.colorbar(scat, ax=ax, fraction=0.046, pad=0.04).set_label(
        'gripper (0=open)', fontsize=7)
    ax.set_title(f'INITIAL home XY — want TIGHT + gripper open\n'
                 f'spread {hxy.std(0).mean()*1000:.0f} mm  (red ring = excluded)', fontsize=9)
    ax.set_xlabel('X (m)', fontsize=8); ax.set_ylabel('Y (m)', fontsize=8)
    ax.legend(fontsize=7); ax.set_aspect('equal', adjustable='datalim')
    ax.tick_params(labelsize=7)

    ax = fig.add_subplot(gs[3, 1])
    order = np.argsort(hg)[::-1]
    ax.bar(range(len(hg)), hg[order],
           color=['tomato' if is_excl[i] else 'steelblue' for i in order])
    ax.axhline(home_grip_max, color='red', ls='--', lw=0.8,
               label=f'not-open thr {home_grip_max}')
    ax.set_xticks(range(len(hg)))
    ax.set_xticklabels([epn[i] for i in order], rotation=90, fontsize=5)
    ax.set_ylabel('initial gripper (0=open)', fontsize=8)
    ax.set_title('Initial gripper — above line = not open', fontsize=9)
    ax.legend(fontsize=7); ax.tick_params(labelsize=7)

    ax = fig.add_subplot(gs[3, 2])
    order = np.argsort(hd)[::-1]
    ax.bar(range(len(hd)), hd[order],
           color=['tomato' if is_excl[i] else 'seagreen' for i in order])
    ax.axhline(home_dev_max * 1000, color='red', ls='--', lw=0.8,
               label=f'off-home thr {home_dev_max*1000:.0f} mm')
    ax.set_xticks(range(len(hd)))
    ax.set_xticklabels([epn[i] for i in order], rotation=90, fontsize=5)
    ax.set_ylabel('arm deviation from home (mm)', fontsize=8)
    ax.set_title('Initial arm off-home — above line = not home', fontsize=9)
    ax.legend(fontsize=7); ax.tick_params(labelsize=7)

    broken = [r['name'].replace('.hdf5', '') for r in rows if r.get('bounds') is None]
    fig.suptitle(
        f'Transition inspection — {os.path.basename(collection_dir.rstrip("/"))}   '
        f'({len(ok)} segmented, {len(broken)} broken)  '
        f'goal: broad states / tight transitions', fontsize=12)
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    print(f'\nDiversity dashboard saved → {out_path}')
    try:
        plt.show()
    except Exception:
        pass


# ── Detail: one episode's phase segmentation over time ──────────────────────────

def plot_episode_detail(r):
    if r.get('bounds') is None:
        print(f'  {r["name"]}: broken state machine — cannot segment.')
        return
    t = np.arange(r['T']) / r['hz']
    pos, grip = r['pos'], r['grip']
    cmap = matplotlib.colormaps['viridis']
    phase_col = {p: cmap(i / (len(PHASES) - 1)) for i, p in enumerate(PHASES)}

    fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
    for ax, sig, lbl in ((axes[0], pos[:, 2], 'Z (m)'),
                         (axes[1], np.linalg.norm(np.gradient(pos, 1/r['hz'], axis=0), axis=1), 'speed (m/s)'),
                         (axes[2], grip, 'gripper')):
        ax.plot(t, sig, 'k-', lw=1)
        for p, (a, b) in r['bounds'].items():
            ax.axvspan(t[a], t[min(b, r['T'] - 1)], color=phase_col[p], alpha=0.25)
        ax.set_ylabel(lbl, fontsize=9)
    for p, (a, b) in r['bounds'].items():
        axes[0].text(t[(a + min(b, r['T'] - 1)) // 2], pos[:, 2].max(), p,
                     fontsize=7, ha='center', rotation=90, va='top')
    axes[2].set_xlabel('time (s)', fontsize=9)
    fig.suptitle(f'{r["name"]}  —  phase segmentation', fontsize=11)
    fig.tight_layout()
    plt.show(block=True)


# ── exclude.txt: full drops + end-crops ─────────────────────────────────────────
#
# Format (backward compatible with the plain `episode_N` drop list):
#     episode_N              drop the whole episode
#     episode_N crop S E     keep frames [S:E) at conversion (end-crop only)
# The original HDF5 files are NEVER modified — the crop is applied downstream by
# training/convert_data.py when it builds the zarr.

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


# ── End-crop suggestion + interactive review ────────────────────────────────────

def suggest_end_crop(pos, grip, hz):
    """Suggest an end-crop frame (exclusive). Targets a trailing gripper uptick
    after the final release — cmd should stay ≈0 there to remain in distribution.
    Returns (crop_end, reason); crop_end == len means no crop needed."""
    T = len(grip)
    opens = [i for i in range(1, T) if grip[i - 1] >= 0.5 > grip[i]]
    if not opens:
        return T, 'no release detected — review manually'
    r = opens[-1]
    settled = next((i for i in range(r, T) if grip[i] < 0.1), None)
    if settled is None:
        return T, 'gripper never settles to 0 after release'
    for i in range(settled, T):
        if grip[i] > 0.15:
            return i, f'gripper upticks to {grip[i]:.2f} at frame {i} (should stay ≈0)'
    return T, 'clean tail — no crop needed'


def _plot_crop(name, pos, grip, hz, crop_end):
    """Show gripper / Z / speed vs frame with the crop line and dropped tail shaded."""
    T = len(grip)
    x = np.arange(T)
    speed = np.linalg.norm(np.gradient(pos, 1.0 / hz, axis=0), axis=1)
    fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
    for ax, sig, lbl in ((axes[0], grip, 'gripper cmd'),
                         (axes[1], pos[:, 2], 'Z (m)'),
                         (axes[2], speed, 'speed (m/s)')):
        ax.plot(x, sig, 'k-', lw=1)
        if crop_end < T:
            ax.axvspan(crop_end, T - 1, color='tomato', alpha=0.25)
            ax.axvline(crop_end, color='red', lw=1.5)
        ax.set_ylabel(lbl, fontsize=9)
        ax.grid(alpha=0.3)
    kept = crop_end if crop_end < T else T
    axes[0].axhline(0.0, color='green', ls=':', lw=0.8, alpha=0.7)
    axes[2].set_xlabel('frame', fontsize=9)
    fig.suptitle(f'{name}  —  keep [0:{kept}) of {T}   '
                 f'({"CROP " + str(T - kept) + " frames off end" if kept < T else "no crop"})',
                 fontsize=11)
    fig.tight_layout()
    plt.show(block=True)
    plt.close(fig)


def crop_review(collection_dir, exclude_file, targets):
    """Interactive end-crop review. Shows each candidate, records approved crops
    to exclude.txt. Never touches the source HDF5 files."""
    full, crops = parse_exclude(exclude_file)
    paths = _sort(glob.glob(os.path.join(collection_dir, 'episode_*.hdf5')))
    paths = [p for p in paths if os.path.basename(p).replace('.hdf5', '') not in full]

    want = set(targets) if targets else None
    reviewed = 0
    for p in paths:
        name = os.path.basename(p).replace('.hdf5', '')
        num = name.replace('episode_', '')
        if want is not None and num not in want:
            continue
        with h5py.File(p, 'r') as f:
            hz = float(f.attrs.get('collection_rate_hz', 30))
            pos = f['action/pose'][:, :3].astype(np.float64)
            grip = f['action/gripper'][()].astype(np.float64)
        T = len(grip)
        existing = crops.get(name, (0, T))[1]
        suggested, reason = suggest_end_crop(pos, grip, hz)
        crop_end = existing if name in crops else suggested
        # With no explicit targets, only prompt for episodes that need a crop.
        if want is None and name not in crops and suggested >= T:
            continue

        reviewed += 1
        print(f'\n── {name}  (T={T}) ──  suggestion: {reason}')
        while True:
            _plot_crop(name, pos, grip, hz, crop_end)
            print(f'  crop_end = {crop_end}  (keeps [0:{crop_end}), drops {T - crop_end} frames)')
            try:
                resp = input('  [Enter]=approve  e <frame>=set  f=full/no-crop  '
                             's=skip  q=quit : ').strip()
            except EOFError:
                resp = 's'
            if resp in ('', 'a'):
                if crop_end < T:
                    crops[name] = (0, int(crop_end))
                    print(f'  ✓ recorded crop [0:{crop_end})')
                else:
                    crops.pop(name, None)
                    print('  ✓ no crop')
                break
            if resp.startswith('e'):
                try:
                    crop_end = int(np.clip(int(resp.split()[1]), 1, T))
                except (ValueError, IndexError):
                    print('  ! usage: e <frame>')
                continue
            if resp == 'f':
                crop_end = T
                continue
            if resp == 's':
                print('  – skipped (unchanged)')
                break
            if resp == 'q':
                write_exclude(exclude_file, full, crops)
                print(f'\nSaved → {exclude_file}  (quit)')
                return
        write_exclude(exclude_file, full, crops)   # persist after each episode

    if reviewed == 0:
        print('No episodes needed cropping (clean tails). '
              'Pass explicit numbers, e.g. --crop 3 9, to review any episode.')
    else:
        print(f'\nReviewed {reviewed} episode(s). exclude.txt → {exclude_file}')


# ── Main ─────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description='State/transition inspector for a collection')
    ap.add_argument('collection', help='Path to collection directory')
    ap.add_argument('--detail', nargs='*', metavar='N',
                    help='Episode numbers to open in phase-segmentation detail view')
    ap.add_argument('--exclude', nargs='*', metavar='N',
                    help='Write exclude.txt (omit numbers to accept the recommendation)')
    ap.add_argument('--crop', nargs='*', metavar='N',
                    help='Interactive end-crop review; writes crop indices to '
                         'exclude.txt (omit numbers to review episodes with a '
                         'suggested crop). Originals are never modified.')
    ap.add_argument('--out', default=None,
                    help='Dashboard path (default: timestamped, never overwrites)')
    ap.add_argument('--home-grip-max', type=float, default=0.5,
                    help='start→servo: flag if initial gripper > this (0=open, default 0.5)')
    ap.add_argument('--home-dev-max', type=float, default=0.08,
                    help='start→servo: flag if arm starts > this many metres off '
                         'the median home pose (default 0.08)')
    args = ap.parse_args()

    paths = _sort(glob.glob(os.path.join(args.collection, 'episode_*.hdf5')))
    if not paths:
        print(f'No episode_*.hdf5 files found in {args.collection}')
        sys.exit(1)

    exclude_file = os.path.join(args.collection, 'exclude.txt')

    # Interactive crop review is its own mode — do it and exit.
    if args.crop is not None:
        crop_review(args.collection, exclude_file, args.crop)
        return

    full, crops = parse_exclude(exclude_file)
    if full:
        print(f'Skipping {len(full)} dropped: {", ".join(sorted(full, key=_epnum))}')
    if crops:
        print(f'Applying {len(crops)} end-crop(s): '
              f'{", ".join(f"{n}[:{e}]" for n, (s, e) in sorted(crops.items(), key=lambda kv: _epnum(kv[0])))}')
    paths = [p for p in paths
             if os.path.basename(p).replace('.hdf5', '') not in full]

    print(f'Loading {len(paths)} episodes from {args.collection} …')
    rows = [s for p in paths
            if (s := compute_stats(
                p, crops.get(os.path.basename(p).replace('.hdf5', '')))) is not None]
    print(f'Loaded {len(rows)} episodes.')

    rows, div, place_med, home_med = score(
        rows, home_grip_max=args.home_grip_max, home_dev_max=args.home_dev_max)
    for r in rows:                       # stash for the table's place-dev column
        if r.get('bounds') is not None:
            r['_place_med'] = place_med
    print_table(rows)

    out = args.out or os.path.join(
        args.collection,
        f'transition_inspection_{_dt.datetime.now():%Y%m%d_%H%M%S}.png')
    plot_dashboard(rows, div, place_med, home_med, args.collection, out,
                   home_grip_max=args.home_grip_max, home_dev_max=args.home_dev_max)

    if args.exclude is not None:
        valid = {r['name'].replace('episode_', '').replace('.hdf5', '') for r in rows}
        if args.exclude:
            chosen = [e for e in args.exclude if e in valid]
            bad = [e for e in args.exclude if e not in valid]
            if bad:
                print(f'[WARN] Unknown episode numbers ignored: {bad}')
        else:
            chosen = [r['name'].replace('episode_', '').replace('.hdf5', '')
                      for r in rows if r.get('recommend_exclude')]
            print(f'Accepting recommendation: {chosen}')
        full |= {f'episode_{e}' for e in chosen}     # preserve existing crops
        write_exclude(exclude_file, full, crops)
        print(f'\nExclude list written → {exclude_file}')
        print('  dropped: ' + '  '.join(sorted(full, key=_epnum)))

    if args.detail is not None:
        plt.close('all')
        wanted = set(args.detail) or {r['name'].replace('episode_', '').replace('.hdf5', '')
                                      for r in rows if r.get('recommend_exclude')}
        for r in rows:
            if r['name'].replace('episode_', '').replace('.hdf5', '') in wanted:
                print(f'\nDetail: {r["name"]}')
                plot_episode_detail(r)


if __name__ == '__main__':
    main()
