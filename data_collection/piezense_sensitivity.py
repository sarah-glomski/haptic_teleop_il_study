#!/usr/bin/env python3
"""Is the piezense reporting as much pressure as it used to?

The wrong-mode fault (see piezense_reconfig.py) is obvious: the resting
baseline sits at ~117 kPa instead of ~109 and grasps barely register. A
PARTIAL config apply is not obvious at all — baseline healthy, response shape
healthy, but the gain is down. The only way to see it is to compare recorded
demos against earlier sessions, which is what this does.

The metric is kPa of grip pressure per unit of gripper closure, so it does not
care whether the operator gripped harder or softer on the day:

    sensitivity = mean(pressure - baseline, while closed) / mean(closure, while closed)

Measured on the grape task:

    Task1Collection5  (2026-08-14, n=14)   6.85
    Task1Collection6  (2026-08-20, n=25)   6.82     <- two sessions, 0.4% apart
    2026-08-25 demos  (n=5)                5.04     <- 26% low, same closure

Usage:
    python3.12 piezense_sensitivity.py demo_data              # newest staging demos
    python3.12 piezense_sensitivity.py demo_data/Task1Collection6
    python3.12 piezense_sensitivity.py demo_data --ref Task1Collection5 Task1Collection6
"""

import argparse
import glob
import os
import re
import sys

import h5py
import numpy as np

DEFAULT_REF = ('Task1Collection5', 'Task1Collection6')
HERE = os.path.dirname(os.path.abspath(__file__))


def episode_sensitivity(path):
    """(sensitivity, baseline_kPa, mean_closure) for one episode, or None."""
    with h5py.File(path) as f:
        if 'piezense' not in f:
            return None
        pz = f['piezense/pressure_input'][:] / 1000.0
        g = f['observation/gripper'][:]
    m = pz.mean(axis=1)
    base = float(np.median(m[:24]))
    closed = g > 0.5
    if not closed.any() or g[closed].mean() <= 0:
        return None
    return float((m[closed] - base).mean() / g[closed].mean()), base, float(g[closed].mean())


def folder_stats(d):
    d = d if os.path.isdir(d) else os.path.join(HERE, 'demo_data', d)
    out = []
    for p in sorted(glob.glob(os.path.join(d, 'episode_*.hdf5')),
                    key=lambda p: int(re.search(r'_(\d+)', p).group(1))):
        r = episode_sensitivity(p)
        if r:
            out.append((os.path.basename(p)[:-5],) + r)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('target', help='folder of episode_*.hdf5 to check')
    ap.add_argument('--ref', nargs='+', default=list(DEFAULT_REF),
                    help=f'reference collections (default: {" ".join(DEFAULT_REF)})')
    a = ap.parse_args()

    tgt = folder_stats(a.target)
    if not tgt:
        print(f'No episodes with piezense data in {a.target}'); return 2
    ref = [r for c in a.ref for r in folder_stats(c)]
    if not ref:
        print(f'No reference episodes found in {a.ref}'); return 2

    rs = np.array([r[1] for r in ref]); ts = np.array([r[1] for r in tgt])
    rb = np.array([r[2] for r in ref]); tb = np.array([r[2] for r in tgt])
    rc = np.array([r[3] for r in ref]); tc = np.array([r[3] for r in tgt])

    print(f'reference {a.ref}: n={len(ref)}')
    print(f'  sensitivity {rs.mean():.2f} +- {rs.std():.2f} kPa per unit closure'
          f'   baseline {rb.mean():.2f}   closure {rc.mean():.3f}')
    print(f'\n{a.target}: n={len(tgt)}')
    print(f'  sensitivity {ts.mean():.2f} +- {ts.std():.2f} kPa per unit closure'
          f'   baseline {tb.mean():.2f}   closure {tc.mean():.3f}')
    for name, s, b, c in tgt:
        print(f'    {name:12s} {s:5.2f}   baseline {b:6.2f}   closure {c:.3f}')

    drop = 100 * (rs.mean() - ts.mean()) / rs.mean()
    print()
    if tb.mean() > 113:
        print(f'WRONG MODE: baseline {tb.mean():.1f} kPa (want ~109). Run '
              f'piezense_reconfig.py and re-record.')
        return 1
    if abs(drop) < 10:
        print(f'NORMAL: sensitivity within {abs(drop):.0f}% of the reference '
              f'(sessions normally agree to a few percent).')
        return 0
    print(f'{"LOW" if drop > 0 else "HIGH"}: sensitivity {abs(drop):.0f}% '
          f'{"below" if drop > 0 else "above"} the reference, at comparable gripper '
          f'closure ({tc.mean():.3f} vs {rc.mean():.3f}).')
    print('  The baseline is healthy, so this is not the actuator-mode fault.')
    print('  Most likely a partial config apply — relaunch (the reconfig watcher '
          'now primes the full config per BLE link) and re-measure.')
    print('  If it stays low after a clean relaunch, look at the hardware: pad '
          'seating on the fingers, tubing seated at both ends, air supply level.')
    return 1


if __name__ == '__main__':
    sys.exit(main())
