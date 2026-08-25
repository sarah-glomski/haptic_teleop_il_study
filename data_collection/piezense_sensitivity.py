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
    """(sensitivity, baseline_kPa, mean_closure, peak_kPa, grip_cycles) or None.

    Two measures, because the obvious one is confounded:

      sensitivity  mean grip pressure per unit closure. Comparable only
                   BETWEEN SIMILAR GRIP BEHAVIOUR. The blind block episodes
                   grip and release ~7 times an episode while the known ones
                   grip once, so most of their closed time is light
                   exploratory contact and the mean reads 22% low with a
                   perfectly healthy sensor.
      peak         highest pressure reached while closed. Nearly behaviour
                   independent — the blind and known block sets peak at 18.6
                   and 18.3 kPa — so this is the number to judge the hardware
                   on. grip_cycles is reported alongside so a behaviour
                   difference is visible rather than inferred.

    Neither controls for the OBJECT. A riper grape deforms more and transmits
    less pressure at the same closure, so compare grape sessions with that in
    mind and use a rigid block to test the sensor itself.
    """
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
    cycles = int(((~closed[:-1]) & closed[1:]).sum())
    return (float((m[closed] - base).mean() / g[closed].mean()), base,
            float(g[closed].mean()), float((m[closed] - base).max()), cycles)


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

    col = lambda rows, i: np.array([r[i] for r in rows])
    rs, ts = col(ref, 1), col(tgt, 1)
    rb, tb = col(ref, 2), col(tgt, 2)
    rc, tc = col(ref, 3), col(tgt, 3)
    rp, tp = col(ref, 4), col(tgt, 4)
    ry, ty = col(ref, 5), col(tgt, 5)

    print(f'reference {a.ref}: n={len(ref)}')
    print(f'  PEAK {rp.mean():6.2f} +- {rp.std():.2f} kPa   mean/closure {rs.mean():.2f}'
          f'   baseline {rb.mean():.2f}   closure {rc.mean():.3f}   grip cycles {ry.mean():.1f}')
    print(f'\n{a.target}: n={len(tgt)}')
    print(f'  PEAK {tp.mean():6.2f} +- {tp.std():.2f} kPa   mean/closure {ts.mean():.2f}'
          f'   baseline {tb.mean():.2f}   closure {tc.mean():.3f}   grip cycles {ty.mean():.1f}')
    for name, sv, b, c, pk, cy in tgt:
        print(f'    {name:12s} peak {pk:6.2f}   mean/closure {sv:5.2f}   '
              f'baseline {b:6.2f}   closure {c:.3f}   cycles {cy}')

    if abs(ty.mean() - ry.mean()) > 1.0:
        print(f'\nNOTE: grip behaviour differs ({ty.mean():.1f} vs {ry.mean():.1f} '
              f'grip cycles per episode) — judge by PEAK, not mean/closure.')
    drop = 100 * (rp.mean() - tp.mean()) / rp.mean()
    print()
    if tb.mean() > 113:
        print(f'WRONG MODE: baseline {tb.mean():.1f} kPa (want ~109). Run '
              f'piezense_reconfig.py and re-record.')
        return 1
    if abs(drop) < 10:
        print(f'NORMAL: peak pressure within {abs(drop):.0f}% of the reference.')
        return 0
    print(f'{"LOW" if drop > 0 else "HIGH"}: peak pressure {abs(drop):.0f}% '
          f'{"below" if drop > 0 else "above"} the reference, at comparable gripper '
          f'closure ({tc.mean():.3f} vs {rc.mean():.3f}).')
    print('  Baseline is healthy, so this is NOT the actuator-mode fault.')
    print('  Before suspecting the sensor, rule out the object: a softer or riper '
          'specimen transmits less pressure at the same closure. Grip a RIGID '
          'block and compare against a block collection — that isolates the '
          'hardware from the specimen.')
    print('  If a block also reads low: relaunch (a fresh driver runs upstream\'s '
          'own config sequence), then check pad seating, tubing at both ends, '
          'and air supply level.')
    return 1


if __name__ == '__main__':
    sys.exit(main())
