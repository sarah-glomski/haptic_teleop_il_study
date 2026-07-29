#!/usr/bin/env python3
"""
Make a "hold-continuation" copy of a collection: for every episode, append
N seconds of the FINAL resting frame so the terminal "gripper-open at place"
state becomes a stable attractor the policy can settle into.

Motivation (see lab notes / transition inspection of Collection5): `release`
is the lowest-coverage (~0.85 cm) and briefest (~0.1 s ≈ 1 frame @10Hz) state
in the task, while the preceding `place` holds the gripper CLOSED ~1.3 s. A
policy parked at the place pose therefore sits in a ~13:1 closed:open
neighborhood and stalls closed. Padding each episode with a held open-at-rest
segment rebalances that. Confirmed better than Collection5 on hardware.

What gets held (per appended frame):
  - EVERY dataset (observation/*, images/*, piezense/*, hololens/*,
    observation/joint_states, action/*) is repeated from its LAST frame, so
    all channels stay frame-aligned.
  - EXCEPT, in the appended segment only:
      action/pose    := observation/pose[last]   (command the rest pose exactly
                                                   -> zero creep)
      action/gripper := 0.0                       (fully open)
    so the held frames are an unambiguous "stay here, gripper fully open".

CURATION IS BAKED IN. The source collection's exclude.txt is applied here
rather than copied forward:
  - fully-dropped episodes are not written at all
  - end-cropped episodes are truncated BEFORE the hold is appended
The second point is the whole reason this script reads exclude.txt. Appending
to the full episode and copying exclude.txt forward would leave the crop
slicing [0:E) off the *front* of a now-longer file, discarding the very hold
frames this script exists to add. The output therefore needs no exclude.txt —
it is already the curated, ready-to-convert collection. Originals are never
modified, so re-run with different curation any time.

Usage:
    python3.12 append_hold_continuation.py \
        --input  demo_data/Task1Collection1 \
        --output demo_data/Task1Collection1.5 \
        --seconds 1.0
"""
import argparse
import os
import pathlib

import h5py
import numpy as np


def _epnum(name):
    digits = ''.join(c for c in name.replace('episode_', '') if c.isdigit())
    return int(digits) if digits else 0


def parse_exclude(path):
    """Return (full_drops:set[str], crops:dict[str,(start,end)]).

    Same format as inspect_collection.py / inspect_transitions.py /
    training/convert_data.py — keep them in sync.
    """
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


def append_hold(src_path: pathlib.Path, dst_path: pathlib.Path, n_hold: int,
                crop=None):
    with h5py.File(src_path, "r") as fin:
        rate = int(fin.attrs.get("collection_rate_hz", 30))

        # Collect every dataset (recursively) and its data.
        datasets = {}
        def collect(name, obj):
            if isinstance(obj, h5py.Dataset):
                datasets[name] = obj[()]
        fin.visititems(collect)

        # Apply the end-crop FIRST so the held frame is the last kept frame,
        # not something from the discarded tail.
        if crop is not None:
            s, e = crop
            datasets = {n: a[s:e] for n, a in datasets.items()}

        n_orig = len(datasets["observation/pose"])

        # Resting state to hold: last observed pose, fully-open gripper.
        obs_pose_last = datasets["observation/pose"][-1].copy()  # 7D quat pose

        with h5py.File(dst_path, "w") as fout:
            for name, arr in datasets.items():
                last = arr[-1:]                              # (1, ...) keep dims
                pad = np.repeat(last, n_hold, axis=0)        # (n_hold, ...)

                if name == "action/pose":
                    pad[:] = obs_pose_last                   # command rest pose
                elif name == "action/gripper":
                    pad[:] = 0.0                             # fully open

                out = np.concatenate([arr, pad], axis=0)
                fout.create_dataset(name, data=out, compression="gzip"
                                    if arr.ndim > 1 and arr.dtype == np.uint8 else None)

            # Preserve + update attrs.
            for k, v in fin.attrs.items():
                fout.attrs[k] = v
            fout.attrs["num_frames"] = n_orig + n_hold
            fout.attrs["hold_frames_appended"] = n_hold
            fout.attrs["hold_seconds_appended"] = n_hold / rate
            if crop is not None:
                fout.attrs["cropped_from"] = f"{crop[0]}:{crop[1]}"
    return n_orig, n_orig + n_hold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--seconds", type=float, default=1.0,
                    help="Seconds of held rest frame to append (default 1.0).")
    ap.add_argument("--ignore-exclude", action="store_true",
                    help="Process every episode as-is, ignoring the source "
                         "exclude.txt (drops and crops).")
    args = ap.parse_args()

    src = pathlib.Path(args.input)
    dst = pathlib.Path(args.output)
    dst.mkdir(parents=True, exist_ok=True)

    episodes = sorted(src.glob("episode_*.hdf5"),
                      key=lambda p: int(p.stem.split("_")[1]))
    if not episodes:
        raise SystemExit(f"No episode_*.hdf5 found in {src}")

    full, crops = (set(), {}) if args.ignore_exclude else \
        parse_exclude(str(src / "exclude.txt"))
    if full:
        print(f"exclude.txt: dropping {len(full)} episode(s): "
              f"{', '.join(sorted(full, key=_epnum))}")
    if crops:
        print(f"exclude.txt: end-cropping {len(crops)} episode(s) BEFORE the "
              f"hold: {', '.join(f'{n}[{s}:{e}]' for n, (s, e) in sorted(crops.items(), key=lambda kv: _epnum(kv[0])))}")

    kept = [p for p in episodes if p.stem not in full]

    # Determine hold length from the first kept episode's recorded rate.
    with h5py.File(kept[0], "r") as f0:
        rate = int(f0.attrs.get("collection_rate_hz", 30))
    n_hold = int(round(args.seconds * rate))
    print(f"\nRate {rate} Hz -> appending {n_hold} frames ({args.seconds}s) "
          f"per episode\n")

    for ep in kept:
        crop = crops.get(ep.stem)
        o, n = append_hold(ep, dst / ep.name, n_hold, crop)
        tag = f"  (cropped {crop[0]}:{crop[1]})" if crop else ""
        print(f"  {ep.name:16s} {o:4d} -> {n:4d} frames{tag}")

    skipped = len(episodes) - len(kept)
    print(f"\nDone. {len(kept)} episodes -> {dst}"
          + (f"  ({skipped} dropped)" if skipped else ""))
    print("No exclude.txt written — curation is baked into these files.")


if __name__ == "__main__":
    main()
