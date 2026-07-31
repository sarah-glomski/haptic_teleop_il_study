#!/usr/bin/env python3
"""
Merge several collections into one folder, renumbering episodes end to end.

Every collection numbers its episodes from episode_0, so they cannot simply be
copied into a shared folder — the names collide. This walks the sources in the
order given and renumbers as it copies, so Task1Collection1's 16 episodes land
as 0-15, Task1Collection2's as 16-31, and so on.

EVERY episode is copied and only the episode_*.hdf5 files are copied. The source
exclude.txt files are not read and not carried over, so curation done before the
merge does NOT survive it — inspect the merged folder and build a fresh
exclude.txt there:

    python3.12 inspect_collection.py demo_data/Task1Merged/ --exclude
    python3.12 inspect_transitions.py demo_data/Task1Merged/ --exclude --crop

Nothing else is brought across either: inspection PNGs, zarr archives and any
existing exclude.txt stay in the source folders.

Sources are never modified. Re-run into a fresh destination any time.

Usage:
    python3.12 merge_collections.py \
        --inputs demo_data/Task1Collection1 demo_data/Task1Collection2 \
                 demo_data/Task1Collection3 demo_data/Task1Collection4 \
        --output demo_data/Task1Merged

    # save disk: hardlink instead of copying (same filesystem only)
    python3.12 merge_collections.py --inputs ... --output ... --link

    # see the plan without writing anything
    python3.12 merge_collections.py --inputs ... --output ... --dry-run

Then convert as usual:
    python3.12 ../training/convert_data.py --input demo_data/Task1Merged \
        --output demo_data/Task1Merged/kinova_teleop_umi.zarr.zip
"""
import argparse
import os
import pathlib
import shutil

import h5py

# What training/convert_data.py reads out of every episode. A file missing one
# of these breaks conversion, so it is worth catching here rather than after a
# multi-GB copy.
REQUIRED_KEYS = (
    'observation/pose',
    'observation/gripper',
    'action/pose',
    'action/gripper',
    'images/dji_wrist',
)


def _epnum(name):
    digits = ''.join(c for c in name.replace('episode_', '') if c.isdigit())
    return int(digits) if digits else 0


def inspect_episode(path: pathlib.Path):
    """(dataset key set, frame count, rate_hz) — or None if unreadable."""
    try:
        with h5py.File(path, 'r') as f:
            keys = set()
            f.visititems(lambda n, o: keys.add(n) if isinstance(o, h5py.Dataset) else None)
            n = len(f['observation/pose']) if 'observation/pose' in f else 0
            rate = int(f.attrs.get('collection_rate_hz', 30))
        return keys, n, rate
    except Exception as e:
        print(f'[WARN] Could not read {path}: {e}')
        return None


# ── Planning ───────────────────────────────────────────────────────────────────

def build_plan(inputs, start_index):
    """[(src_path, dst_name, src_collection)] plus a per-source note.

    Numbering continues across sources in the order given, so the destination
    reads as one collection recorded back to back.
    """
    plan, notes, next_i = [], [], start_index

    for src in inputs:
        src = pathlib.Path(src)
        if not src.is_dir():
            raise SystemExit(f'Not a directory: {src}')

        episodes = sorted(src.glob('episode_*.hdf5'), key=lambda p: _epnum(p.stem))
        if not episodes:
            raise SystemExit(f'No episode_*.hdf5 found in {src}')

        first_new = next_i
        for p in episodes:
            plan.append((p, f'episode_{next_i}.hdf5', src.name))
            next_i += 1

        notes.append(dict(src=src, count=len(episodes),
                          first=first_new, last=next_i - 1))

    return plan, notes


def check_schema(plan):
    """Warn about anything that would surprise convert_data.py downstream."""
    key_sets, rates, frames, bad = {}, {}, {}, []

    for src_path, dst_name, _coll in plan:
        info = inspect_episode(src_path)
        if info is None:
            bad.append(src_path)
            continue
        keys, n, rate = info
        key_sets[dst_name] = keys
        rates.setdefault(rate, []).append(dst_name)
        frames[dst_name] = n

        missing = [k for k in REQUIRED_KEYS if k not in keys]
        if missing:
            print(f'[ERROR] {src_path} is missing {", ".join(missing)} — '
                  f'conversion would fail on it')
            bad.append(src_path)

    if len(rates) > 1:
        # Everything downstream (obs_down_sample_steps, the hold-continuation
        # frame count, latency compensation) is derived from one source rate.
        print('[WARN] sources disagree on collection_rate_hz: '
              + ', '.join(f'{hz} Hz ({len(v)} eps)' for hz, v in sorted(rates.items())))

    if key_sets:
        common = set.intersection(*key_sets.values())
        union = set.union(*key_sets.values())
        partial = sorted(union - common)
        if partial:
            # Not fatal — convert_data.py zero-fills a missing piezense group,
            # for instance — but a key present in only some episodes means the
            # merged collection is not uniform, which is worth knowing before
            # training on it.
            print(f'[WARN] {len(partial)} dataset(s) present in only some episodes:')
            for k in partial:
                have = sum(1 for ks in key_sets.values() if k in ks)
                print(f'         {k}  ({have}/{len(key_sets)} episodes)')

    return bad, sum(frames.values())


# ── Copying ────────────────────────────────────────────────────────────────────

def transfer(src: pathlib.Path, dst: pathlib.Path, link: bool):
    if link:
        try:
            os.link(src, dst)
            return
        except OSError as e:
            raise SystemExit(
                f'Hardlink failed ({e}). --link needs source and destination on '
                f'the same filesystem; re-run without it to copy.')
    shutil.copy2(src, dst)


def write_manifest(path, plan, notes):
    """Where every merged episode came from — the only record of provenance
    once the files are renumbered."""
    lines = ['# merge_collections.py provenance',
             '# new_episode <- source_collection/source_episode',
             '']
    for note in notes:
        lines.append(f'# {note["src"]}: {note["count"]} episodes '
                     f'-> episode_{note["first"]}..episode_{note["last"]}')
    lines.append('')
    for src_path, dst_name, coll in plan:
        lines.append(f'{dst_name} <- {coll}/{src_path.name}')
    with open(path, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description='Merge several collections into one renumbered folder')
    ap.add_argument('--inputs', nargs='+', required=True,
                    help='Source collection directories, in the order they '
                         'should be numbered')
    ap.add_argument('--output', required=True, help='Destination directory')
    ap.add_argument('--link', action='store_true',
                    help='Hardlink instead of copying (same filesystem only). '
                         'Saves the disk space of a second full set.')
    ap.add_argument('--append', action='store_true',
                    help='Add to a destination that already holds episodes, '
                         'continuing the numbering instead of refusing.')
    ap.add_argument('--manifest', action='store_true',
                    help='Also write merge_manifest.txt recording which source '
                         'episode each new number came from. Off by default so '
                         'the destination holds nothing but episode hdf5s.')
    ap.add_argument('--dry-run', action='store_true',
                    help='Print the plan and the schema check, write nothing.')
    args = ap.parse_args()

    dst = pathlib.Path(args.output)
    existing = sorted(dst.glob('episode_*.hdf5'), key=lambda p: _epnum(p.stem)) \
        if dst.is_dir() else []
    if existing and not args.append:
        raise SystemExit(
            f'{dst} already holds {len(existing)} episode(s). Use --append to '
            f'continue numbering after them, or pick an empty destination.')

    start = _epnum(existing[-1].stem) + 1 if existing else 0

    plan, notes = build_plan(args.inputs, start)

    print(f'\nMerging {len(args.inputs)} collection(s) -> {dst}')
    if existing:
        print(f'  appending after {len(existing)} existing episode(s), '
              f'starting at episode_{start}')
    for note in notes:
        print(f'  {note["src"].name:<24} {note["count"]:>3} eps  '
              f'-> episode_{note["first"]}..episode_{note["last"]}')

    print(f'\nChecking {len(plan)} episodes …')
    bad, total_frames = check_schema(plan)
    if bad:
        raise SystemExit(f'\n{len(bad)} episode(s) unusable — nothing written. '
                         f'Fix or remove them and re-run.')

    total_bytes = sum(p.stat().st_size for p, _, _ in plan)
    print(f'OK — {len(plan)} episodes, {total_frames} frames, '
          f'{total_bytes / 1e9:.2f} GB'
          + (' (hardlinked, no extra disk)' if args.link else ''))

    if args.dry_run:
        print('\n--dry-run: nothing written.')
        for src_path, dst_name, coll in plan:
            print(f'  {dst_name:<20} <- {coll}/{src_path.name}')
        return

    dst.mkdir(parents=True, exist_ok=True)
    verb = 'Linking' if args.link else 'Copying'
    print()
    for i, (src_path, dst_name, coll) in enumerate(plan, 1):
        out = dst / dst_name
        if out.exists():
            raise SystemExit(f'{out} already exists — refusing to overwrite.')
        transfer(src_path, out, args.link)
        print(f'\r  {verb} {i}/{len(plan)}  {coll}/{src_path.name} -> {dst_name}   ',
              end='', flush=True)
    print()

    if args.manifest:
        write_manifest(dst / 'merge_manifest.txt', plan, notes)
        print(f'\nProvenance written to {dst / "merge_manifest.txt"}')

    print(f'\nDone. {len(plan)} episodes -> {dst}')
    print('No exclude.txt carried over — curate the merged folder:')
    print(f'  python3.12 inspect_collection.py {dst}/ --exclude')


if __name__ == '__main__':
    main()
