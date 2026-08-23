#!/usr/bin/env python3
"""Unattended: when one training run finishes, turn the episodes staged in
data_collection/demo_data/ into a collection, convert, verify, and launch the
next training run. Written 2026-08-22 so the blind-condition (Task2Collection3)
run starts the moment the Collection2 run (block_sort_c2) releases the GPU.

    nohup python3.12 queue_training.py --wait-for block_sort_c2 \
        --collection Task2Collection3 --task block_sort_c3 --base-task block_sort_c2 \
        > queue_training.log 2>&1 &

Steps, each logged to queue_training.log and to --collection/QUEUE_REPORT.txt:
  1. wait until no `train.py ... task=<wait-for>` process exists (GPU free)
  2. wait until no hdf5_data_collector is running (never move files under it)
  3. move staging episodes + annotations into demo_data/<collection>/
  4. audit every episode (footage complete, labels complete, piezense in sense
     mode, no teleport across lost time); failures are MOVED to
     <collection>/excluded/ (never deleted) so the converter skips them
  5. measure the active rate, write config/task/<task>.yaml from <base-task>
  6. convert -> verify zarr (episode count + step count vs HDF5)
  7. launch training detached; record the wandb URL in QUEUE_REPORT.txt

Pass --correct-only to train only on correctly-placed blind episodes
(ablation); default trains on all of them — wrong guesses are data.
"""
import argparse, csv, glob, os, re, shutil, subprocess, sys, time
from datetime import datetime

import h5py, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DC   = os.path.abspath(os.path.join(HERE, '..', 'data_collection'))
STAGING = os.path.join(DC, 'demo_data')
sys.path.insert(0, DC)
from annotations import AnnotationStore, TaskSpec   # noqa: E402

SHAPE_OF = {'light': 'triangle', 'middle': 'circle', 'heavy': 'square'}
REPORT = []


def log(msg):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True); REPORT.append(line)


def running(pattern):
    return subprocess.run(['pgrep', '-f', pattern], capture_output=True).returncode == 0


def num(ep): return int(ep.split('_')[1])


def audit_episode(path):
    probs = []
    with h5py.File(path) as f:
        t = f['observation/timestamp'][:]; dt = np.diff(t)
        pose = f['observation/pose'][:, :3]; grip = f['observation/gripper'][:]
        n = int(f.attrs['num_frames'])
        img = f['images/dji_wrist'].shape[0] if 'images' in f and 'dji_wrist' in f['images'] else 0
        pz = f['piezense/pressure_input'][:].mean(axis=1) / 1000.0
        base = float(np.median(pz[:24]))
    if img != n: probs.append(f'footage {img}/{n}')
    if base > 113: probs.append(f'piezense wrong mode (baseline {base:.1f} kPa)')
    if (dt <= 0).any(): probs.append('non-monotonic time')
    for i in np.where(dt > 0.5)[0]:
        j = np.linalg.norm(pose[i+1] - pose[i]) * 1000; g = abs(grip[i+1] - grip[i])
        if j > 100 or g > 0.2: probs.append(f'{j:.0f}mm/grip{g:.2f} across {dt[i]:.1f}s gap')
    return probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wait-for', required=True, help='task name of the run to wait for (e.g. block_sort_c2)')
    ap.add_argument('--collection', required=True, help='folder name under demo_data/ to create')
    ap.add_argument('--task', required=True, help='new task config name (config/task/<task>.yaml)')
    ap.add_argument('--base-task', required=True, help='existing task yaml to derive from')
    ap.add_argument('--correct-only', action='store_true')
    ap.add_argument('--config-name', default='train_diffusion_unet_timm_kinova')
    a = ap.parse_args()

    coll = os.path.join(STAGING, a.collection)
    log(f"queued: wait for '{a.wait_for}' -> build {a.collection} -> train task={a.task}")

    # 1. GPU free
    while running(f'train.py.*task={a.wait_for}'):
        time.sleep(60)
    log(f"'{a.wait_for}' training process is gone — GPU free")
    # 2. collector closed
    waited = 0
    while running('hdf5_data_collector.py'):
        if waited % 600 == 0: log('collector still running — waiting for Q before touching staging')
        time.sleep(30); waited += 30
    # 3. move staging -> collection
    eps = sorted(glob.glob(os.path.join(STAGING, 'episode_*.hdf5')))
    if not eps:
        log('NO EPISODES in staging — nothing to do'); return 2
    os.makedirs(coll, exist_ok=True)
    for p in eps: shutil.move(p, os.path.join(coll, os.path.basename(p)))
    for f in ('annotations.csv', 'annotations.xlsx'):
        if os.path.exists(os.path.join(STAGING, f)): shutil.move(os.path.join(STAGING, f), os.path.join(coll, f))
    log(f'moved {len(eps)} episodes + ledger into {coll}')

    # 4. audit; exclude failures (moved, not deleted)
    spec = TaskSpec.load('block_sort')
    st = AnnotationStore(coll, spec)
    rows = {r['episode']: r for r in st.read_rows() if r['status'] == 'keep'}
    excl_dir = os.path.join(coll, 'excluded'); excluded = []
    for p in sorted(glob.glob(os.path.join(coll, 'episode_*.hdf5')), key=lambda p: num(os.path.basename(p)[:-5])):
        ep = os.path.basename(p)[:-5]; r = rows.get(ep); probs = []
        if r is None: probs.append('no annotation row')
        else:
            if not (r.get('block') and r.get('placed') and r.get('condition')): probs.append('incomplete labels')
            if a.correct_only and SHAPE_OF.get(r.get('block')) != r.get('placed'): probs.append('wrong placement (--correct-only)')
        probs += audit_episode(p)
        if probs:
            os.makedirs(excl_dir, exist_ok=True); shutil.move(p, os.path.join(excl_dir, ep + '.hdf5'))
            excluded.append((ep, probs))
    kept = sorted(glob.glob(os.path.join(coll, 'episode_*.hdf5')))
    log(f'audit: {len(kept)} kept, {len(excluded)} excluded -> {excl_dir if excluded else "(none)"}')
    for ep, probs in excluded: log(f'  excluded {ep}: ' + '; '.join(probs))
    if not kept: log('nothing survived the audit'); return 3
    # summary of what is being trained on
    from collections import Counter
    krows = [rows[os.path.basename(p)[:-5]] for p in kept if os.path.basename(p)[:-5] in rows]
    pres = Counter(r['block'] for r in krows); corr = Counter(r['block'] for r in krows if SHAPE_OF.get(r['block']) == r['placed'])
    log(f"training set: {len(krows)} episodes | presented {dict(pres)} | correct {dict(corr)} | conditions {dict(Counter(r['condition'] for r in krows))}")

    # 5. rate + task yaml
    n = d = 0
    for p in kept:
        dt = np.diff(h5py.File(p)['observation/timestamp'][:]); act = dt[dt < 0.5]; n += len(act); d += act.sum()
    rate = round(n / d, 2)
    base_yaml = os.path.join(HERE, 'config', 'task', f'{a.base_task}.yaml')
    new_yaml  = os.path.join(HERE, 'config', 'task', f'{a.task}.yaml')
    zarr_rel  = f'../data_collection/demo_data/{a.collection}/{a.task}_umi.zarr.zip'
    s = open(base_yaml).read()
    s = re.sub(r'^name: .*$', f'name: {a.task}', s, count=1, flags=re.M)
    s = re.sub(r'^task_name: &task_name .*$', f'task_name: &task_name {a.task}', s, count=1, flags=re.M)
    s = re.sub(r'^dataset_frequeny: .*$', f'dataset_frequeny: {rate}', s, count=1, flags=re.M)
    s = re.sub(r'^dataset_path: .*$', f'dataset_path: {zarr_rel}', s, count=1, flags=re.M)
    s = (f'# AUTO-GENERATED by training/queue_training.py on {datetime.now():%Y-%m-%d %H:%M} from '
         f'{a.base_task}.yaml — {a.collection}: {len(kept)} episodes, active rate {rate} Hz '
         f'(pause gaps excluded){", CORRECT placements only" if a.correct_only else ", all placements incl. wrong guesses"}.\n') + s
    open(new_yaml, 'w').write(s)
    log(f'wrote {new_yaml}: rate {rate} Hz, path {zarr_rel}')

    # 6. convert + verify
    out = os.path.join(coll, f'{a.task}_umi.zarr.zip')
    r = subprocess.run(['/usr/bin/python3.12', os.path.join(HERE, 'convert_data.py'), '--input', coll, '--output', out],
                       cwd=HERE, capture_output=True, text=True)
    if r.returncode != 0: log('CONVERT FAILED:\n' + r.stdout[-2000:] + r.stderr[-2000:]); return 4
    import zarr
    z = zarr.open(out); ends = z['meta']['episode_ends'][:]
    frames = sum(int(h5py.File(p).attrs['num_frames']) for p in kept)
    if len(ends) != len(kept) or int(ends[-1]) != frames:
        log(f'ZARR MISMATCH: {len(ends)} eps / {int(ends[-1])} steps vs {len(kept)} / {frames}'); return 5
    log(f'zarr verified: {len(ends)} episodes, {int(ends[-1])} steps, keys {sorted(z["data"].keys())}')

    # 7. launch training, detached
    tlog = open(os.path.join(HERE, f'train_{a.task}.log'), 'ab')
    proc = subprocess.Popen(['conda', 'run', '--no-capture-output', '-n', 'umi', 'python', 'train.py',
                             f'--config-name={a.config_name}', f'task={a.task}'],
                            cwd=HERE, stdout=tlog, stderr=subprocess.STDOUT, start_new_session=True)
    log(f'training launched: pid {proc.pid}, task={a.task}')
    # wandb url (best effort)
    time.sleep(120)
    runs = sorted(glob.glob(os.path.join(HERE, 'wandb', 'run-*')), key=os.path.getmtime)
    if runs:
        rid = runs[-1].rsplit('-', 1)[-1]
        q = subprocess.run(['conda', 'run', '-n', 'umi', 'python', '-c',
                            "import wandb,sys; a=wandb.Api(); r=a.run(f\"{a.default_entity}/Haptic Teleop IL Study - UMI migration/%s\"); print(r.url)" % rid],
                           capture_output=True, text=True, cwd=HERE)
        url = [l for l in q.stdout.splitlines() if l.startswith('https://')]
        log('wandb: ' + (url[-1] if url else f'(run id {rid}, url lookup failed)'))
    open(os.path.join(coll, 'QUEUE_REPORT.txt'), 'w').write('\n'.join(REPORT) + '\n')
    return 0


if __name__ == '__main__':
    code = main()
    try:
        coll_dir = None
    finally:
        pass
    sys.exit(code)
