"""Measure how much the val split really holds out, beyond episode identity.

The split is upstream's own train/val and is disjoint by episode, but an episode is not
a scene: the same station, task and object layout can appear on both sides, in which case
the evaluation measures within-scene generalisation and not novel-scene generalisation.
Nothing in the annotations names a scene, so this approximates it with the one signal
that is recorded -- the robot's starting configuration. Two episodes that begin from
near-identical 14-D joint states are the same rig set up the same way.

Distances are computed in the normalised space the model trains in (state_01/state_99
from stat.json), so a distance is comparable across joints with different ranges.

    python3 scripts/analyze_split_overlap.py --data_root <data/abc_mcap> \
        --stat <dataset_meta_info/abc_mcap/stat.json> --out <report.json>
"""
import argparse
import collections
import glob
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np


def read_one(path):
    with open(path) as fh:
        d = json.load(fh)
    states = d.get('states')
    if not states:
        return None
    return d['episode_id'], d['task'], np.asarray(states[0], dtype=np.float32)


def load_split(data_root, split, workers):
    paths = sorted(glob.glob(os.path.join(data_root, 'annotation', split, '*.json')))
    with ProcessPoolExecutor(workers) as ex:
        recs = [r for r in ex.map(read_one, paths, chunksize=32) if r is not None]
    ids = [r[0] for r in recs]
    tasks = [r[1] for r in recs]
    starts = np.stack([r[2] for r in recs])
    return ids, tasks, starts


def normalise(x, lo, hi):
    # Same bound normalisation Dataset_mix.normalize_bound applies, so a unit here is a
    # unit of the model's input, not of radians.
    span = np.where((hi - lo) == 0, 1.0, hi - lo)
    return (x - lo) / span


def nearest(query, ref):
    """Min L2 from each row of query to any row of ref, and the argmin."""
    if len(ref) == 0:
        return np.full(len(query), np.inf), np.full(len(query), -1)
    d = np.linalg.norm(query[:, None, :] - ref[None, :, :], axis=2)
    return d.min(axis=1), d.argmin(axis=1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', required=True)
    p.add_argument('--stat', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--workers', type=int, default=8)
    a = p.parse_args()

    stat = json.load(open(a.stat))
    lo = np.asarray(stat['state_01'], dtype=np.float32)
    hi = np.asarray(stat['state_99'], dtype=np.float32)

    tr_ids, tr_tasks, tr_start = load_split(a.data_root, 'train', a.workers)
    va_ids, va_tasks, va_start = load_split(a.data_root, 'val', a.workers)
    print(f'train {len(tr_ids)} episodes, val {len(va_ids)} episodes')

    tr_n = normalise(tr_start, lo, hi)
    va_n = normalise(va_start, lo, hi)

    d_any, idx_any = nearest(va_n, tr_n)
    # Same-task nearest neighbour: the interesting question is whether a val episode has a
    # twin among the train episodes of its own task, not across tasks.
    by_task = collections.defaultdict(list)
    for i, t in enumerate(tr_tasks):
        by_task[t].append(i)
    d_task = np.full(len(va_n), np.inf)
    twin = [None] * len(va_n)
    for i, t in enumerate(va_tasks):
        rows = by_task.get(t, [])
        if not rows:
            continue
        sub = tr_n[rows]
        d = np.linalg.norm(sub - va_n[i], axis=1)
        j = int(d.argmin())
        d_task[i] = float(d[j])
        twin[i] = tr_ids[rows[j]]

    # val-to-val nearest neighbour calibrates the scale: it is how close two *different*
    # episodes of the same held-out set sit, which is the floor for "meaningfully apart".
    d_vv, _ = nearest(va_n, va_n)  # includes self at 0
    dd = np.linalg.norm(va_n[:, None, :] - va_n[None, :, :], axis=2)
    np.fill_diagonal(dd, np.inf)
    d_vv = dd.min(axis=1)

    def pct(d):
        f = d[np.isfinite(d)]
        return {q: round(float(np.percentile(f, q)), 4) for q in (5, 25, 50, 75, 95)}

    report = {
        'n_train': len(tr_ids), 'n_val': len(va_ids),
        'tasks_train': len(set(tr_tasks)), 'tasks_val': len(set(va_tasks)),
        'val_tasks_not_in_train': sorted(set(va_tasks) - set(tr_tasks)),
        'nn_val_to_train_any_task': pct(d_any),
        'nn_val_to_train_same_task': pct(d_task),
        'nn_val_to_val': pct(d_vv),
        'closer_to_train_than_to_any_other_val': int((d_task < d_vv).sum()),
        'counts_below': {str(t): int((d_task < t).sum()) for t in (0.05, 0.1, 0.25, 0.5, 1.0)},
        'examples': [
            {'val': va_ids[i], 'task': va_tasks[i], 'nearest_train': twin[i],
             'distance': round(float(d_task[i]), 4)}
            for i in np.argsort(d_task)[:10]],
    }
    json.dump(report, open(a.out, 'w'), indent=2)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
