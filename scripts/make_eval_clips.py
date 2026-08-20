"""Build a fixed evaluation clip list for scripts/eval_video_metrics.py.

The training meta info (dataset_meta_info/create_meta_info.py) emits one window per
start frame, which gives tens of thousands of near-duplicate clips per split. That is
fine for training but wrong for evaluation: it double-counts long episodes and the
windows are almost entirely overlapping. This script instead draws a small, fixed,
task-stratified sample of non-overlapping clips and writes it to JSON, so every
checkpoint is scored on exactly the same clips.

Selection rules:

  * A clip is valid only if `start + needed_frames <= video_length`. The rollout reads
    `pred_step * interact_num + 8` frames of ground truth, and `get_traj_info` silently
    clamps past the end of an episode, which would freeze the ground truth while the
    model keeps predicting. Enforcing the bound here keeps that from happening.
  * Starts are drawn uniformly at random from the valid range, seeded per episode.
    Fixed fractional offsets (say 25% and 70% of the episode) would bias the sample
    towards particular phases of a sort-and-place task.
  * At most `--max_per_episode` clips per episode, spaced at least one clip apart, so
    no single episode dominates.
  * Clips are allocated across tasks in proportion to the number of validation
    episodes per task, subject to per-task capacity.

Example:

    python3 scripts/make_eval_clips.py \
        --val_dataset_dir data/abc_rigid \
        --out dataset_meta_info/abc_rigid/eval_clips_v1.json
"""

import argparse
import hashlib
import glob
import json
import os
from collections import defaultdict

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--val_dataset_dir', type=str, required=True,
                   help='dataset root holding annotation/val/*.json')
    p.add_argument('--out', type=str, required=True)
    p.add_argument('--n_clips', type=int, default=256,
                   help="paper's Table 1 sample size")
    p.add_argument('--pred_step', type=int, default=5)
    p.add_argument('--interact_num', type=int, default=12)
    p.add_argument('--max_per_episode', type=int, default=2)
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args()


def clip_frames(pred_step, interact_num):
    """Number of frames a rollout covers: the initial observation plus 4 per round."""
    return 1 + (pred_step - 1) * interact_num


def needed_frames(pred_step, interact_num):
    """Ground-truth frames get_traj_info requests, which is more than it consumes."""
    return pred_step * interact_num + 8


def load_episodes(val_dataset_dir):
    episodes = []
    for path in sorted(glob.glob(f'{val_dataset_dir}/annotation/val/*.json')):
        with open(path) as f:
            ann = json.load(f)
        episodes.append({
            'episode_id': os.path.basename(path)[:-len('.json')],
            'length': int(ann['video_length']),
            'task': ann['texts'][0],
        })
    return episodes


def stable_seed(base_seed, key):
    """Seed that survives a restart.

    Python salts str.__hash__ per process (PYTHONHASHSEED), so hash() here would make
    the clip list differ between runs despite the fixed --seed.
    """
    digest = hashlib.sha256(str(key).encode()).digest()
    return (base_seed * 1000003 + int.from_bytes(digest[:4], 'big')) % (2 ** 31)


def draw_starts(rng, low, high, count, min_gap):
    """Draw up to `count` starts in [low, high] pairwise at least min_gap apart."""
    starts = []
    for _ in range(200):
        if len(starts) == count:
            break
        candidate = int(rng.integers(low, high + 1))
        if all(abs(candidate - s) >= min_gap for s in starts):
            starts.append(candidate)
    return sorted(starts)


def allocate(per_task_capacity, total):
    """Largest-remainder allocation of `total` clips proportional to capacity."""
    capacity_sum = sum(per_task_capacity.values())
    total = min(total, capacity_sum)
    exact = {t: total * c / capacity_sum for t, c in per_task_capacity.items()}
    alloc = {t: min(int(np.floor(v)), per_task_capacity[t]) for t, v in exact.items()}
    # hand out what rounding left over, largest remainder first, respecting capacity
    remainder = sorted(per_task_capacity, key=lambda t: exact[t] - np.floor(exact[t]),
                       reverse=True)
    while sum(alloc.values()) < total:
        progressed = False
        for task in remainder:
            if sum(alloc.values()) == total:
                break
            if alloc[task] < per_task_capacity[task]:
                alloc[task] += 1
                progressed = True
        if not progressed:
            break
    return alloc


def main():
    args = parse_args()
    n_frames = clip_frames(args.pred_step, args.interact_num)
    n_needed = needed_frames(args.pred_step, args.interact_num)

    episodes = load_episodes(args.val_dataset_dir)
    if not episodes:
        raise SystemExit(f'no annotations under {args.val_dataset_dir}/annotation/val')

    usable = [e for e in episodes if e['length'] >= n_needed]
    dropped = len(episodes) - len(usable)
    print(f'{len(episodes)} val episodes, {len(usable)} long enough for '
          f'{n_needed} frames ({dropped} dropped)')

    # candidate clips per episode, drawn before allocation so capacity is exact
    candidates = defaultdict(list)  # task -> list of clip dicts
    for ep in usable:
        rng = np.random.default_rng(stable_seed(args.seed, ep['episode_id']))
        starts = draw_starts(rng, 0, ep['length'] - n_needed,
                             args.max_per_episode, n_frames)
        for start in starts:
            candidates[ep['task']].append({
                'episode_id': ep['episode_id'],
                'start_idx': start,
                'task': ep['task'],
                'length': ep['length'],
            })

    capacity = {t: len(v) for t, v in candidates.items()}
    alloc = allocate(capacity, args.n_clips)

    rng = np.random.default_rng(args.seed)
    clips = []
    for task in sorted(candidates):
        pool = candidates[task]
        take = alloc.get(task, 0)
        # prefer to keep one clip from as many distinct episodes as possible
        by_episode = defaultdict(list)
        for c in pool:
            by_episode[c['episode_id']].append(c)
        order = list(by_episode)
        rng.shuffle(order)
        ranked = []
        for rank in range(args.max_per_episode):
            for episode_id in order:
                if rank < len(by_episode[episode_id]):
                    ranked.append(by_episode[episode_id][rank])
        clips.extend(ranked[:take])

    clips.sort(key=lambda c: (str(c['episode_id']), c['start_idx']))
    if len(clips) < args.n_clips:
        print(f'WARNING: only {len(clips)} clips available, wanted {args.n_clips}')

    task_counts = defaultdict(int)
    episode_ids = set()
    for c in clips:
        task_counts[c['task']] += 1
        episode_ids.add(c['episode_id'])

    out = {
        'version': 'eval_clips_v1',
        'val_dataset_dir': args.val_dataset_dir,
        'seed': args.seed,
        'n_clips': len(clips),
        'n_episodes': len(episode_ids),
        'clip_frames': n_frames,
        'needed_frames': n_needed,
        'pred_step': args.pred_step,
        'interact_num': args.interact_num,
        'max_per_episode': args.max_per_episode,
        'episodes_dropped_too_short': dropped,
        'task_counts': dict(sorted(task_counts.items(), key=lambda kv: -kv[1])),
        'clips': clips,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2)

    print(f'wrote {len(clips)} clips over {len(episode_ids)} episodes to {args.out}')
    print(f'clip length {n_frames} frames ({n_frames / 5:.1f}s at 5Hz), '
          f'{n_needed} frames of ground truth required per clip')
    for task, count in out['task_counts'].items():
        flag = '  (too few for a per-task number)' if count < 10 else ''
        print(f'  {count:4d}  {task}{flag}')


if __name__ == '__main__':
    main()
