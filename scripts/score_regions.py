#!/usr/bin/env python3
"""Gate 1: score a seeder's masks against the hand-made reference.

    python scripts/score_regions.py --reference_dir DIR --candidate_dir DIR [...]

Prints per-frame IoU, the per-region mean, and whether the candidate clears the bar in
`clipeval.regions.agree`. Pass several `--candidate_dir` with `--label` to compare
seeders side by side; the losers are worth recording too, so the comparison is not
re-litigated later from memory.
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clipeval.regions import agree


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--reference_dir', required=True)
    p.add_argument('--candidate_dir', action='append', required=True)
    p.add_argument('--label', action='append', default=None,
                   help='one per --candidate_dir; defaults to the directory name')
    p.add_argument('--candidate_suffix', default='sam3',
                   help='candidate files are <stem>_<view>_<suffix>.npz')
    p.add_argument('--out')
    return p.parse_args()


def key_of(path, suffix):
    """<stem>_<view>_<suffix>.npz -> (stem, view)."""
    base = os.path.basename(path)[:-len(f'_{suffix}.npz')]
    for view in ('left_wrist', 'right_wrist', 'top'):
        if base.endswith('_' + view):
            return base[:-len(view) - 1], view
    raise ValueError(f'cannot read a view from {path}')


def main():
    args = parse_args()
    labels = args.label or [os.path.basename(d.rstrip('/'))
                            for d in args.candidate_dir]
    refs = {key_of(p, 'reference'): p
            for p in glob.glob(os.path.join(args.reference_dir, '*_reference.npz'))}
    report = {'reference_dir': args.reference_dir, 'candidates': {}}

    for label, cdir in zip(labels, args.candidate_dir):
        rows, detail = [], []
        for (stem, view), rpath in sorted(refs.items()):
            cpath = os.path.join(cdir, f'{stem}_{view}_{args.candidate_suffix}.npz')
            if not os.path.exists(cpath):
                continue
            with np.load(rpath) as z:
                unscored = [str(u) for u in z['unscored']] if 'unscored' in z else []
            ref = agree.load_masks(rpath, regions=agree.REGIONS)
            cand = agree.load_masks(cpath, regions=agree.REGIONS)
            row = agree.compare(cand, ref, unscored=unscored)
            rows.append(row)
            clip = stem.split('_traj_')[-1].split('_8_5_')[0]
            detail.append({'clip': clip, 'view': view, 'unscored': unscored, **row})
            cells = ' '.join(
                f'{r}={row[r]:.2f}({row[f"{r}_cand_px"]}/{row[f"{r}_ref_px"]}px)'
                if r in row else f'{r}=--' for r in agree.REGIONS)
            print(f'{label:>10s}  {clip:>7s} {view:<12s} {cells}')
        summary = agree.summarise(rows)
        print(f'{label:>10s}  {agree.sentence(summary)}\n')
        report['candidates'][label] = {'summary': summary, 'per_frame': detail}

    if args.out:
        with open(args.out, 'w') as f:
            json.dump(report, f, indent=1)
        print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
