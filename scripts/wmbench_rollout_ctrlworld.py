"""Roll a Ctrl-World checkpoint over the evaluation clips and write a wmbench manifest.

The same rollout `scripts/eval_video_metrics.py` runs, with the scoring taken out. That
script generates frames and scores them in one process, which was right when the metric
set was five numbers from `clipeval`; it is wrong now that scoring is sixteen metrics with
their own backbones and their own virtual environment. A rollout costs GPU-hours and a
score costs CPU-minutes, so the frames are written down once (manifest v1: `manifest.json`
plus one lossless npz per clip) and scored as often as the metric set grows, on another
machine if that is where the backbones are.

The frames are the same frames. `bridges/ctrlworld_wm.py` holds the rollout arithmetic and
`wmbench.core.worldmodel.replay` holds the loop, and the npz this writes is byte-identical
to `eval_video_metrics.py --dump_frames` on the same clips and checkpoint. That is what
makes the two comparable: numbers already in `docs/experiments.md` were produced by the
older path.

Example:

    python3 scripts/wmbench_rollout_ctrlworld.py \\
        --ckpt_path outputs/0003_abc_mcap/model/checkpoint-200000.pt \\
        --clips dataset_meta_info/abc_mcap/eval_clips_v1.json \\
        --val_dataset_dir $CTRLWORLD_DATA/abc_mcap \\
        --data_stat_path $CTRLWORLD_META/abc_mcap/stat.json \\
        --tag 0003_abc_mcap --batch_size 4

Score it from the benchmark environment, which needs no Ctrl-World and no GPU rollout:

    wmbench score --manifest outputs/0003_abc_mcap/wmbench/<run> --out <metrics.json>
"""

import argparse
import json
import os
import sys

import numpy as np
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from abc130k.bench_source import ABC_PROFILE, AnnotationSource  # noqa: E402
from bridges.ctrlworld_wm import HISTORY_IDX, CtrlWorldModel  # noqa: E402
from wmbench.core.clips import load_clip_list  # noqa: E402
from wmbench.core.manifest import ManifestWriter  # noqa: E402
from wmbench.core.worldmodel import replay  # noqa: E402


def parse_args():
    # Flag names follow eval_video_metrics.py, so a command line written for that script
    # runs here with only --out changing meaning: a manifest directory, not a JSON file.
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt_path', type=str, default=None,
                   help='checkpoint-<step>.pt; omit or pass "none" to roll out the SVD '
                        'initialization, the model before any ABC training')
    p.add_argument('--clips', type=str, required=True,
                   help='clip list from scripts/make_eval_clips.py or `wmbench clips`')
    p.add_argument('--val_dataset_dir', type=str, default=None,
                   help='defaults to the value recorded in the clip list')
    p.add_argument('--data_stat_path', type=str, default=None)
    p.add_argument('--svd_model_path', type=str, default=None)
    p.add_argument('--clip_model_path', type=str, default=None)
    p.add_argument('--out', type=str, default=None,
                   help='manifest directory; defaults to '
                        '<run_dir>/wmbench/<checkpoint name>')
    p.add_argument('--tag', type=str, default=None,
                   help='<exp_id>_<exp_name>, which names the run directory')
    p.add_argument('--run_dir', type=str, default=None,
                   help='absolute home for the run outputs; see --tag')
    p.add_argument('--split', type=str, default='val')
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--rounds', type=int, default=None,
                   help='rollout rounds; defaults to the clip list\'s interact_num')
    p.add_argument('--limit', type=int, default=None,
                   help='roll out only the first N clips (smoke tests)')
    p.add_argument('--teacher_forced', action='store_true',
                   help='feed ground-truth latents into the history buffer instead of '
                        'the model\'s own predictions')
    p.add_argument('--history_idx', type=str, default=None,
                   help='comma-separated history buffer indexes; defaults to the uniform '
                        '"-6,-5,-4,-3,-2,-1". The authors use "0,0,-12,-9,-6,-3".')
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    from config import merge_args, wm_args

    clips_payload = json.load(open(args.clips))
    history_idx = ([int(x) for x in args.history_idx.split(',')] if args.history_idx
                   else HISTORY_IDX)

    cfg = wm_args(task_type='abc_replay')
    cfg = merge_args(cfg, args)
    # The rollout shape is the clip list's, not the config's: a clip list drawn for 12
    # rounds of 5 states reserves exactly that much ground truth per clip, and reading it
    # back with another shape would run off the end of the episodes it was drawn from.
    cfg.pred_step = clips_payload['pred_step']
    cfg.interact_num = clips_payload['interact_num']
    if args.val_dataset_dir is None:
        args.val_dataset_dir = clips_payload['val_dataset_dir']
    if args.data_stat_path is None:
        args.data_stat_path = cfg.data_stat_path
    # A relative outputs/ is empty in a fresh `cluster submit` checkout, so the run home
    # is overridable for the same reason train_wm.py's is.
    run_dir = args.run_dir or (f'outputs/{args.tag}' if args.tag else cfg.run_dir)

    if args.ckpt_path in (None, '', 'none'):
        args.ckpt_path = None
    wm = CtrlWorldModel(cfg, ckpt_path=args.ckpt_path,
                        val_dataset_dir=args.val_dataset_dir,
                        data_stat_path=args.data_stat_path, split=args.split,
                        history_idx=history_idx, teacher_forced=args.teacher_forced)
    op = wm.operating_point
    rounds = args.rounds or cfg.interact_num

    source = AnnotationSource(root=args.val_dataset_dir, fps=op.fps,
                              geometry=f'{op.width}x{op.height}',
                              preprocess_id=op.preprocess_id, profile=ABC_PROFILE)

    clips, _ = load_clip_list(args.clips, fps=op.fps)
    if args.limit:
        clips = clips[:args.limit]

    ckpt_name = (os.path.basename(args.ckpt_path).removesuffix('.pt')
                 if args.ckpt_path else 'checkpoint-0')
    out = args.out or os.path.join(run_dir, 'wmbench', ckpt_name)
    writer = ManifestWriter(out, meta={
        'world_model': 'bridges.ctrlworld_wm:CtrlWorldModel',
        'checkpoint': os.path.abspath(args.ckpt_path) if args.ckpt_path else 'svd_init',
        'source': 'abc130k.bench_source:AnnotationSource',
        'source_args': {'root': os.path.abspath(args.val_dataset_dir)},
        'provenance': source.provenance.to_dict(),
        'profile': source.profile.to_dict(),
        'operating_point': op.to_dict(),
        'views': list(source.profile.view_names),
        'rounds': rounds,
        'frames_per_step': wm.frames_per_step,
        'gt_frames_per_clip': 1 + rounds * wm.frames_per_step,
        'split': args.split,
        'seed': args.seed,
        'clip_list': os.path.abspath(args.clips),
        'clip_list_version': clips_payload.get('version'),
        'mode': 'teacher_forced' if args.teacher_forced else 'free_running',
        'ground_truth': 'raw mp4 frames at the 5Hz latent rate (no VAE round-trip)',
        # Everything below changes the frames without changing the command line, so it is
        # recorded rather than reconstructed from whichever config.py a reader has.
        'config': {
            'history_idx': history_idx,
            'num_inference_steps': cfg.num_inference_steps,
            'guidance_scale': cfg.guidance_scale,
            'pred_step': cfg.pred_step,
            'interact_num': cfg.interact_num,
            'num_history': cfg.num_history,
            'num_frames': cfg.num_frames,
            'decode_chunk_size': cfg.decode_chunk_size,
            'data_stat_path': os.path.abspath(args.data_stat_path),
        },
    })

    latent_mse = []
    for start in tqdm(range(0, len(clips), args.batch_size), desc='rollout'):
        batch = clips[start:start + args.batch_size]
        frames = replay(wm, source, batch, rounds, seed=args.seed, split=args.split)
        for clip, clip_frames in zip(batch, frames):
            writer.add(clip_frames, clip)
        if wm.last_latent_mse is not None:
            latent_mse.append(wm.last_latent_mse)

    if latent_mse:
        # The one diagnostic measured in the space the model predicts in, so it carries no
        # VAE reconstruction error. Kept in the manifest because a scorer reading pixels
        # cannot recover it.
        stacked = np.concatenate(latent_mse, axis=0)
        writer.meta['latent_mse'] = {'per_round': stacked.mean(axis=0).tolist(),
                                     'mean': float(stacked.mean())}
    manifest = writer.close()

    print(f'wrote {len(manifest)} clips ({writer.bytes / 2 ** 30:.2f} GiB) to {out}')
    if latent_mse:
        per_round = writer.meta['latent_mse']['per_round']
        print(f'  latent MSE first round {per_round[0]:.4f} -> '
              f'last round {per_round[-1]:.4f}')


if __name__ == '__main__':
    main()
