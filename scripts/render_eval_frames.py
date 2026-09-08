"""Render the npz frame dumps from eval_video_metrics.py as watchable mp4.

The dump is lossless npz because H.264 artifacts land differently on ground truth and
prediction, and that difference would read as model error. That argument governs scoring,
not viewing, so this writes ordinary mp4 for review.

One file per clip: ground truth above prediction, the three views side by side. The
failures worth looking for -- wrist appearance that flickers between frames, and views
that stop agreeing about the same scene -- are only visible with the views adjacent and
the reference directly above.

    python3 scripts/render_eval_frames.py --frames_dir <dir> --out_dir <dir>
"""
import argparse
import glob
import os
from concurrent.futures import ProcessPoolExecutor

import imageio.v2 as imageio
import numpy as np

SEP = 2  # separator width, px


def render(args_tuple):
    path, out_dir, fps = args_tuple
    d = np.load(path)
    gt, pred = d['gt'], d['pred']
    # Rounds overlap, so each round regenerates the frame it was conditioned on and the
    # scorer drops it. gt therefore carries one frame more than pred: the clip's first
    # frame, which the model was given rather than asked to predict. Drop it to align.
    gt = gt[:, gt.shape[1] - pred.shape[1]:]

    n_views, n_frames, h, w, _ = pred.shape
    rows = 2 * h + SEP
    cols = n_views * w + (n_views - 1) * SEP
    out = os.path.join(out_dir, os.path.basename(path).replace('.npz', '.mp4'))
    # macro_block_size=1 keeps the exact pixel grid; the default rescales to a multiple
    # of 16 and would resample every frame.
    with imageio.get_writer(out, fps=fps, codec='libx264', quality=8,
                            macro_block_size=1) as w_:
        for t in range(n_frames):
            canvas = np.full((rows, cols, 3), 255, dtype=np.uint8)
            for v in range(n_views):
                x = v * (w + SEP)
                canvas[:h, x:x + w] = gt[v, t]
                canvas[h + SEP:, x:x + w] = pred[v, t]
            w_.append_data(canvas)
    return os.path.getsize(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--frames_dir', required=True)
    p.add_argument('--out_dir', required=True)
    # The dump is one frame per latent step and the latents are 5 Hz, so 5 fps plays at
    # wall-clock speed. Slower reads the per-frame flicker more easily.
    p.add_argument('--fps', type=int, default=5)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--limit', type=int, default=None)
    a = p.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(a.frames_dir, '*.npz')))[:a.limit]
    if not paths:
        raise SystemExit(f'no npz in {a.frames_dir}')
    print(f'rendering {len(paths)} clips at {a.fps} fps')

    with ProcessPoolExecutor(a.workers) as ex:
        sizes = list(ex.map(render, [(q, a.out_dir, a.fps) for q in paths]))
    print(f'wrote {len(sizes)} mp4, {sum(sizes) / 1e6:.1f} MB total, to {a.out_dir}')


if __name__ == '__main__':
    main()
