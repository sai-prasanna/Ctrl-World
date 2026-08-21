#!/usr/bin/env python3
"""Render tracked points over a clip, coloured by how far the ground truth moves them.

The point of this view is to make the displacement strata visible. `clipeval` reports
tracking error separately for points that barely move, move a little, and move a lot,
because a median pooled over a 16x16 grid is dominated by background where there is
nothing to measure. Reading those numbers means knowing which dots are in which bucket,
and that is much easier to see than to describe.

Colour is the bucket. A trail shows where the point has been over the last few frames,
so a still frame already reads as motion. With `--pred_npz` the ground-truth and
predicted tracks are drawn side by side over their own frames.

Input is the .npz written by `tracker_noise_floor.py --save_tracks`, so this runs on a
laptop with no GPU and no tracker install:

    python scripts/viz_tracks.py --npz outputs/.../101599_64_top_tracks.npz \
        --out outputs/.../101599_64_top_buckets.mp4
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clipeval import DISPLACEMENT_BINS
from clipeval.tracking.seeding import STATIC_MAX_DISPLACEMENT_PX

# One colour per displacement stratum, in the order clipeval reports them. Grey for the
# static bucket because it is context, not signal; warm colours for the points that
# carry the metric.
BUCKET_COLORS = [
    ((0, STATIC_MAX_DISPLACEMENT_PX), (120, 120, 120), 'static <4px'),
    ((4, 16), (60, 160, 255), '4-16px'),
    ((16, 64), (255, 170, 40), '16-64px'),
    ((64, float('inf')), (255, 60, 60), '>64px'),
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--npz', required=True,
                   help='from tracker_noise_floor.py --save_tracks')
    p.add_argument('--pred_npz', default=None,
                   help='optional predicted-side tracks; renders gt | pred side by side')
    p.add_argument('--out', required=True, help='output .mp4')
    p.add_argument('--trail', type=int, default=8,
                   help='frames of history to draw behind each point (0 for none)')
    p.add_argument('--scale', type=int, default=3,
                   help='upscale factor; 192x192 is too small to read at native size')
    p.add_argument('--fps', type=int, default=5,
                   help='ABC-130k is recorded at 5 Hz after subsampling')
    p.add_argument('--only_bucket', default=None,
                   help="draw one stratum only, e.g. '64' or '16-64'; hides the rest")
    p.add_argument('--dim', type=float, default=0.55,
                   help='darken the video so the dots read against it (1.0 = no dimming)')
    return p.parse_args()


def bucket_of(displacement):
    """(K,) displacement -> (K,) index into BUCKET_COLORS."""
    out = np.zeros(len(displacement), dtype=int)
    for i, ((lo, hi), _, _) in enumerate(BUCKET_COLORS):
        out[(displacement >= lo) & (displacement < hi)] = i
    return out


def select(bucket, only):
    """Boolean (K,) of which points to draw."""
    if not only:
        return np.ones(len(bucket), dtype=bool)
    names = [name for _, _, name in BUCKET_COLORS]
    for i, name in enumerate(names):
        if only in name or only == str(int(BUCKET_COLORS[i][0][0])):
            return bucket == i
    raise SystemExit(f'--only_bucket {only!r} matched none of {names}')


def draw_disc(canvas, x, y, color, radius, alpha=1.0):
    """Alpha-blend a small square at (x, y). Squares, not circles: at 3x upscale the
    difference is invisible and this stays dependency-free."""
    h, w = canvas.shape[:2]
    xi, yi = int(round(x)), int(round(y))
    y0, y1 = max(0, yi - radius), min(h, yi + radius + 1)
    x0, x1 = max(0, xi - radius), min(w, xi + radius + 1)
    if y0 >= y1 or x0 >= x1:
        return
    patch = canvas[y0:y1, x0:x1].astype(np.float32)
    canvas[y0:y1, x0:x1] = (patch * (1 - alpha)
                            + np.asarray(color, dtype=np.float32) * alpha).astype(np.uint8)


def render(frames, tracks, visible, displacement, args):
    """(T, H, W, 3) frames plus (T, K, 2) tracks -> upscaled, annotated frames."""
    scale = args.scale
    bucket = bucket_of(displacement)
    keep = select(bucket, args.only_bucket)
    t_max = min(len(frames), len(tracks))
    out = []
    for t in range(t_max):
        base = frames[t].astype(np.float32) * args.dim
        canvas = np.repeat(np.repeat(base.astype(np.uint8), scale, axis=0), scale, axis=1)
        for k in np.nonzero(keep)[0]:
            color = BUCKET_COLORS[bucket[k]][1]
            # Trail first so the current position paints over it.
            for back in range(args.trail, 0, -1):
                if t - back < 0:
                    continue
                x, y = tracks[t - back, k] * scale
                draw_disc(canvas, x, y, color, radius=max(0, scale // 3),
                          alpha=0.5 * (1 - back / (args.trail + 1)))
            x, y = tracks[t, k] * scale
            vis = bool(visible[t, k]) if visible is not None else True
            # A point the tracker has lost is drawn hollow-dark rather than hidden:
            # disappearing dots read as "nothing there", which is the wrong story.
            draw_disc(canvas, x, y, color if vis else (30, 30, 30),
                      radius=max(1, scale // 2), alpha=1.0 if vis else 0.8)
        out.append(canvas)
    return np.stack(out)


def legend_counts(displacement, keep):
    bucket = bucket_of(displacement)
    return {name: int(((bucket == i) & keep).sum())
            for i, (_, _, name) in enumerate(BUCKET_COLORS)}


def load(path):
    d = np.load(path)
    tracks = d['tracks']
    displacement = (d['displacement'] if 'displacement' in d
                    else np.linalg.norm(tracks - tracks[0][None], axis=-1).max(axis=0))
    return (d['frames'], tracks, d['visible'] if 'visible' in d else None, displacement)


def main():
    args = parse_args()
    import mediapy

    frames, tracks, visible, displacement = load(args.npz)
    panels = [render(frames, tracks, visible, displacement, args)]
    if args.pred_npz:
        p_frames, p_tracks, p_visible, _ = load(args.pred_npz)
        # Colour the predicted side by the GROUND TRUTH's displacement, so a dot keeps
        # its bucket across both panels and the eye can pair them.
        panels.append(render(p_frames, p_tracks, p_visible, displacement, args))
        t = min(len(panels[0]), len(panels[1]))
        gap = np.zeros((t, panels[0].shape[1], 4, 3), dtype=np.uint8)
        video = np.concatenate([panels[0][:t], gap, panels[1][:t]], axis=2)
    else:
        video = panels[0]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    mediapy.write_video(args.out, video, fps=args.fps)

    keep = select(bucket_of(displacement), args.only_bucket)
    print(f'wrote {args.out}  ({len(video)} frames, {video.shape[2]}x{video.shape[1]})')
    print('points per stratum: ' + ', '.join(
        f'{name} {n}' for name, n in legend_counts(displacement, keep).items()))
    if args.pred_npz:
        print('left panel: ground truth   right panel: prediction '
              '(coloured by ground-truth displacement)')


if __name__ == '__main__':
    main()
