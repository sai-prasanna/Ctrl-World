#!/usr/bin/env python3
"""Gate 1: text-prompted arm and object masks on frame 0 of a saved rollout.

The plan's load-bearing assumption is that we can get usable masks autonomously, at
192x192 with a 24 px letterbox, from text prompts alone. This renders them at 6x so the
answer can be decided by looking rather than by a number.

The masks come from the ground-truth half of the stacked rollout mp4, which is the half
they are always taken from; the prediction is drawn beside them only so the same region
can be seen on both.

    python scripts/segment_rollout.py --video <rollout.mp4> --out_dir <dir>
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clipeval.regions.segment import ARM_PROMPT, RegionSegmenter, overlay
from clipeval.tracking.seeding import content_bbox

VIEWS = ('top', 'left_wrist', 'right_wrist')

# Object prompts per task, matched against the rollout filename. The manipulated thing
# only. Containers were in this list once and it was a mistake: 'bin' and 'container'
# outscore a lego brick a few dozen pixels across, so the union that was supposed to be
# the object came back as the red bin and the checkered tray, and the legos -- the thing
# the model erases -- sat in the background region. A container is scenery here.
TASK_PROMPTS = {
    'screwdriver': ['screwdriver', 'tool'],
    'legos': ['toy', 'small colorful toy'],
    'sleeve': ['shirt', 'folded cloth'],
    'shirt': ['shirt', 'folded cloth'],
}
DEFAULT_PROMPTS = ['object']


def prompts_for(video):
    stem = os.path.basename(video).lower()
    for key, phrases in TASK_PROMPTS.items():
        if key in stem:
            return phrases
    return DEFAULT_PROMPTS


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--video', required=True, help='stacked gt-over-pred rollout mp4')
    p.add_argument('--out_dir', required=True)
    p.add_argument('--views', nargs='+', default=list(VIEWS), choices=VIEWS)
    p.add_argument('--object_prompt', nargs='+', default=None,
                   help='overrides the per-task default')
    p.add_argument('--arm_prompt', default=ARM_PROMPT)
    p.add_argument('--frame', type=int, default=0)
    p.add_argument('--box_threshold', type=float, default=0.2)
    p.add_argument('--object_top_k', type=int, default=3,
                   help='union this many object detections; above 1 for a pile')
    p.add_argument('--arm_top_k', type=int, default=2,
                   help='2 by default: ABC-130k is bimanual and one box is one arm')
    p.add_argument('--upscale', type=int, default=4, help='before detection')
    p.add_argument('--zoom', type=int, default=6, help='of the rendered overlay')
    p.add_argument('--device', default='cuda')
    return p.parse_args()


def split_panels(video, view):
    """(T, 2H, 3W, 3) -> the ground-truth and predicted halves of one view."""
    t, h2, w3, _ = video.shape
    h, w = h2 // 2, w3 // len(VIEWS)
    c = VIEWS.index(view)
    return video[:, :h, c * w:(c + 1) * w], video[:, h:, c * w:(c + 1) * w]


def main():
    args = parse_args()
    from decord import VideoReader, cpu
    from PIL import Image

    vr = VideoReader(args.video, ctx=cpu(0))
    video = vr.get_batch(list(range(len(vr)))).asnumpy()
    phrases = args.object_prompt or prompts_for(args.video)

    seg = RegionSegmenter(device=args.device, upscale=args.upscale)
    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.video))[0]
    report = {'video': args.video, 'frame': args.frame,
              'object_prompt': phrases, 'arm_prompt': args.arm_prompt, 'views': {}}

    strips = []
    for view in args.views:
        gt, pred = split_panels(video, view)
        frame = gt[args.frame]
        bbox = content_bbox(gt)
        r = seg.regions(frame, phrases, arm_phrases=args.arm_prompt, bbox=bbox,
                        box_threshold=args.box_threshold,
                        object_top_k=args.object_top_k, arm_top_k=args.arm_top_k)
        masks = {k: r[k] for k in ('background', 'arm', 'object')}
        strip = np.concatenate([
            overlay(frame, {}, upscale=args.zoom),
            overlay(frame, masks, upscale=args.zoom),
            overlay(pred[args.frame], masks, upscale=args.zoom)], axis=1)
        strips.append(strip)
        area = frame.shape[0] * frame.shape[1]
        report['views'][view] = {
            'arm_px': int(r['arm'].sum()), 'object_px': int(r['object'].sum()),
            'object_area_fraction': float(r['object'].sum() / area),
            'arm_dets': r['arm_dets'], 'object_dets': r['object_dets'],
            'content_bbox': list(r['bbox'])}
        np.savez_compressed(os.path.join(args.out_dir, f'{stem}_{view}_regions.npz'),
                            arm=r['arm'], object=r['object'],
                            background=r['background'], frame=frame,
                            content_bbox=np.asarray(r['bbox']))
        print(f'{view}: arm {r["arm"].sum():5d} px from '
              f'{[d["label"] for d in r["arm_dets"]]}, object '
              f'{r["object"].sum():5d} px from '
              f'{[(d["label"], round(d["score"], 2)) for d in r["object_dets"]]}')

    sheet = os.path.join(args.out_dir, f'{stem}_regions.png')
    Image.fromarray(np.concatenate(strips, axis=0)).save(sheet)
    with open(os.path.join(args.out_dir, f'{stem}_regions.json'), 'w') as f:
        json.dump(report, f, indent=1)
    print(f'wrote {sheet}  (rows: {", ".join(args.views)}; '
          f'columns: frame, masks on real, same masks on generated)')


if __name__ == '__main__':
    main()
