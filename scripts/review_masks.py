#!/usr/bin/env python3
"""Render SAM 3's masks across a whole episode, for a human to look at. `.venv-seed`.

Gate 1 scores frame 0 because that is the frame the world model is conditioned on. That
is the right thing to score and the wrong thing to review: at frame 0 the manipulated
object is often still in a pile, untouched, and in the wrist views frequently out of shot.
The interesting frames are the ones where the arm is holding the thing.

So this samples frames across the clip and renders one labelled sheet per view, which is
what you want in front of you when deciding whether a prompt is working.

    python scripts/review_masks.py --video EPISODE.mp4 --out_dir DIR --n_frames 8

Takes either a plain single-view episode mp4 (`data_local/abc_rigid/videos/val/<id>/0.mp4`)
or a stacked gt-over-pred rollout, detected from the frame shape.
"""

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

from clipeval.regions.segment import VIEWS, disjoint, overlay, split_panels
from clipeval.tracking.seeding import content_bbox
from segment_sam3 import ARM_PROMPT, ConceptSegmenter, REPO, TASK_PROMPTS, prompts_for


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--video', required=True)
    p.add_argument('--out_dir', required=True)
    p.add_argument('--label', help='name for the output file; defaults to the stem')
    p.add_argument('--views', nargs='+', default=['top'], choices=VIEWS)
    p.add_argument('--n_frames', type=int, default=8)
    p.add_argument('--object_prompt')
    p.add_argument('--container_prompt', nargs='+',
                   help='overrides the per-task default; several phrases allowed')
    p.add_argument('--task',
                   help="the episode's instruction text, when the filename does not "
                        "carry it. A raw episode video is called 0.mp4, so there is "
                        "nothing to match a prompt against without this.")
    p.add_argument('--arm_prompt', default=ARM_PROMPT)
    p.add_argument('--threshold', type=float, default=0.2)
    p.add_argument('--model', default=REPO)
    p.add_argument('--device', default='cuda')
    p.add_argument('--zoom', type=int, default=4)
    p.add_argument('--stacked', choices=('auto', 'yes', 'no'), default='auto')
    return p.parse_args()


def label_strip(width, text, height=18):
    """A caption bar, drawn with PIL so the sheet is readable without a viewer."""
    from PIL import Image, ImageDraw
    img = Image.new('RGB', (width, height), (16, 16, 16))
    ImageDraw.Draw(img).text((4, 4), text, fill=(235, 235, 235))
    return np.asarray(img)


def main():
    args = parse_args()
    from decord import VideoReader, cpu
    from PIL import Image

    vr = VideoReader(args.video, ctx=cpu(0))
    video = vr.get_batch(list(range(len(vr)))).asnumpy()
    stacked = (args.stacked == 'yes' or
               (args.stacked == 'auto' and video.shape[2] == 3 * video.shape[1] // 2))
    spec = prompts_for(args.task or args.video)
    object_prompt = args.object_prompt or spec['object']
    container_prompt = args.container_prompt or spec['containers']

    seg = ConceptSegmenter(args.model, args.device)
    os.makedirs(args.out_dir, exist_ok=True)
    label = args.label or os.path.splitext(os.path.basename(args.video))[0]

    for view in args.views:
        frames = split_panels(video, view)[0] if stacked else video
        idx = np.linspace(0, len(frames) - 1, args.n_frames).astype(int)
        bbox = content_bbox(frames)
        rows = []
        for i in idx:
            frame = frames[i]
            arm, arm_scores = seg.mask(frame, args.arm_prompt, args.threshold)
            obj, obj_scores = seg.mask(frame, object_prompt, args.threshold)
            con, _ = seg.mask(frame, container_prompt, args.threshold)
            masks = disjoint(arm, obj & ~con, bbox=bbox)
            masks['containers'] = con & ~arm & masks['background']
            pair = np.concatenate([overlay(frame, {}, upscale=args.zoom),
                                   overlay(frame, masks, upscale=args.zoom)], axis=1)
            caption = (f'f{i:03d}  arm {masks["arm"].sum():5d}px '
                       f'({len(arm_scores)} inst)  object {masks["object"].sum():5d}px '
                       f'({len(obj_scores)} inst)  '
                       f'containers {masks["containers"].sum():5d}px')
            rows.append(np.concatenate([label_strip(pair.shape[1], caption), pair]))
            print(f'{label} {view} {caption}')
        header = label_strip(rows[0].shape[1],
                             f'{label}  {view}   arm="{args.arm_prompt}"  '
                             f'object="{object_prompt}"  '
                             f'containers="{container_prompt}"  '
                             f'threshold={args.threshold}   '
                             f'blue=arm  red=object  yellow=containers  green=background',
                             height=22)
        out = os.path.join(args.out_dir, f'{label}_{view}_review.png')
        Image.fromarray(np.concatenate([header] + rows)).save(out)
        print(f'wrote {out}')


if __name__ == '__main__':
    main()
