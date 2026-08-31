#!/usr/bin/env python3
"""Write a mask overlay video for a whole episode, top view. Runs in `.venv-seed`.

Contact sheets answer "is the mask right on this frame". They do not answer "does the
mask stay on the object while the arm crosses in front of it", which is the question that
decides whether a region metric means anything over 49 frames. That needs video.

Output is a side-by-side mp4: the real frames, the arm region, and the manipulated-object
region, so all three run in lockstep and a mask that drifts is obvious.

    python scripts/segment_video_out.py --video EPISODE.mp4 --task "sort the legos ..." \
        --out_dir DIR

Every frame is segmented independently by default. Per-frame detection has no drift to
accumulate, which is the property we want while judging the prompt itself; `--track` uses
SAM 3's video half instead, which is what the metrics will eventually run.
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
from segment_sam3 import (ARM_PROMPT, ConceptSegmenter, REPO, load_video_model,
                          prompts_for, segment_video)

# (region key used for the overlay colour, caption shown on the panel). The key has to
# be a real region name or `overlay` falls back to yellow, which reads as a fourth region
# that does not exist.
PANELS = ((None, 'ground truth'), ('arm', 'arm'), ('object', 'manipulated object'))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--video', required=True)
    p.add_argument('--out_dir', required=True)
    p.add_argument('--label')
    p.add_argument('--task', help="the episode's instruction text, when the filename "
                                  'does not carry it')
    p.add_argument('--view', default='top', choices=VIEWS,
                   help='top only, in practice. SAM 3 returns an empty arm mask on every '
                        'wrist frame checked -- a gripper filling a wrist frame is not '
                        'something it calls a robot arm -- and the left-wrist object mask '
                        'locks onto the yellow container. The wrist views are out.')
    p.add_argument('--object_prompt')
    p.add_argument('--container_prompt', nargs='+')
    p.add_argument('--arm_prompt', default=ARM_PROMPT)
    p.add_argument('--threshold', type=float, default=0.2)
    p.add_argument('--track', action='store_true',
                   help="use SAM 3's video half instead of per-frame detection")
    p.add_argument('--max_frames', type=int, default=None)
    p.add_argument('--fps', type=int, default=5, help='the rate the frames were saved at')
    p.add_argument('--zoom', type=int, default=1,
                   help='nearest-neighbour display zoom. 1 keeps the output pixel-exact '
                        'with the input, which is what you want when judging a mask; '
                        'anything else is a viewing convenience that makes the frames '
                        'look resampled.')
    p.add_argument('--model', default=REPO)
    p.add_argument('--device', default='cuda')
    p.add_argument('--stacked', choices=('auto', 'yes', 'no'), default='auto')
    return p.parse_args()


def caption(width, text, height=16):
    from PIL import Image, ImageDraw
    img = Image.new('RGB', (width, height), (16, 16, 16))
    ImageDraw.Draw(img).text((4, 3), text, fill=(235, 235, 235))
    return np.asarray(img)


def panel(frame, mask, region, label, zoom):
    """One labelled panel: the frame with `mask` painted, or bare when `mask` is None."""
    img = overlay(frame, {} if mask is None else {region: mask}, upscale=zoom)
    return np.concatenate([caption(img.shape[1], label), img])


def main():
    args = parse_args()
    from decord import VideoReader, cpu
    import mediapy

    vr = VideoReader(args.video, ctx=cpu(0))
    video = vr.get_batch(list(range(len(vr)))).asnumpy()
    stacked = (args.stacked == 'yes' or
               (args.stacked == 'auto' and video.shape[2] == 3 * video.shape[1] // 2))
    frames = split_panels(video, args.view)[0] if stacked else video
    if args.max_frames:
        frames = frames[:args.max_frames]
    bbox = content_bbox(frames)

    spec = prompts_for(args.task or args.video)
    prompts = {'arm': args.arm_prompt,
               'object': args.object_prompt or spec['object'],
               'containers': args.container_prompt or spec['containers']}
    label = args.label or os.path.splitext(os.path.basename(args.video))[0]

    if args.track:
        model, processor = load_video_model(args.model, args.device, args.threshold)
        tracked = segment_video(model, processor, frames, prompts, args.device,
                                len(frames))
        per_frame = [disjoint(tracked['arm'][i],
                              tracked['object'][i] & ~tracked['containers'][i], bbox=bbox)
                     for i in range(len(frames))]
    else:
        seg = ConceptSegmenter(args.model, args.device)
        per_frame = []
        for i, frame in enumerate(frames):
            arm, _ = seg.mask(frame, prompts['arm'], args.threshold)
            obj, _ = seg.mask(frame, prompts['object'], args.threshold)
            con, _ = seg.mask(frame, prompts['containers'], args.threshold)
            per_frame.append(disjoint(arm, obj & ~con, bbox=bbox))
            if (i + 1) % 25 == 0:
                print(f'  {label}: {i + 1}/{len(frames)} frames')

    out_frames = []
    for frame, masks in zip(frames, per_frame):
        out_frames.append(np.concatenate(
            [panel(frame, None if region is None else masks[region], region, label,
                   args.zoom)
             for region, label in PANELS], axis=1))

    os.makedirs(args.out_dir, exist_ok=True)
    suffix = 'tracked' if args.track else 'perframe'
    path = os.path.join(args.out_dir, f'{label}_{args.view}_{suffix}.mp4')
    mediapy.write_video(path, np.stack(out_frames), fps=args.fps)
    empty = sum(1 for m in per_frame if not m['object'].any())
    print(f'wrote {path}  ({len(out_frames)} frames, object empty on {empty})')


if __name__ == '__main__':
    main()
