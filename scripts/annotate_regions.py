#!/usr/bin/env python3
"""Hand-made reference masks for the region metrics. Runs in `.venv-seed`.

Gate 1 needs something to be right about. This turns a human's box — drawn on a frame, or
written down in a JSON — into a mask, using SAM 3's visual-prompt branch. The human
decides *what*, SAM 3 decides *which pixels*, which is the division of labour that makes
twelve reference masks ten minutes of work instead of an hour of pixel painting.

Two ways in, the same output either way:

    # boxes written down, no display needed
    python scripts/annotate_regions.py --video ROLLOUT.mp4 --boxes boxes.json --out_dir DIR

    # click them
    python scripts/annotate_regions.py --video ROLLOUT.mp4 --interactive --out_dir DIR

The boxes JSON is `{view: {region: [x0, y0, x1, y1] or [[...], ...]}}` in the view's own
192x192 pixels. Two ways to say "no box", and they mean different things:

  * `[]` — the region is genuinely not in this frame. A real answer, and the scorer holds
    a seeder to it: finding an arm where there is none is a miss.
  * `null` — the annotator could not tell. The region is dropped from scoring entirely. A
    screwdriver a human cannot pick out of a 192x192 frame is not evidence about the
    segmenter either way, and scoring it as empty would hand a free 1.0 to anything that
    also found nothing.

Masks are always taken from the ground-truth half of the stacked rollout. That asymmetry
is the point of the whole layer: the reference says where the object *is*, and the
prediction is scored on whatever it put there.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clipeval.regions.segment import VIEWS, overlay, split_panels
from clipeval.tracking.seeding import content_bbox

REPO = 'jetjodh/sam3'
REGIONS = ('arm', 'object')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--video', required=True, help='stacked gt-over-pred rollout mp4')
    p.add_argument('--out_dir', required=True)
    p.add_argument('--boxes',
                   help='JSON of {view: {region: boxes}}, optionally nested under the '
                        'video stem so one file covers every clip')
    p.add_argument('--interactive', action='store_true',
                   help='draw the boxes with the mouse instead')
    p.add_argument('--views', nargs='+', default=list(VIEWS), choices=VIEWS)
    p.add_argument('--frame', type=int, default=0)
    p.add_argument('--model', default=REPO)
    p.add_argument('--device', default='cuda')
    p.add_argument('--zoom', type=int, default=6)
    args = p.parse_args()
    if not args.boxes and not args.interactive:
        p.error('pass --boxes or --interactive')
    return args


class BoxSegmenter:
    """SAM 3's visual-prompt branch: a box in, the mask of what is inside it out."""

    def __init__(self, model=REPO, device='cuda'):
        import torch
        from transformers import Sam3Model, Sam3Processor
        self.torch = torch
        self.device = device
        self.model = Sam3Model.from_pretrained(
            model, dtype=torch.float32).to(device).eval()
        self.processor = Sam3Processor.from_pretrained(model)

    def mask(self, frame, boxes, threshold=0.2, margin=2):
        """Union of the masks SAM 3 returns for `boxes`, as one (H, W) bool.

        A region is often several things — two arms, a scatter of legos — so the caller
        passes several boxes and gets one mask. Instance identity is not needed here: the
        region metrics pool features over a region, not over an object.

        Each box is run on its own and its mask is clipped to the box, grown by `margin`
        pixels. That clipping is not tidying. SAM 3's box prompt is a *concept exemplar*,
        not an instance selector: a box drawn round one lego came back with every small
        colourful thing in the frame, the metal rail included. Concept generalisation is
        the right behaviour for the segmenter under test and the wrong behaviour for the
        reference it is tested against, where the human's box is what makes the mask
        ground truth. Inside the box, SAM 3 still decides which pixels, which is the part
        a human is bad at.
        """
        from PIL import Image
        h, w = np.asarray(frame).shape[:2]
        boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
        out_mask = np.zeros((h, w), dtype=bool)
        img = Image.fromarray(np.asarray(frame).astype(np.uint8))
        for box in boxes:
            inputs = self.processor(
                images=img, input_boxes=[[box.tolist()]], input_boxes_labels=[[1]],
                return_tensors='pt').to(self.device)
            with self.torch.no_grad():
                out = self.model(**inputs)
            res = self.processor.post_process_instance_segmentation(
                out, threshold=threshold, mask_threshold=0.5,
                target_sizes=inputs.get('original_sizes').tolist())[0]
            keep = np.zeros((h, w), dtype=bool)
            x0, y0, x1, y1 = box
            keep[max(0, int(y0) - margin):int(y1) + margin + 1,
                 max(0, int(x0) - margin):int(x1) + margin + 1] = True
            inside = [m.cpu().numpy().astype(bool) & keep for m in res['masks']]
            inside = [m for m in inside if m.any()]
            if inside:
                # The instance the box was drawn around is the one that fills it best.
                out_mask |= max(inside, key=lambda m: m.sum())
            else:
                out_mask |= keep      # SAM 3 found nothing here; the box itself is the
                                      # best statement of where the thing is.
        return out_mask


def draw_boxes(frame, zoom):
    """Click-drag boxes for each region on one frame. Returns {region: [boxes]}."""
    import matplotlib.pyplot as plt
    from matplotlib.widgets import RectangleSelector

    picked = {r: [] for r in REGIONS}
    for region in REGIONS:
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(np.repeat(np.repeat(frame, zoom, 0), zoom, 1))
        ax.set_title(f'drag boxes around: {region}\nclose the window when done '
                     f'(no box = not visible)')

        def on_select(press, release, region=region):
            x0, x1 = sorted((press.xdata, release.xdata))
            y0, y1 = sorted((press.ydata, release.ydata))
            picked[region].append([x0 / zoom, y0 / zoom, x1 / zoom, y1 / zoom])

        selector = RectangleSelector(ax, on_select, useblit=True, button=[1],
                                     minspanx=3, minspany=3, interactive=True)
        plt.show()
        del selector
    return picked


def main():
    args = parse_args()
    from decord import VideoReader, cpu
    from PIL import Image

    vr = VideoReader(args.video, ctx=cpu(0))
    video = vr.get_batch(list(range(len(vr)))).asnumpy()
    boxes_in = json.load(open(args.boxes)) if args.boxes else {}

    seg = BoxSegmenter(args.model, args.device)
    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.video))[0]
    # One boxes file can hold every clip, keyed by video stem, so the whole reference set
    # lives in one committed artefact rather than four.
    if stem in boxes_in:
        boxes_in = boxes_in[stem]
    strips, report = [], {'video': args.video, 'frame': args.frame, 'views': {}}

    for view in args.views:
        gt, _ = split_panels(video, view)
        frame = gt[args.frame]
        bbox = content_bbox(gt)
        spec = (draw_boxes(frame, args.zoom) if args.interactive
                else boxes_in.get(view, {}))
        masks, unscored = {}, []
        for region in REGIONS:
            boxes = spec.get(region, [])
            if boxes is None:
                unscored.append(region)
                boxes = []
            boxes = [boxes] if boxes and np.ndim(boxes[0]) == 0 else boxes
            masks[region] = seg.mask(frame, boxes)
        # The arm wins where the two overlap: during a grasp the gripper occludes the
        # thing it holds, so those pixels really are gripper.
        masks['object'] = masks['object'] & ~masks['arm']
        inside = np.zeros(frame.shape[:2], dtype=bool)
        y0, y1, x0, x1 = bbox
        inside[y0:y1, x0:x1] = True
        masks = {k: v & inside for k, v in masks.items()}

        np.savez_compressed(
            os.path.join(args.out_dir, f'{stem}_{view}_reference.npz'),
            frame=frame.astype(np.uint8), content_bbox=np.asarray(bbox),
            unscored=np.asarray(unscored, dtype='<U16'), **masks)
        strips.append(np.concatenate(
            [overlay(frame, {}, upscale=args.zoom),
             overlay(frame, masks, upscale=args.zoom)], axis=1))
        report['views'][view] = {'boxes': {r: spec.get(r, []) for r in REGIONS},
                                 'unscored': unscored,
                                 **{f'{r}_px': int(masks[r].sum()) for r in REGIONS}}
        print(f'{view}: ' + ', '.join(
            f'{r} ' + ('not identifiable' if r in unscored
                       else f'{masks[r].sum()} px') for r in REGIONS))

    sheet = os.path.join(args.out_dir, f'{stem}_reference.png')
    Image.fromarray(np.concatenate(strips, axis=0)).save(sheet)
    with open(os.path.join(args.out_dir, f'{stem}_reference.json'), 'w') as f:
        json.dump(report, f, indent=1)
    print(f'wrote {sheet}')


if __name__ == '__main__':
    main()
