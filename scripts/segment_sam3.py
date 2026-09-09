#!/usr/bin/env python3
"""Region masks from text alone, with SAM 3. Runs in `.venv-seed`.

This is the candidate gate 1 scores. It replaces the GroundingDINO detector bolted to a
SAM segmenter that failed: SAM 3 does promptable concept segmentation, so one model turns
a noun phrase into instance masks, and a bad phrase match is visible as a low score rather
than as a confidently wrong box.

Two modes, the same output format:

    --mode frame0   masks on the conditioning frame, which is what gate 1 scores
    --mode video    masks propagated across the clip, which is what gate 2 needs

Masks always come from the ground-truth half of the stacked rollout and are written to
disk. Nothing here ever looks at the prediction. That asymmetry is the whole point of the
region layer: on the prediction the object may not exist, and a segmenter run there would
happily mask the plausible thing the model painted instead.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clipeval.regions.segment import VIEWS, disjoint, overlay, split_panels
from clipeval.tracking.seeding import content_bbox

REPO = 'jetjodh/sam3'

# Noun phrases per region, matched against the rollout filename. Two things learned the
# expensive way from the GroundingDINO attempt are baked in here. Containers are their own
# region and never unioned into the object: 'bin' and 'container' outscore a lego brick a
# few dozen pixels across, and the object region came back as the red bin. And the arm
# prompt has to find two instances, because ABC-130k is bimanual.
TASK_PROMPTS = {
    'screwdriver': {'object': 'screwdriver', 'containers': ['bin', 'tray']},
    'legos': {'object': 'small toy',
              'containers': ['plastic container', 'red bin', 'tray']},
    'bottle': {'object': 'plastic bottle', 'containers': ['bin', 'tray']},
    'stationery': {'object': 'pen', 'containers': ['plastic container', 'tray']},
    'trash bag': {'object': 'trash bag', 'containers': ['trash bin']},
    'pill': {'object': 'pill', 'containers': ['plastic container', 'tray']},
    'hair-cutting': {'object': 'scissors', 'containers': ['plastic container', 'tray']},
    'utensil': {'object': 'fork', 'containers': ['plastic container', 'tray']},
    'screws and nuts': {'object': 'screw', 'containers': ['plastic container', 'tray']},
    'tools': {'object': 'hand tool', 'containers': ['plastic container', 'tray']},
    'sleeve': {'object': 'folded shirt', 'containers': ['basket', 'tray']},
    'shirt': {'object': 'folded shirt', 'containers': ['basket', 'tray']},
}
DEFAULT_PROMPTS = {'object': 'object', 'containers': ['container', 'tray']}
ARM_PROMPT = 'robot arm'

# A manipulated object is never most of the frame. The GroundingDINO attempt accepted a
# whole-table box because the bound was 0.9; at 0.35 that mask is rejected instead.
MAX_AREA_FRACTION = 0.35
MIN_AREA_PX = 8


def prompts_for(video):
    """Region prompts for a clip, matched against its filename or instruction text."""
    stem = os.path.basename(video).lower()
    for key, spec in TASK_PROMPTS.items():
        if key in stem:
            spec = dict(spec)
            spec['containers'] = list(spec['containers'])
            return spec
    spec = dict(DEFAULT_PROMPTS)
    spec['containers'] = list(spec['containers'])
    return spec


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--video', required=True, help='stacked gt-over-pred rollout mp4')
    p.add_argument('--out_dir', required=True)
    p.add_argument('--mode', default='frame0', choices=('frame0', 'video'))
    p.add_argument('--views', nargs='+', default=['top'], choices=VIEWS,
                   help='top only by default. The wrist views were dropped after gate 1: '
                        'the arm prompt returned nothing on five of them, the object is '
                        'usually out of frame at the start, and the tracking metrics are '
                        'already top-view-only because point drift on a moving camera '
                        'is indistinguishable from real motion.')
    p.add_argument('--object_prompt', help='overrides the per-task default')
    p.add_argument('--container_prompt', help='overrides the per-task default')
    p.add_argument('--arm_prompt', default=ARM_PROMPT)
    p.add_argument('--frame', type=int, default=0, help='frame0 mode only')
    p.add_argument('--threshold', type=float, default=0.2,
                   help='detection score below which an instance is dropped')
    p.add_argument('--max_frames', type=int, default=49, help='video mode only')
    p.add_argument('--model', default=REPO)
    p.add_argument('--device', default='cuda')
    p.add_argument('--zoom', type=int, default=6)
    return p.parse_args()


def area_ok(mask, shape):
    return MIN_AREA_PX <= mask.sum() <= MAX_AREA_FRACTION * shape[0] * shape[1]


class ConceptSegmenter:
    """SAM 3 on one frame: a noun phrase in, the union of its instances out."""

    def __init__(self, model=REPO, device='cuda'):
        import torch
        from transformers import Sam3Model, Sam3Processor
        self.torch = torch
        self.device = device
        # float32: the weights are 3.4 GB and the frames are 192x192, so half precision
        # would buy nothing worth the risk of a silently different mask.
        self.model = Sam3Model.from_pretrained(
            model, dtype=torch.float32).to(device).eval()
        self.processor = Sam3Processor.from_pretrained(model)

    def mask(self, frame, text, threshold=0.2):
        """(H, W) bool union of every instance of `text`, plus the instance scores.

        `text` may be several noun phrases. SAM 3 takes one concept per call, so a region
        that is genuinely several things — the bin and the tray are both containers — is
        the union of one call each.
        """
        from PIL import Image
        frame = np.asarray(frame)
        h, w = frame.shape[:2]
        phrases = [text] if isinstance(text, str) else list(text)
        img = Image.fromarray(frame.astype(np.uint8))
        kept, scores = [], []
        for phrase in phrases:
            inputs = self.processor(images=img, text=phrase,
                                    return_tensors='pt').to(self.device)
            with self.torch.no_grad():
                out = self.model(**inputs)
            res = self.processor.post_process_instance_segmentation(
                out, threshold=threshold, mask_threshold=0.5,
                target_sizes=inputs.get('original_sizes').tolist())[0]
            for m, s in zip(res['masks'], res['scores']):
                m = m.cpu().numpy().astype(bool)
                if area_ok(m, (h, w)):
                    kept.append(m)
                    scores.append(float(s))
        mask = np.any(kept, axis=0) if kept else np.zeros((h, w), dtype=bool)
        return mask, scores


def load_video_model(model_id, device, threshold=0.2):
    """The video model and its processor, built once and reused across views.

    Building one per view leaks the previous model on the GPU and runs a 16 GB card out
    of memory on the third view of the first clip.
    """
    import torch
    from transformers import Sam3VideoConfig, Sam3VideoModel, Sam3VideoProcessor
    # The video half ships far stricter gates than the image half: 0.5 to report a
    # detection and 0.7 to open a new track, against the 0.2 we score images at. Left at
    # the defaults it returned empty masks on every clip where the image model found the
    # object, which reads as "SAM 3 cannot see it" and is really "SAM 3 was not asked".
    # Both are pinned to the same threshold so image and video mode are comparable.
    config = Sam3VideoConfig.from_pretrained(model_id)
    config.score_threshold_detection = threshold
    config.new_det_thresh = max(threshold, 0.3)
    model = Sam3VideoModel.from_pretrained(
        model_id, config=config, dtype=torch.float32).to(device).eval()
    return model, Sam3VideoProcessor.from_pretrained(model_id)


def segment_video(model, processor, frames, prompts, device, max_frames):
    """SAM 3's video half: detect and track across the clip, one pass for all prompts.

    Detection runs on every frame, not only the first. That matters here more than it
    would at 480p: at frame 0 the manipulated object is often not in the wrist views at
    all, so a frame-0-only seed has nothing to find and the region is lost for the whole
    clip.

    Returns:
        dict of region name -> (T, H, W) bool.
    """
    import torch
    frames = np.asarray(frames)[:max_frames]
    session = processor.init_video_session(
        video=list(frames), inference_device=device, processing_device='cpu',
        video_storage_device='cpu')
    texts = list(prompts.values())
    processor.add_text_prompt(session, texts)

    t, h, w = frames.shape[:3]
    out = {r: np.zeros((t, h, w), dtype=bool) for r in prompts}
    text_to_region = {v: k for k, v in prompts.items()}
    with torch.no_grad():
        for model_outputs in model.propagate_in_video_iterator(
                inference_session=session, max_frame_num_to_track=t):
            res = processor.postprocess_outputs(session, model_outputs)
            idx = model_outputs.frame_idx
            by_id = {int(o): m.cpu().numpy().astype(bool)
                     for o, m in zip(res['object_ids'], res['masks'])}
            for text, obj_ids in res['prompt_to_obj_ids'].items():
                region = text_to_region.get(text)
                if region is None:
                    continue
                for oid in obj_ids:
                    m = by_id.get(int(oid))
                    if m is not None and area_ok(m, (h, w)):
                        out[region][idx] |= m
    return out


def main():
    args = parse_args()
    from decord import VideoReader, cpu
    from PIL import Image

    vr = VideoReader(args.video, ctx=cpu(0))
    video = vr.get_batch(list(range(len(vr)))).asnumpy()
    spec = prompts_for(args.video)
    prompts = {'arm': args.arm_prompt,
               'object': args.object_prompt or spec['object'],
               'containers': args.container_prompt or spec['containers']}

    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.video))[0]
    report = {'video': args.video, 'mode': args.mode, 'prompts': prompts,
              'threshold': args.threshold, 'model': args.model, 'views': {}}
    seg = None if args.mode == 'video' else ConceptSegmenter(args.model, args.device)
    video_model = (load_video_model(args.model, args.device, args.threshold)
                   if args.mode == 'video' else (None, None))
    strips = []

    for view in args.views:
        gt, _ = split_panels(video, view)
        bbox = content_bbox(gt)
        if args.mode == 'frame0':
            frame = gt[args.frame]
            raw, scores = {}, {}
            for region, text in prompts.items():
                raw[region], scores[region] = seg.mask(frame, text, args.threshold)
            # Containers come out of the object region, not just out of the background.
            # This is the same confusion that sank the GroundingDINO attempt, arriving by
            # a different route: 'small toy' matches the red bin the legos are sorted
            # into, and a mask that is mostly bin scores the model on rendering a bin.
            masks = disjoint(raw['arm'], raw['object'] & ~raw['containers'], bbox=bbox)
            masks['containers'] = raw['containers'] & ~raw['arm'] & masks['background']
            strips.append(np.concatenate(
                [overlay(frame, {}, upscale=args.zoom),
                 overlay(frame, masks, upscale=args.zoom)], axis=1))
            report['views'][view] = {
                **{f'{r}_px': int(masks[r].sum()) for r in masks},
                'scores': {r: [round(s, 2) for s in v] for r, v in scores.items()}}
            np.savez_compressed(
                os.path.join(args.out_dir, f'{stem}_{view}_sam3.npz'),
                frame=frame.astype(np.uint8), content_bbox=np.asarray(bbox), **masks)
            print(f'{view}: ' + ', '.join(
                f'{r} {masks[r].sum()} px' for r in ('arm', 'object', 'containers')))
        else:
            tracked = segment_video(*video_model, gt, prompts, args.device,
                                    args.max_frames)
            per_frame = [disjoint(tracked['arm'][i],
                                  tracked['object'][i] & ~tracked['containers'][i],
                                  bbox=bbox)
                         for i in range(len(tracked['arm']))]
            masks = {r: np.stack([p[r] for p in per_frame])
                     for r in ('arm', 'object', 'background')}
            masks['containers'] = tracked['containers'] & masks['background']
            report['views'][view] = {
                f'{r}_px_per_frame': [int(x.sum()) for x in masks[r]]
                for r in ('arm', 'object')}
            np.savez_compressed(
                os.path.join(args.out_dir, f'{stem}_{view}_sam3_video.npz'),
                frames=gt[:args.max_frames].astype(np.uint8),
                content_bbox=np.asarray(bbox), **masks)
            first, last = masks['object'][0].sum(), masks['object'][-1].sum()
            print(f'{view}: object {first} px at frame 0, {last} px at frame '
                  f'{len(masks["object"]) - 1}, '
                  f'{sum(1 for m in masks["object"] if not m.any())} frames empty')

    suffix = 'sam3' if args.mode == 'frame0' else 'sam3_video'
    if strips:
        sheet = os.path.join(args.out_dir, f'{stem}_{suffix}.png')
        Image.fromarray(np.concatenate(strips, axis=0)).save(sheet)
        print(f'wrote {sheet}')
    with open(os.path.join(args.out_dir, f'{stem}_{suffix}.json'), 'w') as f:
        json.dump(report, f, indent=1)


if __name__ == '__main__':
    main()
