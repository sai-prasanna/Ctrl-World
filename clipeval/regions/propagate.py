"""Carrying a frame-0 mask through a clip.

Gate 2 of the plan: a mask on frame 0 has to follow the object for 48 frames. Two ways,
and the plan's fallback is the one that always works.

  * `redetect` runs the text-prompted segmenter on every frame, keeping the detection
    nearest the previous box. More expensive, and has no drift to accumulate because
    nothing is carried forward except the association.
  * `sam2_propagate` uses SAM2's video predictor, which is cheaper and tracks through
    occlusion, but is a package outside transformers 4.48.1. Absent, it raises, and the
    caller falls back.

Both run on the ground truth only. The mask is then applied to the prediction unchanged
(see `clipeval.regions`): propagating on the prediction would let the propagator paint a
mask onto empty space wherever the model erased the object, which is the failure this
whole layer exists to catch.
"""

import numpy as np


def box_of(mask):
    """(H, W) bool -> (x0, y0, x1, y1), or None for an empty mask."""
    ys, xs = np.nonzero(np.asarray(mask))
    if len(ys) == 0:
        return None
    return float(xs.min()), float(ys.min()), float(xs.max()) + 1, float(ys.max()) + 1


def _centre(box):
    return np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])


def redetect(segmenter, frames, phrases, prev_box=None, box_threshold=0.25,
             max_jump_px=48.0, top_k=3):
    """Per-frame detection of one phrase, chained by nearest box.

    At 5 fps an object moves tens of pixels between frames, so "nearest to the last box"
    is a strong enough association; `max_jump_px` refuses a detection that teleports,
    which is the failure mode where the prompt latches onto a second, similar object.

    Args:
        segmenter: a `RegionSegmenter`.
        frames: (T, H, W, 3) uint8 ground-truth frames.
        phrases: text prompt for the region.
        prev_box: box to chain the first frame to; None means take the best detection.
        max_jump_px: reject a detection whose centre moved further than this in a frame.

    Returns:
        dict with `masks` (T, H, W) bool, `boxes` list of box-or-None, and `misses`, the
        frames where nothing was accepted. A missed frame keeps the previous mask, so the
        region stays scoreable; `misses` is what says how much to trust it.
    """
    frames = np.asarray(frames)
    t, h, w = frames.shape[0], frames.shape[1], frames.shape[2]
    masks = np.zeros((t, h, w), dtype=bool)
    boxes, misses = [], []
    last = prev_box
    for i in range(t):
        dets = segmenter.detect(frames[i], phrases, box_threshold=box_threshold)[:top_k]
        pick = None
        if dets and last is not None:
            near = min(dets, key=lambda d: np.linalg.norm(_centre(d['box'])
                                                          - _centre(last)))
            if np.linalg.norm(_centre(near['box']) - _centre(last)) <= max_jump_px:
                pick = near
        elif dets:
            pick = dets[0]
        if pick is None:
            misses.append(i)
            masks[i] = masks[i - 1] if i else False
            boxes.append(last)
            continue
        m = segmenter.segment(frames[i], [pick['box']])
        masks[i] = m[0]
        last = box_of(m[0]) or pick['box']
        boxes.append(last)
    return {'masks': masks, 'boxes': boxes, 'misses': misses}


def sam2_propagate(frames, mask0, device='cuda',
                   checkpoint='facebook/sam2.1-hiera-large'):
    """SAM2 video propagation of a frame-0 mask. Raises if `sam2` is not installed."""
    try:
        from sam2.sam2_video_predictor import SAM2VideoPredictor
    except ImportError as exc:                   # gate 2's stated fallback path
        raise ImportError(
            'sam2 is not installed; use redetect() instead') from exc
    import torch
    frames = np.asarray(frames)
    predictor = SAM2VideoPredictor.from_pretrained(checkpoint, device=device)
    with torch.inference_mode():
        state = predictor.init_state(video_path=frames)
        predictor.add_new_mask(state, frame_idx=0, obj_id=1,
                               mask=torch.from_numpy(np.asarray(mask0)))
        out = np.zeros(frames.shape[:3], dtype=bool)
        for idx, _, logits in predictor.propagate_in_video(state):
            out[idx] = (logits[0] > 0).cpu().numpy()[0]
    return {'masks': out, 'boxes': [box_of(m) for m in out], 'misses': []}


def propagate(segmenter, frames, phrases, mask0=None, prefer_sam2=False, **kw):
    """SAM2 if asked for and available, per-frame redetection otherwise."""
    if prefer_sam2 and mask0 is not None:
        try:
            return sam2_propagate(frames, mask0)
        except ImportError:
            pass
    return redetect(segmenter, frames, phrases,
                    prev_box=box_of(mask0) if mask0 is not None else None, **kw)
