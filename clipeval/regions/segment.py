"""Frame bookkeeping for the region layer: panels, colours, and overlays.

This module used to hold a GroundingDINO detector bolted to a SAM segmenter. That design
failed gate 1 — the detector decides *what* from a text phrase and SAM only decides
*where*, so a bad phrase match is unrecoverable, and on 192x192 frames the phrase matched
the table. SAM 3 does both jobs in one model and replaced it; see
`scripts/segment_sam3.py`.

What is left here is what had nothing to do with the segmenter: splitting a stacked
rollout into panels, and painting masks over a frame so a human can check them. It is
numpy only, so `.venv-seed` can import it without dragging in the main venv's stack.
"""

import numpy as np

VIEWS = ('top', 'left_wrist', 'right_wrist')

REGION_COLORS = {'arm': (60, 130, 255), 'object': (255, 70, 70),
                 'background': (70, 220, 120), 'containers': (255, 210, 60)}


def split_panels(video, view):
    """(T, 2H, 3W, 3) -> the ground-truth and predicted halves of one view.

    `rollout_replay_traj.py` saves the real video on the top row and the generated one
    below, three views across.
    """
    _, h2, w3, _ = np.asarray(video).shape
    h, w = h2 // 2, w3 // len(VIEWS)
    c = VIEWS.index(view)
    return video[:, :h, c * w:(c + 1) * w], video[:, h:, c * w:(c + 1) * w]


def inside_bbox(shape, bbox):
    """(H, W) bool that is True inside `bbox`, the content box of a letterboxed clip."""
    out = np.zeros(shape[:2], dtype=bool)
    y0, y1, x0, x1 = bbox if bbox is not None else (0, shape[0], 0, shape[1])
    out[y0:y1, x0:x1] = True
    return out


def disjoint(arm, obj, bbox=None, shape=None):
    """Arm, object and background as three disjoint masks covering the content box.

    The arm wins where it overlaps the object: during a grasp the gripper occludes the
    thing it is holding, so the visible pixels there are gripper.
    """
    arm = np.asarray(arm, dtype=bool)
    obj = np.asarray(obj, dtype=bool) & ~arm
    inside = inside_bbox(shape if shape is not None else arm.shape, bbox)
    return {'arm': arm & inside, 'object': obj & inside,
            'background': inside & ~arm & ~obj}


def overlay(frame, masks, colors=None, alpha=0.55, upscale=6):
    """Masks painted over a frame, upscaled, for looking at.

    Args:
        frame: (H, W, 3) uint8.
        masks: dict of name -> (H, W) bool.
        colors: name -> (r, g, b); defaults cover arm, object, background, containers.
        upscale: nearest-neighbour zoom, because the point is to see which pixels.

    Returns:
        (H*upscale, W*upscale, 3) uint8.
    """
    colors = {**REGION_COLORS, **(colors or {})}
    out = np.asarray(frame).astype(np.float32).copy()
    for name, mask in masks.items():
        mask = np.asarray(mask)
        if mask.dtype != bool or not mask.any():
            continue
        c = np.array(colors.get(name, (255, 255, 0)), dtype=np.float32)
        out[mask] = (1 - alpha) * out[mask] + alpha * c
    out = out.clip(0, 255).astype(np.uint8)
    return np.repeat(np.repeat(out, upscale, axis=0), upscale, axis=1)
