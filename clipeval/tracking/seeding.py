"""Choosing which points to track.

Three ways to seed, in increasing order of what the caller has to supply:

  * `grid_queries` needs nothing but the frame size. Combined with `split_static_dynamic`
    it separates background from moving content after the fact, using ground-truth
    displacement, so a caller with only frames still gets a static-drift number and a
    dynamic-error number.
  * `mask_queries` needs a mask on the conditioning frame, which is how the manipulated
    object gets its own bucket.
  * `point_queries` needs explicit pixels, which is how gripper positions enter, either
    projected from recorded actions through a camera calibration or from a detector.

All query arrays are (K, 3) of (frame_index, x, y), CoTracker's convention.
"""

import numpy as np

# A ground-truth point that travels less than this over the clip counts as background.
# Set above the measured tracker noise floor: on the top view of 10 ABC-130k validation
# clips, with the letterbox excluded, CoTracker3's drift on true background had a median
# of 0.48 px and a p99 of 1.89 px (outputs/0002_abc_rigid/eval/tracker_noise_floor.json).
# 4 px clears that comfortably; much below 2 px the split would start sorting the
# tracker's own jitter into the dynamic bucket. On the wrist views the same drift has a
# median of 31.68 px, indistinguishable from real motion, which is why the tracking
# metrics are top-view-only.
STATIC_MAX_DISPLACEMENT_PX = 4.0


def content_bbox(frames, dark=20, min_fraction=0.8):
    """Bounding box of the real image inside a letterboxed clip.

    ABC-130k frames are 192x192 with a 24 px black bar top and bottom, a quarter of the
    frame. Points seeded there track perfectly and never move, so they silently pad the
    static bucket and drag any background-drift measurement toward zero. Every seeding
    function takes the box so this cannot happen by accident.

    Args:
        frames: (T, H, W, 3) or (H, W, 3).
        dark: luminance at or below which a pixel counts as black.
        min_fraction: a row or column is letterbox when this fraction of it is black.

    Returns:
        (y0, y1, x0, x1), half-open, in pixels.
    """
    a = np.asarray(frames)
    if a.ndim == 4:
        a = a[0]
    lum = a.astype(np.float64).mean(axis=-1)
    rows = (lum <= dark).mean(axis=1) < min_fraction
    cols = (lum <= dark).mean(axis=0) < min_fraction
    if not rows.any() or not cols.any():        # entirely dark frame; use the whole thing
        return 0, lum.shape[0], 0, lum.shape[1]
    ys, xs = np.where(rows)[0], np.where(cols)[0]
    return int(ys[0]), int(ys[-1]) + 1, int(xs[0]), int(xs[-1]) + 1


def grid_queries(height, width, grid=16, margin=8, frame=0, bbox=None):
    """A `grid` x `grid` lattice of query points, inset by `margin` pixels.

    The margin keeps points off the frame border, where a tracker has no context on one
    side and where our decoder's artifacts concentrate.

    Pass `bbox` from `content_bbox` on a letterboxed clip; without it a quarter of the
    points on ABC-130k land on the black bars.
    """
    y0, y1, x0, x1 = bbox if bbox is not None else (0, height, 0, width)
    ys = np.linspace(y0 + margin, y1 - 1 - margin, grid)
    xs = np.linspace(x0 + margin, x1 - 1 - margin, grid)
    gy, gx = np.meshgrid(ys, xs, indexing='ij')
    q = np.stack([np.full(gy.size, frame), gx.ravel(), gy.ravel()], axis=1)
    return q.astype(np.float64)


def mask_queries(mask, n_points=64, frame=0, seed=0, bbox=None):
    """Sample `n_points` query points uniformly inside a boolean (H, W) mask."""
    mask = np.asarray(mask).astype(bool)
    if bbox is not None:
        keep = np.zeros_like(mask)
        y0, y1, x0, x1 = bbox
        keep[y0:y1, x0:x1] = True
        mask = mask & keep
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        raise ValueError('empty mask: nothing to seed')
    rng = np.random.default_rng(seed)
    take = rng.choice(len(ys), size=min(n_points, len(ys)), replace=False)
    return np.stack([np.full(len(take), frame), xs[take], ys[take]],
                    axis=1).astype(np.float64)


def point_queries(points, frame=0, jitter=0.0, n_per_point=1, seed=0):
    """Query points at given (x, y) pixels, optionally with a small jittered cloud.

    A single pixel on a textureless gripper is exactly the case CoTracker3 is documented
    to lose, so seeding a few points per gripper and taking a median across them is more
    robust than trusting one.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if n_per_point > 1 and jitter > 0:
        rng = np.random.default_rng(seed)
        offsets = rng.normal(0, jitter, size=(len(points), n_per_point, 2))
        offsets[:, 0] = 0.0                      # keep the nominal point itself
        points = (points[:, None, :] + offsets).reshape(-1, 2)
    return np.concatenate([np.full((len(points), 1), frame), points],
                          axis=1).astype(np.float64)


def split_static_dynamic(gt_tracks, threshold=STATIC_MAX_DISPLACEMENT_PX):
    """Split tracked points by how far the ground truth moves them.

    Args:
        gt_tracks: (T, K, 2) ground-truth tracks.
        threshold: displacement in pixels below which a point is background.

    Returns:
        dict with boolean (K,) masks `static` and `dynamic`, and the per-point
        displacement used to split them.
    """
    tracks = np.asarray(gt_tracks, dtype=np.float64)
    start = tracks[0]
    displacement = np.linalg.norm(tracks - start[None], axis=-1).max(axis=0)
    static = displacement < threshold
    return {'static': static, 'dynamic': ~static, 'displacement_px': displacement,
            'threshold_px': float(threshold)}
