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
# Chosen so that camera noise and VAE-scale jitter stay on the static side; revisit once
# the tracker noise floor on ABC-130k is measured.
STATIC_MAX_DISPLACEMENT_PX = 4.0


def grid_queries(height, width, grid=16, margin=8, frame=0):
    """A `grid` x `grid` lattice of query points, inset by `margin` pixels.

    The margin keeps points off the frame border, where a tracker has no context on one
    side and where our decoder's artifacts concentrate.
    """
    ys = np.linspace(margin, height - 1 - margin, grid)
    xs = np.linspace(margin, width - 1 - margin, grid)
    gy, gx = np.meshgrid(ys, xs, indexing='ij')
    q = np.stack([np.full(gy.size, frame), gx.ravel(), gy.ravel()], axis=1)
    return q.astype(np.float64)


def mask_queries(mask, n_points=64, frame=0, seed=0):
    """Sample `n_points` query points uniformly inside a boolean (H, W) mask."""
    ys, xs = np.nonzero(np.asarray(mask).astype(bool))
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
