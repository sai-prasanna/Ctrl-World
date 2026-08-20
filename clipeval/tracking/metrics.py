"""Trajectory metrics.

The Hausdorff, dynamic-time-warping and dynamics terms are adapted from EWMBench,
`EWMBench/trajectory_consistency.py` at commit 3a5531c0e6b6be51431251ed5617b9e48c852098
(https://github.com/AgibotTech/EWMBench, arXiv 2505.09694). Only the metric math is
taken; the surrounding harness, which wants Qwen2.5-VL-7B, two CLIP variants and
fine-tuned DINOv2 and YOLO-World checkpoints, is not.

Two deliberate departures from the original:

  * EWMBench reports 1/distance, rescaled by constants (hsd_max=24.979, dyn_max=22.519,
    ndtw_max=50.202) fitted to its own data, so a score of 1.0 means "at least as good as
    the best model AgibotTech measured". Those constants say nothing about ABC-130k, so
    the functions here return the raw distances in pixels, which are interpretable on
    their own. `ewmbench_scores` applies the original transform for anyone who wants a
    number comparable to the published table.
  * DTW is computed exactly rather than with `fastdtw`. Our trajectories are ~50 frames,
    where the exact O(T^2) recursion costs nothing, and fastdtw is an approximation with
    its own radius parameter. Normalization matches the original: total warped distance
    divided by the number of pairs on the alignment path.

Trajectories are (T, K, 2) arrays of pixel coordinates for K points over T frames.
Missing points are marked (-1, -1), as in EWMBench.
"""

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.distance import directed_hausdorff
from scipy.stats import wasserstein_distance

MISSING = -1.0


# ------------------------------------------------------------------ gap filling


def _fill_one(data):
    """Linearly interpolate over missing entries of a (T, 2) track.

    Returns (filled, invalid). A track is invalid, and left alone, when more than 80% of
    its frames are missing: interpolating across that much of a trajectory invents the
    trajectory rather than repairing it.
    """
    mask = (data != np.array([MISSING, MISSING])).any(axis=1)
    if 1 - mask.mean() > 0.80:
        return data, True
    n = data.shape[0]
    prev = np.full(n, -1, dtype=int)
    nxt = np.full(n, n, dtype=int)
    last = -1
    for i in range(n):
        if mask[i]:
            last = i
        prev[i] = last
    last = n
    for i in range(n - 1, -1, -1):
        if mask[i]:
            last = i
        nxt[i] = last

    missing = np.where(~mask)[0]
    if len(missing) == 0:
        return data, False
    p_vals, q_vals = prev[missing], nxt[missing]
    no_prev = p_vals == -1
    no_next = q_vals == n
    both = ~no_prev & ~no_next
    if np.any(no_prev):
        data[missing[no_prev]] = data[q_vals[no_prev]]
    if np.any(no_next):
        data[missing[no_next]] = data[p_vals[no_next]]
    if np.any(both):
        idx, p, q = missing[both], p_vals[both], q_vals[both]
        alpha = ((idx - p) / (q - p).astype(float))[:, None]
        data[idx] = (1 - alpha) * data[p] + alpha * data[q]
    return data, False


def fill_gaps(traj):
    """traj: (T, K, 2). Returns (filled, invalid_mask) with invalid_mask of length K."""
    filled, invalid = [], []
    for k in range(traj.shape[1]):
        one, bad = _fill_one(traj[:, k].copy().astype(np.float64))
        filled.append(one)
        invalid.append(bad)
    return np.stack(filled, axis=1), np.array(invalid)


# ------------------------------------------------------------- point selection


def farthest_distance(points):
    """Diameter of a (T, 2) point set, via rotating calipers on its convex hull."""
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 2:
        return 0.0
    uniq = np.unique(points, axis=0)
    if len(uniq) < 3:
        # ConvexHull needs a 2-simplex; a degenerate track's diameter is a plain max.
        return float(max(np.linalg.norm(a - b) for a in uniq for b in uniq))
    try:
        hull = ConvexHull(uniq)
    except Exception:                       # collinear points
        return float(np.max(np.linalg.norm(uniq[:, None] - uniq[None], axis=-1)))
    verts = uniq[hull.vertices]
    n = len(verts)
    pts = np.vstack([verts, verts[0]])
    max_dist, k = 0.0, 1
    for i in range(n):
        j = (i + 1) % n
        while True:
            next_k = (k + 1) % n
            cross = ((pts[j, 0] - pts[i, 0]) * (pts[next_k, 1] - pts[k, 1])
                     - (pts[j, 1] - pts[i, 1]) * (pts[next_k, 0] - pts[k, 0]))
            if cross < 0:
                k = next_k
            else:
                break
        max_dist = max(max_dist, float(np.linalg.norm(pts[i] - pts[k])),
                       float(np.linalg.norm(pts[i] - pts[next_k])))
    return max_dist


def most_dynamic_index(traj_gt, invalid_gt):
    """Index of the ground-truth track that covers the most ground.

    EWMBench scores one trajectory per clip, the one whose ground truth moves farthest,
    on the grounds that a metric averaged over near-stationary points measures nothing.
    """
    best_idx, best = None, -1.0
    for k in range(traj_gt.shape[1]):
        if invalid_gt[k]:
            continue
        d = farthest_distance(traj_gt[:, k])
        if d > best:
            best_idx, best = k, d
    if best_idx is None:
        raise ValueError('every ground-truth track is invalid')
    return best_idx


# -------------------------------------------------------------------- distances


def hausdorff(a, b):
    """Symmetric Hausdorff distance in pixels between two (T, 2) tracks."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(max(directed_hausdorff(a, b)[0], directed_hausdorff(b, a)[0]))


def dtw(a, b):
    """Exact DTW between two (T, 2) tracks, normalized by alignment path length."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n, m = len(a), len(b)
    cost = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
    acc = np.full((n + 1, m + 1), np.inf)
    acc[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            acc[i, j] = cost[i - 1, j - 1] + min(acc[i - 1, j], acc[i, j - 1],
                                                 acc[i - 1, j - 1])
    # Walk the path back to get its length; the original divides by the same count.
    i, j, steps = n, m, 0
    while i > 0 and j > 0:
        steps += 1
        options = (acc[i - 1, j - 1], acc[i - 1, j], acc[i, j - 1])
        pick = int(np.argmin(options))
        if pick == 0:
            i, j = i - 1, j - 1
        elif pick == 1:
            i -= 1
        else:
            j -= 1
    steps += i + j
    return float(acc[n, m] / max(steps, 1))


def dynamics(pred, gt):
    """Velocity and acceleration agreement for two (T, 2) tracks.

    Returns the Wasserstein distance between the speed distributions and between the
    acceleration distributions, plus EWMBench's range ratios `vr` and `ar`, which
    penalize a prediction whose dynamic range differs from ground truth even when the
    distributions happen to overlap.
    """
    eps = 1e-8
    vel_p = np.linalg.norm(np.diff(pred, axis=0), axis=-1)
    vel_g = np.linalg.norm(np.diff(gt, axis=0), axis=-1)
    acc_p, acc_g = np.diff(vel_p), np.diff(vel_g)

    def ratio(p, g):
        rp, rg = float(p.max() - p.min()), float(g.max() - g.min())
        return (min(rg, rp) + eps) / (max(rg, rp) + eps)

    return {
        'vel_wasserstein': float(wasserstein_distance(vel_p, vel_g)),
        'acc_wasserstein': float(wasserstein_distance(acc_p, acc_g)),
        'vel_range_ratio': ratio(vel_p, vel_g),
        'acc_range_ratio': ratio(acc_p, acc_g),
    }


def trajectory_metrics(traj_pred, traj_gt):
    """Score a (T, K, 2) prediction against a (T, K, 2) ground truth.

    Follows EWMBench: fill gaps, pick the ground truth's most dynamic track, and score
    that one track. Returns raw pixel distances; see `ewmbench_scores` for the published
    normalized form.
    """
    traj_pred, invalid_pred = fill_gaps(np.asarray(traj_pred, dtype=np.float64))
    traj_gt, invalid_gt = fill_gaps(np.asarray(traj_gt, dtype=np.float64))
    k = most_dynamic_index(traj_gt, invalid_gt)
    out = {'track_index': int(k),
           'gt_extent_px': farthest_distance(traj_gt[:, k]),
           'pred_track_lost': bool(invalid_pred[k])}
    if invalid_pred[k]:
        # The prediction lost the point for most of the clip. There is no trajectory to
        # score, so report it as lost rather than as a large-but-finite distance.
        out.update({'hsd_px': float('nan'), 'ndtw_px': float('nan')})
        out.update({key: float('nan') for key in
                    ('vel_wasserstein', 'acc_wasserstein', 'vel_range_ratio',
                     'acc_range_ratio')})
        return out
    out['hsd_px'] = hausdorff(traj_pred[:, k], traj_gt[:, k])
    out['ndtw_px'] = dtw(traj_pred[:, k], traj_gt[:, k])
    out.update(dynamics(traj_pred[:, k], traj_gt[:, k]))
    return out


# EWMBench's per-metric maxima, fitted to its own benchmark data. Reproduced so a number
# can be placed on the published scale; they carry no meaning for ABC-130k on their own.
EWMBENCH_MAX = {'hsd': 24.979, 'dyn': 22.519, 'ndtw': 50.202}


def ewmbench_scores(metrics):
    """Convert `trajectory_metrics` output into EWMBench's 0-1 scores."""
    def inv(x):
        return 0.0 if not np.isfinite(x) or x == 0 else 1.0 / x

    dyn = (0.007 * metrics['vel_range_ratio'] * inv(metrics['vel_wasserstein'])
           + 0.003 * metrics['acc_range_ratio'] * inv(metrics['acc_wasserstein']))
    raw = {'hsd': inv(metrics['hsd_px']), 'ndtw': inv(metrics['ndtw_px']), 'dyn': dyn}
    return {k: float(min(v / EWMBENCH_MAX[k], 1.0)) for k, v in raw.items()}


# ------------------------------------------------- point-set error (not EWMBench)


THRESHOLDS_PX = (2, 4, 8, 16)


def endpoint_error(traj_pred, traj_gt):
    """Per-frame, per-point L2 distance in pixels. (T, K, 2) x2 -> (T, K)."""
    return np.linalg.norm(np.asarray(traj_pred, dtype=np.float64)
                          - np.asarray(traj_gt, dtype=np.float64), axis=-1)


def error_summary(err, thresholds=THRESHOLDS_PX):
    """Summarize a (T, K) error array.

    The headline is a median, not a mean: CoTracker3's error is unbounded when a track is
    lost, so one lost point would otherwise decide the clip.
    """
    err = np.asarray(err, dtype=np.float64)
    finite = err[np.isfinite(err)]
    if finite.size == 0:
        return {'median_px': float('nan'), 'mean_px': float('nan'),
                **{f'acc_{t}px': float('nan') for t in thresholds}}
    out = {'median_px': float(np.median(finite)), 'mean_px': float(finite.mean())}
    for t in thresholds:
        out[f'acc_{t}px'] = float((finite < t).mean())
    return out
