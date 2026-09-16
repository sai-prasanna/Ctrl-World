"""Does a candidate mask agree with the reference one.

Gate 1 was originally "render the masks at 6x and look at them", and we looked twice and
came to two different verdicts. This replaces the opinion with intersection over union
against a small set of hand-made masks, so a seeder can be scored, re-scored after a
change, and compared against the one it replaced.

The bar is stated in `plans/object-region-metrics.md`: mean object IoU >= 0.5 and mean arm
IoU >= 0.5. That is not demanding. It is the level at which a mask is good enough to pool
DINOv2 features under, which is all the region metrics ask of it.
"""

import numpy as np

REGIONS = ('arm', 'object')
PASS_IOU = 0.5


def iou(a, b):
    """Intersection over union of two boolean (H, W) masks.

    Two empty masks agree perfectly and score 1.0: "there is no arm in this view" is a
    correct answer when the reference says the same. One empty and one not scores 0.0,
    which falls out of the definition and needs no special case.
    """
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    union = (a | b).sum()
    if union == 0:
        return 1.0
    return float((a & b).sum() / union)


def compare(candidate, reference, regions=REGIONS, unscored=()):
    """IoU per region for one frame.

    Args:
        candidate: dict of region name -> (H, W) bool, from a seeder.
        reference: dict of region name -> (H, W) bool, hand-made.
        regions: which regions to score. Background is excluded on purpose: it is the
            complement of the other two inside the content box, so its IoU is near 1
            whatever happens and would only flatter the average.
        unscored: regions the annotator could not identify in this frame. They are
            dropped, not counted as empty. A screwdriver that a human cannot pick out of a
            192x192 frame is not evidence about the segmenter either way, and scoring it
            as an empty reference would hand a free 1.0 to anything that also found
            nothing.

    Returns:
        dict of region name -> IoU, plus `n_ref_px` and `n_cand_px` per region, because an
        IoU of 0 reads differently when the candidate found nothing than when it found the
        wrong thing.
    """
    out = {}
    for r in regions:
        ref = reference.get(r)
        cand = candidate.get(r)
        if ref is None or r in unscored:
            continue
        cand = np.zeros_like(ref, dtype=bool) if cand is None else np.asarray(cand, bool)
        out[r] = iou(cand, ref)
        out[f'{r}_ref_px'] = int(np.asarray(ref, bool).sum())
        out[f'{r}_cand_px'] = int(cand.sum())
    return out


def summarise(rows, regions=REGIONS):
    """Mean IoU per region over many frames, and the gate-1 verdict.

    Args:
        rows: iterable of dicts from `compare`.

    Returns:
        dict with `<region>_iou_mean`, `<region>_iou_min`, `n_frames`, `passed`, and
        `missed`, the count of frames where the candidate produced nothing for a region
        the reference has. A seeder that finds the right thing half the time and nothing
        the other half has a different problem from one that is consistently sloppy, and
        the mean alone does not separate them. `<region>_n` is how many frames the mean is
        over, which is not `n_frames` once unscored regions are dropped.
    """
    rows = list(rows)
    out = {'n_frames': len(rows)}
    if not rows:
        out['passed'] = False
        return out
    for r in regions:
        vals = np.array([row[r] for row in rows if r in row], dtype=float)
        if not len(vals):
            continue
        out[f'{r}_n'] = int(len(vals))
        out[f'{r}_iou_mean'] = float(vals.mean())
        out[f'{r}_iou_min'] = float(vals.min())
        out[f'{r}_missed'] = int(sum(1 for row in rows
                                     if row.get(f'{r}_cand_px', 0) == 0
                                     and row.get(f'{r}_ref_px', 0) > 0))
    out['passed'] = all(out.get(f'{r}_iou_mean', 0.0) >= PASS_IOU for r in regions)
    return out


def load_masks(path, regions=None):
    """Read a mask npz written by any seeder or by the annotator.

    Returns:
        dict of region name -> (H, W) or (T, H, W) bool. Keys that are not masks, such as
        `frame` and `content_bbox`, are left out.
    """
    with np.load(path) as z:
        names = regions if regions is not None else [
            k for k in z.files if z[k].dtype == bool]
        return {k: z[k].astype(bool) for k in names if k in z.files}


def sentence(s):
    """The gate-1 verdict as one line."""
    parts = []
    for r in REGIONS:
        if f'{r}_iou_mean' not in s:
            continue
        miss = s.get(f'{r}_missed', 0)
        parts.append(f'{r} IoU {s[f"{r}_iou_mean"]:.2f} over {s[f"{r}_n"]} frames '
                     f'(worst {s[f"{r}_iou_min"]:.2f}'
                     + (f', {miss} empty' if miss else '') + ')')
    verdict = 'passes' if s.get('passed') else 'fails'
    return f'{"; ".join(parts)}: {verdict} the {PASS_IOU} bar'
