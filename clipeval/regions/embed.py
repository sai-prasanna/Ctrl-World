"""Region consistency: DINOv2 patch features under a mask, real versus generated.

Cosine similarity between two whole-frame embeddings is dominated by the background,
which both videos get right, so it is close to 1 whatever happens to the object. Pooling
patch features under a mask first makes the number local: the object region is scored on
the object region alone, and a model that painted a red bin where the legos were is
compared against legos.

Two numbers per frame, because they fail differently:

  * `pooled`, cosine between the mask-averaged feature vectors. Semantic, and blind to
    where inside the region things sit.
  * `patch`, the mean over masked patches of the per-patch cosine. Spatially resolved,
    and the one that punishes an object that moved within its own mask.

DINOv2 patches are 14 px. A 192x192 frame upscaled 4x gives 54x54 patches, so a lego
brick of a few dozen pixels still covers several of them; at native resolution it would
cover one, and `patch` would be a single number dressed up as a map.
"""

import numpy as np
import torch

EMBEDDER = 'facebook/dinov2-base'
# Below this many masked patches the mean is over too few samples to read as a score.
MIN_PATCHES = 4


class DinoEmbedder:
    """DINOv2 patch features for a clip, on a grid, with masks pooled to that grid."""

    def __init__(self, device='cuda', model=EMBEDDER, upscale=4, batch=16,
                 dtype=torch.float16):
        from transformers import AutoImageProcessor, AutoModel
        self.device = device
        self.upscale = upscale
        self.batch = batch
        self.dtype = dtype if str(device).startswith('cuda') else torch.float32
        self.processor = AutoImageProcessor.from_pretrained(model)
        self.model = AutoModel.from_pretrained(
            model, torch_dtype=self.dtype).to(device).eval()
        self.patch = self.model.config.patch_size

    @torch.no_grad()
    def patch_features(self, frames):
        """(T, H, W, 3) uint8 -> L2-normalised (T, gh, gw, C) float32 patch features."""
        frames = np.asarray(frames)
        feats = []
        for i in range(0, len(frames), self.batch):
            chunk = [f for f in frames[i:i + self.batch]]
            if self.upscale != 1:
                chunk = [np.repeat(np.repeat(f, self.upscale, 0), self.upscale, 1)
                         for f in chunk]
            inputs = self.processor(images=chunk, return_tensors='pt').to(self.device)
            inputs['pixel_values'] = inputs['pixel_values'].to(self.dtype)
            out = self.model(**inputs).last_hidden_state[:, 1:]     # drop CLS
            gh = inputs['pixel_values'].shape[-2] // self.patch
            gw = inputs['pixel_values'].shape[-1] // self.patch
            feats.append(out.float().reshape(len(chunk), gh, gw, -1).cpu())
        f = torch.cat(feats)
        return torch.nn.functional.normalize(f, dim=-1).numpy()

    def grid_shape(self, frames):
        f = self.patch_features(np.asarray(frames)[:1])
        return f.shape[1], f.shape[2]


def pool_mask(mask, grid):
    """(H, W) bool -> (gh, gw) float fraction of each patch the mask covers.

    Averaging rather than nearest keeps a mask a few pixels wide from vanishing, and the
    fractions double as the weights the pooled feature is averaged with.
    """
    m = torch.from_numpy(np.asarray(mask).astype(np.float32))[None, None]
    return torch.nn.functional.adaptive_avg_pool2d(m, grid)[0, 0].numpy()


def region_similarity(gt_feats, pred_feats, masks, min_coverage=0.25):
    """Per-frame cosine similarity between two clips inside a per-frame mask.

    Args:
        gt_feats, pred_feats: (T, gh, gw, C) normalised patch features.
        masks: (T, H, W) bool, from the *ground truth*, applied to both clips.
        min_coverage: a patch counts as inside the region when the mask covers at least
            this fraction of it.

    Returns:
        dict with (T,) arrays `pooled`, `patch` and `n_patches`. Frames whose mask covers
        fewer than `MIN_PATCHES` patches are nan in both scores rather than 0: the object
        left the view, which is not the same as the model getting it wrong.
    """
    gt_feats, pred_feats = np.asarray(gt_feats), np.asarray(pred_feats)
    t, gh, gw, _ = gt_feats.shape
    pooled, patchwise, counts = [], [], []
    for i in range(t):
        w = pool_mask(masks[i], (gh, gw))
        sel = w >= min_coverage
        counts.append(int(sel.sum()))
        if sel.sum() < MIN_PATCHES:
            pooled.append(np.nan)
            patchwise.append(np.nan)
            continue
        wsel = w[sel][:, None]
        g = (gt_feats[i][sel] * wsel).sum(0)
        p = (pred_feats[i][sel] * wsel).sum(0)
        pooled.append(float(g @ p / max(np.linalg.norm(g) * np.linalg.norm(p), 1e-8)))
        patchwise.append(float((gt_feats[i][sel] * pred_feats[i][sel]).sum(-1).mean()))
    return {'pooled': np.asarray(pooled), 'patch': np.asarray(patchwise),
            'n_patches': np.asarray(counts)}


def summarise(sim):
    """Region similarity over a clip as three readable numbers.

    The mean is the headline, the minimum is the worst frame, and `frames_scored` says
    how much of the clip the region was actually visible for.
    """
    keep = np.isfinite(sim['pooled'])
    if not keep.any():
        return {'pooled_mean': float('nan'), 'patch_mean': float('nan'),
                'patch_min': float('nan'), 'frames_scored': 0}
    return {'pooled_mean': float(np.mean(sim['pooled'][keep])),
            'patch_mean': float(np.mean(sim['patch'][keep])),
            'patch_min': float(np.min(sim['patch'][keep])),
            'frames_scored': int(keep.sum())}
