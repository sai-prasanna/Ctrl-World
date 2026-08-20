"""Corpus-level distribution metrics: FID and FVD.

Both need features accumulated over every clip before a single number exists, so this is
an accumulator rather than a pure function. Moved unchanged out of
scripts/eval_video_metrics.py.

The numbers are comparable across checkpoints scored with the same backbone and nothing
else: FID and FVD move by large amounts with the feature extractor and the resize path,
and published tables rarely pin either. FeatureBank.backbones records what was used.
"""

from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F


class FeatureBank:
    """Accumulates Inception / I3D features so FID and FVD can be computed at the end."""

    def __init__(self, device, dtype=None, fid=False, i3d_ckpt=None):
        self.device = device
        self.dtype = dtype
        self.inception = None
        self.i3d = None
        self.backbones = {}
        if fid:
            from torchvision.models import Inception_V3_Weights, inception_v3
            net = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
            net.fc = torch.nn.Identity()
            self.inception = net.eval().to(device)
            self.backbones['fid'] = 'torchvision inception_v3 IMAGENET1K_V1 (pool 2048)'
        if i3d_ckpt:
            self.i3d = torch.jit.load(i3d_ckpt).eval().to(device)
            self.backbones['fvd'] = f'torchscript I3D from {i3d_ckpt}'
        self.fid_feats = defaultdict(list)
        self.fvd_feats = defaultdict(list)

    @property
    def enabled(self):
        return self.inception is not None or self.i3d is not None

    @torch.no_grad()
    def add_frames(self, key, frames):
        """frames: (N, H, W, 3) float32 in [0, 255]."""
        if self.inception is None:
            return
        x = frames.permute(0, 3, 1, 2) / 255.0
        x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device)[None, :, None, None]
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device)[None, :, None, None]
        x = (x - mean) / std
        for i in range(0, x.shape[0], 64):
            self.fid_feats[key].append(self.inception(x[i:i + 64]).float().cpu())

    @torch.no_grad()
    def add_videos(self, key, videos):
        """videos: (B, T, H, W, 3) float32 in [0, 255]."""
        if self.i3d is None:
            return
        x = videos.permute(0, 4, 1, 2, 3) / 127.5 - 1.0  # (B, C, T, H, W) in [-1, 1]
        x = x.contiguous()
        feats = self.i3d(x, rescale=False, resize=True, return_features=True)
        self.fvd_feats[key].append(feats.float().cpu())

    @staticmethod
    def _frechet(a, b):
        from scipy import linalg
        a, b = a.numpy().astype(np.float64), b.numpy().astype(np.float64)
        mu_a, mu_b = a.mean(0), b.mean(0)
        cov_a = np.cov(a, rowvar=False)
        cov_b = np.cov(b, rowvar=False)
        covmean, _ = linalg.sqrtm(cov_a @ cov_b, disp=False)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        diff = mu_a - mu_b
        return float(diff @ diff + np.trace(cov_a) + np.trace(cov_b) - 2 * np.trace(covmean))

    def results(self, view_names):
        out = {}
        for view in view_names:
            if self.inception is not None:
                pred = torch.cat(self.fid_feats[f'{view}/pred'])
                real = torch.cat(self.fid_feats[f'{view}/real'])
                out.setdefault('fid', {})[view] = self._frechet(pred, real)
            if self.i3d is not None:
                pred = torch.cat(self.fvd_feats[f'{view}/pred'])
                real = torch.cat(self.fvd_feats[f'{view}/real'])
                out.setdefault('fvd', {})[view] = self._frechet(pred, real)
        return out
