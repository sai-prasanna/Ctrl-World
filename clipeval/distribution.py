"""Corpus-level distribution metrics: FID and FVD.

Both need features accumulated over every clip before a single number exists, so this is
an accumulator rather than a pure function. Moved unchanged out of
scripts/eval_video_metrics.py.

The numbers are comparable across checkpoints scored with the same backbone and nothing
else: FID and FVD move by large amounts with the feature extractor and the resize path,
and published tables rarely pin either. FeatureBank.backbones records what was used.

Both are corpus-level, so there is no per-clip value to average and the per-clip
bootstrap the other metric families use does not apply. Instead `results(n_boot=...)`
resamples *episodes* with replacement and recomputes the Frechet distance over the
resampled corpus, pairing the predicted and real sets on the same draw. That gives a CI
for the sampling variability of the corpus the metric was computed over. It does not
remove the small-sample bias of the estimator itself, which is why the reported point
value stays the full-sample one rather than the bootstrap mean: two checkpoints scored
over the same clips share that bias, so their difference is readable, but a raw
magnitude is not.
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
        # One episode id per accumulated feature *row*, parallel to the tensors above.
        self.fid_episodes = defaultdict(list)
        self.fvd_episodes = defaultdict(list)

    @property
    def enabled(self):
        return self.inception is not None or self.i3d is not None

    @torch.no_grad()
    def add_frames(self, key, frames, episode=None):
        """frames: (N, H, W, 3) float32 in [0, 255]."""
        if self.inception is None:
            return
        self.fid_episodes[key].extend([episode] * int(frames.shape[0]))
        x = frames.permute(0, 3, 1, 2) / 255.0
        x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device)[None, :, None, None]
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device)[None, :, None, None]
        x = (x - mean) / std
        for i in range(0, x.shape[0], 64):
            self.fid_feats[key].append(self.inception(x[i:i + 64]).float().cpu())

    @torch.no_grad()
    def add_videos(self, key, videos, episode=None):
        """videos: (B, T, H, W, 3) float32 in [0, 255]."""
        if self.i3d is None:
            return
        self.fvd_episodes[key].extend([episode] * int(videos.shape[0]))
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
        # scipy dropped the `disp` keyword in 1.17; older versions need it to return
        # quietly rather than printing on a badly conditioned product.
        try:
            covmean = linalg.sqrtm(cov_a @ cov_b, disp=False)[0]
        except TypeError:
            covmean = linalg.sqrtm(cov_a @ cov_b)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        diff = mu_a - mu_b
        return float(diff @ diff + np.trace(cov_a) + np.trace(cov_b) - 2 * np.trace(covmean))

    @staticmethod
    def _episode_rows(episodes):
        """episode id -> row indices, for an episode-level bootstrap."""
        rows = defaultdict(list)
        for i, ep in enumerate(episodes):
            rows[ep].append(i)
        return {k: np.asarray(v) for k, v in rows.items()}

    def _frechet_ci(self, pred, real, pred_eps, real_eps, n_boot, seed):
        """Full-sample distance, plus an episode bootstrap CI when n_boot > 0.

        The predicted and real sets are resampled on the *same* episode draw: they are
        two halves of the same clips, so drawing them independently would add variance
        the paired comparison does not have.
        """
        value = self._frechet(pred, real)
        pred_rows = self._episode_rows(pred_eps)
        real_rows = self._episode_rows(real_eps)
        keys = [k for k in pred_rows if k is not None and k in real_rows]
        if not n_boot or len(keys) < 2:
            return {'value': value}
        rng = np.random.default_rng(seed)
        draws = []
        for _ in range(int(n_boot)):
            drawn = [keys[i] for i in rng.integers(0, len(keys), len(keys))]
            pi = np.concatenate([pred_rows[k] for k in drawn])
            ri = np.concatenate([real_rows[k] for k in drawn])
            draws.append(self._frechet(pred[pi], real[ri]))
        return {'value': value,
                'ci95': [float(np.percentile(draws, 2.5)),
                         float(np.percentile(draws, 97.5))],
                'bootstrap': {'n_boot': int(n_boot), 'n_episodes': len(keys),
                              'mean': float(np.mean(draws))}}

    def results(self, view_names, n_boot=0, seed=0):
        """Per-view FID / FVD as {'value': float} plus 'ci95' when bootstrapped.

        Args:
            view_names: views to report.
            n_boot: episode-bootstrap draws. 0 (the default) reports the point value
                only. Every draw recomputes a matrix square root, which for the
                2048-dim FID features costs seconds, so this is deliberately not the
                1000 draws the per-clip metrics use.
            seed: bootstrap seed.
        """
        out = {}
        for view in view_names:
            key_p, key_r = f'{view}/pred', f'{view}/real'
            if self.inception is not None:
                out.setdefault('fid', {})[view] = self._frechet_ci(
                    torch.cat(self.fid_feats[key_p]), torch.cat(self.fid_feats[key_r]),
                    self.fid_episodes[key_p], self.fid_episodes[key_r], n_boot, seed)
            if self.i3d is not None:
                out.setdefault('fvd', {})[view] = self._frechet_ci(
                    torch.cat(self.fvd_feats[key_p]), torch.cat(self.fvd_feats[key_r]),
                    self.fvd_episodes[key_p], self.fvd_episodes[key_r], n_boot, seed)
        return out
