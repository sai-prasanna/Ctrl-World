"""Scoring for world-model video rollouts.

The package takes ground-truth and generated frames and returns metrics. It knows
nothing about Ctrl-World, or about any particular model: `scripts/eval_video_metrics.py`
keeps the rollout and the checkpoint loading and calls in here for the scoring.

Two things shape the API.

First, frames alone are enough for some metrics and not others. PSNR, SSIM and LPIPS are
pure functions of two frame arrays; FID and FVD need a corpus but no extra input.
Region metrics need to know where the manipulated object is in the conditioning frame,
which frames do not say. So `add` takes frames as the required argument and `mask` as an
optional one. Each optional input unlocks more metrics, and `results` reports what was
skipped and why rather than failing.

Second, FID and FVD are corpus-level. They need features from every clip before a single
number exists, so this is an accumulator, not only a set of pure functions:

    import clipeval

    scorer = clipeval.Scorer(['psnr', 'ssim', 'lpips', 'fvd'],
                             rounds=12, per_round=4, view_names=['top'])
    for clip in clips:
        scorer.add(gt=clip.gt, pred=clip.pred, view='top', clip_id=clip.id,
                   episode_id=clip.episode, mask=clip.mask)  # mask optional
    results = scorer.results()

`add` returns that clip's metrics so a caller can log them per clip; `results` returns
everything, including the metrics that only exist in aggregate.
"""

from collections import defaultdict

import numpy as np
import torch

from . import pixel, stats
from .distribution import FeatureBank

__all__ = ['Scorer', 'pixel', 'stats', 'FeatureBank', 'keep_indices', 'anchor_indices']

PIXEL_METRICS = ('psnr', 'ssim', 'lpips')
# Static baselines. Raw PSNR is not comparable across clips: a clip where little moves
# scores well no matter what the model does. Repeating a real frame gives a per-clip
# floor to measure against, and the two floors answer different questions.
#
# `static_first` freezes the clip's first real frame for the whole rollout. Beating it is
# the minimum bar: it says the model predicted motion that is better than no motion at
# all. `static_round` instead repeats the frame each round was conditioned on, so it
# re-anchors on real pixels every round. That one is an oracle - it is handed ground
# truth twelve times per clip while a free-running model conditions on its own drifted
# prediction - so trailing it is expected early in training and is not the same finding
# as trailing `static_first`.
BASELINES = {'static_first': 'psnr_static_first',
             'static_round': 'psnr_static_round'}
DISTRIBUTION_METRICS = ('fid', 'fvd')


def keep_indices(rounds, per_round):
    """Absolute frame indices the model genuinely predicts.

    Rounds overlap by one frame, and the frame the model re-generates is the one it was
    conditioned on, so scoring it rewards copying. Each round's frame 0 is dropped.
    """
    return np.concatenate([np.arange(i * per_round + 1, i * per_round + per_round + 1)
                           for i in range(rounds)])


def anchor_indices(rounds, per_round):
    """For each kept frame, the frame its round was conditioned on."""
    return np.concatenate([np.full(per_round, i * per_round) for i in range(rounds)])


class Scorer:
    """Accumulates clip scores and reports per-view, per-group and per-round results.

    Args:
        metrics: any of 'psnr', 'ssim', 'lpips', 'fid', 'fvd'. The PSNR baselines come
            with 'psnr'.
        rounds, per_round: rollout shape. `pred` passed to `add` must have
            rounds * per_round frames, and `gt` must be the whole clip so the baselines
            can see the conditioning frames.
        view_names: cameras scored separately.
        view_groups: optional {group: [view, ...]} for rows pooled over cameras.
    """

    def __init__(self, metrics=PIXEL_METRICS, *, rounds, per_round,
                 view_names=('view',), view_groups=None, device='cpu',
                 i3d_ckpt=None, lpips_net='alex', bootstrap=1000, seed=0):
        self.metrics = list(metrics)
        self.rounds = int(rounds)
        self.per_round = int(per_round)
        self.n_pred = self.rounds * self.per_round
        self.view_names = list(view_names)
        self.view_groups = dict(view_groups or {})
        self.device = torch.device(device)
        self.bootstrap = bootstrap
        self.seed = seed
        self.skipped = {}

        self.keep = keep_indices(self.rounds, self.per_round)
        self.anchor = anchor_indices(self.rounds, self.per_round)

        self.lpips = None
        if 'lpips' in self.metrics:
            self.lpips = pixel.Lpips(self.device, net=lpips_net)

        self.bank = FeatureBank(self.device, fid='fid' in self.metrics,
                                i3d_ckpt=i3d_ckpt if 'fvd' in self.metrics else None)
        if 'fvd' in self.metrics and self.bank.i3d is None:
            self._skip('fvd', 'no I3D checkpoint given (--i3d_ckpt)')

        # per-clip, per-view arrays of per-frame scores
        self.series = defaultdict(lambda: defaultdict(list))
        # per-clip, per-view scalars (threshold accuracies, trajectory metrics)
        self.scalars = defaultdict(lambda: defaultdict(list))
        self.episode_ids = []
        self.clip_ids = []
        self._seen = set()

    def _skip(self, metric, reason):
        self.skipped.setdefault(metric, reason)

    # ------------------------------------------------------------------ adding

    def add(self, gt, pred, view, clip_id, episode_id=None, mask=None):
        """Score one clip in one view.

        Args:
            gt: (T, H, W, 3) uint8 or float in [0, 255], the whole clip's real frames.
            pred: (rounds * per_round, H, W, 3), the genuinely predicted frames.
            view: which camera; must be in `view_names`.
            clip_id: identifier, used only for reporting.
            episode_id: clips from one episode share a scene, so confidence intervals
                bootstrap over episodes. Defaults to `clip_id`, which assumes one clip
                per episode.
            mask: optional boolean (H, W) on the conditioning frame, marking the
                manipulated object. This is what gives the region metrics their bucket.

        Returns:
            This clip's metrics, as a flat dict of floats.
        """
        if view not in self.view_names:
            raise ValueError(f'unknown view {view!r}; expected one of {self.view_names}')
        gt = self._as_float(gt)
        pred = self._as_float(pred)
        if pred.shape[0] != self.n_pred:
            raise ValueError(f'expected {self.n_pred} predicted frames '
                             f'({self.rounds} rounds x {self.per_round}), '
                             f'got {pred.shape[0]}')
        if gt.shape[0] <= int(self.keep[-1]):
            raise ValueError(f'gt has {gt.shape[0]} frames but frame {self.keep[-1]} '
                             'must be scored; pass the whole clip, not just the '
                             'predicted range')

        key = (clip_id, view)
        if key in self._seen:
            raise ValueError(f'clip {clip_id!r} already added for view {view!r}')
        self._seen.add(key)
        if view == self.view_names[0]:
            self.clip_ids.append(clip_id)
            self.episode_ids.append(episode_id if episode_id is not None else clip_id)

        target = gt[self.keep]
        out = {}
        if 'psnr' in self.metrics:
            self._series('psnr', view, pixel.psnr(pred, target))
            self._series('psnr_static_round', view, pixel.psnr(gt[self.anchor], target))
            first = gt[self.anchor[0]].expand_as(target)
            self._series('psnr_static_first', view, pixel.psnr(first, target))
        if 'ssim' in self.metrics:
            self._series('ssim', view, pixel.ssim(pred, target))
        if self.lpips is not None:
            self._series('lpips', view, self.lpips(pred, target))
        if self.bank.enabled:
            self.bank.add_frames(f'{view}/pred', pred)
            self.bank.add_frames(f'{view}/real', target)
            self.bank.add_videos(f'{view}/pred', pred[None])
            self.bank.add_videos(f'{view}/real', target[None])
        for metric, per_view in self.series.items():
            if per_view[view]:
                out[metric] = float(np.mean(per_view[view][-1]))
        for metric, per_view in self.scalars.items():
            if per_view[view]:
                out[metric] = per_view[view][-1]
        return out

    def _as_float(self, frames):
        if isinstance(frames, np.ndarray):
            frames = torch.from_numpy(frames)
        return frames.to(self.device).float()

    def _series(self, metric, view, values):
        if torch.is_tensor(values):
            values = values.detach().cpu().numpy()
        self.series[metric][view].append(np.asarray(values, dtype=np.float64))

    def results(self):
        """Per-view, per-group and per-round results with bootstrap CIs."""
        n_clips = len(self.clip_ids)
        out = {
            'n_clips': n_clips,
            'n_episodes': len(set(self.episode_ids)),
            'rounds': self.rounds,
            'frames_per_round': self.per_round,
            'frames_scored_per_clip': self.n_pred,
            'note_excluded_frames': "each round's frame 0 is the model re-generating its "
                                    'conditioning frame and is excluded',
            'metrics_requested': self.metrics,
            'metrics_skipped': dict(self.skipped),
            'per_view': {}, 'per_group': {}, 'per_round': {},
            'distribution_metrics_backbones': self.bank.backbones,
        }
        for metric, per_view in list(self.series.items()):
            for view in self.view_names:
                runs = per_view.get(view)
                if not runs:
                    continue
                per_clip = np.stack(runs)                      # (clips, frames)
                mean, lo, hi = stats.bootstrap_ci(
                    np.nanmean(per_clip, axis=1), self.episode_ids,
                    self.bootstrap, self.seed)
                out['per_view'].setdefault(view, {})[metric] = {'mean': mean,
                                                                'ci95': [lo, hi]}
                rounds = np.nanmean(
                    per_clip.reshape(len(per_clip), self.rounds, self.per_round),
                    axis=(0, 2))
                out['per_round'].setdefault(view, {})[metric] = rounds.tolist()
            self._add_groups(out, metric, per_view, series=True)

        for metric, per_view in list(self.scalars.items()):
            for view in self.view_names:
                values = per_view.get(view)
                if not values:
                    continue
                finite = [v for v in values if np.isfinite(v)]
                episodes = [e for e, v in zip(self.episode_ids, values)
                            if np.isfinite(v)]
                if not finite:
                    out['per_view'].setdefault(view, {})[metric] = {'mean': None,
                                                                     'n': 0}
                    continue
                mean, lo, hi = stats.bootstrap_ci(finite, episodes, self.bootstrap,
                                                  self.seed)
                out['per_view'].setdefault(view, {})[metric] = {
                    'mean': mean, 'ci95': [lo, hi], 'n': len(finite)}
            self._add_groups(out, metric, per_view, series=False)

        self._add_psnr_gains(out)

        for metric, per_view in self.bank.results(self.view_names).items():
            for view, value in per_view.items():
                out['per_view'].setdefault(view, {})[metric] = {'value': value}
            for group, members in self.view_groups.items():
                out['per_group'].setdefault(group, {})[metric] = {
                    'value': float(np.mean([per_view[v] for v in members]))}
        return out

    def _add_groups(self, out, metric, per_view, series):
        for group, members in self.view_groups.items():
            present = [v for v in members if per_view.get(v)]
            if not present:
                continue
            if series:
                stacked = np.concatenate([np.stack(per_view[v]) for v in present])
                values = np.nanmean(stacked, axis=1)
            else:
                values = np.concatenate([np.asarray(per_view[v], dtype=np.float64)
                                         for v in present])
            episodes = list(self.episode_ids) * len(present)
            keep = np.isfinite(values)
            if not keep.any():
                continue
            mean, lo, hi = stats.bootstrap_ci(
                values[keep], [e for e, k in zip(episodes, keep) if k],
                self.bootstrap, self.seed)
            out['per_group'].setdefault(group, {})[metric] = {'mean': mean,
                                                              'ci95': [lo, hi]}

    @staticmethod
    def _add_psnr_gains(out):
        """Model minus frozen-frame baseline, in dB.

        PSNR is already logarithmic, so the difference is exactly the ratio of mean
        squared errors; a quotient of two PSNR numbers would have no fixed meaning.
        """
        for table in (out['per_view'], out['per_group']):
            for row in table.values():
                if 'psnr' not in row:
                    continue
                for baseline in BASELINES.values():
                    if baseline in row:
                        row[f'{baseline}_gain_db'] = row['psnr']['mean'] - row[baseline]['mean']
        for rounds in out['per_round'].values():
            for baseline in BASELINES.values():
                if 'psnr' in rounds and baseline in rounds:
                    rounds[f'{baseline}_gain_db'] = [
                        m - b for m, b in zip(rounds['psnr'], rounds[baseline])]

