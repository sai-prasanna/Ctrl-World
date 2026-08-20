"""Scoring for world-model video rollouts.

The package takes ground-truth and generated frames and returns metrics. It knows
nothing about Ctrl-World, or about any particular model: `scripts/eval_video_metrics.py`
keeps the rollout and the checkpoint loading and calls in here for the scoring.

Two things shape the API.

First, frames alone are enough for some metrics and not others. PSNR, SSIM and LPIPS are
pure functions of two frame arrays; FID and FVD need a corpus but no extra input; the
static-versus-dynamic tracking split seeds a grid and separates points by how far the
ground truth moves them. End-effector and manipulated-object metrics need to know where
the gripper or the object is in the conditioning frame, which frames do not say. So
`add` takes frames as the required argument and `queries` and `mask` as optional ones.
Each optional input unlocks more metrics, and `results` reports what was skipped and why
rather than failing.

Second, FID and FVD are corpus-level. They need features from every clip before a single
number exists, so this is an accumulator, not only a set of pure functions:

    import clipeval

    scorer = clipeval.Scorer(['psnr', 'ssim', 'lpips', 'fvd', 'tracking'],
                             rounds=12, per_round=4, view_names=['top'])
    for clip in clips:
        scorer.add(gt=clip.gt, pred=clip.pred, view='top', clip_id=clip.id,
                   episode_id=clip.episode, queries=clip.queries)  # queries optional
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
# floor to measure against. `static_round` repeats the frame each round was conditioned
# on; `static_first` repeats the clip's initial observation for the whole rollout.
BASELINES = {'static_round': 'psnr_static_round', 'static_first': 'psnr_static_first'}
DISTRIBUTION_METRICS = ('fid', 'fvd')
TRACKING_METRICS = ('tracking',)

# Per-frame tracking series, aggregated per round exactly as the pixel metrics are.
TRACK_SERIES = ('track_err_all', 'track_err_static', 'track_err_dynamic',
                'track_err_frozen')


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
        metrics: any of 'psnr', 'ssim', 'lpips', 'fid', 'fvd', 'tracking'. The PSNR
            baselines come with 'psnr'.
        rounds, per_round: rollout shape. `pred` passed to `add` must have
            rounds * per_round frames, and `gt` must be the whole clip so the baselines
            and the tracker can see the conditioning frames.
        view_names: cameras scored separately.
        view_groups: optional {group: [view, ...]} for rows pooled over cameras.
        tracker: a `clipeval.tracking.extract.Tracker`, or None to skip tracking.
        grid: grid size for seeding when a caller passes no queries.
    """

    def __init__(self, metrics=PIXEL_METRICS, *, rounds, per_round,
                 view_names=('view',), view_groups=None, device='cpu',
                 i3d_ckpt=None, lpips_net='alex', tracker=None, grid=16,
                 bootstrap=1000, seed=0):
        self.metrics = list(metrics)
        self.rounds = int(rounds)
        self.per_round = int(per_round)
        self.n_pred = self.rounds * self.per_round
        self.view_names = list(view_names)
        self.view_groups = dict(view_groups or {})
        self.device = torch.device(device)
        self.bootstrap = bootstrap
        self.seed = seed
        self.grid = grid
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

        self.tracker = tracker
        if 'tracking' in self.metrics and tracker is None:
            self._skip('tracking', 'no tracker given; pass a clipeval.tracking Tracker')

        # per-clip, per-view arrays of per-frame scores
        self.series = defaultdict(lambda: defaultdict(list))
        # per-clip, per-view scalars (threshold accuracies, trajectory metrics)
        self.scalars = defaultdict(lambda: defaultdict(list))
        self.episode_ids = []
        self.clip_ids = []
        self._seen = set()

    def _skip(self, metric, reason):
        self.skipped.setdefault(metric, reason)

    @property
    def _tracking_on(self):
        return 'tracking' in self.metrics and self.tracker is not None

    # ------------------------------------------------------------------ adding

    def add(self, gt, pred, view, clip_id, episode_id=None, queries=None, mask=None):
        """Score one clip in one view.

        Args:
            gt: (T, H, W, 3) uint8 or float in [0, 255], the whole clip's real frames.
            pred: (rounds * per_round, H, W, 3), the genuinely predicted frames.
            view: which camera; must be in `view_names`.
            clip_id: identifier, used only for reporting.
            episode_id: clips from one episode share a scene, so confidence intervals
                bootstrap over episodes. Defaults to `clip_id`, which assumes one clip
                per episode.
            queries: optional (K, 3) of (frame_index, x, y) to track. Without it a grid
                is seeded, which still gives the static/dynamic split.
            mask: optional boolean (H, W) on the conditioning frame, sampled for queries
                when `queries` is not given. This is how the manipulated object gets its
                own bucket.

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
            self._series('psnr_static_first', view,
                         pixel.psnr(gt[0:1].expand_as(target), target))
        if 'ssim' in self.metrics:
            self._series('ssim', view, pixel.ssim(pred, target))
        if self.lpips is not None:
            self._series('lpips', view, self.lpips(pred, target))
        if self.bank.enabled:
            self.bank.add_frames(f'{view}/pred', pred)
            self.bank.add_frames(f'{view}/real', target)
            self.bank.add_videos(f'{view}/pred', pred[None])
            self.bank.add_videos(f'{view}/real', target[None])
        if self._tracking_on:
            out.update(self._add_tracking(gt, pred, view, queries, mask))

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

    # ---------------------------------------------------------------- tracking

    def _add_tracking(self, gt, pred, view, queries, mask):
        from .tracking import extract, metrics as tmetrics, seeding

        target = gt[self.keep]
        # Both sequences start with the same real conditioning frame, so the two tracks
        # begin from identical pixels and every later difference is the model's.
        cond = gt[0:1]
        seq_gt = torch.cat([cond, target]).cpu().numpy().astype(np.uint8)
        seq_pred = torch.cat([cond, pred]).cpu().numpy().astype(np.uint8)

        if queries is None:
            h, w = gt.shape[1], gt.shape[2]
            queries = (seeding.mask_queries(mask, seed=self.seed) if mask is not None
                       else seeding.grid_queries(h, w, grid=self.grid))

        tracks = self.tracker.track_pair(seq_gt, seq_pred, queries)
        gt_tracks = tracks['gt_tracks']
        pred_tracks = tracks['pred_tracks']

        split = seeding.split_static_dynamic(gt_tracks)
        err = tmetrics.endpoint_error(pred_tracks, gt_tracks)[1:]      # drop frame 0
        # A model that freezes the conditioning frame would leave every point where it
        # started, so the ground truth's own displacement is that baseline's error. It
        # plays the role static_round PSNR plays for the pixel metrics: a clip in which
        # little moves scores well however the model behaves.
        frozen = np.linalg.norm(gt_tracks - gt_tracks[0][None], axis=-1)[1:]

        buckets = {'all': np.ones(gt_tracks.shape[1], dtype=bool),
                   'static': split['static'], 'dynamic': split['dynamic']}
        for name, sel in buckets.items():
            series = (np.median(err[:, sel], axis=1) if sel.any()
                      else np.full(err.shape[0], np.nan))
            self._series(f'track_err_{name}', view, series)
            summary = tmetrics.error_summary(err[:, sel] if sel.any() else np.array([]))
            for stat, value in summary.items():
                if stat.startswith('acc_'):
                    self.scalars[f'track_{name}_{stat}'][view].append(value)
        self._series('track_err_frozen', view,
                     np.median(frozen, axis=1) if frozen.size else
                     np.full(err.shape[0], np.nan))
        self.scalars['track_n_dynamic'][view].append(float(split['dynamic'].sum()))

        # HSD, nDTW and DYN score a single trajectory, the one the ground truth moves
        # farthest, following EWMBench.
        traj = tmetrics.trajectory_metrics(
            extract.mark_invisible(pred_tracks, tracks['pred_visible']),
            extract.mark_invisible(gt_tracks, tracks['gt_visible']))
        for name in ('hsd_px', 'ndtw_px', 'gt_extent_px', 'vel_wasserstein',
                     'acc_wasserstein'):
            self.scalars[name][view].append(float(traj[name]))
        for name, value in tmetrics.ewmbench_scores(traj).items():
            self.scalars[f'ewmbench_{name}'][view].append(float(value))
        return {'track_index': traj['track_index'],
                'pred_track_lost': traj['pred_track_lost']}

    # ------------------------------------------------------------------ output

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
        if self.tracker is not None:
            out['tracker'] = self.tracker.info

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


def __getattr__(name):
    # Lazy so `import clipeval` does not pull in cotracker, which most callers, and
    # every run without --track, do not need.
    if name == 'tracking':
        from . import tracking
        return tracking
    raise AttributeError(name)
