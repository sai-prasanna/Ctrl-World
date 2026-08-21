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

import importlib
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
# on. It is an oracle: it receives the real frame at each round start, while a
# free-running model conditions on its own drifted prediction.
BASELINES = {'static_round': 'psnr_static_round'}
DISTRIBUTION_METRICS = ('fid', 'fvd')
TRACKING_METRICS = ('tracking',)

# Ground-truth displacement strata, in pixels. Tracking error is reported per stratum
# because error and tracker noise both scale with how far a point travels, so one pooled
# median mixes points where the metric has resolution with points where it has none.
DISPLACEMENT_BINS = ((4, 16), (16, 64), (64, float('inf')))

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
        track_horizon: how many frames a tracking pass covers, per view. `None` tracks
            the whole clip in one pass; an int re-seeds every `track_horizon` frames; a
            dict maps view name to either. The right value depends on how fast the view
            moves, and the two differ by an order of magnitude on ABC-130k: a fixed
            overhead camera accumulates only 0.21 px of median motion per round, so it
            needs the whole clip to have anything to measure, while a wrist camera sweeps
            13 px per round and loses the tracker entirely over a longer span. Measured in
            outputs/0002_abc_rigid/eval/tracker_noise_floor{,_h5}.json.
    """

    def __init__(self, metrics=PIXEL_METRICS, *, rounds, per_round,
                 view_names=('view',), view_groups=None, device='cpu',
                 i3d_ckpt=None, lpips_net='alex', tracker=None, grid=16,
                 track_horizon=None, hsd_views=None, track_views=None,
                 bootstrap=1000, dist_bootstrap=0, seed=0):
        self.metrics = list(metrics)
        self.rounds = int(rounds)
        self.per_round = int(per_round)
        self.n_pred = self.rounds * self.per_round
        self.view_names = list(view_names)
        self.view_groups = dict(view_groups or {})
        self.device = torch.device(device)
        self.bootstrap = bootstrap
        # FID/FVD are corpus-level: each draw recomputes a matrix square root, so this
        # gets its own, much smaller, draw count. 0 disables the CI entirely.
        self.dist_bootstrap = dist_bootstrap
        self.seed = seed
        self.grid = grid
        self.track_horizon = track_horizon
        # Views whose tracking is reliable enough in the tail for a max-based statistic.
        self.hsd_views = hsd_views
        # Views where a tracked pixel displacement means what the metric claims. On a
        # camera rigidly mounted to a moving arm it does not: image motion there is
        # dominated by ego-motion, which the replayed ground-truth actions already
        # determine, and the end effector is very nearly stationary in that frame. So a
        # wrist track measures how well the model re-renders its own conditioning input,
        # which the pixel metrics already cover, and cannot measure the end effector at
        # all. Default to every view; callers with a moving camera should pass the fixed
        # ones only.
        self.track_views = track_views
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
        if 'ssim' in self.metrics:
            self._series('ssim', view, pixel.ssim(pred, target))
        if self.lpips is not None:
            self._series('lpips', view, self.lpips(pred, target))
        if self.bank.enabled:
            episode = episode_id if episode_id is not None else clip_id
            self.bank.add_frames(f'{view}/pred', pred, episode)
            self.bank.add_frames(f'{view}/real', target, episode)
            self.bank.add_videos(f'{view}/pred', pred[None], episode)
            self.bank.add_videos(f'{view}/real', target[None], episode)
        if self._tracking_on and (self.track_views is None
                                  or view in self.track_views):
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
            # Exclude letterbox: a quarter of an ABC-130k frame is a black bar, and
            # points seeded there track perfectly and pad the static bucket.
            bbox = seeding.content_bbox(seq_gt)
            queries = (seeding.mask_queries(mask, seed=self.seed, bbox=bbox)
                       if mask is not None
                       else seeding.grid_queries(h, w, grid=self.grid, bbox=bbox))

        horizon = self.track_horizon
        if isinstance(horizon, dict):
            horizon = horizon.get(view)
        if horizon:
            tracks = self._track_windowed(seq_gt, seq_pred, queries, int(horizon))
        else:
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

        # Stratify by how far the ground truth actually moves the point. A median over
        # every seeded point is dominated by background that barely moves, where the
        # tracker's own error is most of the number: on the top view that median sits at
        # 0.54 of the signal. Points that move a lot are where the metric has resolution,
        # and they have to be reported separately rather than averaged in.
        displacement = split['displacement_px']
        buckets = {'all': np.ones(gt_tracks.shape[1], dtype=bool),
                   'static': split['static'], 'dynamic': split['dynamic']}
        for lo, hi in DISPLACEMENT_BINS:
            name = f'disp{lo}_{hi}' if np.isfinite(hi) else f'disp{lo}plus'
            buckets[name] = (displacement >= lo) & (displacement < hi)
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
        for name, sel in buckets.items():
            self.scalars[f'track_n_{name}'][view].append(float(sel.sum()))

        # HSD, nDTW and DYN score a single trajectory, the one the ground truth moves
        # farthest, following EWMBench.
        traj = tmetrics.trajectory_metrics(
            extract.mark_invisible(pred_tracks, tracks['pred_visible']),
            extract.mark_invisible(gt_tracks, tracks['gt_visible']))
        # HSD is a Hausdorff distance, a maximum over the trajectory, so it is decided by
        # the worst-tracked frame. On the wrist views the tracker's p90 error equals the
        # real motion, so that maximum is noise and the metric is dropped there; the
        # median-based point errors and nDTW survive. See the plan's step 1 result.
        traj_names = ['ndtw_px', 'gt_extent_px', 'vel_wasserstein', 'acc_wasserstein']
        if self.hsd_views is None or view in self.hsd_views:
            traj_names.append('hsd_px')
        for name in traj_names:
            self.scalars[name][view].append(float(traj[name]))
        for name, value in tmetrics.ewmbench_scores(traj).items():
            self.scalars[f'ewmbench_{name}'][view].append(float(value))
        return {'track_index': traj['track_index'],
                'pred_track_lost': traj['pred_track_lost']}

    def _track_windowed(self, seq_gt, seq_pred, queries, horizon):
        """Track in overlapping windows, re-seeding at each window start.

        A tracker's error grows with the motion it has to follow, and on a wrist camera a
        whole clip at 5 Hz sweeps the scene many times over; re-seeding each round keeps
        every pass inside the span the tracker can actually hold. Windows overlap by one
        frame, exactly as rollout rounds do.

        This changes what the metric means, and the change is the point. A whole-clip pass
        measures accumulated drift: where has the point ended up after 48 frames. A
        per-round pass measures whether the model got *this round's motion* right, given
        whatever frame it started from. Both are worth having; only the second survives on
        a wrist view.
        """
        step = horizon - 1
        gt_parts, pred_parts, gt_vis, pred_vis = [], [], [], []
        for start in range(0, len(seq_gt) - 1, step):
            g = seq_gt[start:start + horizon]
            p = seq_pred[start:start + horizon]
            if len(g) < 2:
                break
            out = self.tracker.track_pair(g, p, queries)
            # drop each window's frame 0 except the first, since windows overlap there
            keep = slice(0 if start == 0 else 1, None)
            gt_parts.append(out['gt_tracks'][keep])
            pred_parts.append(out['pred_tracks'][keep])
            gt_vis.append(out['gt_visible'][keep])
            pred_vis.append(out['pred_visible'][keep])
        return {'gt_tracks': np.concatenate(gt_parts),
                'pred_tracks': np.concatenate(pred_parts),
                'gt_visible': np.concatenate(gt_vis),
                'pred_visible': np.concatenate(pred_vis)}

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

        dist = self.bank.results(self.view_names, self.dist_bootstrap, self.seed)
        for metric, per_view in dist.items():
            for view, entry in per_view.items():
                out['per_view'].setdefault(view, {})[metric] = entry
            for group, members in self.view_groups.items():
                # A group average of two corpus-level distances is not itself a Frechet
                # distance, so it carries no CI even when the members have one.
                out['per_group'].setdefault(group, {})[metric] = {
                    'value': float(np.mean([per_view[v]['value'] for v in members]))}
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
        return importlib.import_module('.tracking', __name__)
    raise AttributeError(name)
