"""Measure CoTracker3's error floor on ground-truth ABC-130k clips.

Step 1 of plans/clipeval-package.md. Every tracking metric in `clipeval` is bounded below
by the tracker's own error on real video, so that floor has to be known before any model
number from it means anything. If the floor is 6 px on ABC-130k textures, a model whose
gripper lands 4 px from the right place is indistinguishable from a perfect one.

The plan sketched comparing tracks against recorded gripper actions projected into the
image. That needs camera calibration, which is an open question for this dataset, so this
script measures the floor two ways that need no labels at all, and separately reports
whichever calibration-shaped keys the annotations actually carry.

  * Forward-backward cycle consistency. Track a grid forward through the clip, then track
    the endpoints backward through the reversed clip, and measure how far each point lands
    from where it started. A perfect tracker returns exactly. This is the headline number.
  * Static-background drift. A pixel whose value barely changes over the whole clip is
    background, and a fixed camera means background does not move. Points seeded in
    low-variance regions should hold still; whatever motion the tracker reports there is
    noise. This is the floor for the static bucket specifically.

Neither number is an upper bound on the true error: a tracker can drift consistently in
both directions and pass the cycle test. They are floors, which is what is wanted.

Overlay videos go to outputs/ so the failure modes can be looked at rather than inferred
from a percentile.

Example:

    python3 scripts/tracker_noise_floor.py \
        --val_dataset_dir $WORK/sraman00/ctrlworld/data/abc_rigid \
        --clips dataset_meta_info/abc_rigid/eval_clips_v1.json \
        --tracker_ckpt $WORK/sraman00/ctrlworld/cotracker/scaled_offline.pth \
        --n_clips 10 \
        --out outputs/0002_abc_rigid/eval/tracker_noise_floor.json
"""

import argparse
import json
import os
import sys

import numpy as np
from decord import VideoReader, cpu

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import clipeval  # noqa: E402
from clipeval.tracking import metrics as tmetrics, seeding  # noqa: E402

VIEW_NAMES = ['top', 'left_wrist', 'right_wrist']

# Keys that would let gripper positions be projected into the image without a detector.
# Reported, not required: whether ABC-130k ships any of them is an open question, and
# the answer decides how end-effector seeding gets built in step 6 of the plan.
CALIBRATION_KEYS = ['camera', 'cameras', 'calibration', 'intrinsics', 'extrinsics',
                    'camera_intrinsics', 'camera_extrinsics', 'K', 'T_cam', 'pose']


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--val_dataset_dir', type=str, default=None,
                   help='defaults to the value recorded in the clip list')
    p.add_argument('--clips', type=str, required=True)
    p.add_argument('--tracker_ckpt', type=str, required=True,
                   help='local path to CoTracker3 scaled_offline.pth; torch.hub.load '
                        'reaches GitHub at call time and compute nodes have no network')
    p.add_argument('--n_clips', type=int, default=10)
    p.add_argument('--grid', type=int, default=16)
    p.add_argument('--views', type=str, nargs='+', default=VIEW_NAMES)
    p.add_argument('--horizon', type=int, default=0,
                   help='track in windows of this many frames, re-seeding at each '
                        'window start, instead of one pass over the whole clip. The '
                        'rollout advances 4 frames per round, so --horizon 5 measures '
                        'the floor over exactly the span a round has to get right. '
                        '0 means the whole clip.')
    p.add_argument('--static_percentile', type=float, default=25.0,
                   help='points whose local temporal variance falls below this '
                        'percentile count as background')
    p.add_argument('--out', type=str, required=True)
    p.add_argument('--video_dir', type=str, default=None,
                   help='write overlay videos here (default: alongside --out)')
    p.add_argument('--no_videos', action='store_true')
    p.add_argument('--save_tracks', action='store_true',
                   help='also dump tracks and frames to .npz so scripts/viz_tracks.py '
                        'can render without re-running the tracker on a GPU')
    p.add_argument('--device', type=str, default='cuda')
    return p.parse_args()


def load_gt_frames(root, clip, n_frames):
    """(views, T, H, W, 3) uint8 of raw mp4 frames, plus the annotation."""
    with open(f'{root}/annotation/val/{clip["episode_id"]}.json') as f:
        ann = json.load(f)
    ids = np.arange(clip['start_idx'], clip['start_idx'] + n_frames)
    if ids[-1] >= ann['video_length']:
        raise ValueError(f'clip {clip["episode_id"]}@{clip["start_idx"]} needs frame '
                         f'{ids[-1]} but the episode has {ann["video_length"]}')
    frames = []
    for view in range(3):
        vr = VideoReader(f'{root}/{ann["videos"][view]["video_path"]}', ctx=cpu(0),
                         num_threads=2)
        batch = vr.get_batch(ids.tolist())
        arr = batch.asnumpy() if hasattr(batch, 'asnumpy') else batch.numpy()
        frames.append(arr)
    return np.stack(frames), ann


def temporal_variance(frames):
    """(T, H, W, 3) -> (H, W) variance of each pixel over time, averaged over channels.

    High where something moved, low on background. Cheaper and more direct than a
    segmentation model, and this is a fixed camera.
    """
    return frames.astype(np.float32).var(axis=0).mean(axis=-1)


def static_mask_from_variance(frames, percentile, bbox, margin=8):
    """Boolean (H, W) marking the quietest pixels inside the real image.

    The percentile is taken over the content region only. Including the letterbox would
    make the black bars the quietest quarter of every frame, so the mask would select
    padding and the drift it measured would be zero by construction.
    """
    var = temporal_variance(frames)
    y0, y1, x0, x1 = bbox
    inside = np.zeros(var.shape, dtype=bool)
    inside[y0 + margin:y1 - margin, x0 + margin:x1 - margin] = True
    cutoff = np.percentile(var[inside], percentile)
    return (var <= cutoff) & inside, var


def windowed_cycle_error(tracker, frames, queries_fn, horizon):
    """Cycle consistency measured over short windows instead of the whole clip.

    A point tracker's error accumulates with the motion it has to follow, and on an
    ego-centric camera a 49-frame clip at 5 Hz sweeps the whole scene several times. The
    metric that matters is per round, and a round spans `horizon` frames, so this seeds a
    fresh grid at each window start and scores only that span.

    Returns the concatenated per-point errors and the per-window displacement, so the
    floor can be read against the motion in the same window rather than the whole clip.
    """
    errs, disps = [], []
    step = horizon - 1                    # windows overlap by a frame, as rounds do
    for start in range(0, len(frames) - 1, step):
        window = frames[start:start + horizon]
        if len(window) < 3:               # too short for a there-and-back
            break
        queries = queries_fn(window)
        err, tracks, _ = cycle_error(tracker, window, queries)
        errs.append(err)
        disps.append(np.linalg.norm(tracks - tracks[0][None], axis=-1).max(axis=0))
    return np.concatenate(errs), np.concatenate(disps)


def cycle_error(tracker, frames, queries):
    """Forward then backward. Returns (error_px per point, forward tracks, visibility).

    The backward pass re-seeds at the forward pass's endpoints in the reversed clip, so
    the two passes see the same pixels in the opposite order.
    """
    fwd_tracks, fwd_vis = tracker.track(frames, queries)
    t_last = len(frames) - 1
    ends = fwd_tracks[t_last]                                    # (K, 2)
    back_queries = np.concatenate([np.zeros((len(ends), 1)), ends], axis=1)
    back_tracks, _ = tracker.track(frames[::-1].copy(), back_queries)
    returned = back_tracks[t_last]                               # back at frame 0
    err = np.linalg.norm(returned - queries[:, 1:], axis=-1)
    return err, fwd_tracks, fwd_vis


def draw_overlay(frames, tracks, visible, radius=1):
    """Paint track positions onto a copy of the clip. Green when visible, red when not."""
    out = np.array(frames, dtype=np.uint8, copy=True)
    t_max, h, w, _ = out.shape
    for t in range(t_max):
        for k in range(tracks.shape[1]):
            x, y = tracks[t, k]
            xi, yi = int(round(x)), int(round(y))
            if not (0 <= xi < w and 0 <= yi < h):
                continue
            color = ((0, 255, 0) if visible[t, k] else (255, 0, 0))
            y0, y1 = max(0, yi - radius), min(h, yi + radius + 1)
            x0, x1 = max(0, xi - radius), min(w, xi + radius + 1)
            out[t, y0:y1, x0:x1] = color
    return out


def summarize(values, thresholds=tmetrics.THRESHOLDS_PX):
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {'n': 0}
    return {
        'n': int(finite.size),
        'median_px': float(np.median(finite)),
        'mean_px': float(finite.mean()),
        'p90_px': float(np.percentile(finite, 90)),
        'p99_px': float(np.percentile(finite, 99)),
        'max_px': float(finite.max()),
        **{f'within_{t}px': float((finite < t).mean()) for t in thresholds},
    }


def main():
    args = parse_args()
    with open(args.clips) as f:
        clip_list = json.load(f)
    root = args.val_dataset_dir or clip_list['val_dataset_dir']
    clips = clip_list['clips'][:args.n_clips]
    n_frames = clip_list['clip_frames']

    video_dir = args.video_dir or os.path.join(
        os.path.dirname(os.path.abspath(args.out)), 'tracker_overlays')
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    tracker = clipeval.tracking.Tracker(args.tracker_ckpt, device=args.device)
    print(f'tracker: {tracker.info}')

    per_clip = []
    pooled = {view: {'cycle': [], 'static_drift': [], 'displacement': []}
              for view in args.views}
    calibration_seen = {}

    for clip in clips:
        frames, ann = load_gt_frames(root, clip, n_frames)
        for key in CALIBRATION_KEYS:
            if key in ann:
                calibration_seen.setdefault(key, str(type(ann[key]).__name__))
        clip_id = f'{clip["episode_id"]}@{clip["start_idx"]}'
        record = {'clip_id': clip_id, 'task': clip.get('task'), 'views': {}}

        for view in args.views:
            v = VIEW_NAMES.index(view)
            clip_frames = frames[v]
            h, w = clip_frames.shape[1:3]

            bbox = seeding.content_bbox(clip_frames)
            queries = seeding.grid_queries(h, w, grid=args.grid, bbox=bbox)
            if args.horizon:
                err, displacement = windowed_cycle_error(
                    tracker, clip_frames,
                    lambda w_frames: seeding.grid_queries(h, w, grid=args.grid,
                                                          bbox=bbox),
                    args.horizon)
                # overlays still show the whole-clip pass, which is what a reader wants
                _, tracks, vis = cycle_error(tracker, clip_frames, queries)
            else:
                err, tracks, vis = cycle_error(tracker, clip_frames, queries)
                displacement = np.linalg.norm(
                    tracks - tracks[0][None], axis=-1).max(axis=0)

            mask, var = static_mask_from_variance(clip_frames, args.static_percentile,
                                                  bbox)
            static_q = seeding.mask_queries(mask, n_points=64, seed=0, bbox=bbox)
            static_tracks, _ = tracker.track(clip_frames, static_q)
            # On a fixed camera, background does not move; whatever motion the tracker
            # reports on these points is its own noise.
            static_drift = np.linalg.norm(
                static_tracks - static_tracks[0][None], axis=-1).max(axis=0)

            record['views'][view] = {
                'cycle_consistency': summarize(err),
                'static_background_drift': summarize(static_drift),
                'gt_displacement': summarize(displacement),
                'horizon': args.horizon or n_frames,
                'frame_size': [int(h), int(w)],
                'content_bbox': [int(v) for v in bbox],
                'static_pixel_variance_cutoff': float(np.percentile(var[mask], 100.0))
                if mask.any() else None,
            }
            pooled[view]['cycle'].append(err)
            pooled[view]['static_drift'].append(static_drift)
            pooled[view]['displacement'].append(displacement)

            if args.save_tracks:
                os.makedirs(video_dir, exist_ok=True)
                # Persist the tracks so a visualization does not need a GPU re-run.
                # Frames go in too: they are small at 192x192 and it keeps the file
                # self-contained.
                np.savez_compressed(
                    os.path.join(video_dir,
                                 f'{clip_id.replace("@", "_")}_{view}_tracks.npz'),
                    frames=clip_frames.astype(np.uint8), tracks=tracks, visible=vis,
                    queries=queries, displacement=displacement,
                    content_bbox=np.asarray(bbox))

            if not args.no_videos:
                os.makedirs(video_dir, exist_ok=True)
                import mediapy
                overlay = draw_overlay(clip_frames, tracks, vis)
                mediapy.write_video(
                    os.path.join(video_dir, f'{clip_id.replace("@", "_")}_{view}.mp4'),
                    overlay, fps=5)

        per_clip.append(record)
        top = record['views'][args.views[0]]
        print(f'{clip_id:24s} cycle median '
              f'{top["cycle_consistency"]["median_px"]:.2f} px   '
              f'static drift median '
              f'{top["static_background_drift"]["median_px"]:.2f} px   '
              f'gt displacement median {top["gt_displacement"]["median_px"]:.2f} px')

    results = {
        'what': 'CoTracker3 error floor on ground-truth ABC-130k validation clips',
        'tracker': tracker.info,
        'clip_list': args.clips,
        'n_clips': len(clips),
        'grid': args.grid,
        'horizon': args.horizon or n_frames,
        'static_percentile': args.static_percentile,
        'reading': {
            'cycle_consistency': 'forward then backward through the reversed clip; the '
                                 'distance a point lands from where it started. A floor, '
                                 'not a bound: consistent drift in both directions passes.',
            'static_background_drift': 'motion the tracker reports on low-variance '
                                       'background pixels of a fixed camera, which is '
                                       'noise by construction. VALID FOR THE TOP VIEW '
                                       'ONLY: a wrist camera moves, so nothing is static '
                                       'in image space and this measures real motion of '
                                       'mislabelled points rather than tracker error.',
            'gt_displacement': 'how far the grid points actually travel, for scale. A '
                               'floor near this number means the metric measures nothing.',
        },
        'calibration_keys_found': calibration_seen or None,
        'calibration_note': 'keys that would allow projecting recorded gripper positions '
                            'into the image without a detector; empty means step 6 of the '
                            'plan needs segmentation or a detector instead',
        'per_view': {},
        'per_clip': per_clip,
    }
    for view in args.views:
        results['per_view'][view] = {
            name: summarize(np.concatenate(vals))
            for name, vals in (('cycle_consistency', pooled[view]['cycle']),
                               ('static_background_drift', pooled[view]['static_drift']),
                               ('gt_displacement', pooled[view]['displacement']))}

    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)

    print(f'\n{len(clips)} clips, grid {args.grid}x{args.grid}')
    for view in args.views:
        row = results['per_view'][view]
        print(f'  {view:14s} cycle median {row["cycle_consistency"]["median_px"]:6.2f} px '
              f'(p90 {row["cycle_consistency"]["p90_px"]:6.2f})   '
              f'static drift median '
              f'{row["static_background_drift"]["median_px"]:6.2f} px   '
              f'gt displacement median {row["gt_displacement"]["median_px"]:6.2f} px')
    print(f'calibration keys in annotations: {calibration_seen or "none found"}')
    if not args.no_videos:
        print(f'overlays in {video_dir}')
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
