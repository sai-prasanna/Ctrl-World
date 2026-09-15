# Plan: a reusable package for scoring world-model clips

Status: proposed, 2026-08-20
Owner: sai-prasanna
Branch: `abc-130k`

## Goal

Extract clip scoring out of `scripts/eval_video_metrics.py` into a standalone package,
`clipeval`, that takes ground-truth and generated frames and returns metrics. The package
covers the pixel metrics already in use (PSNR, SSIM, LPIPS, FID, FVD) plus tracking-based
metrics that measure whether the end effector, the static scene, and the manipulated object
behave correctly.

Two things motivate the split. First, pixel metrics score appearance rather than control
accuracy: PSNR and SSIM reward blurred mean-futures, and FID and FVD cannot tell whether an
object moved to the right place. Second, the scoring code has no dependency on Ctrl-World.
Keeping it in a script tied to `BatchRollout` means any other model, or a later port, has to
reimplement it.

## Scope of the API

The proposal is that inputs are "just ground-truth frames and generated frames". That covers
most of the surface, but not all of it, and the gap is worth naming before any code is written.

Frames alone are enough for:

- PSNR, SSIM, and LPIPS, which are pure functions of two frame arrays.
- FID and FVD, which need a corpus of clips rather than a single clip, but no extra inputs.
- The static-versus-dynamic tracking split, which seeds a grid of query points and separates
  them by how far they move in the ground truth.

Frames alone are *not* enough for:

- End-effector metrics, which need to know where each gripper is in the conditioning frame.
  Recorded actions plus camera calibration give that directly; a detector is the fallback.
- Manipulated-object metrics, which need a mask or a set of query points on the object in the
  conditioning frame.

So the API takes frames as the required argument and accepts optional `queries`, `masks`, and
`actions`. Each optional input unlocks more metrics, and the package reports which metrics it
skipped and why rather than failing. A caller that has only frames gets a useful subset.

The second API constraint is that FID and FVD are dataset-level, not clip-level. They need
features accumulated across every clip before a single number exists. The package therefore
exposes an accumulator, not only pure functions:

```python
import clipeval

scorer = clipeval.Scorer(metrics=['psnr', 'ssim', 'lpips', 'fvd', 'tracking'])
for clip in clips:
    scorer.add(gt=clip.gt, pred=clip.pred, view='top', clip_id=clip.id,
               queries=clip.queries)      # optional
results = scorer.results()                # per-view, per-round, with bootstrap CIs
```

`add` returns the clip-level metrics so a caller can log them per clip. `results` returns
everything, including the metrics that only exist in aggregate.

## Metrics

The following table lists what the package computes and what each metric needs.

| Metric | Level | Extra input | Source |
|---|---|---|---|
| PSNR, SSIM, LPIPS | Clip | None | Already in `eval_video_metrics.py` |
| PSNR against frozen-frame baselines | Clip | None | Already in `eval_video_metrics.py` |
| FID, FVD | Corpus | None | Already in `eval_video_metrics.py` |
| Static-point drift | Clip | None | New |
| Dynamic-point error | Clip | None | New |
| End-effector HSD, nDTW, DYN | Clip | Gripper seed per arm | Adapted from EWMBench |
| Object trajectory error | Clip | Object seed or mask | Adapted from MTV-World |

Report tracking error as a median and as threshold accuracy, the fraction of points within
2, 4, 8, and 16 pixels. CoTracker3 produces unbounded errors when it fails, so a mean over
points lets one lost track dominate a clip.

Report every tracking metric per round, as the pixel metrics already are. Growth in tracking
error across the rollout is a direct readout of drift.

Normalize tracking error by ground-truth displacement, or report the frozen-frame baseline
alongside. Raw pixel error has the same flaw as raw PSNR: a clip in which little moves scores
well whatever the model does.

## What to reuse

Reuse the metric math from two published benchmarks rather than deriving it:

- EWMBench supplies symmetric Hausdorff distance, normalized dynamic time warping, and a
  dynamic-consistency term that compares velocity and acceleration distributions with a
  Wasserstein distance. Vendor those functions with attribution. Skip the surrounding harness:
  it wants Qwen2.5-VL-7B, two CLIP variants, fine-tuned DINOv2 and YOLO-World checkpoints, a
  fixed directory layout, and a preprocessing pass, to produce four dimensions of which one is
  wanted. Its Python and CUDA pins also conflict with this repo's torch 2.7.1.
- MTV-World supplies the manipulated-object formulation: segment the object in both videos and
  compare positions frame by frame. Skip RVOS, which that paper needs because it evaluates
  across arbitrary tasks. Seeding query points on the object in the conditioning frame is
  cheaper here.

Both benchmarks separate trajectory extraction from trajectory scoring, and EWMBench reads
trajectories from a `traj.npy` file per episode. Keep that separation, so swapping the tracker
later touches one function.

## Tracker choice

Start with CoTracker3, and treat its reliability on generated frames as an open question.

CoTracker3 is no longer the most accurate tracker available: it scores 64.45 on TAP-Vid DAVIS
against 66.56 for TAPNext, and LocoTrack matches it while running roughly six times faster. It
wins on ecosystem maturity, which matters more than a two-point margin for a metric that runs
periodically. Its offline mode interpolates through occlusion better than its online mode, so
use offline.

Three documented failure modes bear on manipulation video:

- Textureless surfaces are the top reported failure. Robot arms and plain tabletops are
  textureless, so this is the common case rather than a corner case.
- Errors are unbounded when tracking fails, which is why the headline metric is a median.
- Accuracy drops by more than 30 points from static to dynamic subsets. The gripper and the
  manipulated object sit in the hard regime; static background sits in the easy one.

No published work characterizes point tracking on diffusion-generated frames, so the noise
floor on this data is unknown.

## Repository layout

Keep the package inside this repo as a subpackage, under a top-level `clipeval/` directory,
with self-contained code and no import of any external benchmark repository. Give it its own
`pyproject.toml` later, once a second project needs it. Packaging it early costs a release
process before there is a second caller.

Self-contained applies to EWMBench only: copy its metric functions into
`clipeval/tracking/metrics.py` with attribution and a note of the source commit, rather than
adding the EWMBench repository as a submodule or a dependency. Its Python and CUDA pins
conflict with this repo, and only a few functions are wanted.

CoTracker is a normal dependency, not vendored. `clipeval/tracking/extract.py` imports it.

```
clipeval/
  __init__.py        # Scorer
  pixel.py           # psnr, ssim, lpips, frozen-frame baselines
  distribution.py    # FeatureBank: FID and FVD
  tracking/
    extract.py       # CoTracker3 wrapper: frames + queries -> tracks
    metrics.py       # HSD, nDTW, DYN, threshold accuracy (vendored, attributed)
    seeding.py       # grid seeding, static/dynamic split, gripper and object seeds
  stats.py           # bootstrap CIs over episodes
```

`scripts/eval_video_metrics.py` keeps the Ctrl-World-specific parts, the `BatchRollout` class
and the checkpoint loading, and calls `clipeval` for scoring. `psnr`, `ssim`, and `FeatureBank`
move across unchanged, so the pixel numbers stay comparable to runs already recorded.

## Implementation order

Each step produces a usable result on its own, and the risky work comes last.

1. Measure the tracker noise floor. Run CoTracker3 on about 10 ground-truth validation clips,
   compare the tracks against recorded gripper actions projected into the image, and write
   overlay videos to `outputs/`. The floor sets the smallest model error the metric can
   detect. If the floor is large on ABC-130k textures, the rest of the plan changes.
2. Add `--dump_frames DIR` to `scripts/eval_video_metrics.py`, writing lossless per-clip
   arrays. Write npz or PNG sequences, not mp4: H.264 artifacts perturb a point tracker, and
   ground truth and prediction would take different amounts of that distortion. Check the total
   size for 256 clips across 3 views before choosing between npz and PNG.
3. Extract `clipeval` with the pixel and distribution metrics only, and confirm the numbers
   match a recorded run of `eval_video_metrics.py` exactly.
4. Add tracking with grid seeding and the static-versus-dynamic split. This needs no per-episode
   setup and validates the whole path.
5. Vendor HSD, nDTW, and DYN, and run them on the tracks from step 4.
6. Add gripper and object seeding, per arm. This is the only step that needs per-episode input,
   and it depends on whether calibration or segmentation is the cheaper route for ABC-130k.

## Environment

The repo uses a single virtualenv, `$ROOT/venv`, where `ROOT=$WORK/sraman00/ctrlworld`, as
`jobs/train.sbatch` and `jobs/rollout.sbatch` both show. Keep it that way: add CoTracker3 to
`requirements.txt` and install it into that venv. Do not create a second environment for
evaluation.

CoTracker coexists with the pinned torch 2.7.1, diffusers 0.34.0, and transformers 4.48.1.
Its `setup.py` declares `install_requires=[]`, so it pins nothing and uses whatever torch is
already installed. Add it to `requirements.txt` as a git reference:

```
cotracker @ git+https://github.com/facebookresearch/co-tracker.git
```

Leonardo compute nodes have no network access, and the batch scripts set `HF_HUB_OFFLINE=1`.
Run the install on a login node. Download the checkpoint there too and load it from a local
path: `torch.hub.load` reaches GitHub at call time, which fails on a compute node, so
`extract.py` takes a `--tracker_ckpt` flag instead.

## Step 1 result: the tracker noise floor, measured

Run on 2026-08-20 over 10 validation clips, 16x16 grid, CoTracker3 offline
(`scaled_offline.pth`), by `scripts/tracker_noise_floor.py`. Full numbers in
`outputs/0002_abc_rigid/eval/tracker_noise_floor.json`, overlays beside it.

Camera calibration turned out not to be available, so the floor is measured two ways that
need no labels: forward-backward cycle consistency, and drift on low-variance background
pixels of a fixed camera. Both are floors rather than bounds — a tracker that drifts
consistently in both directions passes the cycle test.

| View | Cycle median | Cycle p90 | Background drift median | GT displacement p90 |
|---|---|---|---|---|
| top | 0.38 px | 5.80 px | 0.48 px | 26.67 px |
| left_wrist | 28.82 px | 91.83 px | 31.68 px | 112.03 px |
| right_wrist | 23.27 px | 84.27 px | 37.11 px | 96.53 px |

A first pass seeded its grid over the whole 192x192 frame. ABC-130k pads to that size with
a 24 px black bar top and bottom, a quarter of the image, so 64 of 256 points sat on dead
pixels that track perfectly and never move. That padded the static bucket and pulled the
background-drift floor toward zero, and on the wrist views the surviving points were
almost entirely letterbox. `seeding.content_bbox` now excludes it; the table is from the
corrected run.

Three conclusions, and they do change the plan.

**The top view works.** Noise sits at 0.22 of the signal at p90, background drift reaches
1.89 px at p99, with a 7.85 px worst case over 2560 points, and threshold accuracy at 4, 8 and 16 px is
meaningful. The 4 px static/dynamic threshold sits above the p99, so under 1% of
genuinely static points are misfiled as moving. The 2 px
accuracy bucket is inside the noise for the tail and should be read as a floor check
rather than a score.

**The wrist views are excluded from tracking, on validity rather than precision.** The
tracker's ability to follow them turns out not to be the deciding question. Over a whole
clip it cannot: cycle error is 0.82 to 0.87 of displacement at p90 and only 21 to 25% of
points survive a 4 px round trip. Over a single round it can: error falls to 3.53 px
against 13.48 px of real motion, a median noise-to-signal of 0.26, better than the top
view's own 0.54 over a whole clip.

| View | Horizon | Cycle med | Cycle p90 | Disp med | N/S med | N/S p90 | <4px |
|---|---|---|---|---|---|---|---|
| top | clip (49) | 0.38 px | 5.80 px | 0.70 px | 0.54 | 0.22 | 88.7% |
| top | round (5) | 0.15 px | 1.53 px | 0.21 px | 0.68 | 0.23 | 93.7% |
| left_wrist | clip (49) | 28.82 px | 91.83 px | 41.85 px | 0.69 | 0.82 | 21.5% |
| left_wrist | round (5) | 3.53 px | 49.82 px | 13.48 px | 0.26 | 1.00 | 51.4% |
| right_wrist | clip (49) | 23.27 px | 84.27 px | 37.74 px | 0.62 | 0.87 | 25.3% |
| right_wrist | round (5) | 2.42 px | 43.60 px | 13.89 px | 0.17 | 0.94 | 55.1% |

But a trackable pixel is not an interpretable one. A wrist camera is bolted to a moving
arm, so a point's image motion is ego-motion composed with the object's own motion, and
the 13.5 px of per-round wrist displacement against 0.21 px in the top view says
ego-motion is essentially the whole signal. That ego-motion is set by the recorded actions
the model is conditioned on, so the metric would largely grade the model on re-rendering
its own input, which is a warp-consistency check that the pixel metrics already perform on
those views.

Two of the three stated targets invert outright. The end effector is very nearly
stationary in its own wrist frame, so that view cannot measure end-effector motion at all;
and the static-versus-dynamic split has no meaning when the whole scene sweeps, so the
manipulated object cannot be separated from the background either.

Ego-motion compensation is the obvious repair and is not worth attempting. A homography is
a misspecified motion model at wrist range, where scene depth varies a great deal, so
parallax would leak into the object residual and manufacture error that scales with arm
speed; gripper-relative coordinates share the defect. The only sound version is tracking
segmented object points, which needs a segmenter and per-clip object identity, and
top-view tracking already answers where the object went.

So `track_views` defaults to the fixed camera. The per-view `track_horizon` machinery
stays, since it is what a segmentation-based reformulation would need, and the wrist
per-round numbers are recorded here so that decision does not have to be re-measured.

**Report tracking error stratified by displacement, not pooled.** A median over every
seeded point is dominated by background that barely moves, where the tracker's own error
is most of the number: the top view's pooled median noise-to-signal is 0.54, against 0.22
at p90 where the moving points are. `Scorer` therefore buckets points by ground-truth
displacement (under 4 px, 4 to 16, 16 to 64, above 64) and reports each separately, with
the frozen-frame baseline alongside. The pooled figure should not be quoted on its own.

**The failure tail is real.** Even on the top view, cycle error reaches 50.17 px at p99
against a 0.38 px median. That is the unbounded-error failure mode the plan anticipated,
and it confirms the median-plus-threshold-accuracy reporting over a mean.

The overlays show why a whole-clip pass fails on the wrist views. Frame 0
is fully tracked; by frame 12 essentially every point inside the image is flagged
occluded, and the five sampled frames show unrelated scenes, because at 5 Hz an
ego-centric camera moves the whole image between consecutive frames. Before the letterbox
fix the wrist background drift looked healthy at 1.5 px, which suggested the low-variance
region might be gripper hardware rigidly attached to the camera and therefore genuinely
still; it was the black bars. Corrected, that number is 31.68 px. Shortening the horizon, rather than swapping trackers, is what fixes this: the
tracker holds fine across one round and only collapses when asked to carry points through
many rounds of ego-motion.

## Open questions

- ~~Whether camera calibration for ABC-130k is available.~~ Settled: it is not. The
  validation annotations carry no camera, intrinsics, extrinsics or pose keys, so step 6
  needs a detector or segmentation rather than projected gripper positions.
- Whether to report a single aggregate score. EWMBench and WorldArena both do, and WorldArena
  calls its version EWMScore. An aggregate hides the per-bucket detail that motivates this work,
  so the default is to report the buckets separately.

## References

- EWMBench: Evaluating Scene, Motion, and Semantic Quality in Embodied World Models,
  arXiv 2505.09694, code at https://github.com/AgibotTech/EWMBench
- MTV-World: Towards High-Consistency Embodied World Model with Multi-View Trajectory Videos,
  arXiv 2511.12882
- Genie Envisioner, arXiv 2508.05635, for the arm-versus-object trajectory-error split
- CoTracker3: Simpler and Better Point Tracking by Pseudo-Labelling Real Videos,
  https://cotracker3.github.io/
- LocoTrack: Local All-Pair Correspondence for Point Tracking,
  https://cvlab-kaist.github.io/locotrack/
