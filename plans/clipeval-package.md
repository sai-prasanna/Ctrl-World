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
`train.sbatch` and `rollout.sbatch` both show. Keep it that way: add CoTracker3 to
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

## Open questions

- Whether camera calibration for ABC-130k is available. If it is, projected gripper positions
  give ground-truth end-effector pixels without a detector, and a detector or tracker is needed
  only on the generated video.
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
