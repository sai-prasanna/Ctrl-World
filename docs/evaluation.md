# Video prediction metrics

This page describes how to score a Ctrl-World checkpoint on held-out ABC-130k
validation clips, following the protocol behind Table 1 of the Ctrl-World paper
(arXiv 2510.10125).

The evaluation replays recorded actions through the world model autoregressively and
compares the generated frames against ground truth with peak signal-to-noise ratio
(PSNR), structural similarity index measure (SSIM), learned perceptual image patch
similarity (LPIPS), Fréchet inception distance (FID), and Fréchet video distance (FVD).

## Pieces

| Path | Purpose |
|---|---|
| `scripts/make_eval_clips.py` | Draws the fixed clip list from the validation annotations. |
| `scripts/eval_video_metrics.py` | Rolls out a batch of clips and scores them. |
| `clipeval/` | Model-independent scoring package: pixel, distribution, and tracking metrics. |
| `scripts/selftest_eval_metrics.py` | Checks the metrics on synthetic data, on CPU. |
| `scripts/eval_leonardo.sh` | Wraps both for Leonardo: `setup`, `clips`, and `run`. |
| `dataset_meta_info/abc_rigid/eval_clips_v1.json` | The committed clip list. |
| `scripts/wmbench_rollout_ctrlworld.py` | Writes the same rollout as a `wmbench` manifest, for the 16-metric protocol below. |
| `scripts/wmbench_leonardo.sh` | Wraps the benchmark environment for Leonardo: `setup`, `run`, `check`. |

## Protocol

Each clip runs for 12 autoregressive rounds. A round conditions on the previous round's
final frame, receives a 5-step action chunk, and predicts 5 latent frames. Rounds overlap
by one frame, so a clip spans `1 + 4 x 12 = 49` frames, which is 9.8 seconds at the 5 Hz
latent rate. The paper uses a 10-second horizon, and 12 rounds is the closest match.

The following choices differ from `scripts/rollout_replay_traj.py`, and each one changes
the numbers:

- **Ground truth is the raw video.** The rollout script builds its reference by decoding
  encoded latents, which hides the variational autoencoder's (VAE's) own reconstruction
  error inside every metric. The evaluation reads the `.mp4` frames instead. The files on
  disk are already written at 5 Hz and 192x192, aligned one-to-one with the latents.
- **Each round's first frame is excluded.** Because rounds overlap, the frame a round
  regenerates is the frame it was conditioned on, so scoring it rewards copying. With
  `pred_step=5` that frame is a quarter of the rollout. Metrics cover the other 48 frames.
- **Metrics are reported per round.** Drift across a rollout is what the memory mechanism
  exists to control, and a clip-level average hides it.
- **Sampling is seeded per clip.** The diffusion sampler is stochastic, and the rollout
  script sets no seed, so repeat runs of one checkpoint disagree.

`num_inference_steps`, `guidance_scale`, and the history indices are recorded in the
output file. Changing any of them moves the numbers more than most checkpoint deltas.

### History indices

The evaluation pins `history_idx = [-6, -5, -4, -3, -2, -1]`, matching
`scripts/rollout_replay_traj.py`. The buffer holds one entry per round and rounds advance
4 latent frames, so those indices are 24, 20, 16, 12, 8 and 4 frames back: evenly spaced,
ending at the most recent round.

That is the `skip=1` regime of `dataset_droid_exp33.py`, which builds history at
`skip_his = 4 * skip` and future frames at `skip`, for `skip` in `{1, 2}`. A rollout
predicts consecutive frames, so it is a `skip=1` clip and takes `skip=1` history. Pairing
`skip=2` history with `skip=1` future, as `[0, 0, -8, -6, -4, -2]` did, is a combination
training never draws.

The paper conditions on `o_{t-km}, ..., o_{t-m}, o_t`: evenly spaced, ending at the
current frame, with no anchor on the clip's first observation. The two leading zero slots
of the earlier value were an artifact of the rollout buffer being pre-filled with copies
of the first latent, not something the paper asks for. The paper's stated history interval
is 1-2 seconds against the 0.8 seconds here, which would argue for the `skip=2` spacing;
this pin follows what the checkpoint was trained to predict instead.

The `[0, 0, -12, -9, -6, -3]` in `config.py` implies a 12-frame spacing, which training
never draws. Only the policy-in-the-loop scripts read that value. Training itself reads
neither, because the dataset lays out history by frame offset.

Numbers recorded before 2026-08-25 used `[0, 0, -8, -6, -4, -2]` and are not comparable
across this change.

## Clip selection

The training meta info in `dataset_meta_info/create_meta_info.py` emits one window per
start frame, giving roughly 46,000 near-duplicate windows across the validation split.
That suits training and not evaluation, so `make_eval_clips.py` draws a separate sample:

- A clip is valid only when `start + 68 <= video_length`. `get_traj_info` clamps reads
  past the end of an episode, which freezes the ground truth while the model keeps
  predicting, and reports no error when it happens.
- Starts are drawn uniformly from the valid range with a per-episode seed. Fixed
  fractional offsets would bias the sample toward particular phases of a sort-and-place
  task.
- Each episode contributes at most two clips, spaced at least one clip apart.
- Clips are allocated across tasks in proportion to the validation episodes per task.

The committed list holds 226 clips over 121 episodes. Two properties of it matter when
you read the results:

- The count is 226 rather than the paper's 256. Of the 145 validation episodes, 24 are
  shorter than the 68 frames a 12-round rollout needs, which is 13.6 seconds. Raising the
  per-episode cap to three would reach 256, at the cost of more correlated clips. Since
  every checkpoint scores on the same list, the exact count only matters for comparing
  against the paper, and FID and FVD are not comparable to the paper regardless.
- Excluding short episodes biases the sample toward long tasks. "Put the screwdriver in
  the bin" has 21 validation episodes and contributes 2 clips, so that task is effectively
  unmeasured. Treat any task with fewer than 10 clips as unreported.

To regenerate the list, run `scripts/eval_leonardo.sh clips` on a login node and commit
the result. Keep old versions rather than overwriting them, so earlier numbers stay
readable.

### The split holds out episodes, not scenes

The validation split is upstream's own and is disjoint by episode, but an episode is not a
scene. `scripts/analyze_split_overlap.py` approximates scene identity with the one signal
the annotations record — the robot's starting joint configuration, in the normalized space
the model trains in — and finds the two splits heavily overlapped:

| Measure | Value |
|---|---|
| Validation tasks absent from training | 0 of 11 |
| Median distance, val episode to nearest train episode | 0.079 |
| Median distance, val episode to nearest other val episode | 0.209 |
| Val episodes closer to a train episode than to any other val episode | 209 of 224 |
| Val episodes with a train neighbor within 0.05 | 97 of 224 |

A validation episode is about 2.6 times closer to some training episode than to its own
nearest neighbor inside the split. Every number in [experiments.md](experiments.md)
therefore measures within-scene generalization: the same station, task, and object layout
as training, on a different take. Read those numbers as an upper bound on novel-scene
performance, and do not quote them as evidence the model generalizes to a new rig.

Fixing this needs a held-out scene set, which the ABC-130k annotations do not currently
support without new metadata. To reproduce the measurement:

```bash
python3 scripts/analyze_split_overlap.py --data_root <data/abc_mcap> \
    --stat <dataset_meta_info/abc_mcap/stat.json> --workers 8 --out <report.json>
```

## Reading the results

The output JSON reports each metric per camera (`top`, `left_wrist`, `right_wrist`) and
per group. The `third_view` group is the top camera and `wrist_view` is the mean of the
two wrist cameras, matching the paper's two rows.

Splitting by camera is sound here. The three views are stacked vertically in latent space
only, and `eval_video_metrics.py` splits them before the VAE decode, so the decoder never
sees a seam.

### PSNR baseline

Raw PSNR does not compare across clips: a clip in which little moves scores well whatever
the model does. The evaluation therefore scores a frozen-frame baseline,
`psnr_static_round`, against the same targets. It repeats the frame each round was
conditioned on, and answers whether the model beat freezing the image for that second.

The baseline anchors on a real frame, so it does not degrade as the model drifts. That
makes it an oracle: it receives the ground truth at each round start, while a free-running
model conditions on its own prediction. Part of any negative gain is therefore accumulated
drift the baseline is immune to by construction.

The comparison is reported as `_gain_db`, the difference in decibels. PSNR is already
logarithmic, so subtracting gives the ratio of mean squared errors. A quotient of two
PSNR values has no fixed meaning, because it moves with the data range.

### Confidence intervals

Intervals on PSNR, SSIM, and LPIPS come from a bootstrap over episodes, not clips,
because two clips from one episode share a scene and lighting.

FID and FVD carry no interval. Both are corpus-level, so there is no per-clip value to
average and the same bootstrap does not apply, and both are biased at small sample sizes.
Compare them only at a fixed clip count, and treat a move in either as directional. To
decide whether one checkpoint beats another, use PSNR, SSIM, and LPIPS, which are
per-clip and carry episode-bootstrapped intervals.

### FID and FVD

These two are comparable across your own checkpoints and nothing else. The paper pins
neither its feature extractor nor its resize path, and both metrics move by large amounts
with those choices. The evaluation uses a torchvision Inception-v3 backbone for FID and
the stylegan-v TorchScript I3D for FVD, and records both in the output under
`distribution_metrics_backbones`. When the I3D file is absent, FVD is skipped rather than
approximated.

### Latent mean squared error

`latent_mse` compares predicted and ground-truth latents per round. It costs nothing,
because both tensors already exist, and it ranks checkpoints without a VAE decode. Use it
for cheap sweeps and keep the pixel metrics for reported numbers.

## The wmbench protocol

`wmbench` scores the same rollouts against WorldArena's 16 metrics. It is a separate
repository with no Ctrl-World inside it, so the protocol above gains four things rather
than changing: a declared frame space, a reference column beside every no-reference metric,
bounds fitted where the scoring happens, and a vision-language judge.
`plans/wmbench-package.md` holds the design and the milestone status.

The stages are separate commands, because a rollout belongs on a GPU node and scoring does
not. `scripts/wmbench_rollout_ctrlworld.py` writes a manifest — `manifest.json` plus one
lossless npz per clip — and `wmbench score` reads it from the other environment, as often
as the metric set grows.

### The operating point is applied to ground truth

A world model declares the frame rate, resolution, and crop it works in. Ctrl-World's is
5 Hz, 256x192, `fov_crop` to 88.6 degrees horizontal field of view, named
`ctrlworld_fovcrop_v1`, and the harness puts ground truth into that space before any pixel
metric runs. A source whose pixels already carry a different preprocessing is refused
instead of compared: a crop cannot be undone, and scoring a model against pixels it was
never shown reports the difference as model error. Every result file records the operating
point, and two checkpoints are comparable only at the same one.

### Reference columns and bounds

Most of the 16 metrics are no-reference: they read the prediction alone. A MUSIQ imaging
quality of 0.42 says nothing until you know what this dataset's own footage scores at
192-pixel height and 5 Hz, so every such metric is also run on the ground-truth frames of
the same clip and reported as `<metric>_gt`. Read the pair, not the number.

`wmbench fit-bounds` turns that into a scale. It reads a manifest three ways — the recorded
footage, the first real frame held for the whole rollout, and each round's conditioning
frame held for that round — and takes the 1st and 99th percentiles per metric. Zero on the
normalized scale then means "no better than a frozen frame here" and one means "as good as
this dataset's own footage", both measured at this geometry. WorldArena's hardcoded table
cannot be reused: `flow_score` is a pixel count, so halving the frame height halves it.
Bounds are keyed by dataset and operating point, and belong to the run
(`experiments/<tag>/eval/`), not to the benchmark. Normalized columns appear only with
`--norm-bounds`; raw values are always reported.

The anti-static penalty is additive for the same reason. A model that emits a frozen frame
scores perfect subject consistency, background consistency, and photometric consistency, so
those three are multiplied by `dynamic_degree` when it falls at or below the threshold
(0.1213 unless the bounds file sets another). WorldArena does that in place; `wmbench`
writes `<metric>_penalized` beside the raw column, records whether the gate fired, and
leaves the raw value readable. Read the `_penalized` column whenever the model might be
standing still — `photometric_consistency` in particular maxes out on a clip that does not
move, because flow you cannot invert is flow through a surface that changed.

### What is comparable to the clipeval numbers

PSNR, SSIM, LPIPS, FID, and FVD are the same code, copied from `clipeval/` with its
headers, over the same clips and the same excluded frames. They are comparable to
[experiments.md](experiments.md) with one qualification.

The rollout decodes differently. `eval_video_metrics.py` decodes a whole 48-frame rollout
in chunks of `decode_chunk_size`, 7 by default; the bridge has to return each round's
frames as it produces them, so it decodes 4 at a time. The Stable Video Diffusion temporal
decoder mixes across the frames of a chunk, so the chunk a frame lands in changes it: the
pixels differ by 1 to 2.5 grey levels on average and PSNR by up to 0.05 dB. The predicted
latents are bit-identical either way, and the manifest records `decode_chunk_size`. Treat a
difference of that size between a `wmbench` number and a `clipeval` number as the decoder,
not the checkpoint.

The 16 metrics themselves are comparable only across checkpoints scored by this package.
Several backbones are substitutions — torchvision RAFT for `raft-things.pth` and SEA-RAFT,
RIFE for VFIMamba — and the interpolator or flow network is part of the metric's
definition. `tools/parity_worldarena.py` in the benchmark repository measures the gap on
real clips; it runs 0.975 to 1.038 of WorldArena's values across the six metrics that can
be checked, with the flow metrics widest.

### Does the model obey its actions

Every metric above compares a rollout with the recording, so a model that ignores the
action chunk and continues the scene plausibly still scores well on all of them. The test
that separates the two is counterfactual: roll the same clips out again on another clip's
motion, that clip's joint trajectory shifted to start from this clip's own first state,
with the same conditioning frame and the same seed, and score both rollouts against the
clip's own recording. `jobs/wmbench_rollout.sbatch` with `COUNTERFACTUAL=shift` writes
that manifest, `wmbench compare` pairs it with the true-action rollout, and the paired
PSNR and LPIPS difference is the reading. A margin near zero means the actions are not
what drives the frames, and no other number in this document matters for VLAW until it
is fixed.

### Seeing the cases

`wmbench gallery` is the page to open before trusting any row of the tables above. For
every metric it plays the clip the checkpoint scored best on and the clip it scored worst
on, prediction against recording under a wipe, with the metric's own signal drawn onto
the frames and its per-frame value running underneath. Object survival paints exactly
the pixels it counts and each colour's centroid path; depth blends the two
Depth-Anything maps; the flow metrics paint RAFT magnitude with arrows; the pixel metrics
paint the difference. When a metric's worst clip does not look worse than its best one,
the metric is not measuring what its name says at this operating point, and that is the
finding. `jobs/wmbench_gallery.sbatch` builds it on Leonardo from a scored manifest;
the videos are files beside the page, so copy the directory, not the HTML.

### The judge

Three of WorldArena's metrics are Likert scores from a vision-language model: interaction
quality, perspectivity, and instruction following. `wmbench judge --rubric triad` runs
Qwen3-VL-8B-Instruct locally over 16 sampled frames, greedy, and gets all three from one
answer. That is WorldArena's prompt and WorldArena's arrangement: the three scores are
conditioned on each other through the shared context, so asking for one alone is a
different measurement. The 8B model in bfloat16 is about 17 GB of weights before
activations, so plan for a card with 24 GB or more. Scores are not comparable across model
sizes, any more than a metric is comparable across backbones.

`--rubric success` is the policy track's evaluator. A world model has pixels and no `qpos`,
so the only thing that can say whether a rollout completed the task is something that looks
at it. The judge is shown the rollout and its recorded clip as a reference, judges only the
rollout, and treats uncertainty as failure. The answer is written into the `success` field
of each world record in ABC's `summary.json`, where `build_summary` reports it like sim
success, alongside a `judged_success` block naming the model that wrote it.

### Validating the policy track

The policy track is only trustworthy if it orders the obvious cases correctly, so run those
before reading any number from it. On 16 clips, judged success must come out
ground-truth-action replay (`--policy echo`) >= `abc_dit_xl_200k` >> random actions. Check
the judge's own stability first: two runs at temperature 0 on the same frames must agree.
An ordering that fails here is a finding about the world model or the judge, not a result
about the policy.

## Check the metrics without a GPU

`scripts/selftest_eval_metrics.py` exercises the `clipeval` metrics, both
distribution-metric backbones, and the aggregation on synthetic data. It runs on CPU in
seconds and needs no checkpoint, so you can catch a broken metric before spending GPU
time. It does not cover the rollout, which needs a model and a GPU.

```bash
python3 scripts/selftest_eval_metrics.py --i3d_ckpt $WORK/sraman00/ctrlworld/i3d_torchscript.pt
```

Each check prints `PASS` or `FAIL`, and the command exits nonzero when anything fails.

## Run an evaluation on Leonardo

The `setup` and `clips` subcommands run on a login node, because compute nodes have no
internet access and the clip list is built from the validation annotations.

1. Install the dependencies and cache the backbone weights. Run this once:

   ```bash
   ssh leonardo
   cd $WORK/sraman00/ctrlworld/repo
   bash scripts/eval_leonardo.sh setup
   ```

   The command installs `lpips` into the venv, caches the AlexNet and Inception-v3
   weights under `$WORK/sraman00/ctrlworld/torchhome`, and downloads the I3D backbone,
   verifying its MD5.

2. Submit one job per checkpoint. The `cluster` CLI snapshots the working tree at an
   exact commit, so each run records the code that produced it:

   ```bash
   cluster submit leonardo --gpus 1 --cpus 8 --time 08:00:00 -- \
       bash scripts/eval_leonardo.sh run 10000
   ```

   To smoke-test the pipeline in a few minutes, append `--limit 8`.

3. Read the results:

   ```bash
   cluster logs leonardo RUNID
   ```

   The job writes `$WORK/sraman00/ctrlworld/eval/metrics_step<step>.json`.

Prefer `cluster submit` over copying the repository to `$WORK/sraman00/ctrlworld/repo` by
hand. An earlier untracked copy of `jobs/rollout.sbatch` pointed at the wrong dataset, and the
runs it produced looked normal.

### Job sizing

`AIFAC_S07_034` is a shared account. When another user saturates it, `sbatch` rejects new
work with `More processors requested than permitted`, and the fix is to wait rather than
to shrink the request.

A full 226-clip run does 226 x 12 x 50 denoising steps. Raise `BATCH_SIZE` to fill the
GPU, and set the walltime from a `--limit` run before committing to the full sweep.
