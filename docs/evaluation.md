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
