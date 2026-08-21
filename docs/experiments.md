# Experiments

One entry per evaluated run. Numbers here are copied from the JSON in
`outputs/<tag>/eval/`, which is force-added to git despite the `output*` ignore rule so
the record cannot drift from the prose. Protocol, deviations from the paper, and the
reasoning behind each metric live in [evaluation.md](evaluation.md); this file is the
results log.

## 0002_abc_rigid — SVD world model on ABC-130k (`abc_rigid`)

Branch `abc-130k`. 4x A100 x batch 4 x grad-accum 4 = effective batch 64, matching the
paper's 2x8 H100. Trained from SVD init; checkpoints every 2500 steps.

Evaluated at **step 5000** and **step 10000** on `eval_clips_v1`: 226 clips over 121
episodes of the validation split, 12 autoregressive rounds each. Each round predicts 5
frames and the next round overlaps by one, so a clip is 49 frames (9.8 s at 5 Hz) and
48 are scored — every round's frame 0 is dropped because it is the conditioning frame,
not a prediction. Free-running: after the first round the model conditions only on its
own output. Ground truth is the raw mp4 frames at the 5 Hz latent rate, with no VAE
round-trip, so the reported error includes the autoencoder's own reconstruction loss.

Views are scored in two groups. `third_view` is the single `top` camera; `wrist_view`
pools `left_wrist` and `right_wrist`. All three are denoised jointly as one latent
stacked vertically, then split per camera before the VAE decode.

### Pixel metrics

Brackets are 95% bootstrap CIs resampled over **episodes**, not clips, since clips from
the same episode are not independent.

**third_view (`top`)**

| | step 5000 | step 10000 |
|---|---|---|
| PSNR (dB) | 22.667 [22.367, 22.809] | **23.293** [22.987, 23.448] |
| SSIM | 0.7996 [0.7915, 0.8071] | **0.8100** [0.8013, 0.8175] |
| LPIPS | 0.0984 [0.0961, 0.1025] | **0.0913** [0.0889, 0.0952] |
| vs copy-frame | −1.692 dB | **−1.065 dB** |

**wrist_view (`left_wrist` + `right_wrist`)**

| | step 5000 | step 10000 |
|---|---|---|
| PSNR (dB) | 17.167 [16.898, 17.562] | **17.793** [17.522, 18.173] |
| SSIM | 0.6577 [0.6489, 0.6744] | **0.6694** [0.6601, 0.6865] |
| LPIPS | 0.3151 [0.3037, 0.3232] | **0.2984** [0.2873, 0.3059] |
| vs copy-frame | −2.004 dB | **−1.379 dB** |

+0.63 dB on both groups with disjoint PSNR CIs, so the improvement from 5k to 10k is
real and the model is still far from converged — the paper trains for 100k steps.

The copy-frame baseline (`psnr_static_round`, 24.358 dB third-view / 19.172 wrist) is
identical at both steps by construction: it repeats each round's conditioning frame, so
it does not depend on the checkpoint. It is still ahead of the model, but the gap
halved in 5000 steps. **It is an oracle**: it is handed the real frame at every round
start, while the model is free-running on its own drifted prediction. Read `gain_db` as
"how far behind an oracle that never moves", not as a fair competitor.

### Distribution metrics

| | step 5000 | step 10000 |
|---|---|---|
| FID third_view | 24.950 | 23.783 |
| FVD third_view | 290.375 | 270.030 |
| FID wrist_view | 37.574 | 36.965 |
| FVD wrist_view | 413.982 | 403.324 |

FVD moves −7.0% third-view but only −2.6% wrist, against −7.3% / −5.3% on LPIPS. The
wrist rollouts are landing closer to the right frames without their distribution of
motion looking much more real.

Neither metric carries a confidence interval: both are computed once over the whole
pooled feature set rather than per episode. On 226 clips a ~10-point FVD move is within
the range where Fréchet estimators are still biased by sample size, so the wrist −10.7
should not be read as a real change. Bootstrapping the feature bank over episodes would
fix this and needs no re-rollout, since the per-clip features are already computed.

FVD uses stylegan-v's I3D (MD5-pinned in `scripts/eval_leonardo.sh`), the de facto
standard, so the absolute scale is comparable to published numbers. **FID is not
portable**: it uses torchvision's ImageNet Inception rather than the TF-ported
`pt_inception` that the literature quotes. Compare it only against our own runs.

### Latent MSE by round

Third-view latent error saturates rather than diverging — 0.211 at round 1, 0.301 by
round 4, 0.317 at round 12 (mean 0.296). Per-round PSNR shows the same shape:

| round | 1 | 2 | 3 | 4 | 6 | 8 | 10 | 12 |
|---|---|---|---|---|---|---|---|---|
| step 5000 | 25.37 | 23.59 | 22.92 | 22.65 | 22.32 | 22.20 | 22.02 | 22.03 |
| step 10000 | 25.73 | 24.13 | 23.56 | 23.28 | 22.98 | 22.87 | 22.73 | 22.73 |

Most of the degradation happens in the first three rounds and then flattens, so the
rollout is not accumulating unbounded drift over 9.8 s.

### Exact configuration

```json
{"history_idx": [0, 0, -8, -6, -4, -2], "num_inference_steps": 50, "guidance_scale": 1.0,
 "pred_step": 5, "interact_num": 12, "num_history": 6, "num_frames": 5, "seed": 0}
```

`history_idx` matches `rollout_replay_traj.py`, not `config.py` — see the history-index
section of [evaluation.md](evaluation.md) for why.

### Reproducing

All of this runs on Leonardo. Compute nodes have no internet, so the two `setup` steps
must run on a login node first.

```bash
# once: deps, LPIPS AlexNet, FID Inception, and the MD5-pinned I3D
scripts/eval_leonardo.sh setup

# once: draw the fixed clip list (already committed as eval_clips_v1.json —
# only rerun to make a new version, and bump the filename if you do)
scripts/eval_leonardo.sh clips

# per checkpoint, inside a GPU job
cluster submit leonardo --gpus 1 --cpus 8 --time 04:00:00 --name eval_leonardo \
  -- scripts/eval_leonardo.sh run 10000
```

One checkpoint takes ~2.5 h on a single A100 at `BATCH_SIZE=4` (step 10000 ran 2:33:12,
step 5000 ran 2:39:11). Override `CKPT`, `OUT`, or `BATCH_SIZE` in the environment if
you need a path outside the default `outputs/0002_abc_rigid/model/` layout.

Verify the scoring code without a GPU or the dataset:

```bash
python scripts/selftest_eval_metrics.py   # 18 checks, CPU only
```

### Known gaps

- **Not converged.** The next run should go to at least 30000 steps. 20000 more steps at
  ~4.71 s/it is ~26 h, past the 24 h wall limit, so it needs two chained jobs:
  `MAX_STEPS=30000 RESUME=--resume sbatch train.sbatch`, submitted twice. `--resume`
  with no `--ckpt_path` auto-picks the newest `checkpoint-<step>.pt`.
- **No tracking metrics yet.** Pixel metrics score appearance, not control accuracy —
  PSNR and SSIM reward a blurred mean-future. `clipeval/tracking` and the CoTracker3
  noise floor exist; `--track --tracker_ckpt <path>` has not been run on either
  checkpoint.
- **No fair per-round baseline.** A baseline repeating the model's *own* conditioning
  frame would separate "this round predicted no motion" from "the rollout has drifted",
  which the current oracle baseline confounds.
- **No CIs on FID/FVD**, as above.
