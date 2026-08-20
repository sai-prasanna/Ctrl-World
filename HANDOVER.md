# Handover: Ctrl-World on ABC-130k (rigid subset)

The 10,000-step training run on the `abc_rigid` subset is complete, and its results are
written up in `outputs/0002_abc_rigid/README.md`. This file covers what the run produced,
what is worth doing next, and the cluster details you need to do it.

## Run summary

Job 52904703 trained on one Leonardo node with 4 A100 GPUs, on the `boost_usr_prod`
partition.

| Property | Value |
|---|---|
| State | `COMPLETED`, exit code 0:0 |
| Steps | 10,000 |
| Wall time | 13 h 09 m at 4.71 s/it |
| Loss | 16.2 at step 155, 12.1 at step 10,000 |
| Dates | 2026-08-19 to 2026-08-20 |

Three replay rollouts on held-out validation episodes (jobs 53229682, 53231044, and
53231046) hold scene structure across all 49 frames, with no collapse and no object loss.
The 0001 overfit run, by contrast, drifted by frame 24 and collapsed by frame 48. For the
per-episode numbers and the reason raw pixel error doesn't compare across episodes, see
`outputs/0002_abc_rigid/README.md`.

Treat this as evidence that the model rolls out stably, not that it is accurate enough for
downstream policy evaluation. The evaluation covers 3 episodes with a pixel-level metric
and no task-success measure.

## Artifacts

Checkpoints stay on Leonardo at
`$WORK/sraman00/ctrlworld/repo/outputs/0002_abc_rigid/model/`, at steps 2500, 5000, 7500,
and 10,000. Each step writes two files:

- `checkpoint-<step>.pt`, 9.3 GB. A bare `state_dict`, and what the rollout scripts load.
- `trainstate-<step>.pt`, 12.2 GB. Optimizer, scheduler, and step, for `--resume`.

Videos, figures, and the writeup live in `outputs/0002_abc_rigid/` in the repo, which is
gitignored so the binaries stay untracked.

## What to do next

Consider these in any order. None of them blocks the others.

1. **Train longer.** The paper specifies 100,000 steps. Resume from step 10,000 with
   `MAX_STEPS=100000 RESUME=--resume sbatch train.sbatch`.
2. **Measure the 4-node speedup.** See the multi-node section.
3. **Evaluate more episodes.** The validation set holds 145; the rollouts used 3.
4. **Try the per-frame subtask annotations.** ABC ships `language_events` covering about
   25% of episodes. This run conditions on the per-episode task string instead, matching
   how Ctrl-World conditions on DROID's instruction.

## Fixes to keep

`train.sbatch` and `rollout.sbatch` are tracked in this repo. Copy them to
`$WORK/sraman00/ctrlworld/` before submitting, because `sbatch` runs them from there.

- `rollout.sbatch` hardcoded `--val_dataset_dir` to `data/abc_smoke`, the garment smoke
  set from the 0001 overfit run, which silently evaluated against the wrong dataset. It
  now reads `${DATA:-$ROOT/data/abc_rigid}`. Keeping these files untracked is what let
  that survive, so make changes here and copy them across, not the other way round.
- Passing several validation IDs in one `sbatch --export` silently evaluates only the
  first. `sbatch` treats commas as its own separator, so
  `--export=ALL,CKPT=...,VALIDS=a,b,c` exports `VALIDS=a` and discards the rest. Submit
  one job per validation ID.

To roll out the final checkpoint on three validation episodes, run:

```bash
cd $WORK/sraman00/ctrlworld
CK=$PWD/repo/outputs/0002_abc_rigid/model/checkpoint-10000.pt
for v in $(ls data/abc_rigid/annotation/val/*.json | xargs -n1 basename | sed 's/.json//' | head -3); do
  sbatch --export=ALL,CKPT=$CK,VALIDS=$v rollout.sbatch
done
```

Each job takes about 8 minutes and writes to `rollouts/Rollouts_abc_replay/video/`. The
videos show ground truth on top and the prediction below, with the top, left wrist, and
right wrist views across.

## Training configuration

The target is exact replication of the Ctrl-World recipe, verified against both the paper
(arXiv 2510.10125) and the released repo.

| Setting | Value | Source |
|---|---|---|
| Learning rate | 1e-5, constant, no warmup | Paper and repo; the repo has no scheduler |
| Effective batch | 64 = 4 GPUs x batch 4 x grad_accum 4 | Paper: "2x8 H100, total batch 64" |
| Max steps | 10,000 | Released checkpoint is `checkpoint-10000.pt` |
| Gradient clip | 1.0, fp16 | Repo |

The sources disagree on step count. The paper says 100,000, the repo default was 500,000,
and the released checkpoint is at 10,000.

These deviations from upstream are deliberate, and required by the dataset: `action_dim`
14 for bimanual rather than 7, `width` 192 for square 192x192 frames rather than 192x320,
`down_sample` 6 for 30 Hz rather than 15 Hz, and `ckpt_path=None` to train from the Stable
Video Diffusion (SVD) initialization.

## Data

`abc_rigid` holds 11 rigid-body "sort into containers or bins" tasks from ABC-130k:
15,477 episodes, 285 hours, and 5,031,807 training samples against 46,000 validation
samples. Extraction is complete, with 180 GB of latents at `data/abc_rigid/` and no errors.

Conditioning uses the per-episode task string, of which there are 11.

For scale, the full ABC-130k dataset covers 201 tasks and 3,541 hours, or 12.4 times the
rigid subset. Extrapolating from this run, the full set needs roughly 2.2 TB of latents
and about 15 TB of raw downloads. The raw data exceeds the free space on `$WORK`, so
download, extract, and delete it in waves.

Adding data does not change training wall time, which depends on step count alone. At
10,000 steps the model sees 640,000 samples, or 1% of one epoch over the full set.

## Multi-node training

A 4-node attempt, job 52896891, did not respond: `accelerate launch` never spawned workers
on any node, and the GPUs stayed at 0%. Nobody found the root cause. Budget debugging time
before retrying, because the single-node path reaches the same effective batch.

If it works, 16 GPUs hold batch 64 with `grad_accum 1`, so each step runs one
forward-backward pass instead of four. Measured single-node time is 1.19 s per microbatch,
which suggests 1.3 to 1.7 s/it once the larger cross-node allreduce is included, or 3.6 to
4.7 hours for 10,000 steps. That estimate is unverified. To settle it, submit a 4-node job
for about 200 steps and read the actual s/it.

## Cluster access

`ssh leonardo` needs a smallstep certificate that expires within hours. When it fails with
`Permission denied (publickey)`, renew it:

```bash
step ssh login 'sai.raman@uni-tuebingen.de' --provisioner cineca-hpc
```

The command prints a single sign-on URL and starts a listener on `127.0.0.1:10000`. Open
the URL, and when the browser redirect to that address fails, `curl` the failed URL from
the same machine to complete the handshake. Then `ssh-add -l` shows `ECDSA-CERT`.

Two failure modes are worth knowing:

- A stale `step oauth` process holding port 10000 makes `step` fall back to a port the
  identity provider rejects, which surfaces as `Invalid parameter: redirect_uri`. Clear it
  with `pkill -x step`. Avoid `pkill -f 'step ssh login'`, which also matches the shell
  running it.
- The listener expires after a few minutes. Restart the login to get a fresh URL.

Pushing to GitHub needs an explicit credential helper:

```bash
git -c credential.helper='!gh auth git-credential' push origin abc-130k
```

## Cluster gotchas

- Compute nodes have no internet. Run anything that downloads on `lrd_all_serial`
  (login08 or login13). GPU jobs set `HF_HUB_OFFLINE=1`.
- `ffmpeg` is absent on compute nodes. The venv symlinks to the `imageio_ffmpeg` binary,
  and `train.sbatch` puts `venv/bin` on `PATH`. Keep that symlink.
- `create_meta_info.py` runs out of memory above about 30 GB without the `sample['states']`
  fix in commit b433b4a.
- Bulk downloads from Hugging Face hit HTTP 429. The downloader retries with backoff at 6
  workers.
- `sbatch` often returns "Socket timed out on send/recv". Check `squeue` before retrying,
  because timed-out submissions sometimes still land. This project has produced duplicate
  jobs twice.
- Step time oscillates between roughly 4.50 and 5.80 s/it, most likely from `$WORK`
  filesystem contention. It is not a degradation trend.
- Two jobs died with `CANCELLED by 134235`, the user's own uid, and no error. Ask before
  resubmitting after that.

## Branch

All work is on `abc-130k`, and pushed. Ask before merging to `main`.
