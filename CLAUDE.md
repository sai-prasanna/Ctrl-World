# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

**Goal: replicate VLAW (arXiv 2602.12063) on ABC-130k.** VLAW is the iterative
co-improvement of a VLA policy and the world model; this repo is the world-model half.
Work order is the Ctrl-World pipeline first — pretrain the world model on ABC, score it,
then post-train on downstream tasks — before the policy-in-the-loop co-improvement loop.
Judge changes by whether they move that goal forward.

The code is the official Ctrl-World implementation (action-conditioned video world model
built on Stable Video Diffusion), plus an in-progress port to the bimanual **ABC-130k**
dataset on branch `abc-130k`. Upstream targets DROID (7-DoF Franka, 192x320); the port
targets ABC (14-D bimanual joints, 192x192). Both paths coexist — check `action_dim`/`width`
before assuming which one a script is set up for.

Known gap for VLAW on ABC: the policy-in-the-loop scripts (`rollout_interact_pi*.py`) are
DROID-specific — 7-DoF Franka forward kinematics and the joint-velocity-to-cartesian
action adapter — so a bimanual policy needs that layer rewritten, not just reconfigured.

## Commands

Training (per-device batch; effective batch = devices x batch x grad_accum, paper uses 64):

```bash
WANDB_MODE=offline accelerate launch --main_process_port 29501 scripts/train_wm.py \
  --dataset_root_path dataset_example --dataset_meta_info_path dataset_meta_info \
  --dataset_names abc_subset
```

Data prep for ABC (see readme.md section 4 for the full flow):

```bash
python dataset_example/select_abc_episodes.py --dry_run          # inspect tasks/hours
python dataset_example/select_abc_episodes.py --task_regex ... --max_hours 350 --output_path dataset_example/abc_subset
accelerate launch dataset_example/extract_latent_abc.py --episode_list ... --output_path ... --svd_path ...
python dataset_meta_info/create_meta_info.py --droid_output_path dataset_example/abc_subset --dataset_name abc_subset
```

`dataset_example/extract_latent.py` is the DROID equivalent of `extract_latent_abc.py`.

The LeRobot mirror those scripts read (`lerobot/abc_130k_v3_train`) letterboxes the 4:3
cameras into 224x224, so a quarter of every latent is encoded black. `extract_latent_abc_mcap.py`
reads the original release (`XDOF/ABC-130k`, one MCAP per episode) instead and writes
256x192 with no padding. The release mixes camera rigs of different resolution and field
of view, so `fov_crop` frames them to a common view before the resize; check it before
trusting cross-episode geometry.

It runs in two stages because no single machine can do both halves: pulling from the Hub
needs a network, which only login nodes have, and the decode and VAE passes need cores and
a GPU, which only the offline boost nodes have.

```bash
./launch_download.sh <shard> <num_shards> <workers> [split]   # login node, once per shard
sbatch --array=0-3 --export=ALL,NSHARD=4 abc_gpu.sbatch      # boost: decode, then encode
sbatch meta_and_train.sbatch                                  # index, then start training
```

The download shards stage MCAP blobs into `mcap_cache` and stop once `--max_staged` blobs
are waiting, so `$WORK` cannot fill while the decoders lag. `abc_gpu.sbatch` decodes those
blobs to `videos/` and `annotation/` and then VAE-encodes `latent_videos/` on the same
allocation; it sweeps the episode list repeatedly, so it can start while downloads are
still arriving and `not staged` is a normal status rather than an error. Both halves shard
by a stride over the same sorted list, so shard *i* of one matches shard *i* of the other.

Everything is keyed on the annotation, written last and atomically, so it doubles as the
resume marker: rerunning any stage skips what is already done. The mp4s stay on disk for
evaluation. Unlike the mirror, the annotations also carry the commanded `action`.

Downloads run with xet enabled (`HF_XET_HIGH_PERFORMANCE=1`): it is worth about 20x, and
disabling it was once the pipeline's real bottleneck. Keep `--workers` low there and let
xet supply the parallelism, because a login node caps a user at 2048 processes.

Rollouts:

```bash
python scripts/rollout_replay_traj.py --task_type abc_replay --ckpt_path <ckpt>   # replay recorded actions
python scripts/rollout_key_board.py --task_type keyboard --keyboard lllrrr        # DROID only
python scripts/rollout_interact_pi.py ...                                          # DROID only, needs openpi + JAX
```

Evaluation (see `docs/evaluation.md`):

```bash
python3 scripts/selftest_eval_metrics.py --i3d_ckpt <i3d_torchscript.pt>   # CPU, no checkpoint, seconds
python scripts/make_eval_clips.py --val_dataset_dir <data> --out dataset_meta_info/<ds>/eval_clips_v1.json
python scripts/eval_video_metrics.py --ckpt_path <ckpt> --clips <clips.json> --out <out.json> [--fid --track ...]
```

There is no pytest suite and no linter config. `selftest_eval_metrics.py` is the only
test-like entry point; run it after touching anything in `clipeval/`.

## Architecture

- `models/ctrl_world.py` — `CrtlWorld` wraps the UNet, the CLIP text/image encoders, and
  `Action_encoder2`, which turns an action chunk (+ optional instruction) into the
  cross-attention conditioning. `frame_level_cond` controls whether conditioning is per
  frame or per clip.
- `models/unet_spatio_temporal_condition.py`, `models/pipeline_ctrl_world.py`,
  `models/pipeline_stable_video_diffusion.py` — forked diffusers components. Edit these
  rather than pinning a diffusers version.
- `dataset/dataset_droid_exp33.py` — `Dataset_mix` serves *both* datasets. It reads
  precomputed VAE latents (never raw video during training), normalizes state with
  `state_01`/`state_99` from `dataset_meta_info/<name>/stat.json`, and lays out history by
  frame offset (`skip_his = 4 * skip`) rather than by `history_idx`.
- All camera views are stacked vertically into **one** latent and denoised jointly, then
  split per camera before the VAE decode. Adding/removing a view changes latent height.
- `clipeval/` — model-independent scoring package. `Scorer` is an accumulator because
  FID/FVD are corpus-level. `pixel` (PSNR/SSIM/LPIPS), `distribution` (FID/FVD),
  `tracking` (CoTracker3 point tracks, binned by ground-truth displacement), `regions`
  (mask-based, WoW-World-Eval style). Each optional `add()` input unlocks more metrics;
  `results()` reports what was skipped instead of failing.
- `scripts/eval_video_metrics.py` owns the rollout and checkpoint loading; `clipeval` knows
  nothing about Ctrl-World. Keep that boundary.

## Config

`config.py` (`wm_args` dataclass) holds training, model, and rollout defaults; `config_eval.py`
is a near-duplicate frozen on the DROID/paper settings. Every entry script builds `wm_args`,
then `merge_args(cfg, cli_args)` overlays non-`None` argparse values. Adding a knob means
adding it to the dataclass *and* to the script's `ArgumentParser` with `default=None`.

`__post_init__` is a big `task_type` switch that sets `val_dataset_dir`, `val_id`,
`start_idx`, and `instruction` per rollout task. Add new eval sets there, not in the script.

Values that silently change results: `down_sample` (DROID 15 Hz -> 3, ABC 30 Hz -> 6),
`num_inference_steps`, `guidance_scale`, `history_idx`. The `history_idx` in `config.py`
(`[0,0,-12,-9,-6,-3]`) is only read by the policy-in-the-loop scripts; the evaluation and
`rollout_replay_traj.py` pin `[-6,-5,-4,-3,-2,-1]`, the evenly spaced `skip=1` history
that pairs with the consecutive frames a rollout predicts.

## Outputs and provenance

Everything a run produces goes to `outputs/{exp_id}_{exp_name}/` — `model/`, `samples/`,
`rollout/`, `figures/`, `eval/`. Set `exp_id`/`exp_name` in `config.py` or pass
`--tag <exp_id>_<exp_name>`. `outputs/` is gitignored, except eval JSON, which is
force-added so `docs/experiments.md` cannot drift from the numbers it quotes.

`docs/experiments.md` is the results log (one section per evaluated run, numbers copied
from the eval JSON); `docs/evaluation.md` is the protocol and the reasoning. Update
`experiments.md` whenever a new checkpoint is scored.

## Cluster (Leonardo)

`train.sbatch` and `rollout.sbatch` are the Slurm entry points; `$ROOT=$WORK/sraman00/ctrlworld`
with a prestaged venv, `HF_HOME`, and `HF_HUB_OFFLINE=1`. Compute nodes have **no internet** —
anything that downloads (HF data, LPIPS/Inception/I3D/CoTracker weights) must run on a login
node first. `scripts/eval_leonardo.sh` encodes that split: `setup`, `clips`, and
`tracker_setup` are login-node commands; `run` and `noisefloor` run inside a job.
The `cluster` skill handles submission and log fetching.

## Style

`plans/` holds design docs for in-progress work — read the relevant one before extending a
half-built subsystem (`clipeval-package.md`, `object-region-metrics.md`). Docs and prose in
this repo follow the Google developer documentation style guide (`google-dev-style` skill).
Comments here explain *why* a value or deviation exists, often citing the paper; match that
density rather than annotating mechanics.
