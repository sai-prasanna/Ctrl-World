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
  --dataset_root_path preprocessing --dataset_meta_info_path dataset_meta_info \
  --dataset_names abc_subset
```

Data prep for ABC — see `docs/data-pipeline.md` for the flow and its invariants:

```bash
python preprocessing/extract_latent_abc_mcap.py --dump_episode_files <out.json>  # login node
jobs/launch_download.sh <shard> <num_shards> <workers> [split]                   # login node
sbatch --array=0-3 --export=ALL,NSHARD=4 jobs/abc_gpu.sbatch                     # decode, encode
sbatch jobs/meta_and_train.sbatch                                                # index, train
```

Two facts about that pipeline change results silently. `preprocessing/rigid_tasks.txt`
records the task selection, a research decision, so the repo tracks it;
`abc_mcap_files.json` only lists the release, so regenerate it rather than copy it. And
`fov_crop` is what makes camera rigs of different field of view comparable — read it
before trusting cross-episode geometry.

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
  `regions`
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

All Slurm entry points live in `jobs/`, one per pipeline stage; the `_mcap` ones are the
live `abc_mcap` generation and `train.sbatch` / `rollout.sbatch` the superseded
`abc_rigid` one. `$ROOT=$WORK/sraman00/ctrlworld`
with a prestaged venv, `HF_HOME`, and `HF_HUB_OFFLINE=1`. Compute nodes have **no internet** —
anything that downloads (HF data, LPIPS/Inception/I3D weights) must run on a login
node first. `scripts/eval_leonardo.sh` encodes that split: `setup`, `clips`, and
`tracker_setup` are login-node commands; `run` and `noisefloor` run inside a job.
The `cluster` skill handles submission and log fetching, and is the only sanctioned way to
start a job: it snapshots the working tree at a commit and checks that SHA out on the login
node, so nothing depends on hand-editing `$ROOT/repo`.

### Running the entry points either way

Every `*.sbatch` file runs unchanged in three contexts: `sbatch <file>` from a checkout,
`cluster submit`, and `bash <file>` on a machine with no Slurm. Two conventions make that
work, so preserve them when you add an entry point.

Each file locates itself with `BASH_SOURCE` and `cd`s to the repo root — `jobs/..`, not its
own directory — instead of naming `$ROOT/repo`, because `cluster submit` stages the run in
`$WORK/runs/<project>/<runid>/code`. The training chain re-submits its own resolved path,
which keeps every link on one commit.

Keep the `#SBATCH` headers even though `cluster submit` writes its own sbatch, which makes
them dead for the *first* job. The chain resubmits with a plain `sbatch`, so every link
after the first takes its resources from them.

Anything that outlives a run lives outside the code snapshot, under overridable paths:

| Variable | Default | Holds |
|---|---|---|
| `CTRLWORLD_ROOT` | `$WORK/sraman00/ctrlworld` | venv, `HF_HOME`, everything below |
| `CTRLWORLD_DATA` | `$ROOT/data` | extracted videos, annotations, latents |
| `CTRLWORLD_META` | `$ROOT/dataset_meta_info` | generated `stat.json` and `*_sample.json` |

Set `CTRLWORLD_ROOT` to reproduce the pipeline anywhere else. The generated index has to sit
outside the repo whichever way you run: `train_sample.json` is around 490 MB, and a
`cluster submit` checkout is discarded when the job ends. Pass `--meta_root` to
`create_meta_info.py` to put it there. Checkpoints go to `$ROOT/outputs/<tag>` for the same
reason — a relative `outputs/` is empty in a fresh checkout, so `--resume` would silently
retrain from scratch. `scripts/train_wm.py --run_dir` sets that home.

## Style

`plans/` holds design docs for in-progress work — read the relevant one before extending a
half-built subsystem (`clipeval-package.md`, `object-region-metrics.md`). Docs and prose in
this repo follow the Google developer documentation style guide (`google-dev-style` skill).
Comments here explain *why* a value or deviation exists, often citing the paper; match that
density rather than annotating mechanics.
