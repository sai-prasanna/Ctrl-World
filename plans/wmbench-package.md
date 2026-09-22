# Plan: wmbench, a WorldArena-style benchmark for Ctrl-World on ABC-130k

Status: built, 2026-09-22. M0-M6 verified locally; M1 and M2 also verified on Leonardo
against the `abc_mcap` checkpoints, M5 and M6 on real data still wait there, and M7 waits
on sim post-training.
Owner: sai-prasanna
Branch: `abc-130k`, plus the sibling repository `~/Desktop/wmbench` (HEAD `a0d4784`)

## Goal

VLAW (arXiv 2602.12063) needs a world model that can stand in for the environment while a
policy improves. Before the co-improvement loop, the world model has to be scored on more
than PSNR, SSIM, LPIPS, FID, and FVD, which is all `clipeval/` gives. WorldArena
(arXiv 2602.08971, github.com/tsinghua-fib-lab/WorldArena) is the reference for embodied
world-model evaluation: 16 video metrics in six sub-dimensions, plus a policy-in-the-loop
track whose figure of merit is the correlation between success judged inside the world
model and success measured in a simulator. Ctrl-World is one of its 14 evaluated models
(EWMScore 59.70, policy-evaluator r = 0.986 on RoboTwin).

WorldArena itself does not run here. It has no model interface — it scores flat
directories of 640x480 mp4s plus a `summary.json`, and its dataset assumptions are
hardcoded (two arms split at `x < 320`, 640x480 in eight places, path-depth parsing,
RoboTwin-calibrated normalization bounds). Its `video_quality/` directory carries no
license and vendors CC-BY-NC code, and needs three mutually incompatible conda
environments. Its policy track knows only RoboTwin 2.0 and pi0.5.

ABC-130k (github.com/amazon-far/abc) ships what the policy track needs: released policies
(`DiTInferencePolicy.infer(obs) -> (30, 14)`), a MuJoCo simulator whose success evaluators
are pure functions of `qpos`, and per-task sim finetunes with published success rates.

So: a standalone benchmark package that is agnostic to both model and dataset. `wmbench`
defines the interfaces and ships everything that needs neither a model nor a dataset. Two
thin implementations live in this repository — `abc130k` gains an `EpisodeSource`,
`bridges/` implements `WorldModel` for Ctrl-World and wires the ABC policy loop.

## Decisions taken

| Question | Decision |
|---|---|
| WorldArena code | Case by case, see [Metric provenance](#metric-provenance). Copy from the licensed upstreams WorldArena itself vendors (VBench Apache-2.0, EvalCrafter, RAFT BSD-3, DINO Apache-2.0, JEDi); write the WorldArena-specific glue; never copy from `video_quality/` (no license) or from AMT (CC-BY-NC). |
| Repository | `wmbench` is its own git repository, consumed here as a pinned dependency. `abc130k` implements its protocols structurally and never imports it. |
| Metric scope | All 16 WorldArena metrics, one environment. Where a backbone cannot be installed there, substitute and record the substitution. |
| `clipeval/` | Untouched. Its pixel and distribution code is copied into `wmbench` so numbers stay comparable with `docs/experiments.md`. |
| Policy-in-the-loop, phase A | World-model-only rollout of `abc_dit_xl_200k`, proprioception dead-reckoned from commanded joint targets, success from a local Qwen3-VL judge against the ground-truth clip. No simulator. |
| Policy-in-the-loop, phase B | Hybrid: MuJoCo owns state and `qpos` scoring, the world model owns pixels. Blocked on sim post-training. |
| Environments | `venv` (torch 2.7.1, diffusers) serves Ctrl-World inference only. `venv_bench` (torch 2.11+cu128) holds the metrics, the judge, and the ABC policy. The two talk over a Unix socket. |
| LeRobot v3 | Step 1 here: `abc130k` writes the existing 5 Hz cropped extraction as LeRobot v3 with its provenance, and `wmbench` gets a generic v3 source. Step 2, a native 30 Hz re-extract, is a separate plan; see [Data facts](#data-facts-for-the-later-native-re-extract). |
| Dataset versus model | Sampling rate, resolution, and crop are the model's `OperatingPoint`, declared by the model and applied to ground truth by the harness. The dataset definition holds recording-level facts only. |
| Normalization | Per-dataset 1st and 99th percentile bounds fitted at the operating point being scored. Raw values always reported; normalized only with `--norm-bounds`; the aggregate is off by default and is not called EWMScore. |
| Trajectory accuracy | Implemented behind the registry, requiring `masks`; no mask provider shipped. `plans/object-region-metrics.md` measured object IoU 0.22 on ABC wrist views, and WorldArena's left/right split is RoboTwin geometry. |

## Layout as built

```
wmbench/
  src/wmbench/
    core/          episode, worldmodel, policy, judge, registry, manifest, clips,
                   stats, frames          # numpy and PyAV only
    metrics/       scorer, pixel, distribution, motion, smoothness, rife, consistency,
                   quality, depth, jepa, semantic, trajectory, normalize
    judges/        vlm_local + prompts/{worldarena_triad,success,caption}.txt
    sources/       lerobot_v3, synthetic (SyntheticEpisodeSource + StaticWorldModel)
    datasets/abc130k/  profile, proprio, policies, eval_clips_v1.json, instructions.json
    ipc/           protocol, wm_server, wm_client
    policy_loop/   env (WorldModelEnv, Cadence, Proprio), correlation
    plugins.py     entry points `wmbench.sources` and `wmbench.world_models`
    cli.py         clips | rollout | score | fit-bounds | export-worldarena | serve |
                   judge | correlate
    selftest.py    109 CPU checks, exit code is the failure count
  tools/           convert_eval_clips_v1.py, parity_worldarena.py
```

In this repository:

```
abc130k/src/abc130k/bench_source.py    AnnotationSource: EpisodeSource over the
                                       annotation layout, 5 Hz, 256x192,
                                       preprocess_id ctrlworld_fovcrop_v1
abc130k/src/abc130k/lerobot_export.py  annotation layout -> LeRobot v3, one dataset per
                                       task, provenance in meta/info.json
bridges/ctrlworld_wm.py                CtrlWorldModel: the only code importing models/
bridges/abc_policy_run.py              venv_bench client: ABC's rollout driver against
                                       the world-model server
scripts/wmbench_rollout_ctrlworld.py   rollout -> manifest, in venv
scripts/wmbench_wm_server.py           the same model behind a Unix socket
scripts/wmbench_leonardo.sh            setup | run | check
jobs/wmbench_{rollout,score,judge,policy,lerobot_export}.sbatch
```

`abc130k` holds no tensor library and never imports `wmbench`: its `DatasetProfile`,
`SourceProvenance`, and `Episode` are duplicated field for field, because the protocols are
structural and the dependency has to point one way only. `abc130k/__init__.py` resolves
names lazily through PEP 562 for the same reason — a scoring environment installs the
reading half and must not fail on an MCAP reader it never calls.

## Core abstractions as built

Three declarations keep dataset facts, storage facts, and model facts apart.

`DatasetProfile(name, view_names, view_groups, control_hz, state_dim, state_layout)` is the
recording. ABC: views `top, left_wrist, right_wrist`; groups `third_view=[top]`,
`wrist_view=[left_wrist, right_wrist]`; 30 Hz; 14-D
`[left_j1..6, left_grip, right_j1..6, right_grip]`. No fps, no resolution — neither is a
property of the robot.

`SourceProvenance(fps, geometry, preprocess_id)` is what one copy on disk holds. This
extraction: 5 Hz, `256x192`, `ctrlworld_fovcrop_v1`.

`OperatingPoint(fps, height, width, preprocess_id, preprocess)` is what the model consumes
and emits. `gt_window` reconciles the two: when the source is raw, it subsamples to the
model's rate and applies the model's preprocess; when the source has preprocessing baked
in, it can serve only a model declaring the same id, and anything else raises. The harness
cannot undo a crop it did not perform, and comparing a model against pixels it never saw
would report the difference as model error.

`EpisodeSource` is `profile`, `provenance`, `list_episodes`, `load`, `frames`, and
`intrinsics`, with times in seconds. Latents are deliberately absent from the protocol; the
Ctrl-World bridge reads `latent_videos/*.pt` itself.

`WorldModel` is `operating_point`, `frames_per_step`, `reset(contexts) -> handle`,
`step(handle, actions) -> {view: (B, frames_per_step, H, W, 3) uint8}`, and `close(handle)`.
Batched, because a 256-clip evaluation one clip at a time is GPU-days, and handle-based so
one instance can serve several rollouts and a remote implementation can hold the state on
the other side of a socket.

Two optional attributes were added during the build, because `replay` cannot derive them
and a wrong guess does not raise. `actions_per_step` defaults to `frames_per_step + 1`, the
conditioning frame's state plus one per predicted frame; Ctrl-World sets it to `pred_step`.
`context_frames` defaults to 1. A model that wants more history than the clip start has in
front of it gets the clip's first frame repeated, which is what a rollout starting at t=0
has to do anyway.

Eval clips are `(episode_id, start_s, duration_s, task, instruction)`, and
`clip_id` is `{episode_id}@{start_s:g}s`. Seconds rather than frame indices throughout: an
id built from a frame index means a different window after a re-extraction at another rate.
`tools/convert_eval_clips_v1.py` converts the committed v1 list with
`start_s = start_idx / 5`, preserving the v1 identity by episode and start second;
`EvalClip.key(fps)` still spells the filename stem `<episode>_<start index>`, which is what
`dump_clip` already writes.

`MetricSpec` declares `level` (series, scalar, corpus), `requires` (a subset of `pred`,
`gt`, `gt_full`, `instruction`, `instruction_variants`, `masks`, `corpus`), `weights`,
`per_view`, and `higher_is_better`. The scorer reads the declarations and decides what can
run, recording `skipped[name] = reason` otherwise. Three additions to the plan came out of
writing the metrics:

- `Resources` and `@resource(name)`. Backbones are built lazily and shared, and a factory
  raises `ResourceUnavailable(reason)` when its weights are missing. Weights are checked
  once, before the first clip is read, so a metric whose backbone cannot be had on this
  node says so up front.
- `MetricInput.cache`, per clip and per view. RAFT flow is the reason: four metrics and the
  anti-static penalty want the same consecutive-frame field, and computing it five times
  would dominate the run. `MetricInput.views_pred` and `views_gt` carry every camera for a
  metric declaring `per_view=False`, which the scorer calls once, after the last view has
  been added.
- `*_gt` columns and an `all_views` section. Every no-reference metric — one requiring
  nothing but `pred` — also runs on the clip's own ground-truth frames. A MUSIQ score of
  0.42 means nothing until you know what real footage scores at this resolution and frame
  rate. Cross-view consistency has no view to sit under, and pooling it over view groups
  would double-count the cameras it already read, so it reports under `all_views`.

Manifest v1 is `manifest.json` plus `clips/<episode>_<start>.npz`, whose keys are exactly
what `scripts/eval_video_metrics.py::dump_clip` writes. A frame dump that script already
produced becomes a valid manifest as soon as a `manifest.json` is written beside it.
Frames are stored as uint8 npz rather than video: H.264 artifacts land differently on
ground truth and on prediction, and that difference would be read as model error.

## Metrics

Every metric runs per camera and is pooled per view group, and every no-reference metric is
also computed on the ground-truth frames of the same clip.

| Metric | Requires | Backbone in `venv_bench` | Deviation from WorldArena |
|---|---|---|---|
| `psnr`, `ssim`, `lpips`, `psnr_static_first`, `psnr_static_round` | pred, gt_full | LPIPS AlexNet | None. Copied from `clipeval`. |
| `fid`, `fvd` | corpus | Inception-v3, TorchScript stylegan-v I3D | None. Copied from `clipeval`. |
| `dynamic_degree` | pred | torchvision `raft_large` (C_T_V2) | WorldArena's sigmoid and its `tau = 6 px at 256 px` kept, resolution-relative. RAFT weights substituted for `raft-things.pth`. |
| `dynamic_degree_raw` | pred | same RAFT | Not in WorldArena. The sigmoid saturates, so a model that hallucinates violent motion and one that tracks the real thing are indistinguishable in `dynamic_degree`; this keeps the tail magnitude in pixels, which `*_gt` gives a scale. |
| `flow_score` | pred | same RAFT | EvalCrafter's mean magnitude, backbone substituted. |
| `photometric_consistency` | pred | same RAFT | WorldArena uses SEA-RAFT and folds the anti-static penalty in place; here the penalty is a separate column. Registered under this name; WorldArena's code calls it `photometric_smoothness`. |
| `motion_smoothness` | pred | RIFE v3.6 HD (MIT) | WorldArena uses VFIMamba, which needs a compiled `mamba-ssm`; VBench uses AMT, which is CC-BY-NC. The interpolator is part of the definition, so these values compare checkpoints scored here and not WorldArena's table. |
| `subject_consistency` | pred | DINO ViT-B/16 via `transformers` | Pooling is per clip, WorldArena's distributed path, not its single-GPU per-frame mean. Penalty separated. |
| `background_consistency` | pred | CLIP ViT-B/32 via `transformers` | Same two. |
| `cross_view_consistency` | pred, gt | DINO | Not in WorldArena, and lower is better. Mean cosine between the top view and each wrist view over time, minus the same on ground truth, so scenes whose cameras simply look alike cancel. Addresses the view drift `docs/experiments.md` records. |
| `imaging_quality` | pred | MUSIQ (SPAQ head) via `pyiqa` | None. VBench's division by 100 kept. |
| `aesthetic_quality` | pred | CLIP ViT-L/14 + LAION linear head | None. |
| `depth_accuracy` | pred, gt | Depth-Anything-V2-Small-hf | Formula reproduced exactly; the hardcoded path parsing that finds its ground truth is dropped, because the harness already pairs the frames. |
| `jepa_similarity` | pred, gt, corpus | V-JEPA ViT-H/16 + SSv2 probe | MMD copied from JEDi. `SIMILARITY_ALPHA = 0.4` applied to the x100 MMD, which is the paper's 40 on the raw MMD — the same number on two scales. The `vjepa` model code is CC-BY-NC, so it is imported, never vendored, and the metric skips with that reason when it is absent. |
| `interaction_quality`, `perspectivity`, `instruction_following` | pred, instruction | Qwen3-VL-8B-Instruct, local | One prompt, not three. WorldArena's triad produces all three scores in a single answer, so they are conditioned on each other; asking for one alone is a different measurement. `--rubric triad` runs the model once for all three. |
| `semantic_alignment` | pred, gt | captioner + CLIP ViT-B/16 text | Captioner is a resource rather than an import, so the metric skips until one is configured. |
| `action_following` | pred, instruction_variants | CLIP ViT-B/32 | None. Needs three rollouts per clip under paraphrases; last priority for an action-conditioned model. |
| `trajectory_accuracy` | pred, gt, masks | none shipped | nDTW is `exp(-DTW / (|reference| * d))`, the vision-and-language-navigation definition, bounded in [0, 1]. WorldArena's `1 / (dtw / len)` is an unbounded inverse mean cost needing an empirical maximum before it can be read. No mask provider ships. |

### Metric provenance

| Metric | Source of code | Why |
|---|---|---|
| pixel, distribution, stats | copied from `clipeval/` | Ours; the numbers must stay comparable. |
| `subject_consistency`, `background_consistency`, `aesthetic_quality`, `imaging_quality`, `dynamic_degree` | copied from VBench (Apache-2.0) | Licensed, and where WorldArena took them. |
| `flow_score` | copied from EvalCrafter | Licensed upstream of WorldArena's version. |
| RAFT wrapper | torchvision `raft_large` | Avoids vendoring RAFT. The parity table below measures the weight swap. |
| `jepa_similarity` | MMD copied from JEDi; V-JEPA loaded with plain torch | JEDi is licensed; only its torch pin is a problem. |
| `motion_smoothness` | written, over RIFE (MIT) | The licensed alternatives need a CUDA build or are CC-BY-NC. |
| `photometric_consistency`, `depth_accuracy`, penalty, normalization, nDTW | written | WorldArena-specific glue, a few dozen lines each, and the only copies are in the unlicensed `video_quality/`. |
| VLM rubrics, caption prompt | prompt text copied from the WorldArena paper | The wording is the metric definition, and the paper is the citable source. |
| `semantic_alignment`, `action_following` | written | Thin: a caption call plus a CLIP cosine, and a pairwise CLIP diversity. |

Every copied file carries a header naming the upstream repository, commit, and license.

### Bounds and the anti-static penalty

`wmbench fit-bounds` reads a manifest three ways — the recorded footage, the first real
frame held for the whole rollout, and each round's conditioning frame held for that round —
and takes the 1st and 99th percentiles per metric. Zero on the normalized scale then means
"no better than a frozen frame here" and one means "as good as this dataset's own footage",
both measured at the resolution and frame rate being scored. WorldArena's single hardcoded
table cannot be read at another geometry: `flow_score` is a pixel count, so halving the
frame height halves it. Bounds are keyed by dataset and by
`operating_point_id` (`5hz_256x192_ctrlworld_fovcrop_v1`), and live with the model run
under `experiments/<tag>/eval/`, not in the benchmark.

The anti-static penalty is additive. WorldArena multiplies the three consistency metrics by
`dynamic_degree` in place whenever it is at or below 0.1213, which makes a faithful still
scene and a dead model indistinguishable after the fact. `normalize.penalize` writes
`<metric>_penalized` beside the raw column instead, records whether the gate fired, and
takes the threshold from the bounds file when one sets it. The 0.1213 has no derivation
anywhere in WorldArena's source and is carried over as an empirical fit.

The aggregate is a plain unweighted mean of normalized, direction-corrected metrics over a
list the caller supplies. It is off unless asked for and is not called EWMScore: no
implementation of EWMScore exists in WorldArena's repository and its weights are not
recoverable from the code.

## Policy-in-the-loop

Phase A is built. `wmbench.policy_loop.env` owns the arithmetic that nothing else checks:
`Cadence.derive(control_hz, fps, frames_per_step)` gives 6 control ticks per predicted
frame and 24 per `step` for ABC at 30 Hz through a 5 Hz model emitting 4 frames. Getting
that wrong does not raise — the policy just sees stale pixels, or the model is conditioned
on states sampled at the wrong rate, and the rollout looks plausible either way.
`WorldModelEnv` knows no view names, no joint layout, and no gripper convention; a
`Proprio` supplies the one thing a world model needs that the recording no longer provides,
the state that follows an action the policy invented.
`wmbench.datasets.abc130k.policies.AbcWorldEnv` presents that loop as ABC's `SimTaskEnv`,
renaming `top/left_wrist/right_wrist` to `top/left/right`, so
`abc_minimal.eval_policy.rollout_worlds` drives it unchanged with RTC
(`rtc_prefix_length=4`, `rtc_inference_lead_steps=4`, `execute_chunk_dim=15`,
`diffusion_steps=10`). Running the model through the simulator's own driver rather than a
reimplementation of it is what makes a world-model success rate and a simulator success
rate the same measurement.

Three differences from a simulator are deliberate and visible in the output.
`evaluate()` reports no success, because there is no `qpos` inside a world model;
`wmbench judge --rubric success` scores the rendered video against its ground-truth clip
afterwards and writes the answer into the `success`, `final_success`, and `final_task_eval`
fields of each world record, where `build_summary` reports it like sim success, plus a
`judged_success` block naming the model that wrote it. `randomization` carries the
evaluation clip rather than a scene randomization, since a world-model rollout is seeded by
a recorded clip. The observation carries `tick`, which ABC's own env does not, and which
the validation policies use to stay aligned with the chunking under RTC.

The horizon is WorldArena's 1.2x the recorded length, capped at the 12 rounds
`docs/experiments.md` has calibrated. The cap does not divide evenly: `rollout_worlds`
counts in chunks of 15 control actions and the loop counts in rounds of 24 ticks, so a
12-round horizon becomes 19 chunks, which reach 11 full rounds. Rounding up would run the
loop past its horizon, where it freezes, and a judge shown a frozen tail reads it as a
scene where nothing happens; losing a fraction of a round is the cheaper error. The
manifest records the rounds reached, not the horizon requested, because `wmbench score`
derives every frame index from it.

The policy sees frames in the world model's space — fov-cropped 256x192 — which no ABC
policy trained on; ABC trained on 224x224 letterboxed 4:3 and 16:9 frames. That gap is a
property of the model under test, and the control experiment for it feeds the policy
fov-cropped and ABC-letterboxed frames from the same real episode and compares action error
against the recorded action.

Validation ordering, on 16 clips: ground-truth-action replay through the world model
(`--policy echo`) >= `abc_dit_xl_200k` >> random actions.

Phase B, after sim post-training, keeps the same driver: `SimTaskEnv` supplies `state` and
`evaluate`, `WorldModelEnv` supplies images, and the world model is conditioned on true sim
joints. The correlation study grades variants of the released finetune
(`diffusion_steps` in {10, 4, 2, 1}, action noise sigma in {0, 0.02, 0.05} rad), re-measures
true success in `abc_sim` (num_worlds 20, seeds 20260511 and 20260512, one physics backend),
and `wmbench correlate` reports Pearson r and Spearman rho with bootstrap intervals. Pearson
asks whether the world model's success rate is an affine image of the simulator's, which a
calibration would need; Spearman asks only whether the ordering survives, which is what
choosing between checkpoints needs and is the weaker, more defensible claim.

## Cluster

`venv_bench` is `uv venv` on Python 3.12 with torch 2.11+cu128, the pinned `amazon-far/abc`
checkout under `$CTRLWORLD_ROOT/third_party/abc`, and
`wmbench[metrics,judge,lerobot,jepa]`. ABC installs with `--no-deps` and an explicit
dependency list, because its own dependencies pull `mujoco-warp`, `warp-lang`, `viser`, and
`mjviser`, and warp compiles kernels at import time against a CUDA toolkit the compute
nodes do not expose. `mujoco` itself is still needed in phase A: `abc_minimal.eval_policy`
imports `abc_sim.randomization.core` at module scope, which imports `mujoco`.
`require_mjwarp()` is a function and only the simulator path calls it.

`scripts/wmbench_leonardo.sh` splits the work the way `eval_leonardo.sh` does. `setup` runs
on a login node — compute nodes have no internet and sixteen metrics pull sixteen backbones
— and `check` reports torch, CUDA, and which weights landed. `run <args>` is what the jobs
invoke: it exports the environment variables behind every `--*-ckpt` flag, so no caller has
to remember where a weight went. Until `wmbench` has a git remote, `setup` mirrors the
laptop checkout to `$CTRLWORLD_ROOT/wmbench` with rsync, driven by `WMBENCH_RSYNC_FROM`,
and installs it editable.

Five Slurm entry points, all following the self-locating `BASH_SOURCE` convention and
keeping their `#SBATCH` headers:

| File | Environment | Does |
|---|---|---|
| `jobs/wmbench_rollout.sbatch` | `venv` | Rolls a checkpoint over the clip list and writes a manifest. The only half needing a GPU and Ctrl-World. |
| `jobs/wmbench_score.sbatch` | `venv_bench` | Scores that manifest, fits bounds, normalizes. |
| `jobs/wmbench_judge.sbatch` | `venv_bench` | Qwen3-VL rubrics over a manifest, and success over a policy run. |
| `jobs/wmbench_policy.sbatch` | both | Starts the world-model server in `venv`, runs `bridges/abc_policy_run.py` in `venv_bench`, over a socket in `$TMPDIR`. The only job needing both. |
| `jobs/wmbench_lerobot_export.sbatch` | `venv_bench` | CPU-only LeRobot v3 export, so it does not queue behind training. |

## Milestones and verification

| Milestone | Status |
|---|---|
| M0 skeleton, registry, manifest, copied pixel and distribution metrics | Done. `python -m wmbench.selftest` passes 109 checks on CPU. `wmbench rollout --world-model static` on the synthetic source writes a manifest whose scored PSNR equals `psnr_static_first` to the last bit — two code paths sharing nothing but the frames, so an off-by-one in the rollout, the manifest, or the scorer breaks the equality while leaving every number plausible. `scripts/selftest_eval_metrics.py` still passes untouched. |
| M1 Ctrl-World bridge | Done. Locally on a DROID checkpoint, 4 clips: predicted latents bit-identical to `scripts/eval_video_metrics.py`, per-view PSNR, SSIM, and LPIPS agreeing to 1.4e-7. On Leonardo, `abc_mcap` step 200000 over all 256 clips: per-view PSNR within 0.0021 dB of the eval script's own rollout at a different `decode_chunk_size`, and no clip further than 0.076 dB. Same frames, different metric code: the five clipeval numbers reproduce to 6e-9 on PSNR and SSIM and 5.8e-6 on LPIPS. FID and FVD do not survive the decode-schedule change; see the parity section of `docs/experiments.md`. |
| M2 metrics | Built and checked against cases whose answer is known in advance: a frozen clip scores dynamic_degree 0.0076 and flow 0.029 px, a translating pattern 0.9628 and a raw magnitude of 6.19 px for a 6 px shift, identical frames score subject and background consistency 1.000000, a prediction identical to the truth has cross-view drift 0.00, nDTW of a trajectory against itself is 1, and MMD of a feature set with itself is 0. The parity table below covers the backbone substitutions. Done on Leonardo: the full 256-clip run on `checkpoint-200000.pt` and `checkpoint-150000.pt`, the paired comparison between them, and the `docs/experiments.md` entry. Twenty-two of the twenty-six registered metrics score; `semantic_alignment` needs a captioner, `action_following` needs instruction variants, and `trajectory_accuracy` and `trajectory_dtw_len` need masks. The fitted bounds are not run. |
| M3 WorldArena export and world-model server | Export and socket built; `replay()` through the socket reproduces the in-process frames in the selftest's IPC check. Acceptance by WorldArena's `preprocess_datasets.py` is untested. |
| M4 LeRobot | Writer and reader built. The round trip — export the val split, then score through the LeRobot source and through the annotation source on the same clips — waits on Leonardo, where the data is. |
| M5 phase A policy loop | Built. Cadence, horizon, and chunk arithmetic covered by the selftest's policy-loop checks against a stub model. A real run of `abc_dit_xl_200k` through the server waits on Leonardo and on the ABC checkpoint being staged. |
| M6 judge | Built, including `--rubric triad` and the write-back into `summary.json`. Temperature-0 self-agreement and the GT-replay >= policy >> random ordering wait on a GPU with 24 GB or more; the 8B judge in bfloat16 is about 17 GB of weights before activations. |
| M7 phase B | Blocked on sim post-training of the world model. The env swap is designed, not written. |

### Parity against the WorldArena clone

`tools/parity_worldarena.py` scores the same frames with both implementations. The ratio
`wmbench / WorldArena`, top view, 4 real clips:

| Metric | Ratio |
|---|---|
| `dynamic_degree` | 0.984 - 1.038 |
| `flow_score` | 0.989 - 1.027 |
| `subject_consistency` | 0.998 - 1.001 |
| `background_consistency` | 1.000 - 1.006 |
| `imaging_quality` | 1.000 |
| `aesthetic_quality` | 0.975 - 1.030 |

Exact equality is not the bar and would mean the backbones had not in fact been swapped.
The flow metrics carry the widest spread, which is where the RAFT weight substitution sits.

## Data facts for the later native re-extract

On Leonardo, `$CTRLWORLD_ROOT/data/abc_mcap` holds 15,477 train and 224 val episodes, three
mp4s each at 256x192, 5 Hz, fov-cropped, plus latents: 365 GB. The whole `ctrlworld` root is
2.0 TB. The project `$WORK` quota is 9.3 of 10 TB used, and fast scratch 751 GB of 1 TB. The
raw MCAP release is 23.1 TB.

Resolution is a station label, and all three cameras of a station share it (728 of 728
shards): 640x480 for about 75% of episodes, 1920x1200 for about 14%, 848x480 for about 11%,
and three rarer sizes under 1%. Task predicts the rig only weakly (Cramér's V 0.38);
`put_the_screwdriver_in_the_bin` is 61% ZED-X. The mean episode is 99 s corpus-wide and 66 s
for the rigid 11.

Native 30 Hz, three cameras, AV1 CRF 30, measured from public ports to within about 10%:
1.06 MB/s at 640x480, 1.28 at 848x480, 5.64 at 1920x1200. Rigid-11 train is about 1.65 TB,
rigid-11 val about 0.03 TB, and the full corpus about 22 TB, which is no saving over keeping
the MCAPs. A padded 848x480 canvas cuts the 1920x1200 bucket roughly fourfold, bringing
rigid-11 train near 1.2 TB. Neither figure fits the 0.7 TB of free quota, so the re-extract
needs a cleanup or a quota request first, which is why it is not in this plan.

## Open items

- `venv_bench` needs `mujoco` for phase A even though nothing simulates, because
  `abc_minimal.eval_policy` imports `abc_sim.randomization.core` at module scope. Add it to
  the explicit install list in `scripts/wmbench_leonardo.sh setup` when the first policy run
  fails on the import.
- `wmbench` has no git remote, so `requirements.txt` carries a commented placeholder instead
  of a pin, and `setup` mirrors the laptop checkout with rsync. Push the repository, then
  replace both with `wmbench @ git+...@<sha>`.
- Leonardo driver compatibility with cu128 wheels. The fallback is cu126 builds of the same
  torch, which breaks ABC's exact pin but not its code.
- `lerobot[dataset]==0.6.1` forces `av<16`, which is why `abc130k.video.write_mp4` passes
  a `Fraction` rate: PyAV 15 refuses a float. Watch that constraint when the LeRobot pin
  moves.
- The decode chunking gap. `bridges/ctrlworld_wm.py` decodes a round's 4 frames at a time
  because a streaming `step` cannot wait for a chunk boundary three rounds later, while
  `eval_video_metrics.py` decodes the whole 48-frame rollout in chunks of 7. SVD's temporal
  decoder mixes across a chunk, so the pixels differ by 1 to 2.5 grey levels on average and
  PSNR by up to 0.05 dB. Latents are bit-identical either way. The manifest records
  `decode_chunk_size`; see `docs/evaluation.md` for what stays comparable.
- No mask provider for `trajectory_accuracy`, and none planned until segmentation on ABC
  wrist views is worth more than the 0.22 IoU `plans/object-region-metrics.md` measured.
