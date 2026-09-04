#!/bin/bash
#SBATCH --job-name=abc_mcap_train
#SBATCH --account=AIFAC_S07_034
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=480G
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out

# 200k steps on the 256x192 re-extraction, extending the paper's 100k budget. No wall clock
# covers that, so a standalone job resubmits itself from the newest checkpoint until it
# gets there.
set -x
# $ROOT holds everything that outlives a run: the venv, the HF cache, the data, the
# generated index and the outputs. It defaults to the Leonardo layout but is overridable,
# so the same file reproduces the run on another machine or from a `cluster submit`
# checkout: CTRLWORLD_ROOT=/path/to/scratch sbatch|bash <this file>. A submitted run needs
# no argument -- `cluster submit` exports CLUSTER_SHARED_DIR for exactly this.
ROOT=${CTRLWORLD_ROOT:-${CLUSTER_SHARED_DIR:-$WORK/sraman00/ctrlworld}}
DATA=${CTRLWORLD_DATA:-$ROOT/data}
META=${CTRLWORLD_META:-$ROOT/dataset_meta_info}
# Resolve this file's own location instead of a fixed checkout. `cluster submit` drops each
# run in $WORK/runs/Ctrl-World/<runid>/code, so the chain must re-submit whichever copy it
# was started from; that also keeps every link of the chain on the same commit.
#
# BASH_SOURCE alone cannot say where that copy is: Slurm runs a *duplicate* of this file
# from /var/spool/slurmd, so only `bash <this file>` -- how `cluster submit`'s wrapper
# starts it -- sees the real path. Under `sbatch` it resolves to the spool, and the cd below
# then lands outside the repo. That is what broke the chain on 2026-09-01: link two ran from
# /var/spool/slurmd, found no outputs/ to resume from, and died. So resolve once, verify the
# answer holds a checkout, and hand it down the chain in CTRLWORLD_SCRIPT.
SELF=jobs/train_mcap.sbatch
SCRIPT=${CTRLWORLD_SCRIPT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")}
[ -f "$(dirname "$SCRIPT")/../scripts/train_wm.py" ] || SCRIPT=${SLURM_SUBMIT_DIR:-$PWD}/$SELF
[ -f "$(dirname "$SCRIPT")/../scripts/train_wm.py" ] || {
  echo "no checkout at $(dirname "$SCRIPT")/..; pass CTRLWORLD_SCRIPT=<repo>/$SELF"; exit 1; }
export CTRLWORLD_SCRIPT=$SCRIPT
cd "$(dirname "$SCRIPT")/.."
export HF_HOME=$ROOT/hf HF_HUB_OFFLINE=1
export PATH=$ROOT/venv/bin:$PATH
export WANDB_MODE=offline SWANLAB_DISABLED=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

# Keep the artifacts under the repository path, but on durable storage. `cluster submit`
# deletes the checkout when the job stops, so a real outputs/ there loses the checkpoints,
# and --resume then retrains from the SVD weights without a word. A link keeps every path
# in the code unchanged.
#
# A checkout always has a real outputs/, because the eval JSONs are tracked. So copy what
# is there into the durable tree first, then replace the directory with the link. cp -n
# never overwrites, so the durable copy always wins. This is idempotent, and it also
# migrates a persistent checkout on its first run.
mkdir -p $ROOT/outputs
if [ ! -L outputs ]; then
  if [ -d outputs ]; then
    cp -rn outputs/. $ROOT/outputs/ || exit 1
    rm -rf outputs
  fi
  ln -s $ROOT/outputs outputs
fi

SVD=$HF_HOME/hub/models--stabilityai--stable-video-diffusion-img2vid/snapshots/9cf024d5bfa8f56622af86c884f26a52f6676f2e
CLIP=$HF_HOME/hub/models--openai--clip-vit-base-patch32/snapshots/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268
TAG=0003_abc_mcap
# Checkpoints and samples want different homes. The chain resumes from the previous job's
# checkpoints, so those stay in the durable tag-keyed tree and are the same directory for
# every runid. Samples and rollouts are the one job's own, so under `cluster submit` they
# go in its run dir, which is what `cluster fetch` copies back; standalone they fall back
# beside the checkpoints. train_wm.py keeps --output_dir independent of --run_dir for
# exactly this.
OUT=outputs/$TAG/model
RUN_DIR=${CLUSTER_OUTPUT_DIR:-outputs/$TAG}
MAX_STEPS=${MAX_STEPS:-200000}

# 256x192 is 1.33x the tokens of the 192x192 run, so the per-device batch is 2. The
# accumulation is derived from the process count below, to hold the effective batch at 64.

LAST_AT_START=$(ls -1 $OUT 2>/dev/null | grep -o "checkpoint-[0-9]\+" | grep -o "[0-9]\+" \
  | sort -n | tail -1)
# Chain the successor now, not after training returns. A 30k-step run at ~6.4 s/it needs
# about 55 h, so every job except the last one is killed at the 24 h wall: Slurm sends
# SIGTERM, the step dies, and anything after the training command never runs. The first
# attempt learned this the hard way, stopping at step 12,591 with nobody to resume it.
#
# afterany means the successor starts whether this job times out, fails or completes, and
# the guard above makes a successor that has nothing left to do exit immediately. That
# also stops the chain: once a job resumes from a checkpoint at MAX_STEPS, it leaves.
if [ -n "$LAST_AT_START" ] && [ "$LAST_AT_START" -ge "$MAX_STEPS" ]; then
  echo "already at step $LAST_AT_START of $MAX_STEPS; nothing to resume"
  exit 0
fi
# Chain only when this file carries its own directives. `cluster submit` writes its own
# wrapper, so a re-submit here would silently drop back to the 1-node header above.
#
# --export=ALL copies this job's environment, SLURM_MEM_PER_* included, and the successor's
# own allocation sets its own; srun then refuses to start ("SLURM_MEM_PER_CPU,
# SLURM_MEM_PER_GPU, and SLURM_MEM_PER_NODE are mutually exclusive"). Drop the parent's in a
# subshell so the successor sees only what Slurm gives it.
#
# The successor writes into the run dir `cluster logs` reads, so one runid covers the whole
# chain. Only the header's %x-%j.out in the checkout is left when there is no run dir.
LOGS=${CLUSTER_RUN_DIR:+$CLUSTER_RUN_DIR/logs}
if [ -z "$CLUSTER_RUNID" ]; then
  ( unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU SLURM_MEM_PER_NODE
    sbatch --dependency=afterany:$SLURM_JOB_ID \
      ${LOGS:+--output=$LOGS/job-%j.out --error=$LOGS/job-%j.err} \
      --export=ALL,MAX_STEPS=$MAX_STEPS,CTRLWORLD_SCRIPT=$SCRIPT "$SCRIPT" )
fi

# The allocation decides the topology. One task holds one node and its 4 GPUs, so the
# process count is nodes x 4, and srun starts one accelerate per node with its own rank.
NNODES=${SLURM_NNODES:-1}
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
NPROC=$((NNODES * GPUS_PER_NODE))
MASTER=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)

# The effective batch stays at the paper's 64 on every topology: 2 x accum x NPROC. More
# GPUs therefore buy less accumulation, not a larger batch.
ACCUM=$((64 / (2 * NPROC)))
[ "$ACCUM" -lt 1 ] && { echo "NPROC=$NPROC is too large for an effective batch of 64"; exit 1; }
echo "topology: $NNODES nodes, $NPROC processes, accum $ACCUM, master $MASTER"

# SLURM_PROCID is quoted so each srun task expands its own rank. Unquoted, the submitting
# shell would expand it once and give every node rank 0, and the job would hang at rendezvous.
srun --ntasks=$NNODES --ntasks-per-node=1 --cpu-bind=none bash -c "
  $ROOT/venv/bin/accelerate launch \
  --num_machines $NNODES --num_processes $NPROC \
  --machine_rank \$SLURM_PROCID \
  --main_process_ip $MASTER --main_process_port ${MASTER_PORT:-29501} \
  --mixed_precision fp16 --dynamo_backend no \
  scripts/train_wm.py \
  --svd_model_path $SVD --clip_model_path $CLIP \
  --dataset_root_path $DATA --dataset_meta_info_path $META \
  --dataset_names abc_mcap \
  --train_batch_size 2 --gradient_accumulation_steps $ACCUM --num_workers 8 \
  --max_train_steps $MAX_STEPS --checkpointing_steps 5000 \
  --validation_steps 2500 --video_num 4 \
  --tag $TAG --run_dir $RUN_DIR --output_dir $OUT \
  --resume
"  # --resume is a no-op on the first job: it just finds nothing to resume

