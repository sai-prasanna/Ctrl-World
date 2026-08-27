#!/bin/bash
#SBATCH --job-name=abc_mcap_train
#SBATCH --account=AIFAC_S07_034
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out

# 30k steps on the 256x192 re-extraction. The 24 h wall clock does not cover 30k steps,
# so the job resubmits itself from the newest checkpoint until it gets there.
set -x
ROOT=$WORK/sraman00/ctrlworld
# Resolve this file's own location instead of a fixed checkout. `cluster submit` drops each
# run in $WORK/runs/Ctrl-World/<runid>/code, so the chain must re-submit whichever copy it
# was started from; that also keeps every link of the chain on the same commit.
SCRIPT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")
cd "$(dirname "$SCRIPT")"
export HF_HOME=$ROOT/hf HF_HUB_OFFLINE=1
export PATH=$ROOT/venv/bin:$PATH
export WANDB_MODE=offline SWANLAB_DISABLED=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

SVD=$HF_HOME/hub/models--stabilityai--stable-video-diffusion-img2vid/snapshots/9cf024d5bfa8f56622af86c884f26a52f6676f2e
CLIP=$HF_HOME/hub/models--openai--clip-vit-base-patch32/snapshots/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268
TAG=0003_abc_mcap
# Artifacts and generated metadata live outside the code snapshot: checkpoints so --resume
# still finds the lineage from a fresh checkout, metadata because train_sample.json is
# ~490 MB and has no business in a git ref.
RUN_DIR=$ROOT/outputs/$TAG
OUT=$RUN_DIR/model
META=$ROOT/dataset_meta_info
MAX_STEPS=${MAX_STEPS:-30000}

# 256x192 is 1.33x the tokens of the 192x192 run, so per-device batch drops 4 -> 2 and
# grad_accum rises 4 -> 8, keeping the paper effective batch of 64.

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
sbatch --dependency=afterany:$SLURM_JOB_ID --export=ALL,MAX_STEPS=$MAX_STEPS "$SCRIPT"

$ROOT/venv/bin/accelerate launch --num_processes 4 --mixed_precision fp16 --dynamo_backend no \
  scripts/train_wm.py \
  --svd_model_path $SVD --clip_model_path $CLIP \
  --dataset_root_path $ROOT/data --dataset_meta_info_path $META \
  --dataset_names abc_mcap \
  --train_batch_size 2 --gradient_accumulation_steps 8 --num_workers 8 \
  --max_train_steps $MAX_STEPS --checkpointing_steps 5000 \
  --validation_steps 2500 --video_num 4 \
  --tag $TAG --run_dir $RUN_DIR \
  --resume  # --resume is a no-op on the first job: it just finds nothing to resume

