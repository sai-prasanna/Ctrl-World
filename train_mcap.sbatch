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
cd $ROOT/repo
export HF_HOME=$ROOT/hf HF_HUB_OFFLINE=1
export PATH=$ROOT/venv/bin:$PATH
export WANDB_MODE=offline SWANLAB_DISABLED=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

SVD=$HF_HOME/hub/models--stabilityai--stable-video-diffusion-img2vid/snapshots/9cf024d5bfa8f56622af86c884f26a52f6676f2e
CLIP=$HF_HOME/hub/models--openai--clip-vit-base-patch32/snapshots/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268
TAG=0003_abc_mcap
OUT=$ROOT/repo/outputs/$TAG/model
MAX_STEPS=${MAX_STEPS:-30000}

# 256x192 is 1.33x the tokens of the 192x192 run, so per-device batch drops 4 -> 2 and
# grad_accum rises 4 -> 8, keeping the paper effective batch of 64.

$ROOT/venv/bin/accelerate launch --num_processes 4 --mixed_precision fp16 --dynamo_backend no \
  scripts/train_wm.py \
  --svd_model_path $SVD --clip_model_path $CLIP \
  --dataset_root_path $ROOT/data --dataset_meta_info_path dataset_meta_info \
  --dataset_names abc_mcap \
  --train_batch_size 2 --gradient_accumulation_steps 8 --num_workers 8 \
  --max_train_steps $MAX_STEPS --checkpointing_steps 2500 \
  --validation_steps 2500 --video_num 4 \
  --tag $TAG --resume  # --resume is a no-op on the first job: it just finds nothing to resume

# Requeue until 30k, unless the run finished or died before writing a checkpoint.
LAST=$(ls -1 $OUT 2>/dev/null | grep -o "checkpoint-[0-9]\+" | grep -o "[0-9]\+" | sort -n | tail -1)
if [ -n "$LAST" ] && [ "$LAST" -lt "$MAX_STEPS" ]; then
  sbatch --export=ALL,MAX_STEPS=$MAX_STEPS $ROOT/train_mcap.sbatch
fi
