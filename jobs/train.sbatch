#!/bin/bash
#SBATCH --job-name=abc_train
#SBATCH --account=AIFAC_S07_034
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
# boost_qos_lprod allows 4 days (the default QoS caps at 24h) for up to 32 GPUs / 8
# nodes per job, so a full run fits in one job with no resume seam. 100k steps at
# ~4.71 s/it is ~131h and still needs chaining; 30k from a 10k checkpoint does not.
#SBATCH --qos=boost_qos_lprod
#SBATCH --time=36:00:00
#SBATCH --output=%x-%j.out

set -x
ROOT=${CTRLWORLD_ROOT:-${CLUSTER_SHARED_DIR:-$WORK/sraman00/ctrlworld}}
# Run from the checkout this file sits in, not a fixed $ROOT/repo; see
# train_mcap.sbatch for why. jobs/ is one level below the repo root.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HF_HOME=$ROOT/hf HF_HUB_OFFLINE=1
export PATH=$ROOT/venv/bin:$PATH
export WANDB_MODE=offline SWANLAB_DISABLED=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

SVD=$HF_HOME/hub/models--stabilityai--stable-video-diffusion-img2vid/snapshots/9cf024d5bfa8f56622af86c884f26a52f6676f2e
CLIP=$HF_HOME/hub/models--openai--clip-vit-base-patch32/snapshots/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268

# 4 A100 x batch 4 x grad_accum 4 = effective batch 64, matching the paper's 2x8 H100 @ 64
# A checkpoint step costs ~21GB on disk (8.6GB weights + 11.4GB optimizer/scheduler
# state), so the interval is 5000 rather than 2500 — halving the count halves the bill.
$ROOT/venv/bin/accelerate launch --num_processes 4 --mixed_precision fp16 --dynamo_backend no \
  scripts/train_wm.py \
  --svd_model_path $SVD --clip_model_path $CLIP \
  --dataset_root_path $ROOT/data --dataset_meta_info_path dataset_meta_info \
  --dataset_names abc_rigid \
  --train_batch_size 4 --gradient_accumulation_steps 4 --num_workers 8 \
  --max_train_steps ${MAX_STEPS:-10000} --checkpointing_steps ${CKPT_STEPS:-5000} \
  --validation_steps 2500 --video_num 4 \
  --tag 0002_abc_rigid ${RESUME:-}
