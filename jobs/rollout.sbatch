#!/bin/bash
#SBATCH --job-name=abc_rollout
#SBATCH --account=AIFAC_S07_034
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out

set -x
ROOT=$WORK/sraman00/ctrlworld
cd $ROOT/repo
export HF_HOME=$ROOT/hf HF_HUB_OFFLINE=1
export PATH=$ROOT/venv/bin:$PATH
export PYTHONUNBUFFERED=1
SVD=$HF_HOME/hub/models--stabilityai--stable-video-diffusion-img2vid/snapshots/9cf024d5bfa8f56622af86c884f26a52f6676f2e
CLIP=$HF_HOME/hub/models--openai--clip-vit-base-patch32/snapshots/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268

$ROOT/venv/bin/python scripts/rollout_replay_traj.py \
  --task_type abc_replay \
  --svd_model_path $SVD --clip_model_path $CLIP \
  --ckpt_path $CKPT \
  --val_dataset_dir ${DATA:-$ROOT/data/abc_rigid} \
  --val_id $VALIDS \
  --save_dir $ROOT/rollouts
