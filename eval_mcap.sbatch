#!/bin/bash
# Score an abc_mcap checkpoint on the 256-clip validation set.
#
# eval_video_metrics.py owns the rollout: it replays each clip's recorded actions for 12
# rounds and scores the predictions as it goes, so this does not consume rollout videos
# produced elsewhere. rollout_replay_traj.py is for looking at, this is for measuring.
#
#   sbatch --export=ALL,STEP=10000 eval_mcap.sbatch
#
#SBATCH -A aifac_s07_034
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -t 08:00:00
#SBATCH -J abc_eval
#SBATCH -o /leonardo_work/AIFAC_S07_034/sraman00/ctrlworld/logs_eval-%j.out
ROOT=$WORK/sraman00/ctrlworld
cd $ROOT/repo
export HF_HOME=$ROOT/hf HF_HUB_OFFLINE=1
export PATH=$ROOT/venv/bin:$PATH
export PYTHONUNBUFFERED=1
export TORCH_HOME=$ROOT/torchhome
SVD=$HF_HOME/hub/models--stabilityai--stable-video-diffusion-img2vid/snapshots/9cf024d5bfa8f56622af86c884f26a52f6676f2e
CLIP=$HF_HOME/hub/models--openai--clip-vit-base-patch32/snapshots/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268
STEP=${STEP:?set STEP, e.g. --export=ALL,STEP=10000}
CKPT=$ROOT/repo/outputs/0003_abc_mcap/model/checkpoint-${STEP}.pt
OUT=$ROOT/repo/outputs/0003_abc_mcap/eval/metrics_step${STEP}.json
mkdir -p $(dirname $OUT)

# FVD is skipped rather than silently faked when the I3D backbone is missing.
I3D=""
[ -f "$ROOT/i3d_torchscript.pt" ] && I3D="--i3d_ckpt $ROOT/i3d_torchscript.pt"
# CoTracker weights are not staged; --track needs a login-node download first, so the
# tracking metrics stay off rather than failing the run half way through.

echo "eval step=$STEP ckpt=$CKPT"
srun --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK --cpu-bind=none \
  $ROOT/venv/bin/python scripts/eval_video_metrics.py \
    --ckpt_path $CKPT \
    --clips dataset_meta_info/abc_mcap/eval_clips_v1.json \
    --val_dataset_dir $ROOT/data/abc_mcap \
    --data_stat_path dataset_meta_info/abc_mcap/stat.json \
    --svd_model_path $SVD --clip_model_path $CLIP \
    --batch_size ${BATCH_SIZE:-4} \
    --fid $I3D \
    --out $OUT
