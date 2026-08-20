#!/bin/bash
# Run scripts/eval_video_metrics.py on Leonardo for one checkpoint.
#
# Usage: bash scripts/run_eval_leonardo.sh <step> [extra args...]
#
# Environment setup mirrors train.sbatch and rollout.sbatch: the venv, the offline HF
# cache, and TORCH_HOME for the LPIPS backbone weights, which are pre-fetched on the
# login node because compute nodes have no internet.
set -euxo pipefail

STEP=${1:?usage: run_eval_leonardo.sh <step> [extra args...]}
shift

ROOT=$WORK/sraman00/ctrlworld
export HF_HOME=$ROOT/hf HF_HUB_OFFLINE=1
export TORCH_HOME=$ROOT/torchhome
export PYTHONUNBUFFERED=1

SVD=$HF_HOME/hub/models--stabilityai--stable-video-diffusion-img2vid/snapshots/9cf024d5bfa8f56622af86c884f26a52f6676f2e
CLIP=$HF_HOME/hub/models--openai--clip-vit-base-patch32/snapshots/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268
CKPT=${CKPT:-$ROOT/repo/outputs/0002_abc_rigid/model/checkpoint-${STEP}.pt}
OUT=${OUT:-$ROOT/eval/metrics_step${STEP}.json}

# FVD is only computed when an I3D TorchScript checkpoint is present.
I3D_ARG=""
if [ -f "$ROOT/i3d_torchscript.pt" ]; then
  I3D_ARG="--i3d_ckpt $ROOT/i3d_torchscript.pt"
fi

$ROOT/venv/bin/python scripts/eval_video_metrics.py \
  --ckpt_path "$CKPT" \
  --clips dataset_meta_info/abc_rigid/eval_clips_v1.json \
  --val_dataset_dir "$ROOT/data/abc_rigid" \
  --data_stat_path dataset_meta_info/abc_rigid/stat.json \
  --svd_model_path "$SVD" --clip_model_path "$CLIP" \
  --batch_size "${BATCH_SIZE:-4}" \
  --fid $I3D_ARG \
  --out "$OUT" \
  "$@"
