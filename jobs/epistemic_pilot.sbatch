#!/bin/bash
#SBATCH -A aifac_s07_034
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -t 02:00:00
#SBATCH -J abc_epistemic
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT=${CTRLWORLD_ROOT:-${CLUSTER_SHARED_DIR:-${WORK:?}/sraman00/ctrlworld}}
export HF_HOME="$ROOT/hf" HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
OUT=${CLUSTER_OUTPUT_DIR:-outputs/epistemic_140k_pilot}
SVD="$HF_HOME/hub/models--stabilityai--stable-video-diffusion-img2vid/snapshots/9cf024d5bfa8f56622af86c884f26a52f6676f2e"
CLIP="$HF_HOME/hub/models--openai--clip-vit-base-patch32/snapshots/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
"$ROOT/venv/bin/python" scripts/pilot_epistemic.py \
  --ckpt_path "$ROOT/outputs/0003_abc_mcap/model/checkpoint-140000.pt" \
  --svd_model_path "$SVD" --clip_model_path "$CLIP" \
  --val_dataset_dir "$ROOT/data/abc_mcap" \
  --data_stat_path "$ROOT/dataset_meta_info/abc_mcap/stat.json" \
  --dataset_root_path "$ROOT/data" \
  --dataset_meta_info_path "$ROOT/dataset_meta_info" \
  --out "$OUT" "$@"
