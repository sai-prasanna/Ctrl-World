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
# $ROOT holds everything that outlives a run: the venv, the HF cache, the data, the
# generated index and the outputs. It defaults to the Leonardo layout but is overridable,
# so the same file reproduces the run on another machine or from a `cluster submit`
# checkout: CTRLWORLD_ROOT=/path/to/scratch sbatch|bash <this file>. A submitted run needs
# no argument -- `cluster submit` exports CLUSTER_SHARED_DIR for exactly this.
ROOT=${CTRLWORLD_ROOT:-${CLUSTER_SHARED_DIR:-$WORK/sraman00/ctrlworld}}
DATA=${CTRLWORLD_DATA:-$ROOT/data}
META=${CTRLWORLD_META:-$ROOT/dataset_meta_info}
# Run from wherever this file was checked out; see train_mcap.sbatch for why.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HF_HOME=$ROOT/hf HF_HUB_OFFLINE=1
export PATH=$ROOT/venv/bin:$PATH
export PYTHONUNBUFFERED=1
export TORCH_HOME=$ROOT/torchhome

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
STEP=${STEP:?set STEP, e.g. --export=ALL,STEP=10000}
RUN_DIR=outputs/0003_abc_mcap
CKPT=$RUN_DIR/model/checkpoint-${STEP}.pt
# HIST names an alternative history pattern, so a comparison run does not overwrite the
# default result. HIST_NAME becomes part of the file name.
HIST=${HIST:-}
if [ -n "$HIST" ]; then
  OUT=$RUN_DIR/eval/metrics_step${STEP}_hist_${HIST_NAME:-alt}.json
  HIST_ARG="--history_idx $HIST"
else
  OUT=$RUN_DIR/eval/metrics_step${STEP}.json
  HIST_ARG=""
fi
mkdir -p $(dirname $OUT)

# DUMP_FRAMES saves every clip's ground truth and prediction losslessly, for the
# qualitative failures the aggregate metrics cannot show: wrist instability and views
# that drift out of agreement. The dump is gigabytes, so it lands beside the metrics in
# the durable tree rather than in $CLUSTER_OUTPUT_DIR, which `cluster fetch` pulls back
# whole. The run prints the total size.
DUMP_ARG=""
if [ -n "${DUMP_FRAMES:-}" ]; then
  DUMP_DIR=${OUT%.json}_frames
  mkdir -p $DUMP_DIR
  DUMP_ARG="--dump_frames $DUMP_DIR"
  echo "dumping frames to $DUMP_DIR"
fi

# FVD is skipped rather than silently faked when the I3D backbone is missing.
I3D=""
[ -f "$ROOT/i3d_torchscript.pt" ] && I3D="--i3d_ckpt $ROOT/i3d_torchscript.pt"

echo "eval step=$STEP ckpt=$CKPT"
srun --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK --cpu-bind=none \
  $ROOT/venv/bin/python scripts/eval_video_metrics.py \
    --ckpt_path $CKPT \
    --clips dataset_meta_info/abc_mcap/eval_clips_v1.json \
    --val_dataset_dir $DATA/abc_mcap \
    --data_stat_path $META/abc_mcap/stat.json \
    --svd_model_path $SVD --clip_model_path $CLIP \
    --batch_size ${BATCH_SIZE:-4} $HIST_ARG $DUMP_ARG \
    --fid $I3D \
    --out $OUT

# The canonical copy lives in the durable tag-keyed tree, because eval JSONs are tracked
# and docs/experiments.md quotes them. `cluster fetch` only reaches inside the run dir,
# so leave a copy there too when this is a submitted run -- it is a few KB.
if [ -n "$CLUSTER_OUTPUT_DIR" ]; then
  cp "$OUT" "$CLUSTER_OUTPUT_DIR/"
fi
