#!/bin/bash
# Replay rollouts against the abc_mcap val split: the model is driven by the recorded
# actions over a long horizon, which is what exposes drift that teacher-forced validation
# samples cannot show.
#SBATCH -A aifac_s07_034
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -t 00:29:00
#SBATCH -J abc_rollout
#SBATCH -o /leonardo_work/AIFAC_S07_034/sraman00/ctrlworld/logs_rollout-%j.out
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
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8

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
# config.py still defaults these to the upstream authors' /cephfs paths, which do not
# exist here; every entry script has to name them explicitly.
SVD=$HF_HOME/hub/models--stabilityai--stable-video-diffusion-img2vid/snapshots/9cf024d5bfa8f56622af86c884f26a52f6676f2e
CLIP=$HF_HOME/hub/models--openai--clip-vit-base-patch32/snapshots/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268
RUN_DIR=outputs/0003_abc_mcap
CKPT=${CKPT:-$RUN_DIR/model/checkpoint-30000.pt}
IDS=${IDS:-00ce9827-879c-4bed-82ef-8bfd07302d60,016f2a41-e549-4b00-b786-c5b0842bcaa2,02eb2789-4234-4d4c-984e-08872ed00c1d}
echo "rollout ckpt=$CKPT"
# The val_dataset_dir override points at the mcap data; config.py's abc_replay entry still
# names the old LeRobot-mirror subset. The instruction comes from each annotation.
srun --cpu-bind=none $ROOT/venv/bin/python scripts/rollout_replay_traj.py \
  --task_type abc_replay \
  --ckpt_path $CKPT \
  --svd_model_path $SVD --clip_model_path $CLIP \
  --val_dataset_dir $DATA/abc_mcap \
  --val_id $IDS \
  --save_dir $RUN_DIR/rollout

# As in eval_mcap.sbatch: the rollouts stay in the durable tree, with a copy in the run
# dir so `cluster fetch` brings them back from a submitted run.
if [ -n "$CLUSTER_OUTPUT_DIR" ]; then
  cp -r "$RUN_DIR/rollout" "$CLUSTER_OUTPUT_DIR/"
fi
