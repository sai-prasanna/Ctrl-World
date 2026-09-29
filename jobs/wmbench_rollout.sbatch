#!/bin/bash
# Roll an abc_mcap checkpoint over the evaluation clips and write a wmbench manifest.
#
# The GPU half of the benchmark, and the only half that needs this repository's venv:
# the manifest it writes is pixels plus provenance, which `wmbench score` reads from
# venv_bench with no Ctrl-World code and no checkpoint. eval_mcap.sbatch does both halves
# in one job against the five clipeval metrics; this one splits them, so the sixteen
# metrics can be re-run without paying for the rollout again.
#
#   sbatch --export=ALL,STEP=200000 wmbench_rollout.sbatch
#
#SBATCH -A aifac_s07_034
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -t 08:00:00
#SBATCH -J wmb_rollout
#SBATCH -o /leonardo_work/AIFAC_S07_034/sraman00/ctrlworld/logs_wmbench_rollout-%j.out
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
# deletes the checkout when the job stops, so a real outputs/ there loses the manifest.
# A link keeps every path in the code unchanged. cp -n never overwrites, so the durable
# copy always wins; this is idempotent and also migrates a persistent checkout.
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
STEP=${STEP:?set STEP, e.g. --export=ALL,STEP=200000}
TAG=${TAG:-0003_abc_mcap}
# STEP=0 is the SVD initialization: no file to load, so the script gets no --ckpt_path and
# builds the model the way train_wm.py does on its first step. It is the floor every
# metric reports for a model that has seen no ABC frame, and costs no training.
if [ "$STEP" = 0 ]; then CKPT=${CKPT:-none}; else CKPT=${CKPT:-outputs/$TAG/model/checkpoint-${STEP}.pt}; fi
# The run name is what distinguishes one manifest from the next under the same tag, so an
# alternative history pattern or a smoke run does not overwrite the default rollout.
RUN=${RUN:-step${STEP}}
OUT=${OUT:-outputs/$TAG/wmbench/$RUN}

# abc130k is on the path from this checkout rather than installed: `cluster submit`
# discards the checkout, and an editable install into the shared venv would outlive it
# pointing at a directory that no longer exists.
export PYTHONPATH=$PWD/abc130k/src${PYTHONPATH:+:$PYTHONPATH}

echo "wmbench rollout step=$STEP ckpt=$CKPT -> $OUT"
srun --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK --cpu-bind=none \
  $ROOT/venv/bin/python scripts/wmbench_rollout_ctrlworld.py \
    --ckpt_path $CKPT \
    --clips ${CLIPS:-dataset_meta_info/abc_mcap/eval_clips_v1.json} \
    --val_dataset_dir $DATA/abc_mcap \
    --data_stat_path $META/abc_mcap/stat.json \
    --svd_model_path $SVD --clip_model_path $CLIP \
    --batch_size ${BATCH_SIZE:-4} \
    ${HIST:+--history_idx $HIST} \
    ${LIMIT:+--limit $LIMIT} \
    ${VARIANTS:+--instruction_variants $VARIANTS} \
    --out $OUT

# The npz files are gigabytes and stay in the durable tree; `cluster fetch` pulls the run
# directory back whole, so only the manifest -- provenance and a few KB -- travels.
if [ -n "$CLUSTER_OUTPUT_DIR" ]; then
  cp "$OUT/manifest.json" "$CLUSTER_OUTPUT_DIR/manifest_${RUN}.json"
fi
