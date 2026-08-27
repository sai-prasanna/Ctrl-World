#!/bin/bash
# Stage half of the split pipeline: download episode blobs on a LOGIN node.
# Only login nodes have a route to the Hub (a boost node cannot even resolve
# huggingface.co), so the network stage runs here and does no decoding at all.
#
# launch_download.sh starts this detached; call it directly only to debug a shard.
# Run --dump_episode_files once before the first shard. It is the other stage that needs
# the Hub, and every later stage reads its output offline:
#
#   python preprocessing/extract_latent_abc_mcap.py \
#     --dump_episode_files $CTRLWORLD_ROOT/abc_mcap_files.json
#
# Resolve the repo root from this file's own location instead of a fixed $ROOT/repo, so
# the same file runs from a checkout, from `cluster submit`, and on a machine with no
# Slurm.
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# $ROOT holds everything that outlives a run: the venv, the HF cache, the data, the
# generated index and the outputs. It defaults to the Leonardo layout but is overridable,
# so the same file reproduces the run on another machine or from a `cluster submit`
# checkout: CTRLWORLD_ROOT=/path/to/scratch sbatch|bash <this file>.
ROOT=${CTRLWORLD_ROOT:-$WORK/sraman00/ctrlworld}
export HF_HOME=$ROOT/hf PYTHONUNBUFFERED=1
export PATH=$ROOT/venv/bin:$PATH
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
# Xet is the whole throughput story: 25 MB/s on one stream against 11 MB/s without it,
# and 141 MB/s aggregate across eight. It spends its own threads per download, so keep
# --workers low and let xet supply the parallelism.
export HF_XET_HIGH_PERFORMANCE=1
unset HF_HUB_OFFLINE
# The repo tracks the task selection beside the code, because it records a research
# decision. The episode index only lists the release, so it lives under $ROOT instead and
# you regenerate it.
TASKS=${CTRLWORLD_TASKS:-preprocessing/rigid_tasks.txt}
exec $ROOT/venv/bin/python preprocessing/extract_latent_abc_mcap.py \
  --episode_files ${CTRLWORLD_EPISODE_FILES:-$ROOT/abc_mcap_files.json} \
  --tasks "$(cat "$TASKS")" \
  --split ${SPLIT:-train} --skip_latent --download_only \
  --output_path ${CTRLWORLD_DATA:-$ROOT/data}/abc_mcap --cache_dir $ROOT/mcap_cache \
  --max_staged ${MAX_STAGED:-400} \
  --shard ${SHARD:-0} --num_shards ${NSHARD:-1} --workers ${WORKERS:-8}
