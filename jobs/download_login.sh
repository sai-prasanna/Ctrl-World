#!/bin/bash
# Stage half of the split pipeline: download episode blobs on a LOGIN node.
# Only login nodes have a route to the Hub (a boost node cannot even resolve
# huggingface.co), so the network stage runs here and does no decoding at all.
ROOT=$WORK/sraman00/ctrlworld
cd $ROOT/repo
export HF_HOME=$ROOT/hf PYTHONUNBUFFERED=1
export PATH=$ROOT/venv/bin:$PATH
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
# Xet is the whole throughput story: 25 MB/s on one stream against 11 MB/s without it,
# and 141 MB/s aggregate across eight. It spends its own threads per download, so keep
# --workers low and let xet supply the parallelism.
export HF_XET_HIGH_PERFORMANCE=1
unset HF_HUB_OFFLINE
exec $ROOT/venv/bin/python dataset_example/extract_latent_abc_mcap.py \
  --episode_files $ROOT/abc_mcap_files.json --tasks $(cat $ROOT/rigid_tasks.txt) \
  --split ${SPLIT:-train} --skip_latent --download_only \
  --output_path $ROOT/data/abc_mcap --cache_dir $ROOT/mcap_cache \
  --max_staged ${MAX_STAGED:-400} \
  --shard ${SHARD:-0} --num_shards ${NSHARD:-1} --workers ${WORKERS:-8}
