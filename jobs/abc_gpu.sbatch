#!/bin/bash
# Compute half of the ABC extraction: decode staged MCAP blobs to mp4s, then VAE-encode
# them to latents. One job per shard does both, because they shard identically and both
# want the same node - splitting them only added a hand-off.
#
# Pairs with launch_download.sh, which stages blobs on a login node. Login nodes are the
# only ones with a route to the Hub; boost nodes are the only ones on this budget with
# real core counts. Hence the split, and hence HF_HUB_OFFLINE=1 here: the cache is the
# only source this job can see.
#
#   ./launch_download.sh 0 4 8 train      # on a login node, once per shard
#   sbatch --array=0-3 --export=ALL,NSHARD=4 abc_gpu.sbatch
#
#SBATCH -A aifac_s07_034
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH -t 08:00:00
#SBATCH -J abc_gpu
#SBATCH -o /leonardo_work/AIFAC_S07_034/sraman00/ctrlworld/logs_abcgpu-%A_%a.out
# $ROOT holds everything that outlives a run: the venv, the HF cache, the data, the
# generated index and the outputs. It defaults to the Leonardo layout but is overridable,
# so the same file reproduces the run on another machine or from a `cluster submit`
# checkout: CTRLWORLD_ROOT=/path/to/scratch sbatch|bash <this file>.
ROOT=${CTRLWORLD_ROOT:-$WORK/sraman00/ctrlworld}
# Run from wherever this file was checked out; see train_mcap.sbatch for why.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HF_HOME=$ROOT/hf HF_HUB_OFFLINE=1
export PATH=$ROOT/venv/bin:$PATH
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
SVD=$HF_HOME/hub/models--stabilityai--stable-video-diffusion-img2vid/snapshots/9cf024d5bfa8f56622af86c884f26a52f6676f2e
DATA=${CTRLWORLD_DATA:-$ROOT/data}/abc_mcap
# The task selection is committed beside the code because it is a scientific choice; the
# episode index is a derived Hub listing, so it lives under $ROOT. Generate it once on a
# login node with --dump_episode_files before the first download shard starts.
TASKS=${CTRLWORLD_TASKS:-preprocessing/rigid_tasks.txt}
EPISODE_FILES=${CTRLWORLD_EPISODE_FILES:-$ROOT/abc_mcap_files.json}
SHARD=${SLURM_ARRAY_TASK_ID:-0}
NSHARD=${NSHARD:-4}
SPLIT=${SPLIT:-train}

echo "shard $SHARD/$NSHARD split=$SPLIT on $(hostname), alloc=$SLURM_CPUS_PER_TASK cpus"

# The batch shell is affinity-bound to one core even though Slurm allocated the whole
# node, so a pool started from here would pile every worker onto that core.
# Stop sweeping half an hour before the wall clock so the job exits between episodes.
srun --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK --cpu-bind=none \
  $ROOT/venv/bin/python preprocessing/extract_latent_abc_mcap.py \
    --episode_files "$EPISODE_FILES" --tasks "$(cat "$TASKS")" \
    --split $SPLIT --skip_latent --decode_cached \
    --output_path $DATA --cache_dir $ROOT/mcap_cache \
    --deadline $(( 7 * 3600 + 1800 )) --sweep_wait 60 \
    --shard $SHARD --num_shards $NSHARD --workers ${WORKERS:-30}

# Encoding needs the GPUs the decode pass left idle. It skips any trajectory whose .pt
# already exists, so a requeued job resumes and overlapping shards cost nothing.
$ROOT/venv/bin/accelerate launch --num_processes 4 --mixed_precision no \
  preprocessing/encode_latents_abc.py \
    --data_path $DATA --svd_path $SVD --fp16 \
    --shard $SHARD --num_shards $NSHARD
