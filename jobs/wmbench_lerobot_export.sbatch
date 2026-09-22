#!/bin/bash
# Write the extracted abc_mcap corpus out as LeRobot v3, one dataset per task.
#
# A second view of the same pixels, not a second extraction: the frames come straight off
# the mp4s that abc_gpu.sbatch produced, so nothing is re-cropped or re-sampled and the
# field-of-view crop already in them is recorded as `preprocess_id` rather than repeated.
# What it buys is every LeRobot consumer - a policy, another world model, a benchmark that
# only speaks that format.
#
# CPU only, and a whole node's worth: the cost is decoding three h264 streams per episode
# and re-encoding them as AV1. No GPU, so this does not queue behind the training runs.
#
# boost_usr_prod without --gres is accepted: `sbatch --test-only` with these exact
# resources schedules onto a booster node (checked 2026-09-22). lrd_all_serial would start
# sooner but caps at 8 CPUs and 4 hours, which is under half this job's walltime, and
# dcgp_usr_prod refuses the project's account outright ("invalid account or expired
# budget"). So the queue wait is the price of the node.
#
#   sbatch --export=ALL,SPLIT=val jobs/wmbench_lerobot_export.sbatch
#   sbatch --export=ALL,SPLIT=train,TASKS_FILE=abc130k/src/abc130k/tasks/rigid.txt ...
#
#SBATCH -A aifac_s07_034
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH -t 12:00:00
#SBATCH -J abc_lerobot
#SBATCH -o /leonardo_work/AIFAC_S07_034/sraman00/ctrlworld/logs_lerobot-%j.out
# $ROOT holds everything that outlives a run: the venv, the HF cache, the data, the
# generated index and the outputs. It defaults to the Leonardo layout but is overridable,
# so the same file reproduces the run on another machine or from a `cluster submit`
# checkout: CTRLWORLD_ROOT=/path/to/scratch sbatch|bash <this file>. A submitted run needs
# no argument -- `cluster submit` exports CLUSTER_SHARED_DIR for exactly this.
ROOT=${CTRLWORLD_ROOT:-${CLUSTER_SHARED_DIR:-$WORK/sraman00/ctrlworld}}
DATA=${CTRLWORLD_DATA:-$ROOT/data}
# Run from wherever this file was checked out; see train_mcap.sbatch for why.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HF_HOME=$ROOT/hf HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
# venv_bench, not venv: the writer is `lerobot`, which pins a torch the model's
# environment cannot hold. Nothing in this job touches a checkpoint.
export PATH=$ROOT/venv_bench/bin:$PATH
# abc130k is a self-contained package rather than a script beside the repo, so it goes on
# the path from the checkout this file resolved. An editable install would pin one path,
# which `cluster submit` discards when the run ends.
export PYTHONPATH="$PWD/abc130k/src${PYTHONPATH:+:$PYTHONPATH}"
# Each episode is one decode plus one encode, and libsvtav1 takes the cores it is given;
# holding every encoder to one thread keeps the parallelism at the episode level.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

SPLIT=${SPLIT:-val}
SRC=${SRC:-$DATA/abc_mcap}
OUT=${OUT:-$DATA/lerobot/abc_mcap_$SPLIT}
# The task selection is a research decision, so it is tracked beside the code; leaving it
# unset exports every task present in the split.
# Arrays, because the list file is one task per line with comments: an unquoted string
# would word-split it into a dozen positional arguments and argparse would refuse them,
# and quoting the string would pass an empty --tasks when TASKS_FILE is unset.
TASKS_ARG=()
[ -n "${TASKS_FILE:-}" ] && TASKS_ARG=(--tasks "$(cat "$TASKS_FILE")")
LIMIT_ARG=()
[ -n "${LIMIT:-}" ] && LIMIT_ARG=(--limit "$LIMIT")

echo "exporting $SRC ($SPLIT) -> $OUT"
srun --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK --cpu-bind=none \
  $ROOT/venv_bench/bin/python -m abc130k.lerobot_export \
    --root $SRC --out $OUT --split $SPLIT "${TASKS_ARG[@]}" "${LIMIT_ARG[@]}"

du -sh $OUT
