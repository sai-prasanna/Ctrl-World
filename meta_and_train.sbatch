#!/bin/bash
# Build the training index from whatever abc_gpu.sbatch has finished, then start training.
# Run this after the extraction shards are done; it is cheap and does no GPU work of its
# own beyond holding the allocation that submits training.
#
#   sbatch meta_and_train.sbatch
#
#SBATCH -A aifac_s07_034
#SBATCH -p lrd_all_serial
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 04:00:00
#SBATCH -J abc_meta
#SBATCH -o /leonardo_work/AIFAC_S07_034/sraman00/ctrlworld/logs_meta-%j.out
ROOT=$WORK/sraman00/ctrlworld
cd $ROOT/repo
export HF_HOME=$ROOT/hf HF_HUB_OFFLINE=1
export PATH=$ROOT/venv/bin:$PATH
export PYTHONUNBUFFERED=1

# The state vector is the same 14-D bimanual joint+gripper reading as abc_rigid in the
# same units, so its p01/p99 carry over; recomputing them would resample the same
# distribution over fewer episodes.
mkdir -p dataset_meta_info/abc_mcap
cp dataset_meta_info/abc_rigid/stat.json dataset_meta_info/abc_mcap/stat.json
$ROOT/venv/bin/python dataset_meta_info/create_meta_info.py \
  --droid_output_path $ROOT/data/abc_mcap --dataset_name abc_mcap

# Submitting training bounces off the per-user GPU job limit while another run holds a
# slot, so a refusal is expected rather than an error.
# Bounded by the 4 h wall above rather than by patience: 36 tries at 5 min is 3 h.
for attempt in $(seq 1 36); do
  if sbatch $ROOT/train_mcap.sbatch; then exit 0; fi
  echo "training submit refused (attempt $attempt); retrying in 5 min"
  sleep 300
done
