#!/bin/bash
# Detach a download shard from the ssh session that started it. Without setsid + </dev/null
# the worker pool dies with a BrokenPipe as soon as the login shell goes away.
# Connect with `ssh -o ControlPath=none` when launching several: a shared ControlMaster
# pins every shard onto whichever login node the first connection happened to pick.
#
#   jobs/launch_download.sh <shard> <num_shards> [workers] [split]
#
# $ROOT holds everything that outlives a run: the venv, the HF cache, the data, the
# generated index and the outputs. It defaults to the Leonardo layout but is overridable,
# so the same file reproduces the run on another machine or from a `cluster submit`
# checkout: CTRLWORLD_ROOT=/path/to/scratch sbatch|bash <this file>.
ROOT=${CTRLWORLD_ROOT:-$WORK/sraman00/ctrlworld}
# Logs outlive the checkout, so they go to $ROOT. Resolve the download script beside this
# file rather than under a fixed $ROOT/repo.
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
setsid nohup env SHARD=$1 NSHARD=$2 WORKERS=${3:-8} SPLIT=${4:-train} \
  "$HERE/download_login.sh" > $ROOT/logs_download_${4:-train}_$1.log 2>&1 < /dev/null &
disown
echo "download shard $1/$2 on $(hostname)"
