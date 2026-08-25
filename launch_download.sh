#!/bin/bash
# Detach a download shard from the ssh session that started it. Without setsid + </dev/null
# the worker pool dies with a BrokenPipe as soon as the login shell goes away.
# Connect with `ssh -o ControlPath=none` when launching several: a shared ControlMaster
# pins every shard onto whichever login node the first connection happened to pick.
ROOT=$WORK/sraman00/ctrlworld
cd $ROOT
setsid nohup env SHARD=$1 NSHARD=$2 WORKERS=${3:-8} SPLIT=${4:-train} \
  ./download_login.sh > $ROOT/logs_download_${4:-train}_$1.log 2>&1 < /dev/null &
disown
echo "download shard $1/$2 on $(hostname)"
