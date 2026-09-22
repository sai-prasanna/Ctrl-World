#!/bin/bash
# Run an ABC policy inside an abc_mcap checkpoint, and write ABC's own summary.json.
#
# The only job in this repository that needs both virtual environments at once. The
# checkpoint stays in `venv` behind scripts/wmbench_wm_server.py; the policy, the loop and
# ABC's rollout driver run in `venv_bench` on torch 2.11, and the two talk over a Unix
# socket in $TMPDIR. Neither environment can hold both pins, so this job holds both.
#
#   sbatch --export=ALL,STEP=200000,POLICY=abc_dit_xl_200k wmbench_policy.sbatch
#   sbatch --export=ALL,STEP=200000,POLICY=echo,RUN=step200000_echo wmbench_policy.sbatch
#
# The validation ordering the plan calls for is three runs of this file: echo (recorded
# actions), abc_dit_xl_200k, and random. Success is not measured here -- the judge scores
# the rendered videos afterwards and writes the answer back into summary.json.
#
#SBATCH -A aifac_s07_034
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -t 04:00:00
#SBATCH -J wmb_policy
#SBATCH -o /leonardo_work/AIFAC_S07_034/sraman00/ctrlworld/logs_wmbench_policy-%j.out
# $ROOT holds everything that outlives a run: the two venvs, the HF cache, the data, the
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
export PYTHONUNBUFFERED=1
export TORCH_HOME=$ROOT/torchhome

# Keep the artifacts under the repository path, but on durable storage. `cluster submit`
# deletes the checkout when the job stops, so a real outputs/ there loses the rollouts.
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
CKPT=${CKPT:-outputs/$TAG/model/checkpoint-${STEP}.pt}
POLICY=${POLICY:-abc_dit_xl_200k}
RUN=${RUN:-step${STEP}_${POLICY}}
OUT=${OUT:-outputs/$TAG/wmbench/policy/$RUN}
# The ABC checkpoint cache, prestaged on a login node: compute nodes have no internet and
# abc_dit_xl_200k_model.pt is 8.1 GB behind a User-Agent header.
ABC_CKPT_DIR=${ABC_CKPT_DIR:-$ROOT/abc_cache}
# The pinned amazon-far/abc checkout, staged rather than vendored; venv_bench installs it,
# and this keeps the path on PYTHONPATH for a checkout that was not installed.
ABC_REPO=${ABC_REPO:-$ROOT/third_party/abc}

# abc130k is on the path from this checkout rather than installed: `cluster submit`
# discards the checkout, and an editable install into a shared venv would outlive it
# pointing at a directory that no longer exists.
export PYTHONPATH=$PWD/abc130k/src:$ABC_REPO${PYTHONPATH:+:$PYTHONPATH}

# A Unix socket address holds 107 bytes, so it lives in $TMPDIR and not beside the run.
SOCK=${SOCK:-${TMPDIR:-/tmp}/cw_$SLURM_JOB_ID.sock}
READY=$SOCK.ready
rm -f "$SOCK" "$READY"

# --overlap on both steps: the server and the client are two steps of one allocation, and
# without it the second waits for the first to exit, which it never does.
echo "serving $CKPT on $SOCK"
srun --overlap --ntasks=1 --cpus-per-task=2 --cpu-bind=none \
  $ROOT/venv/bin/python scripts/wmbench_wm_server.py \
    --ckpt_path $CKPT \
    --clips ${CLIPS:-dataset_meta_info/abc_mcap/eval_clips_v1.json} \
    --val_dataset_dir $DATA/abc_mcap \
    --data_stat_path $META/abc_mcap/stat.json \
    --svd_model_path $SVD --clip_model_path $CLIP \
    ${HIST:+--history_idx $HIST} \
    --socket "$SOCK" --ready_file "$READY" &
SERVER=$!
# Kill the server however this job ends, including a wall-clock kill: it holds the GPU.
trap 'kill $SERVER 2>/dev/null; rm -f "$SOCK" "$READY"' EXIT

# The ready file, not the socket: the socket path exists from bind, minutes before the
# checkpoint has finished loading, so waiting on it races the load.
for _ in $(seq 1 3600); do
  [ -e "$READY" ] && break
  kill -0 $SERVER 2>/dev/null || { echo "the world-model server died before it was ready"; exit 1; }
  sleep 1
done
[ -e "$READY" ] || { echo "the world-model server was not ready within an hour"; exit 1; }
echo "server ready, running $POLICY over ${NUM_WORLDS:-4} world(s) -> $OUT"

srun --overlap --ntasks=1 --cpus-per-task=6 --cpu-bind=none \
  $ROOT/venv_bench/bin/python bridges/abc_policy_run.py \
    --socket "$SOCK" --ready_file "$READY" \
    --val_dataset_dir $DATA/abc_mcap \
    --clips ${CLIPS:-dataset_meta_info/abc_mcap/eval_clips_v1.json} \
    --policy $POLICY --ckpt_dir $ABC_CKPT_DIR \
    --num_worlds ${NUM_WORLDS:-4} \
    ${ROUNDS:+--rounds $ROUNDS} \
    ${ALPHA:+--alpha $ALPHA} \
    --out $OUT
STATUS=$?

# The judge is a separate stage and a separate model, so it is not run here; when
# JUDGE_MODEL names a prestaged Qwen3-VL it runs in the same allocation, because the
# rollouts are already on this node's disk.
if [ $STATUS -eq 0 ] && [ -n "${JUDGE_MODEL:-}" ]; then
  echo "judging $OUT for success"
  srun --overlap --ntasks=1 --cpus-per-task=6 --cpu-bind=none \
    $ROOT/venv_bench/bin/python -m wmbench.cli judge \
      --manifest $OUT --rubric success --model "$JUDGE_MODEL" --view top
fi

kill $SERVER 2>/dev/null
wait $SERVER 2>/dev/null

# summary.json and the manifest are a few KB of provenance; the npz files and the mp4s are
# gigabytes and stay in the durable tree, which `cluster fetch` does not reach into.
if [ -n "$CLUSTER_OUTPUT_DIR" ]; then
  cp "$OUT/summary.json" "$CLUSTER_OUTPUT_DIR/summary_${RUN}.json" 2>/dev/null
  cp "$OUT/manifest.json" "$CLUSTER_OUTPUT_DIR/manifest_${RUN}.json" 2>/dev/null
  cp "$OUT/judge_success.json" "$CLUSTER_OUTPUT_DIR/judge_success_${RUN}.json" 2>/dev/null
fi
exit $STATUS
