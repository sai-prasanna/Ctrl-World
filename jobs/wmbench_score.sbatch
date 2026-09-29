#!/bin/bash
# Score a wmbench manifest with every metric the staged weights allow.
#
# The second half of the split that wmbench_rollout.sbatch starts. It reads pixels and
# provenance, never a checkpoint, so it runs in venv_bench (torch 2.11, transformers 5)
# with no Ctrl-World code on the path. Re-running the sixteen metrics therefore costs a
# scoring pass, not a rollout.
#
#   sbatch --export=ALL,MANIFEST=$ROOT/outputs/0003_abc_mcap/wmbench/step200000 wmbench_score.sbatch
#
# CAPTIONS (a `judge --rubric caption` file), MASKS=grounding-dino and OBJECTS (the
# dataset's object list) unlock semantic_alignment and the trajectory metrics; without
# them those columns are reported as skipped, as before.
#
#SBATCH -A aifac_s07_034
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -t 04:00:00
#SBATCH -J wmb_score
#SBATCH -o /leonardo_work/AIFAC_S07_034/sraman00/ctrlworld/logs_wmbench_score-%j.out
# $ROOT holds everything that outlives a run: the venv, the HF cache, the weights and the
# outputs. It defaults to the Leonardo layout but is overridable, so the same file
# reproduces the run on another machine or from a `cluster submit` checkout:
# CTRLWORLD_ROOT=/path/to/scratch sbatch|bash <this file>. A submitted run needs no
# argument -- `cluster submit` exports CLUSTER_SHARED_DIR for exactly this.
ROOT=${CTRLWORLD_ROOT:-${CLUSTER_SHARED_DIR:-$WORK/sraman00/ctrlworld}}
# Run from wherever this file was checked out; see train_mcap.sbatch for why.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

TAG=${TAG:-0003_abc_mcap}
BENCH_DIR=${BENCH_DIR:-$ROOT/outputs/$TAG/wmbench}
MANIFEST=${MANIFEST:-$BENCH_DIR/step200000}
# The run name, not the manifest name: the same manifest gets scored again when a bound
# or a backbone changes, and neither result should overwrite the other.
RUN=${RUN:-$(basename "$MANIFEST")}
OUT=${OUT:-$BENCH_DIR/scores_${RUN}.json}
mkdir -p "$(dirname "$OUT")"

# wmbench_leonardo.sh owns where every weight landed and exports the environment variables
# the metric resources read, so this file names none of them. FVD's I3D is the exception:
# it has a flag and no variable, and `run` exports the path for this line to pass.
I3D_ARG=""
[ -f "${WMBENCH_I3D_CKPT:-$ROOT/i3d_torchscript.pt}" ] && \
  I3D_ARG="--i3d-ckpt ${WMBENCH_I3D_CKPT:-$ROOT/i3d_torchscript.pt}"

echo "wmbench score manifest=$MANIFEST -> $OUT"
# --per-clip because the aggregate is not the result: the report's distributions and its
# best/median/worst examples are drawn from the per-clip rows, and a run scored without
# them cannot be explained after the fact without paying for the whole pass again.
srun --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK --cpu-bind=none \
  scripts/wmbench_leonardo.sh run score \
    --manifest "$MANIFEST" \
    --metrics "${METRICS:-all}" \
    --device cuda \
    --per-clip \
    $I3D_ARG \
    ${NORM_BOUNDS:+--norm-bounds $NORM_BOUNDS} \
    ${CAPTIONS:+--captions $CAPTIONS} \
    ${MASKS:+--masks $MASKS} \
    ${OBJECTS:+--objects $OBJECTS} \
    ${LIMIT:+--limit $LIMIT} \
    --out "$OUT"

# The page is what a number is read from. It needs the manifest as well as the scores,
# because the examples are frames; JUDGE_JSON folds a `wmbench judge` run into the same
# page when there is one.
REPORT=${OUT%.json}.html
srun --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK --cpu-bind=none \
  scripts/wmbench_leonardo.sh run report \
    --manifest "$MANIFEST" \
    --scores "$OUT" \
    --view "${VIEW:-top}" \
    ${JUDGE_JSON:+--judge $JUDGE_JSON} \
    ${NORM_BOUNDS:+--bounds $NORM_BOUNDS} \
    --out "$REPORT"

# The canonical copy lives in the durable tree, which outlives the checkout. `cluster
# fetch` only reaches inside the run dir, so leave a copy there too when this is a
# submitted run -- it is a few KB, and it is what you commit to experiments/<tag>/eval/.
if [ -n "${CLUSTER_OUTPUT_DIR:-}" ]; then
  cp "$OUT" "$REPORT" "$CLUSTER_OUTPUT_DIR/"
fi
