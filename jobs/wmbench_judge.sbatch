#!/bin/bash
# Run a VLM rubric over a wmbench manifest.
#
# Split from wmbench_score.sbatch because a judge is a different kind of measurement: its
# number is the pair (prompt, model), it costs a forward pass of an 8B VLM per clip rather
# than a backbone per frame, and it is the half that has to be re-run when a rubric is
# reworded. Same venv_bench, same offline weights.
#
#   sbatch --export=ALL,RUBRIC=triad,MANIFEST=$ROOT/outputs/0003_abc_mcap/wmbench/step200000 wmbench_judge.sbatch
#
#SBATCH -A aifac_s07_034
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -t 04:00:00
#SBATCH -J wmb_judge
#SBATCH -o /leonardo_work/AIFAC_S07_034/sraman00/ctrlworld/logs_wmbench_judge-%j.out
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
# The rubric names the metric, so it names the output file too: a triad score and a
# caption run over the same manifest are two results, not two versions of one. `triad` is
# the CLI's alias for the three Likert rubrics WorldArena reads out of one answer
# (interaction_quality, perspectivity, instruction_following); `success` and `caption` are
# the other two names it takes.
RUBRIC=${RUBRIC:-triad}
RUN=${RUN:-$(basename "$MANIFEST")}
OUT=${OUT:-$BENCH_DIR/judge_${RUBRIC}_${RUN}.json}
mkdir -p "$(dirname "$OUT")"

# Temperature 0 is the judge's default and the reason a rerun reproduces its own numbers;
# MODEL is here so a smaller Qwen3-VL variant can stand in for a smoke run without
# anyone editing the file, but a comparable number needs the pinned default.
echo "wmbench judge rubric=$RUBRIC manifest=$MANIFEST -> $OUT"
srun --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK --cpu-bind=none \
  scripts/wmbench_leonardo.sh run judge \
    --manifest "$MANIFEST" \
    --rubric "$RUBRIC" \
    --device cuda \
    ${MODEL:+--model $MODEL} \
    ${VIEW:+--view $VIEW} \
    ${N_FRAMES:+--n-frames $N_FRAMES} \
    ${LIMIT:+--limit $LIMIT} \
    ${SUMMARY:+--summary $SUMMARY} \
    --out "$OUT"

# The canonical copy lives in the durable tree, which outlives the checkout. `cluster
# fetch` only reaches inside the run dir, so leave a copy there too when this is a
# submitted run.
if [ -n "${CLUSTER_OUTPUT_DIR:-}" ]; then
  cp "$OUT" "$CLUSTER_OUTPUT_DIR/"
fi
