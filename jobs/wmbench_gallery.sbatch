#!/bin/bash
# Build the wmbench case gallery for a scored manifest.
#
# The evidence behind wmbench_score.sbatch's numbers: per metric, the best and the worst
# clip as video with the metric drawn onto the frames. It recomputes each signal with the
# scorer's backbones, so it runs in venv_bench on a GPU like scoring does, and reads the
# manifest and the scores JSON that job wrote. A second manifest and scores file add the
# tab that wipes one checkpoint against the other.
#
#   sbatch --export=ALL,MANIFEST=$ROOT/outputs/0003_abc_mcap/wmbench/step200000,SCORES=$ROOT/outputs/0003_abc_mcap/wmbench/scores_step200000.json wmbench_gallery.sbatch
#
#SBATCH -A aifac_s07_034
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -t 02:00:00
#SBATCH -J wmb_gallery
#SBATCH -o /leonardo_work/AIFAC_S07_034/sraman00/ctrlworld/logs_wmbench_gallery-%j.out
# $ROOT holds everything that outlives a run; see wmbench_score.sbatch.
ROOT=${CTRLWORLD_ROOT:-${CLUSTER_SHARED_DIR:-$WORK/sraman00/ctrlworld}}
# Run from wherever this file was checked out; see train_mcap.sbatch for why.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

TAG=${TAG:-0003_abc_mcap}
BENCH_DIR=${BENCH_DIR:-$ROOT/outputs/$TAG/wmbench}
MANIFEST=${MANIFEST:-$BENCH_DIR/step200000}
RUN=${RUN:-$(basename "$MANIFEST")}
SCORES=${SCORES:-$BENCH_DIR/scores_${RUN}.json}
# The earlier checkpoint is optional; without it the page has no two-run tab.
MANIFEST_PREV=${MANIFEST_PREV:-}
RUN_PREV=${RUN_PREV:-${MANIFEST_PREV:+$(basename "$MANIFEST_PREV")}}
SCORES_PREV=${SCORES_PREV:-${MANIFEST_PREV:+$BENCH_DIR/scores_${RUN_PREV}.json}}
OUT=${OUT:-$BENCH_DIR/gallery_${RUN}}
mkdir -p "$OUT"

echo "wmbench gallery manifest=$MANIFEST scores=$SCORES prev=${MANIFEST_PREV:-none} -> $OUT"
srun --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK --cpu-bind=none \
  scripts/wmbench_leonardo.sh run gallery \
    --manifest "$MANIFEST" --scores "$SCORES" --label "$RUN" \
    ${MANIFEST_PREV:+--manifest-prev "$MANIFEST_PREV" --scores-prev "$SCORES_PREV" --label-prev "$RUN_PREV"} \
    --device cuda \
    ${VIEW:+--view $VIEW} \
    ${METRICS:+--metrics $METRICS} \
    ${JUDGE:+--judge $JUDGE} \
    ${CAPTIONS:+--captions $CAPTIONS} \
    ${OBJECTS:+--objects $OBJECTS} \
    ${N_CASES:+--n-cases $N_CASES} \
    --out-dir "$OUT"

# The page is small; the videos are the run's. `cluster fetch` copies the run dir back,
# so leave the page and its listing there and the videos in the durable tree.
if [ -n "${CLUSTER_OUTPUT_DIR:-}" ]; then
  cp "$OUT/gallery.html" "$OUT/gallery.json" "$CLUSTER_OUTPUT_DIR/"
fi
