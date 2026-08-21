#!/bin/bash
# Everything the video-metrics evaluation needs on Leonardo, in one place.
#
#   scripts/eval_leonardo.sh setup          # login node, once: deps and backbone weights
#   scripts/eval_leonardo.sh clips          # login node: regenerate the clip list
#   scripts/eval_leonardo.sh run <step>     # inside an sbatch job: score one checkpoint
#   scripts/eval_leonardo.sh tracker_setup  # login node, once: CoTracker3 and its weights
#   scripts/eval_leonardo.sh noisefloor     # inside an sbatch job: tracker error floor
#
# `setup` and `clips` run on the login node because compute nodes have no internet and
# the clip list is built from the validation annotations. `run` is what the batch job
# invokes. See docs/evaluation.md for the full procedure.
set -euo pipefail

ROOT=${ROOT:-$WORK/sraman00/ctrlworld}
DATA=${DATA:-$ROOT/data/abc_rigid}
CLIPS=${CLIPS:-dataset_meta_info/abc_rigid/eval_clips_v1.json}
PY=$ROOT/venv/bin/python
export TORCH_HOME=${TORCH_HOME:-$ROOT/torchhome}

# stylegan-v's I3D, the usual FVD backbone. Pinned so a silently different file (or an
# HTML error page saved under the same name) is caught rather than changing the numbers.
I3D_URL='https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt?dl=1'
I3D_MD5=8d45542982dc92282a73cb852e3c53d4

cmd_setup() {
    set -x
    uv pip install --python "$PY" lpips

    mkdir -p "$TORCH_HOME/hub/checkpoints"
    # LPIPS pulls AlexNet through torchvision on first use; fetch it here instead.
    "$PY" -c 'import lpips; lpips.LPIPS(net="alex"); print("lpips backbone cached")'
    # FID's Inception-v3, same reason.
    [ -f "$TORCH_HOME/hub/checkpoints/inception_v3_google-0cc3c7bd.pth" ] || \
        curl -sSL --max-time 300 -o "$TORCH_HOME/hub/checkpoints/inception_v3_google-0cc3c7bd.pth" \
            https://download.pytorch.org/models/inception_v3_google-0cc3c7bd.pth

    if [ ! -f "$ROOT/i3d_torchscript.pt" ]; then
        curl -sSL --max-time 300 -o "$ROOT/i3d_torchscript.pt" "$I3D_URL"
    fi
    echo "$I3D_MD5  $ROOT/i3d_torchscript.pt" | md5sum -c -
    set +x
    echo "setup complete"
}

COTRACKER_DIR=${COTRACKER_DIR:-$ROOT/cotracker}
COTRACKER_CKPT=$COTRACKER_DIR/scaled_offline.pth
COTRACKER_URL='https://huggingface.co/facebook/cotracker3/resolve/main/scaled_offline.pth'

cmd_tracker_setup() {
    set -x
    uv pip install --python "$PY" 'cotracker @ git+https://github.com/facebookresearch/co-tracker.git'
    mkdir -p "$COTRACKER_DIR"
    # torch.hub.load reaches GitHub at call time, so the checkpoint is fetched here on a
    # login node and passed to the job by path.
    [ -f "$COTRACKER_CKPT" ] || \
        curl -sSL --max-time 600 -o "$COTRACKER_CKPT" "$COTRACKER_URL"
    "$PY" -c 'import cotracker, os, sys; print("cotracker", cotracker.__file__)'
    ls -la "$COTRACKER_CKPT"
    set +x
    echo "tracker setup complete"
}

cmd_noisefloor() {
    set -x
    export HF_HOME=$ROOT/hf HF_HUB_OFFLINE=1
    export PYTHONUNBUFFERED=1
    local out=${OUT:-$ROOT/eval/tracker_noise_floor.json}
    $PY scripts/tracker_noise_floor.py \
        --clips "$CLIPS" \
        --val_dataset_dir "$DATA" \
        --tracker_ckpt "$COTRACKER_CKPT" \
        --n_clips "${N_CLIPS:-10}" \
        --out "$out" \
        "$@"
}

cmd_clips() {
    "$PY" scripts/make_eval_clips.py --val_dataset_dir "$DATA" --out "$CLIPS" "$@"
}

cmd_run() {
    local step=${1:?usage: eval_leonardo.sh run <step> [extra args...]}
    shift
    set -x
    export HF_HOME=$ROOT/hf HF_HUB_OFFLINE=1
    export PYTHONUNBUFFERED=1

    local svd=$HF_HOME/hub/models--stabilityai--stable-video-diffusion-img2vid/snapshots/9cf024d5bfa8f56622af86c884f26a52f6676f2e
    local clip=$HF_HOME/hub/models--openai--clip-vit-base-patch32/snapshots/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268
    local ckpt=${CKPT:-$ROOT/repo/outputs/0002_abc_rigid/model/checkpoint-${step}.pt}
    local out=${OUT:-$ROOT/eval/metrics_step${step}.json}

    # FVD is skipped rather than silently faked when the I3D backbone is missing.
    local i3d_arg=""
    if [ -f "$ROOT/i3d_torchscript.pt" ]; then
        i3d_arg="--i3d_ckpt $ROOT/i3d_torchscript.pt"
    else
        echo "WARNING: $ROOT/i3d_torchscript.pt missing, FVD will be skipped; run setup" >&2
    fi

    $PY scripts/eval_video_metrics.py \
        --ckpt_path "$ckpt" \
        --clips "$CLIPS" \
        --val_dataset_dir "$DATA" \
        --data_stat_path dataset_meta_info/abc_rigid/stat.json \
        --svd_model_path "$svd" --clip_model_path "$clip" \
        --batch_size "${BATCH_SIZE:-4}" \
        --fid $i3d_arg \
        --dist_bootstrap "${DIST_BOOTSTRAP:-100}" \
        --out "$out" \
        "$@"
}

case "${1:-}" in
    setup) shift; cmd_setup "$@" ;;
    clips) shift; cmd_clips "$@" ;;
    run)   shift; cmd_run "$@" ;;
    tracker_setup) shift; cmd_tracker_setup "$@" ;;
    noisefloor)    shift; cmd_noisefloor "$@" ;;
    *) sed -n '2,12p' "$0" >&2; exit 2 ;;
esac
