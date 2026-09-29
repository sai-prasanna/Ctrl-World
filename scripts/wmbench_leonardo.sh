#!/bin/bash
# Everything the wmbench benchmark needs on Leonardo, in one place.
#
#   scripts/wmbench_leonardo.sh setup       # login node, once: venv_bench and every weight
#   scripts/wmbench_leonardo.sh mamba       # login node: compile VFIMamba's CUDA extensions
#   scripts/wmbench_leonardo.sh check       # anywhere: report torch, CUDA and the weights
#   scripts/wmbench_leonardo.sh run <args>  # inside a job: `wmbench <args>` pointed offline
#
# `setup` runs on a login node because compute nodes have no internet and wmbench's
# sixteen metrics pull sixteen backbones. `run` is what the batch jobs invoke; it exports
# the flags' environment variables so no caller has to remember where a weight landed.
#
# This is the eval_leonardo.sh of the second environment. venv_bench is separate from the
# Ctrl-World venv on purpose: the benchmark pins torch 2.11 and transformers 5, Ctrl-World
# pins torch 2.7.1 and diffusers, and the two resolve against each other only by
# downgrading a metric backbone. The rollout crosses the gap over a socket, not an import
# (wmbench.ipc), so nothing here needs Ctrl-World's packages.
set -euo pipefail

# $ROOT holds everything that outlives a run. Same variable, same default and same
# precedence as the job scripts, so `CTRLWORLD_ROOT=/path/to/scratch` reproduces this
# whole environment on another machine.
ROOT=${CTRLWORLD_ROOT:-${CLUSTER_SHARED_DIR:-$WORK/sraman00/ctrlworld}}
VENV=${VENV_BENCH:-$ROOT/venv_bench}
PY=$VENV/bin/python
# Weights that are not on the Hub and have no cache of their own to fall into.
WEIGHTS=${WMBENCH_WEIGHTS:-$ROOT/weights}
ABC_SRC=${ABC_SRC:-$ROOT/third_party/abc}
# The benchmark source. Until wmbench has a git remote, `setup` mirrors the laptop
# checkout here with rsync; see cmd_setup.
WMBENCH_SRC=${WMBENCH_SRC:-$ROOT/wmbench}

export HF_HOME=${HF_HOME:-$ROOT/hf}
export TORCH_HOME=${TORCH_HOME:-$ROOT/torchhome}
# $HOME is a 50 GB quota shared with everything else; uv's cache holds multi-gigabyte
# CUDA wheels, so it goes on $WORK with the rest of the run's state.
export UV_CACHE_DIR=${UV_CACHE_DIR:-$ROOT/uvcache}

# The repository this file was checked out from, so `setup` can install `abc130k` from it
# and `run` can put it first on the path. jobs/../, not this directory.
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

# The ABC release the policy loop drives, pinned. A SHA rather than a branch: abc_minimal's
# action layout and its RTC evaluation defaults are what the sim success rates were
# measured against, and both have moved since.
ABC_SHA=6c467cebcecf16a4dce79e6fd87a7ca2281c3ef0
ABC_URL=https://github.com/amazon-far/abc.git
# ABC's checkpoint host rejects a request without this User-Agent with a 403 (prepare.py).
ABC_DATA_BASE=https://abc-data.timehorizons.org
ABC_UA='abc-prepare/1.0'

# huggingface.co/MCG-NJU/VFIMamba, 264,414,373 bytes.
VFIMAMBA_SHA256=c1dac5b08f4c41e95452f7a41b35347409fc58b0e25b3ba8de899144ff28a350
# The two CUDA extensions VFIMamba needs, as the release wheels rather than an sdist: both
# projects publish one wheel per (CUDA, torch minor, ABI, Python) and building from source
# takes half an hour of nvcc on a shared login node. cu12torch2.10 under torch 2.11+cu128
# is a deliberate mismatch that holds -- the extension ABI did not move across that torch
# minor, and `selective_scan_fn` agrees with `selective_scan_ref` to 3e-6.
MAMBA_WHEEL=https://github.com/state-spaces/mamba/releases/download/v2.3.2.post1/mamba_ssm-2.3.2.post1+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
CAUSAL_CONV1D_WHEEL=https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.7.0/causal_conv1d-1.7.0+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl

# torch 2.11 has no cu126 wheel newer than this pair, and ABC pins the +cu128 build
# exactly. CU_TAG is the one knob for the driver fallback: Leonardo's driver is 535.274.02
# (CUDA 12.2), which serves a 12.8 runtime through CUDA minor-version compatibility, so
# cu128 is what this installs. Set CU_TAG=cu126 if a node ever reports otherwise.
CU_TAG=${CU_TAG:-cu128}
TORCH_VERSION=${TORCH_VERSION:-2.11.0}
TORCHVISION_VERSION=${TORCHVISION_VERSION:-0.26.0}

hf_get() {  # repo [allow-pattern ...]
    HF_HUB_OFFLINE=0 "$PY" - "$@" <<'PY'
import sys
from huggingface_hub import snapshot_download
repo, allow = sys.argv[1], sys.argv[2:] or None
print(snapshot_download(repo_id=repo, allow_patterns=allow), flush=True)
PY
}

fetch() {  # url dest [curl args...]
    local url=$1 dest=$2; shift 2
    if [ -s "$dest" ]; then echo "have $dest"; return; fi
    mkdir -p "$(dirname "$dest")"
    curl -fsSL --retry 3 --retry-delay 5 -o "$dest.part" "$@" "$url"
    mv "$dest.part" "$dest"
    echo "got $dest ($(du -h "$dest" | cut -f1))"
}

cmd_setup() {
    mkdir -p "$ROOT" "$WEIGHTS" "$UV_CACHE_DIR" "$HF_HOME"

    # ---------------------------------------------------------------- the environment
    if [ ! -x "$PY" ]; then
        uv venv --python 3.12 "$VENV"
    fi

    # torch first, from the CUDA index. Everything after it asks for `torch==2.11.0`,
    # which a `+cu128` local version satisfies, so uv leaves this build in place instead
    # of pulling the PyPI default.
    uv pip install --python "$PY" \
        --index-url "https://download.pytorch.org/whl/$CU_TAG" \
        "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION"
    # ABC decodes video with torchcodec, and only the CPU build is on that index.
    uv pip install --python "$PY" \
        --index-url https://download.pytorch.org/whl/cpu "torchcodec==0.11.0"

    # ---------------------------------------------------------------- ABC, pinned
    # Staged as a checkout at a SHA rather than vendored: it is someone else's code, and
    # the policy loop imports it directly.
    if [ ! -d "$ABC_SRC/.git" ]; then
        mkdir -p "$(dirname "$ABC_SRC")"
        git clone "$ABC_URL" "$ABC_SRC"
    fi
    git -C "$ABC_SRC" fetch --all --tags
    git -C "$ABC_SRC" checkout --detach "$ABC_SHA"

    # --no-deps and an explicit list: abc-minimal's dependencies include mujoco-warp,
    # warp-lang, viser and mjviser, which are the simulator half. Phase A drives the
    # policy against Ctrl-World, not against MuJoCo, and warp compiles kernels at import
    # time against a CUDA toolkit the compute nodes do not expose. Phase B adds them back.
    #
    # Plain `mujoco` stays, because it is not optional in practice: abc_minimal's
    # eval_policy imports abc_sim.randomization.core, which imports mujoco at module
    # scope, so even a sim-free policy run fails at import without it. 3.13.0 rather than
    # the pyproject's `~=3.8.0`: that pin exists for mujoco-warp 3.8.0.3, which is not
    # installed here, and 3.13 is the version verified against torch 2.11 cu128.
    uv pip install --python "$PY" --no-deps -e "$ABC_SRC"
    uv pip install --python "$PY" \
        numpy "imageio[ffmpeg]" imageio-ffmpeg av ftfy regex sentencepiece tyro \
        huggingface_hub mcap mcap-protobuf-support wandb "mujoco==3.13.0"

    # ---------------------------------------------------------------- wmbench
    # The rsync is the working default because the repository is private and the login
    # node authenticates to GitHub with its own key, which does not carry a forwarded
    # agent. Once that key covers wmbench, this replaces everything down to the install:
    #   uv pip install --python "$PY" \
    #     "wmbench[metrics,judge,lerobot,jepa] @ git+ssh://git@github.com/sai-prasanna/wmbench@68109b6"
    # Until then the laptop checkout is the only copy, so mirror it. --delete keeps a
    # stale module from shadowing a renamed one; .venv and .git are the laptop's, not
    # this machine's.
    if [ -n "${WMBENCH_RSYNC_FROM:-}" ]; then
        mkdir -p "$WMBENCH_SRC"
        rsync -a --delete --exclude .venv --exclude .git --exclude '__pycache__' \
            "$WMBENCH_RSYNC_FROM/" "$WMBENCH_SRC/"
    fi
    uv pip install --python "$PY" -e "$WMBENCH_SRC[metrics,judge,lerobot]"

    # `jepa` is installed separately and without its dependencies. vjepa 0.1.2 pins
    # torchvision >=0.19.1,<0.20, which no resolver can reconcile with the 0.26 the
    # metrics need, and uv refuses the whole install rather than the one extra. The pin is
    # nominal: the package is 59 KB of model code that imports plain torch, and
    # wmbench only calls `vit_huge` and `AttentiveClassifier` out of it.
    uv pip install --python "$PY" --no-deps vjepa==0.1.2

    # abc130k stays editable from this checkout: it is this repository's package, and the
    # job scripts put $PWD/abc130k/src first on PYTHONPATH anyway so a `cluster submit`
    # checkout's copy wins. The install is what makes it importable from a login node.
    uv pip install --python "$PY" -e "$REPO/abc130k"

    # The model venv gets wmbench's core too, and only the core. wmbench_rollout_ctrlworld
    # runs there, on torch 2.7.1 and diffusers, and writes a manifest -- which is numpy and
    # PyAV, both already installed. --no-deps is the whole point: the metrics extra would
    # pull torch 2.11 into the environment the checkpoint loads in.
    uv pip install --python "$ROOT/venv/bin/python" --no-deps -e "$WMBENCH_SRC"

    # ---------------------------------------------------------------- weights
    mkdir -p "$TORCH_HOME/hub/checkpoints" "$TORCH_HOME/hub/pyiqa"

    # Hugging Face repositories. HF_HUB_OFFLINE is set for jobs; here it is explicitly
    # off, because this is the one place that is allowed to reach the network.
    hf_get facebook/dino-vitb16                          # subject/cross-view consistency
    hf_get openai/clip-vit-base-patch32                  # background consistency, action following
    hf_get openai/clip-vit-large-patch14                 # aesthetic quality
    hf_get openai/clip-vit-base-patch16                  # semantic alignment
    hf_get depth-anything/Depth-Anything-V2-Small-hf     # depth accuracy
    hf_get IDEA-Research/grounding-dino-base             # trajectory accuracy (object tracks)
    hf_get hzwer/RIFE 'RIFEv3.6_HD_preview.zip'          # motion smoothness
    hf_get Qwen/Qwen3-VL-8B-Instruct                     # judge rubrics
    hf_get Qwen/Qwen2.5-VL-7B-Instruct                   # judge captions

    # RIFE ships one zip and wmbench wants the file inside it.
    if [ ! -s "$WEIGHTS/rife/flownet.pkl" ]; then
        mkdir -p "$WEIGHTS/rife"
        HF_HUB_OFFLINE=0 "$PY" - "$WEIGHTS/rife" <<'PY'
import os
import sys
import zipfile
from huggingface_hub import hf_hub_download
path = hf_hub_download('hzwer/RIFE', 'RIFEv3.6_HD_preview.zip')
with zipfile.ZipFile(path) as z:
    name = next(n for n in z.namelist() if n.endswith('flownet.pkl'))
    with z.open(name) as src, open(os.path.join(sys.argv[1], 'flownet.pkl'), 'wb') as dst:
        dst.write(src.read())
print('extracted flownet.pkl', flush=True)
PY
    fi

    # pyiqa resolves the MUSIQ checkpoint when the model is built, into its own cache
    # under $TORCH_HOME. Building it once here is the only way to fill that cache.
    # pretrained='spaq', not the default: pyiqa's default is the KonIQ fine-tune, and
    # imaging_quality is defined against the SPAQ one, which is what VBench scores with.
    "$PY" -c 'import pyiqa; pyiqa.create_metric("musiq", device="cpu", pretrained="spaq"); print("musiq cached")'

    # torchvision and lpips download their backbones on first use, which a compute node
    # cannot do. Named URLs rather than a warm-up import: the file names are what the
    # hub cache keys on, so a wrong name re-downloads at score time and fails there.
    fetch https://download.pytorch.org/models/raft_large_C_T_V2-1bb1363a.pth \
        "$TORCH_HOME/hub/checkpoints/raft_large_C_T_V2-1bb1363a.pth"
    fetch https://download.pytorch.org/models/inception_v3_google-0cc3c7bd.pth \
        "$TORCH_HOME/hub/checkpoints/inception_v3_google-0cc3c7bd.pth"
    "$PY" -c 'import lpips; lpips.LPIPS(net="alex"); print("lpips backbone cached")'

    # The LAION aesthetic head: 4 KB, and the only thing standing between CLIP ViT-L/14
    # and a number.
    fetch https://github.com/LAION-AI/aesthetic-predictor/raw/main/sa_0_4_vit_l_14_linear.pth \
        "$WEIGHTS/sa_0_4_vit_l_14_linear.pth"

    # VFIMamba, the interpolator WorldArena measures motion_smoothness with. wmbench
    # follows it here rather than substituting, so the weight is the metric's definition
    # and not an implementation detail; RIFE stays as the fallback when the CUDA
    # extensions are missing (`mamba`, below).
    fetch https://huggingface.co/MCG-NJU/VFIMamba/resolve/main/model.pkl \
        "$WEIGHTS/vfimamba/model.pkl"
    # Pinned by digest: the repository has no revision in its URL, and a silently
    # different interpolator changes motion_smoothness without changing any code.
    echo "$VFIMAMBA_SHA256  $WEIGHTS/vfimamba/model.pkl" | sha256sum -c -

    # V-JEPA: 10 GB for one corpus-level number, but jepa_similarity is what WorldArena
    # replaced FVD with, and our clips are 5 fps, outside I3D's temporal prior.
    fetch https://dl.fbaipublicfiles.com/jepa/vith16/vith16.pth.tar \
        "$WEIGHTS/vjepa/vith16.pth.tar"
    fetch https://dl.fbaipublicfiles.com/jepa/vith16/ssv2-probe.pth.tar \
        "$WEIGHTS/vjepa/ssv2-probe.pth.tar"

    # ---------------------------------------------------------------- ABC weights
    # The multi-task sim DiT-XL parent, model-only fp32, and its sidecar. The sidecar
    # carries sim_prompt_map, without which the task prompts do not match training.
    fetch "$ABC_DATA_BASE/checkpoints/abc_dit_xl_200k_model.pt" \
        "$WEIGHTS/abc/abc_dit_xl_200k_model.pt" -H "User-Agent: $ABC_UA"
    fetch "$ABC_DATA_BASE/checkpoints/abc_dit_xl_200k_model.json" \
        "$WEIGHTS/abc/abc_dit_xl_200k_model.json" -H "User-Agent: $ABC_UA"

    # ABC's own CLIP assets, at the path its ClipConfig defaults to. These are OpenAI's
    # original checkpoints, not the HF ports above: the policy's text tower loads the
    # .pt and tokenizes with the gzipped BPE, so the HF repository cannot stand in.
    fetch https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt \
        "$HOME/.cache/clip/ViT-B-32.pt"
    fetch https://github.com/openai/CLIP/raw/main/clip/bpe_simple_vocab_16e6.txt.gz \
        "$HOME/.cache/clip/bpe_simple_vocab_16e6.txt.gz"

    echo "setup complete: $VENV"
    cmd_check
}

# Everything `run` exports. Sourced by the job scripts' `run`, and by `check` so the two
# agree on where a weight is looked for.
bench_env() {
    export HF_HUB_OFFLINE=1
    export PYTHONUNBUFFERED=1
    export PATH=$VENV/bin:$PATH
    # The three flags whose defaults point at ~/.cache/wmbench, which is on the small home
    # quota and is not where setup staged anything.
    export WMBENCH_RAFT_CKPT=${WMBENCH_RAFT_CKPT:-$TORCH_HOME/hub/checkpoints/raft_large_C_T_V2-1bb1363a.pth}
    export WMBENCH_MUSIQ_CKPT=${WMBENCH_MUSIQ_CKPT:-$TORCH_HOME/hub/pyiqa/musiq_spaq_ckpt-358bb6af.pth}
    export WMBENCH_AESTHETIC_HEAD=${WMBENCH_AESTHETIC_HEAD:-$WEIGHTS/sa_0_4_vit_l_14_linear.pth}
    export WMBENCH_RIFE_CKPT=${WMBENCH_RIFE_CKPT:-$WEIGHTS/rife/flownet.pkl}
    export WMBENCH_VJEPA_DIR=${WMBENCH_VJEPA_DIR:-$WEIGHTS/vjepa}
    # VFIMamba's weight. Exported ahead of the metric that reads it, so every prestaged
    # path is declared in one place.
    export WMBENCH_VFIMAMBA_CKPT=${WMBENCH_VFIMAMBA_CKPT:-$WEIGHTS/vfimamba/model.pkl}
    # FVD's I3D has no environment variable, only --i3d-ckpt, so the job scripts read
    # this one and pass the flag.
    export WMBENCH_I3D_CKPT=${WMBENCH_I3D_CKPT:-$ROOT/i3d_torchscript.pt}
    export ABC_CKPT_DIR=${ABC_CKPT_DIR:-$WEIGHTS/abc}
    # This checkout's abc130k wins over the editable install, which may point at a
    # checkout `cluster submit` has already deleted.
    export PYTHONPATH=$REPO/abc130k/src${PYTHONPATH:+:$PYTHONPATH}
}

# VFIMamba's two CUDA extensions, separate from `setup` because they are the only part of
# the environment that can fail on its own without costing anything: motion_smoothness
# falls back to RIFE, and the other fifteen metrics do not know these exist. Login node,
# for the network and, in the fallback, for nvcc.
cmd_mamba() {
    # --no-deps because each wheel's metadata asks for the torch minor it was compiled
    # against (2.10), which would drag this venv off the 2.11 the rest of it is pinned to.
    # The ABI is what matters and it holds; see the wheel URLs above.
    uv pip install --python "$PY" --no-deps "$CAUSAL_CONV1D_WHEEL" "$MAMBA_WHEEL"
    uv pip install --python "$PY" einops
    if "$PY" -c 'import causal_conv1d, mamba_ssm' 2>/dev/null; then
        "$PY" -c 'import causal_conv1d, mamba_ssm; print("causal_conv1d", causal_conv1d.__version__); print("mamba_ssm", mamba_ssm.__version__)'
        return
    fi

    # Both projects build their release wheels on a newer distribution than Leonardo runs:
    # they need GLIBC_2.32 and the login and compute nodes have 2.28 (AlmaLinux 8.10). The
    # import fails at the .so, not at the install, so the wheels have to be replaced rather
    # than diagnosed from pip's output.
    echo "prebuilt wheels do not import here; building from source" >&2
    uv pip uninstall --python "$PY" mamba-ssm causal-conv1d || true
    # gcc as well as cuda: /usr/bin/c++ is 8.5 and torch's headers refuse anything before
    # 9, so a build that only loads the toolkit fails in C++17.h rather than in nvcc.
    module load cuda/12.3 gcc/12.2.0 2>/dev/null || \
        echo "no cuda/gcc modules; using whatever is on PATH" >&2
    export CUDA_HOME=${CUDA_HOME:-${CUDA_ROOT:-/usr/local/cuda}}
    export CC=${CC:-$(command -v gcc)} CXX=${CXX:-$(command -v g++)}
    # The login node is shared and its process limit is low; a full-width build is how one
    # user earns everyone a 2048-process refusal.
    export MAX_JOBS=${MAX_JOBS:-4}
    # --no-build-isolation is mandatory, not a speed-up: both setup.py files import torch
    # to find the ABI they compile against, and an isolated build has no torch to import.
    uv pip install --python "$PY" --no-build-isolation \
        --no-binary causal-conv1d --no-binary mamba-ssm \
        "causal-conv1d==1.7.0" "mamba-ssm==2.3.2.post1"
    "$PY" -c 'import causal_conv1d, mamba_ssm; print("causal_conv1d", causal_conv1d.__version__); print("mamba_ssm", mamba_ssm.__version__)'
}

cmd_run() {
    bench_env
    set -x
    exec "$VENV/bin/wmbench" "$@"
}

cmd_check() {
    bench_env
    "$PY" - <<'PY'
import os, torch, torchvision
print(f'python         {os.sys.version.split()[0]}')
print(f'torch          {torch.__version__}')
print(f'torchvision    {torchvision.__version__}')
print(f'torch cuda     {torch.version.cuda}')
print(f'cuda available {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'device         {torch.cuda.get_device_name(0)}')
    print(f'capability     {torch.cuda.get_device_capability(0)}')
    # The cu128 question, answered by arithmetic rather than by a version comparison.
    x = torch.ones(1024, 1024, device='cuda')
    print(f'matmul         {(x @ x).sum().item():.0f}')
for mod in ('transformers', 'pyiqa', 'timm', 'lpips', 'scipy', 'vjepa', 'lerobot',
            'wmbench', 'abc130k', 'abc_minimal', 'mujoco', 'causal_conv1d',
            'mamba_ssm'):
    try:
        m = __import__(mod)
        print(f'{mod:<14} {getattr(m, "__version__", "ok")}')
    except Exception as exc:
        print(f'{mod:<14} MISSING ({type(exc).__name__}: {exc})')
PY
    echo
    echo "weights:"
    local hf=$HF_HOME/hub
    for p in \
        "$hf/models--facebook--dino-vitb16" \
        "$hf/models--openai--clip-vit-base-patch32" \
        "$hf/models--openai--clip-vit-large-patch14" \
        "$hf/models--openai--clip-vit-base-patch16" \
        "$hf/models--depth-anything--Depth-Anything-V2-Small-hf" \
        "$hf/models--IDEA-Research--grounding-dino-base" \
        "$hf/models--Qwen--Qwen3-VL-8B-Instruct" \
        "$hf/models--Qwen--Qwen2.5-VL-7B-Instruct" \
        "$TORCH_HOME/hub/checkpoints/raft_large_C_T_V2-1bb1363a.pth" \
        "$TORCH_HOME/hub/checkpoints/inception_v3_google-0cc3c7bd.pth" \
        "$TORCH_HOME/hub/checkpoints/alexnet-owt-7be5be79.pth" \
        "$WMBENCH_MUSIQ_CKPT" "$WMBENCH_AESTHETIC_HEAD" "$WMBENCH_RIFE_CKPT" \
        "$WMBENCH_VJEPA_DIR/vith16.pth.tar" "$WMBENCH_VJEPA_DIR/ssv2-probe.pth.tar" \
        "$WMBENCH_VFIMAMBA_CKPT" \
        "$WMBENCH_I3D_CKPT" \
        "$ABC_CKPT_DIR/abc_dit_xl_200k_model.pt" \
        "$ABC_CKPT_DIR/abc_dit_xl_200k_model.json" \
        "$HOME/.cache/clip/ViT-B-32.pt" \
        "$HOME/.cache/clip/bpe_simple_vocab_16e6.txt.gz"
    do
        if [ -e "$p" ]; then
            printf '  ok      %8s  %s\n' "$(du -sh "$p" | cut -f1)" "$p"
        else
            printf '  MISSING %8s  %s\n' '-' "$p"
        fi
    done
}

case "${1:-}" in
    setup) shift; cmd_setup "$@" ;;
    mamba) shift; cmd_mamba "$@" ;;
    check) shift; cmd_check "$@" ;;
    run)   shift; cmd_run "$@" ;;
    *) sed -n '2,12p' "$0" >&2; exit 2 ;;
esac
