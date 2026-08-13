#!/bin/bash
# One-shot rebuild of the whole Prism experiment environment on a fresh GPU box.
#
#   git clone https://github.com/meojun/Prism-singleGPU-A100- prism-exp
#   cd prism-exp && ./bootstrap.sh
#
# Everything is pinned (setup/pins.env + setup/requirements.lock.txt), so this
# reproduces the exact stack that produced exp/results/1-env-verification/ -- it does NOT
# re-resolve dependencies. Re-resolving is what breaks this repo (see §"Traps").
#
# Idempotent: re-running skips work that is already done. Safe to re-run after a
# partial failure.
#
# Options (env vars):
#   SKIP_MODELS=1    don't download the Llama weights
#   SKIP_KVCACHED_MAIN=1  don't clone the standalone kvcached main branch
#   ROOT=<path>      install somewhere other than this repo's directory
set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
cd "$ROOT"
source "$ROOT/setup/pins.env"

VENV=$ROOT/prism-venv
say() { echo -e "\n=== $* ==="; }
have() { command -v "$1" >/dev/null 2>&1; }

# --- 0. preflight -------------------------------------------------------------
say "[0/8] preflight"
have nvidia-smi || { echo "FATAL: no nvidia-smi -- is this a GPU box?" >&2; exit 1; }
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader

CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')
CC_MAJOR=${CC%%.*}
if [ "${CC_MAJOR:-0}" -ge 10 ]; then
    cat >&2 <<WARN

!! WARNING: compute capability $CC (Blackwell or newer).
!! This stack is pinned to torch 2.4.0+cu121, which has NO kernels for this arch.
!! It will install fine and then die with "no kernel image is available for
!! execution on the device" at the first GPU op. This repo targets A100 (8.0) /
!! H100 (9.0). You need a different torch/vllm/flashinfer set for Blackwell,
!! which means re-resolving the whole stack -- the lockfile will not help you.

WARN
    read -r -p "Continue anyway? [y/N] " a; [ "$a" = y ] || exit 1
fi

have git || { echo "FATAL: git missing" >&2; exit 1; }
have cc  || echo "WARN: no C compiler on PATH -- kvcached's extension build will fail"
if ! have uv; then
    say "installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# --- 1. sources at pinned revisions -------------------------------------------
clone_at() {  # clone_at <dir> <repo> <sha> [branch]
    local dir=$1 repo=$2 sha=$3 branch=${4:-}
    if [ -d "$dir/.git" ]; then
        echo "  $dir already present ($(git -C "$dir" rev-parse --short HEAD))"
    else
        if [ -n "$branch" ]; then git clone -b "$branch" "$repo" "$dir"
        else git clone "$repo" "$dir"; fi
    fi
    git -C "$dir" fetch --depth=1 origin "$sha" 2>/dev/null || git -C "$dir" fetch origin
    git -C "$dir" checkout -q "$sha"
    echo "  $dir @ $(git -C "$dir" rev-parse --short HEAD)"
}

say "[1/8] clone sources at pinned revisions"
clone_at "$ROOT/prism-research" "$PRISM_RESEARCH_REPO" "$PRISM_RESEARCH_SHA"
clone_at "$ROOT/kvcached-prism" "$KVCACHED_REPO" "$KVCACHED_PRISM_SHA" "$KVCACHED_PRISM_BRANCH"
if [ "${SKIP_KVCACHED_MAIN:-0}" != 1 ]; then
    clone_at "$ROOT/kvcached" "$KVCACHED_REPO" "$KVCACHED_MAIN_SHA" main
fi

# --- 2. venv ------------------------------------------------------------------
say "[2/8] python $PYTHON_VERSION venv"
[ -d "$VENV" ] || uv venv "$VENV" --python="$PYTHON_VERSION"
source "$VENV/bin/activate"

# --- 3. torch (custom index) --------------------------------------------------
# PyPI is needed as a SECOND index here, not as a convenience. torch 2.4.0+cu121
# pins nvidia-cudnn-cu12==9.1.0.70 exactly, and download.pytorch.org/whl/cu121 has
# since PRUNED that file (it serves 9.0.0.312 and then jumps to 9.2x). With a bare
# --index-url (which *replaces* the default index) the resolve now dies with
# "no version of nvidia-cudnn-cu12==9.1.0.70 ... torch==2.4.0+cu121 cannot be used".
# PyPI still carries the wheel. --index-strategy unsafe-best-match lets uv take the
# nvidia-* deps from PyPI while torch itself still resolves to the local-version
# 2.4.0+cu121 from the pytorch index (a local version sorts above plain 2.4.0).
say "[3/8] torch $TORCH_VERSION (cu121)"
python -c "import torch,sys; sys.exit(0 if torch.__version__.startswith('$TORCH_VERSION') else 1)" 2>/dev/null \
  && echo "  torch already installed" \
  || uv pip install "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION" \
       --index-url "$TORCH_INDEX" \
       --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match

# --- 4. flashinfer (custom index) ---------------------------------------------
say "[4/8] flashinfer $FLASHINFER_VERSION"
python -c "import flashinfer" 2>/dev/null \
  && echo "  flashinfer already installed" \
  || uv pip install "flashinfer==$FLASHINFER_VERSION" -i "$FLASHINFER_INDEX"

# --- 5. everything else, exact pins -------------------------------------------
# We install the LOCKFILE, never `-e python[all]`. That extra has no upper bound
# on transformers, so a fresh resolve pulls 5.x and breaks vLLM 0.6.3.post1 with
# `ImportError: cannot import name 'DTensor'`. The lock also carries the
# pyairports git URL (its PyPI release was pulled) and setuptools<81.
#
# --no-deps is REQUIRED, not an optimisation. The lock is a complete freeze of a
# working environment, but that environment is not internally consistent:
# litellm 1.95.0 declares tokenizers>=0.21 while transformers 4.45.2 pins
# tokenizers 0.20.3, so any real resolve dies with "No solution found". The
# conflict is harmless in practice -- litellm is only reachable from
# sglang/lang/backend/litellm.py, which the multi-model server never imports.
# --no-deps installs the exact frozen set and sidesteps the resolver entirely.
say "[5/8] pinned dependency set (${ROOT}/setup/requirements.lock.txt)"
uv pip install --no-deps -r "$ROOT/setup/requirements.lock.txt"

# --- 6. the two editable packages, no dep resolution --------------------------
say "[6/8] prism-research (sglang fork) + kvcached, editable"
uv pip install -e "$ROOT/prism-research/python" --no-deps
uv pip install -e "$ROOT/kvcached-prism" --no-deps --no-build-isolation
( cd "$ROOT/kvcached-prism" && python setup.py build_ext --inplace )

# --- 7. profiled model table --------------------------------------------------
# Prism's GPU scheduler refuses to start for a model missing from this file
# (`ValueError: Model path ... not found in the profiled model info file`).
# Ours has the paper's Llama/Mistral entries plus the Qwen2.5 ones we profiled.
say "[7/8] profiled model_info.json"
MI=$ROOT/prism-research/python/sglang/multi_model/utils/model_info.json
cp "$MI" "$MI.upstream.bak" 2>/dev/null || true
cp "$ROOT/setup/model_info.json" "$MI"
python -c "import json;print('  %d profiled entries'%len(json.load(open('$MI'))))"

# --- 8. redis -----------------------------------------------------------------
# Prism's controller and GPU scheduler both need it on 127.0.0.1:6379.
say "[8/8] redis"
if redis-cli ping >/dev/null 2>&1; then
    echo "  redis already answering on :6379"
elif have supervisorctl && supervisorctl status redis >/dev/null 2>&1; then
    supervisorctl start redis || true
elif have redis-server; then
    redis-server --daemonize yes
    echo "  started redis-server (NOT supervised -- it dies on reboot)"
else
    echo "  !! redis missing. install it (apt-get install -y redis-server) or Prism"
    echo "     will hang at startup with models stuck in 'activating'."
fi

# --- models -------------------------------------------------------------------
if [ "${SKIP_MODELS:-0}" != 1 ]; then
    say "models"
    "$ROOT/setup/download_models.sh" || echo "  (model download skipped/failed -- rerun setup/download_models.sh)"
fi

# --- verify -------------------------------------------------------------------
say "verify"
python - <<'PY'
import importlib, sys
ok = True
want = {"torch": "2.4.0", "sglang": "0.3.4.post2", "vllm": "0.6.3.post1",
        "transformers": "4.45.2", "flashinfer": "0.1.6"}
for m, exp in want.items():
    try:
        v = getattr(importlib.import_module(m), "__version__", "n/a")
        good = v.startswith(exp)
        ok &= good
        print(f"  {'OK ' if good else 'BAD'} {m:14s} {v}   (want {exp})")
    except Exception as e:
        ok = False
        print(f"  BAD {m:14s} IMPORT FAILED: {type(e).__name__}: {e}")
try:
    import kvcached, kvcached.vmm_ops  # noqa
    print("  OK  kvcached       + vmm_ops extension")
except Exception as e:
    ok = False
    print(f"  BAD kvcached       {type(e).__name__}: {e}")
import torch
print(f"  {'OK ' if torch.cuda.is_available() else 'BAD'} cuda available {torch.cuda.is_available()}")
ok &= torch.cuda.is_available()
sys.exit(0 if ok else 1)
PY

cat <<EOF

=== bootstrap complete ===

  source $ROOT/exp/scripts/env.sh
  ./exp/scripts/run_sanity.sh A     # then B, C
  python exp/scripts/summarize_sanity.py

Expected sanity numbers are in exp/results/1-env-verification/REPORT.md -- compare against
them to confirm the new box behaves. Note they were measured on A100-80G; a
different GPU will shift absolute latencies.
EOF
