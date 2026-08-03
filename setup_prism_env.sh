#!/bin/bash
# Build the Prism (sglang-multi-model v0.3.4.post2 fork) environment natively,
# i.e. without the docker image referenced in prism-research/install.md
# (this container cannot run docker-in-docker).
set -euo pipefail

ROOT=/workspace/prism-exp
VENV=$ROOT/prism-venv

echo "=== [1/6] create python 3.10 venv ==="
uv venv "$VENV" --python=3.10
source "$VENV/bin/activate"
uv pip install --upgrade pip setuptools wheel

echo "=== [2/6] torch 2.4.0 (cu121) ==="
uv pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121

echo "=== [3/6] sglang-multi-model (Prism fork) + vllm 0.6.3.post1 ==="
cd "$ROOT/prism-research"
uv pip install -e "python[all]"

echo "=== [4/6] flashinfer 0.1.6 (cu121/torch2.4) ==="
uv pip install flashinfer==0.1.6 -i https://flashinfer.ai/whl/cu121/torch2.4/

echo "=== [5/6] kvcached (branch prism/shm) ==="
if [ ! -d "$ROOT/kvcached-prism" ]; then
    git clone -b prism/shm https://github.com/ovg-project/kvcached.git "$ROOT/kvcached-prism"
fi
cd "$ROOT/kvcached-prism"
uv pip install -e . --no-build-isolation
python setup.py build_ext --inplace

echo "=== [6/6] extra client/analysis deps ==="
uv pip install redis matplotlib pandas seaborn jsonlines aiohttp datasets

echo "=== versions ==="
python - <<'PY'
import importlib
for m in ["torch", "sglang", "vllm", "flashinfer", "transformers", "kvcached"]:
    try:
        mod = importlib.import_module(m)
        print(f"{m:14s} {getattr(mod, '__version__', 'n/a')}")
    except Exception as e:
        print(f"{m:14s} IMPORT FAILED: {type(e).__name__}: {e}")
PY
echo "=== DONE ==="
