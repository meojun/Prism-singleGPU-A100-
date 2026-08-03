#!/bin/bash
# Fetch the model weights the sanity sweep needs (~21 GB).
#
#   HF_TOKEN=hf_xxx ./setup/download_models.sh
#
# Llama is gated on Hugging Face: you need a token from an account that has
# accepted the meta-llama license, or every download 401s.
set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
export HF_HOME=${HF_HOME:-/workspace/.hf_home}

[ -f "$ROOT/prism-venv/bin/activate" ] && source "$ROOT/prism-venv/bin/activate"
[ -f /workspace/.env ] && { set -a; . /workspace/.env; set +a; }

if [ -z "${HF_TOKEN:-}" ]; then
    echo "HF_TOKEN not set. Put 'HF_TOKEN=hf_xxx' in /workspace/.env or export it." >&2
    exit 1
fi
hf auth whoami >/dev/null 2>&1 || hf auth login --token "$HF_TOKEN" --add-to-git-credential

# Models used by exp/configs/llama_*.json. Both are already present in Prism's
# profiled model_info.json, so no profiling step is needed.
# NOTE: there is no Llama-3.1 3B -- 3B exists only in Llama 3.2.
MODELS=(
    meta-llama/Llama-3.1-8B
    meta-llama/Llama-3.2-3B
)
# Ungated Qwen2.5 fallbacks (exp/configs/qwen_*.json); uncomment if you want them.
# MODELS+=(Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-1.5B-Instruct
#          Qwen/Qwen2.5-3B-Instruct  Qwen/Qwen2.5-7B-Instruct)

for m in "${MODELS[@]}"; do
    echo "=== $m"
    # --exclude original/* skips the duplicate .pth consolidated checkpoints
    hf download "$m" --exclude "original/*"
done

echo "HF_HOME=$HF_HOME"
du -sh "$HF_HOME/hub"/* 2>/dev/null || true
