#!/bin/bash
# Fetch the model weights the experiments need.
#
#   HF_TOKEN=hf_xxx ./setup/download_models.sh
#
# Llama is gated on Hugging Face: the token must come from an account that has
# accepted the meta-llama license, or every Llama download 401s.
#
# The default set is the HETEROGENEOUS 6-model mix used by
# exp/run_paper_faithful_v2.sh.  It is deliberately picked so that KV cell size
# is NOT monotone in parameter count -- Llama-3.2-3B carries 114688 B/token
# against Qwen2.5-3B's 36864 at the same size, and Llama-3.1-8B's 131072
# against Qwen2.5-7B's 57344.  That is what gives Algorithm 1's KVPR objective
# something to discriminate; a set of identical models makes it flat.
#
#   MODEL_SET=v1 ./setup/download_models.sh   # the older 3x Llama-3.1-8B study
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

case "${MODEL_SET:-v2}" in
  v2)
    # size (GiB weights) / KV cell size (B/token), from setup/model_info.json
    MODELS=(
        meta-llama/Llama-3.2-1B          #  2.28 /  32768   small
        Qwen/Qwen2.5-1.5B-Instruct       #  3.01 /  28672   small
        meta-llama/Llama-3.2-3B          #  6.00 / 114688   medium, fat KV
        Qwen/Qwen2.5-3B-Instruct         #  5.84 /  36864   medium, thin KV
        meta-llama/Llama-3.1-8B          # 15.08 / 131072   large, fat KV
        Qwen/Qwen2.5-7B-Instruct         # 14.28 /  57344   large, thin KV
    ) ;;
  v1)
    MODELS=(meta-llama/Llama-3.1-8B meta-llama/Llama-3.2-3B meta-llama/Llama-3.2-1B) ;;
  *) echo "unknown MODEL_SET=$MODEL_SET" >&2; exit 1 ;;
esac

for m in "${MODELS[@]}"; do
    echo "=== $m  $(date -Is)"
    for try in 1 2 3; do
        # --exclude original/* skips the duplicate .pth consolidated checkpoints
        hf download "$m" --exclude "original/*" && break
        echo "  retry $try for $m"; sleep 10
    done
done

echo "ALL_MODELS_DONE $(date -Is)"
echo "HF_HOME=$HF_HOME"
du -sh "$HF_HOME/hub"/* 2>/dev/null || true
