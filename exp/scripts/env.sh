#!/bin/bash
# Common environment for all Prism experiments.
# Usage:  source /workspace/prism-exp/exp/scripts/env.sh
export PRISM_ROOT=/workspace/prism-exp
export PRISM_REPO=$PRISM_ROOT/prism-research
export PRISM_EXP=$PRISM_ROOT/exp
export HF_HOME=/workspace/.hf_home
export PYTHONUNBUFFERED=1
# Prism's kvcached-v0 talks to the engines through /dev/shm; keep it consistent.
export TOKENIZERS_PARALLELISM=false
source "$PRISM_ROOT/prism-venv/bin/activate"

# HF token (gated Llama/Mistral models). Put HF_TOKEN=... in /workspace/.env
if [ -f /workspace/.env ]; then
    set -a; . /workspace/.env; set +a
fi
