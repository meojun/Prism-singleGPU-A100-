#!/bin/bash
# Common environment for all Prism experiments.
# Usage:  source /workspace/prism-exp/exp/scripts/env.sh
export PRISM_ROOT=${PRISM_ROOT:-/workspace/prism-exp}
export PRISM_REPO=$PRISM_ROOT/prism-research
export PRISM_EXP=$PRISM_ROOT/exp
export HF_HOME=/workspace/.hf_home
export PYTHONUNBUFFERED=1
# Datasets. real_trace.pkl (harness default) has synthetic "Hello "*n prompts;
# these two carry real ShareGPT text -- see exp/scripts/build_sharegpt_trace.py.
export DATASETS=${DATASETS:-/workspace/datasets}
export SHAREGPT_JSON=$DATASETS/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json
export SHAREGPT_CONTENT=$DATASETS/sharegpt/sharegpt_content.pkl
export SHAREGPT_FULL=$DATASETS/sharegpt/sharegpt_full.pkl
# Prism's kvcached-v0 talks to the engines through /dev/shm; keep it consistent.
export TOKENIZERS_PARALLELISM=false
source "$PRISM_ROOT/prism-venv/bin/activate"

# HF token (gated Llama/Mistral models). Put HF_TOKEN=... in /workspace/.env
if [ -f /workspace/.env ]; then
    set -a; . /workspace/.env; set +a
fi
