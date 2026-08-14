#!/bin/bash
# Fetch the datasets the experiments read. ~670 MB, a few minutes.
#
#   ./setup/download_datasets.sh
#
# ShareGPT is NOT optional and NOT bundled: build_paired_workload.py and
# profile_v2.py both read it for real prompt/response text, and exp/scripts/env.sh
# hard-codes the path below. Without it every workload build dies with
# FileNotFoundError -- after you have already waited out bootstrap.
set -euo pipefail
DATASETS=${DATASETS:-/workspace/datasets}
SG="$DATASETS/sharegpt"
mkdir -p "$SG"
F="$SG/ShareGPT_V3_unfiltered_cleaned_split.json"
URL=https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json

if [ -s "$F" ] && [ "$(stat -c%s "$F")" -gt 100000000 ]; then
    echo "  ShareGPT already present ($(du -h "$F" | cut -f1))"
else
    echo "=== ShareGPT V3 (~670 MB)"
    for try in 1 2 3; do
        wget -q --show-progress -O "$F" "$URL" && break
        echo "  retry $try"; sleep 10
    done
fi
python3 -c "
import json,sys
d=json.load(open('$F'))
n=sum(1 for c in d if len(c.get('conversations',[]))>=2)
print(f'  {len(d)} records, {n} usable (>=2 turns)')
sys.exit(0 if n > 10000 else 1)
" || { echo "FATAL: ShareGPT looks truncated"; exit 1; }
echo "DATASETS_DONE  $DATASETS"
