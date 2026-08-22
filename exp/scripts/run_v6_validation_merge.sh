#!/bin/bash
# v6 KV-migration validation, on the merged tree and this box.
#
# run_v6_validation.sh cannot be reused: it hardcodes R=/workspace/prism-exp
# (which on this box is the TP branch, with no v6 patch -- hence "unknown
# system: paper-faithful-v6"), points SLO_BASE at that box's profiling output,
# and calls /workspace/shm_clean.sh, a path that does not exist here.  Rather
# than edit a file the other branch owns, this is the same run wired to the
# merged tree and to this box's own calibration.
set -uo pipefail
R=${PRISM_ROOT:-/workspace/prism-merge}
cd "$R"
set -a; . /workspace/.env 2>/dev/null; set +a
source "$R/prism-venv/bin/activate"
export PRISM_ROOT=$R PRISM_REPO=$R/prism-research PRISM_EXP=$R/exp
export PYTHONPATH="$PRISM_REPO/python${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=/workspace/.hf_home PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1
# This box's numbers, not the committed ones and not the other box's.
export SLO_BASE_FILE=$R/exp/configs/v2/slo_base.json
export PREFILL_SPEED_FILE=$R/exp/configs/v2/prefill_speed.json
export SLO_BASE=$SLO_BASE_FILE PREFILL_SPEED=$PREFILL_SPEED_FILE
export KVPR_TAU=${KVPR_TAU:-$(python3 -c "import json;print(json.load(open('$R/exp/results/paper-faithful-tp/calibration/tau.json'))['tau'])" 2>/dev/null || echo 0.171086)}
export NGPU=2 CFG=$R/exp/configs/v2/6model_2gpu.json
export V2_DURATION=420 V2_WARMUP=60 V2_MEASURE=300

[ -x "$R/exp/scripts/shm_clean.sh" ] && "$R/exp/scripts/shm_clean.sh" || \
    rm -f /dev/shm/ipc_* /dev/shm/mp-* /dev/shm/torch_* /dev/shm/cuda.shm.* 2>/dev/null

WL=$R/exp/workloads/paper-faithful-v4
TRACE=$WL/bursty_r8_s1.pkl
if [ ! -f "$TRACE" ]; then
    mkdir -p "$WL"
    python3 exp/scripts/build_paired_workload.py --rate 8 --duration 420 --seed 1 \
        --slo-base "$SLO_BASE_FILE" --outdir "$WL" 2>&1 | tail -2
fi

OUT=${1:-$R/exp/results/paper-faithful-v6/e2e-merge}
rm -rf "$OUT"; mkdir -p "$OUT"
echo "=== paper-faithful-v6 on merged tree  tau=$KVPR_TAU  $(date -Is)"
./exp/scripts/run_v4_case.sh paper-faithful-v6 bursty 8 1 "$TRACE" "$OUT" 2>&1 | tee "$OUT/run.log"
rc=${PIPESTATUS[0]}
echo "--- exit=$rc  $(date -Is)"

echo "=== KV hand-off probe -- which queue key each side used"
grep -rh "KV-PROBE" "$OUT" 2>/dev/null | sed 's/.*KV-PROBE/KV-PROBE/' | sort -u | head -20
echo "=== did the mechanism engage?"
KVT=$(find "$OUT" -name kv_transfers.jsonl 2>/dev/null | head -1)
if [ -n "$KVT" ]; then
    echo "  kv_transfers.jsonl: $(wc -l < "$KVT") records"
else
    echo "  kv_transfers.jsonl ABSENT -- nothing was transferred"
fi
for k in stash fetch inject capture; do
    printf "  %-8s %s\n" "$k" "$(grep -rhc "PAPER-KV-V6\] $k" "$OUT" 2>/dev/null | awk '{s+=$1} END{print s+0}')"
done
grep -rh "fetch timed out" "$OUT" 2>/dev/null | wc -l | xargs echo "  fetch timed out:"
exit "$rc"
