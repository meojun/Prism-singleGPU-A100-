#!/bin/bash
# Short smoke runs + the Section 8 gate.  Nothing else may run until this passes.
#
#   ./exp/scripts/run_sanity_v2.sh [rate]
set -uo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/env.sh"

RATE=${1:-12}
DUR=${SANITY_DURATION:-240}
SEED=${SANITY_SEED:-77}
BASE=$PRISM_EXP/results/paper-faithful-v2/sanity
WL=$BASE/workloads
export SLO_BASE_FILE=${SLO_BASE_FILE:-$PRISM_EXP/configs/v2/slo_base.json}
export PREFILL_SPEED_FILE=${PREFILL_SPEED_FILE:-$PRISM_EXP/configs/v2/prefill_speed.json}
export CFG=${CFG:-$PRISM_EXP/configs/v2/6model_2gpu.json}
export NGPU=2
mkdir -p "$BASE" "$WL"

echo "=== unit tests"
python3 "$PRISM_EXP/tests/test_moore_hodgson.py"  >/dev/null || { echo "FAIL: Alg2 unit tests"; exit 1; }
python3 "$PRISM_EXP/tests/test_kvpr_placement.py" >/dev/null || { echo "FAIL: Alg1 unit tests"; exit 1; }
echo "  Algorithm 1 + Algorithm 2 unit tests pass"

echo "=== build sanity workload (rate=$RATE seed=$SEED duration=${DUR}s)"
V2_DURATION=$DUR
[ -f "$WL/bursty_r${RATE}_s${SEED}.pkl" ] || \
  python3 "$SCRIPT_DIR/build_paired_workload.py" --rate "$RATE" --duration "$DUR" \
    --seed "$SEED" --slo-base "$SLO_BASE_FILE" --outdir "$WL" || exit 1

for sys in paper-faithful released-prototype; do
  for wl in bursty steady; do
    d="$BASE/smoke/${sys}_${wl}"
    [ -f "$d/DONE" ] && { echo "skip $sys/$wl"; continue; }
    mkdir -p "$d"
    if "$SCRIPT_DIR/run_v2_case.sh" "$sys" "$wl" "$RATE" "$SEED" \
         "$WL/${wl}_r${RATE}_s${SEED}.pkl" "$d" 2>&1 | tee "$d/run.log"; then
      python3 "$SCRIPT_DIR/collect_v2_metrics.py" --run-dir "$d" \
        --slo-base "$SLO_BASE_FILE" --warmup 40 --measure $((DUR-80)) \
        --label "sanity/$sys/$wl" -o "$d/metrics.json" && touch "$d/DONE"
    else
      echo "!! sanity run $sys/$wl FAILED"
    fi
    pkill -f "launch_multi_model[_]server" 2>/dev/null || true
    sleep 10
  done
done

echo
echo "=== Section 8 gate (evaluated on the paper-faithful bursty run)"
V2_DURATION=$DUR python3 "$SCRIPT_DIR/sanity_v2.py" \
  --bursty-dir "$BASE/smoke/paper-faithful_bursty" \
  --steady-dir "$BASE/smoke/paper-faithful_steady" \
  --prefill-speed "$PREFILL_SPEED_FILE" \
  --slo-base "$SLO_BASE_FILE" \
  --workload-dir "$WL" -o "$BASE/sanity_gate.json"
RC=$?
echo "sanity gate exit=$RC"
exit $RC
