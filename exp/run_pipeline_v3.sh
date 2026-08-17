#!/bin/bash
# Unattended, resume-safe paper-faithful-v3 sweep.
# Same paired steady/bursty traces, rates, models, SLO scales and prototype arm
# as v2; the second arm is the literal Algorithm-1 v3 implementation.
set -uo pipefail
cd /workspace/prism-exp
source exp/scripts/env.sh

BASE=${V3_BASE:-exp/results/paper-faithful-v3}
WL=${V3_WORKLOAD_DIR:-exp/workloads/paper-faithful-v2}
RATES=(${V3_RATES:-4 8 14 20})
SEEDS=(${V3_SEEDS:-1})
SYSTEMS=(released-prototype paper-faithful-v3)
DURATION=${V3_DURATION:-420}
WARMUP=${V3_WARMUP:-60}
MEASURE=${V3_MEASURE:-300}
export V3_TAU=${V3_TAU:-0.00035}
export KVPR_TAU=$V3_TAU
export V2_DURATION=$DURATION V2_WARMUP=$WARMUP V2_MEASURE=$MEASURE
export SLO_BASE_FILE=${SLO_BASE_FILE:-$PRISM_EXP/configs/v2/slo_base.json}
export PREFILL_SPEED_FILE=${PREFILL_SPEED_FILE:-$PRISM_EXP/configs/v2/prefill_speed.json}
export NGPU=2 CFG=$PRISM_EXP/configs/v2/6model_2gpu.json
mkdir -p "$BASE"/raw "$BASE"/logs "$WL"

stamp() { date -Is; }
echo "V3 pipeline start $(stamp) tau=$V3_TAU rates=${RATES[*]}"
printf 'tau_mode=absolute-line8\ntau=%s\nrates=%s\nworkloads=bursty steady\nsystems=%s\n' \
  "$V3_TAU" "${RATES[*]}" "${SYSTEMS[*]}" > "$BASE/META.txt"

for seed in "${SEEDS[@]}"; do for rate in "${RATES[@]}"; do
  if [ ! -f "$WL/bursty_r${rate}_s${seed}.pkl" ] || [ ! -f "$WL/steady_r${rate}_s${seed}.pkl" ]; then
    python3 exp/scripts/build_paired_workload.py --rate "$rate" --duration "$DURATION" \
      --seed "$seed" --slo-base "$SLO_BASE_FILE" --outdir "$WL" || exit 1
  fi
done; done

failed=0
for seed in "${SEEDS[@]}"; do for rate in "${RATES[@]}"; do
  for workload in bursty steady; do for system in "${SYSTEMS[@]}"; do
    out="$BASE/raw/$system/$workload/rate_$rate/seed_$seed"
    if [ -f "$out/DONE" ]; then echo "skip $system/$workload/$rate/$seed"; continue; fi
    if [ -d "$out" ] && find "$out" -mindepth 1 -print -quit | grep -q .; then
      archived="${out}_failed_attempt_$(date +%Y%m%dT%H%M%S)_$$"
      echo "archive incomplete run $out -> $archived"
      mv "$out" "$archived"
    fi
    mkdir -p "$out"
    trace="$WL/${workload}_r${rate}_s${seed}.pkl"
    if exp/scripts/run_v2_case.sh "$system" "$workload" "$rate" "$seed" "$trace" "$out" \
         2>&1 | tee "$BASE/logs/${system}_${workload}_r${rate}_s${seed}.log"; then
      python3 exp/scripts/collect_v2_metrics.py --run-dir "$out" --slo-base "$SLO_BASE_FILE" \
        --trace "$trace" --ttft-scale 5 --tpot-scale 3 --warmup "$WARMUP" \
        --measure "$MEASURE" --label "$system/$workload/$rate/$seed" -o "$out/metrics.json" \
        && touch "$out/DONE" || failed=1
    else
      echo "FAILED $system/$workload/$rate/$seed"; failed=1
    fi
    pkill -f 'launch_multi_model[_]server' 2>/dev/null || true
    sleep 10
  done; done
done; done

python3 exp/scripts/aggregate_v2.py --base "$BASE" -o "$BASE/processed" || true
python3 exp/scripts/validate_v3_results.py --base "$BASE" --rates "${RATES[@]}" --seed "${SEEDS[0]}" \
  || failed=1
echo "V3 pipeline finished $(stamp) failed=$failed"
if [ "$failed" = 0 ]; then
  echo "PIPELINE_DONE $(stamp)" > /workspace/logs/v3-pipeline.status
else
  echo "PIPELINE_FAILED failed=$failed $(stamp)" > /workspace/logs/v3-pipeline.status
fi
exit "$failed"
