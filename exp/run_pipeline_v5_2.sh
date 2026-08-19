#!/bin/bash
# V5_2 measurement.  Instrumentation only -- nothing about behaviour changes.
#
# (a) Algorithm 2 costs 39% of goodput at steady 8 req/s while deferring
#     nothing (1161 of 1161 selected, 0 deferred), so the cost is not its
#     admission decisions.  released-prototype and v3-alg2only differ in
#     exactly one function inside the GPU scheduler loop; the [V5-LOOP] records
#     time that loop and split it into memory read, Redis round trips,
#     admission_control and dispatch.
#
# (b) A deactivation costs 5.4 s as the server sees it against the engine's own
#     0.96 s teardown.  [V5-HOP] stamps the hops in between.  Needs an arm that
#     actually migrates, hence v3 on bursty.
set -uo pipefail
ulimit -n 65535 2>/dev/null || true
cd /workspace/prism-exp
source exp/scripts/env.sh

BASE=${V52_BASE:-/workspace/prism-exp/exp/results/paper-faithful-v5_2}
WL=/workspace/prism-exp/exp/workloads/paper-faithful-v4
STATE=$BASE/state
SEEDS=(${V52_SEEDS:-1 2 3})
export KVPR_TAU=${V52_TAU:-0.00035}
export V2_DURATION=420 V2_WARMUP=60 V2_MEASURE=300
export SLO_BASE_FILE=$PRISM_EXP/configs/v2/slo_base.json
export PREFILL_SPEED_FILE=$PRISM_EXP/configs/v2/prefill_speed.json
export NGPU=2 CFG=$PRISM_EXP/configs/v2/6model_2gpu.json
export CUDA_VISIBLE_DEVICES=${V52_GPU_PAIR:-0,1}

mkdir -p "$BASE"/{raw,logs,state,aggregated}
mkdir -p "$BASE"/raw/{requests,migrations,scheduler,gpu_metrics}
stamp(){ date -Is; }; say(){ echo "===== $* $(stamp)"; }
mark(){ printf '%s %s\n' "$2" "$(stamp)" > "$STATE/$1.status"; }
FAILED=0

run_case() {
  local sys=$1 wl=$2 rate=$3 seed=$4
  local key="${sys}_${wl}_r${rate}_s${seed}"
  local out="$BASE/raw/$sys/$wl/rate_$rate/seed_$seed"
  [ -f "$out/DONE" ] && { echo "skip $key"; return 0; }
  [ -d "$out" ] && mv "$out" "${out}_failed_$(date +%Y%m%dT%H%M%S)_$$"
  mkdir -p "$out"; mark "$key" RUNNING
  local trace="$WL/${wl}_r${rate}_s${seed}.pkl"
  if exp/scripts/run_v4_case.sh "$sys" "$wl" "$rate" "$seed" "$trace" "$out" \
       2>&1 | tee "$BASE/logs/${key}.log"; then
    if python3 exp/scripts/collect_v2_metrics.py --run-dir "$out" --slo-base "$SLO_BASE_FILE" \
         --trace "$trace" --ttft-scale 5 --tpot-scale 3 --warmup 60 --measure 300 \
         --label "$key" -o "$out/metrics.json"; then
      touch "$out/DONE"; mark "$key" SUCCESS
    else mark "$key" FAILED; FAILED=1; fi
  else echo "FAILED $key"; mark "$key" FAILED; FAILED=1; fi
  pkill -f 'launch_multi_model[_]server' 2>/dev/null || true
  rm -f /dev/shm/ipc_[0-9]*_root /dev/shm/cuda.shm.* 2>/dev/null || true
  sleep 10
}

say "STAGE A  scheduler-loop cost: prototype vs Moore-Hodgson, steady 8"
for seed in "${SEEDS[@]}"; do
  for sys in released-prototype paper-faithful-v3-alg2only; do run_case "$sys" steady 8 "$seed"; done
done

say "STAGE B  deactivation hops, bursty 8 (needs an arm that migrates)"
for seed in "${SEEDS[@]}"; do run_case paper-faithful-v3 bursty 8 "$seed"; done

say "STAGE C  analysis"
python3 exp/scripts/collect_v4_metrics.py --base "$BASE" > "$BASE/logs/aggregate.log" 2>&1 || true
python3 exp/scripts/analyze_v5_2_instr.py --base "$BASE" > "$BASE/REPORT.md" 2>>"$BASE/logs/aggregate.log" || FAILED=1

say "V5_2 pipeline finished failed=$FAILED"
[ "$FAILED" = 0 ] && echo "PIPELINE_DONE $(stamp)" > /workspace/logs/v5_2-pipeline.status \
                  || echo "PIPELINE_FAILED $(stamp)" > /workspace/logs/v5_2-pipeline.status
exit "$FAILED"
