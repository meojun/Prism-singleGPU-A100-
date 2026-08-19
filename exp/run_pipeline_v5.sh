#!/bin/bash
# V5: isolate the per-request tax, and re-validate the P2P fast path.
#
# The v4 study left two things unresolved and this answers both with runs.
#
# 1. At steady 8 req/s the paper-faithful arms lose goodput while pushing
#    identical throughput, deferring nothing, and paying a uniform 6-10%
#    latency penalty that grows with how much control-plane work the arm does.
#    Both algorithms were on in every arm, so which one costs it is not
#    separable from the v4 data.  paper-alg1-only and paper-alg2-only separate
#    them without new code.
#
# 2. P2P migration engaged in the one end-to-end run it was enabled for and
#    then OOMed on a CUDA IPC lifetime bug.  ipc_collect() fixes that and has
#    never been tested under load.
#
# Resume-safe: every stage and run writes a DONE marker and is skipped if it
# exists, so a restart continues at the first incomplete item.
set -uo pipefail
# supervisord hands children 1024 fds; the loaders need far more.
ulimit -n 65535 2>/dev/null || true
cd /workspace/prism-exp
source exp/scripts/env.sh

BASE=${V5_BASE:-/workspace/prism-exp/exp/results/paper-faithful-v5}
WL=${V5_WORKLOAD_DIR:-/workspace/prism-exp/exp/workloads/paper-faithful-v4}
STATE=$BASE/state
SEEDS=(${V5_SEEDS:-1 2 3})
GPU_PAIR=${V5_GPU_PAIR:-0,1}

export KVPR_TAU=${V5_TAU:-0.00035}
export V2_DURATION=420 V2_WARMUP=60 V2_MEASURE=300
export SLO_BASE_FILE=$PRISM_EXP/configs/v2/slo_base.json
export PREFILL_SPEED_FILE=$PRISM_EXP/configs/v2/prefill_speed.json
export NGPU=2 CFG=$PRISM_EXP/configs/v2/6model_2gpu.json
export CUDA_VISIBLE_DEVICES=$GPU_PAIR

mkdir -p "$BASE"/{raw,logs,state,aggregated,figures}
mkdir -p "$BASE"/raw/{requests,migrations,scheduler,gpu_metrics}
stamp() { date -Is; }
say()   { echo "===== $* $(stamp)"; }
is_done()  { [ -f "$STATE/$1.done" ]; }
mark_done(){ stamp > "$STATE/$1.done"; }
mark()     { printf '%s %s\n' "$2" "$(stamp)" > "$STATE/$1.status"; }
FAILED=0

if ! is_done env_check; then
  say "STAGE 0 environment"
  { nvidia-smi --query-gpu=index,name,memory.total,compute_cap,driver_version --format=csv
    echo "--- topology"; nvidia-smi topo -m; echo "--- nvlink"; nvidia-smi nvlink --status
  } > "$BASE/environment.txt" 2>&1
  BUSY=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
         | awk -F', ' -v p="$GPU_PAIR" 'BEGIN{split(p,a,","); for(i in a) mine[a[i]]=1}
                                        !($1 in mine) && $2 > 512 {print $1}')
  [ -n "$BUSY" ] && { echo "FATAL: GPUs outside the pair are busy: $BUSY"; exit 1; }
  mark_done env_check
fi

# run <system> <workload> <rate> <seed> [extra env assignments...]
run_case() {
  local sys=$1 wl=$2 rate=$3 seed=$4; shift 4
  local key="${sys}_${wl}_r${rate}_s${seed}"
  local out="$BASE/raw/$sys/$wl/rate_$rate/seed_$seed"
  [ -f "$out/DONE" ] && { echo "skip $key"; return 0; }
  [ -d "$out" ] && mv "$out" "${out}_failed_$(date +%Y%m%dT%H%M%S)_$$"
  mkdir -p "$out"; mark "$key" RUNNING
  local trace="$WL/${wl}_r${rate}_s${seed}.pkl"
  if env "$@" exp/scripts/run_v4_case.sh "$sys" "$wl" "$rate" "$seed" "$trace" "$out" \
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

# ---- STAGE 1/2: which algorithm costs the per-request tax?
# steady first: Algorithm 2 defers nothing there, so anything it costs is pure
# overhead rather than admission behaviour.
say "STAGE 1 ablation, steady 8"
for seed in "${SEEDS[@]}"; do
  for sys in paper-faithful-v3-alg1only paper-faithful-v3-alg2only; do run_case "$sys" steady 8 "$seed"; done
done
say "STAGE 2 ablation, bursty 8"
for seed in "${SEEDS[@]}"; do
  for sys in paper-faithful-v3-alg1only paper-faithful-v3-alg2only; do run_case "$sys" bursty 8 "$seed"; done
done

# ---- STAGE 3: does the P2P fast path hold under load now?
say "STAGE 3 P2P re-enabled"
for seed in "${SEEDS[@]}"; do
  run_case paper-faithful-v4 bursty 8 "$seed" V5_P2P=1 V5_PAGELOCK=1
done

say "STAGE 4 aggregation"
python3 exp/scripts/collect_v4_metrics.py --base "$BASE" > "$BASE/logs/aggregate.log" 2>&1 || FAILED=1

say "V5 pipeline finished failed=$FAILED"
if [ "$FAILED" = 0 ]; then echo "PIPELINE_DONE $(stamp)" > /workspace/logs/v5-pipeline.status
else echo "PIPELINE_FAILED $(stamp)" > /workspace/logs/v5-pipeline.status; fi
exit "$FAILED"
