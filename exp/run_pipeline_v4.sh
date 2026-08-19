#!/bin/bash
# Unattended, resume-safe paper-faithful-v4 study.
#
# Every stage writes a DONE marker under $BASE/state and is skipped if that
# marker exists, so a restart at any point continues from where it stopped and
# never re-runs work that already succeeded.  The watchdog relies on this.
#
# GPUs: exactly two, fixed for the whole study.  GPUs outside the pair are
# never made visible to any child process.
set -uo pipefail
cd /workspace/prism-exp
source exp/scripts/env.sh

BASE=${V4_BASE:-/workspace/prism-exp/exp/results/paper-faithful-v4}
# Its own directory: the traces embed the per-request SLOs, which come from
# THIS box's baselines, so writing them next to v3's would silently replace
# v3's committed artefacts with numbers measured somewhere else.
WL=${V4_WORKLOAD_DIR:-/workspace/prism-exp/exp/workloads/paper-faithful-v4}
STATE=$BASE/state
RATES=(${V4_RATES:-8 20})
SEEDS=(${V4_SEEDS:-1 2 3})
# Three arms.  v3 and v4 differ in exactly the v4 mechanisms -- page-locked
# host weights and GPU-to-GPU migration -- so v3-vs-v4 isolates what NVLink
# contributes, while the prototype supplies the baseline both are measured
# against.  The prototype is re-run here rather than carried over from an
# older report on a different machine.
SYSTEMS=(${V4_SYSTEMS:-released-prototype paper-faithful-v3 paper-faithful-v4})
DURATION=${V4_DURATION:-420}
WARMUP=${V4_WARMUP:-60}
MEASURE=${V4_MEASURE:-300}
GPU_PAIR=${V4_GPU_PAIR:-0,1}

export KVPR_TAU=${V4_TAU:-0.00035}
export V2_DURATION=$DURATION V2_WARMUP=$WARMUP V2_MEASURE=$MEASURE
export SLO_BASE_FILE=${SLO_BASE_FILE:-$PRISM_EXP/configs/v2/slo_base.json}
export PREFILL_SPEED_FILE=${PREFILL_SPEED_FILE:-$PRISM_EXP/configs/v2/prefill_speed.json}
export NGPU=2 CFG=$PRISM_EXP/configs/v2/6model_2gpu.json
export CUDA_VISIBLE_DEVICES=$GPU_PAIR

mkdir -p "$BASE"/{raw,logs,microbench,tp-validation,aggregated,figures,state} "$WL"
mkdir -p "$BASE"/raw/{requests,migrations,scheduler,gpu_metrics}

stamp() { date -Is; }
say()   { echo "===== $* $(stamp)"; }
done_marker() { echo "$1"; }
is_done() { [ -f "$STATE/$1.done" ]; }
mark_done() { echo "$(stamp)" > "$STATE/$1.done"; }
mark_state() { printf '%s %s\n' "$2" "$(stamp)" > "$STATE/$1.status"; }

FAILED=0

# --------------------------------------------------------------- 0. environment
if ! is_done env_check; then
  say "STAGE 0 environment"
  {
    echo "gpu_pair=$GPU_PAIR"
    nvidia-smi --query-gpu=index,name,memory.total,compute_cap,driver_version --format=csv
    echo "--- topology"; nvidia-smi topo -m
    echo "--- nvlink"; nvidia-smi nvlink --status
    echo "--- cpu"; lscpu | grep -E "^Model name|^CPU\(s\)|^NUMA node\(s\)"
    echo "--- free"; free -g | head -2
  } > "$BASE/environment.txt" 2>&1

  # Refuse to start if a GPU outside the pair is busy: the whole point of the
  # allocation is that the other two stay idle for the duration.
  BUSY=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
         | awk -F', ' -v pair="$GPU_PAIR" '
             BEGIN{split(pair,p,","); for(i in p) mine[p[i]]=1}
             !($1 in mine) && $2 > 512 {print $1":"$2}')
  if [ -n "$BUSY" ]; then
    echo "FATAL: GPUs outside the allocated pair are in use: $BUSY"
    mark_state env_check FAILED; exit 1
  fi
  echo "idle_gpus_outside_pair=confirmed" >> "$BASE/environment.txt"
  mark_done env_check
fi

# --------------------------------------------------- 1. implementation sanity
if ! is_done impl_sanity; then
  say "STAGE 1 implementation sanity"
  {
    python3 patches/paper_faithful/apply_patches.py --repo "$PRISM_REPO" &&
    python3 patches/paper_faithful_v3/apply_v3.py  --repo "$PRISM_REPO" &&
    python3 patches/paper_faithful_v4/apply_v4.py  --repo "$PRISM_REPO" &&
    python3 exp/tests/test_moore_hodgson.py &&
    python3 exp/tests/test_kvpr_placement.py &&
    python3 exp/tests/test_v4_loading.py
  } > "$BASE/logs/impl_sanity.log" 2>&1
  if [ $? -ne 0 ]; then
    echo "FATAL: implementation sanity failed"; tail -30 "$BASE/logs/impl_sanity.log"
    mark_state impl_sanity FAILED; exit 1
  fi
  mark_done impl_sanity
fi

# ------------------------------------------------- 2. loading microbenchmark
if ! is_done micro_loading; then
  say "STAGE 2 parallel-loading microbenchmark"
  mark_state micro_loading RUNNING
  timeout 5400 python3 exp/scripts/microbench_loading_v4.py \
      --models meta-llama/Llama-3.2-1B Qwen/Qwen2.5-1.5B-Instruct meta-llama/Llama-3.2-3B \
               Qwen/Qwen2.5-3B-Instruct meta-llama/Llama-3.1-8B Qwen/Qwen2.5-7B-Instruct \
      --gpu-ids 0,1 --target-gpu 0 --reps 3 \
      --out "$BASE/microbench/loading.json" \
      > "$BASE/logs/micro_loading.log" 2>&1 \
    && mark_done micro_loading || { echo "micro_loading FAILED"; mark_state micro_loading FAILED; FAILED=1; }
fi

# ----------------------------------------------- 3. migration microbenchmark
if ! is_done micro_migration; then
  say "STAGE 3 migration microbenchmark"
  mark_state micro_migration RUNNING
  timeout 5400 python3 exp/scripts/microbench_migration_v4.py \
      --models meta-llama/Llama-3.2-1B meta-llama/Llama-3.2-3B meta-llama/Llama-3.1-8B \
      --gpu-ids 0,1 --source-gpu 0 --target-gpu 1 --reps 3 \
      --out "$BASE/microbench/migration.json" \
      > "$BASE/logs/micro_migration.log" 2>&1 \
    && mark_done micro_migration || { echo "micro_migration FAILED"; mark_state micro_migration FAILED; FAILED=1; }
fi

# ----------------------------------------------------- 4. TP=2 validation
if ! is_done tp2; then
  say "STAGE 4 TP=2 validation"
  mark_state tp2 RUNNING
  pkill -f 'launch_multi_model[_]server' 2>/dev/null || true
  rm -f /dev/shm/ipc_0_model_*_root 2>/dev/null || true
  timeout 3600 ./exp/scripts/run_tp2_validation.sh "$BASE/tp-validation" \
      > "$BASE/logs/tp2.log" 2>&1
  # A TP=2 failure is a finding, not a reason to abandon the study.
  [ -f "$BASE/tp-validation/tp2_validation.json" ] && mark_done tp2 \
    || { echo "tp2 produced no record"; mark_state tp2 FAILED; FAILED=1; }
fi

# --------------------------------------------------------- 5. workloads
if ! is_done workloads; then
  say "STAGE 5 workloads"
  ok=1
  for seed in "${SEEDS[@]}"; do for rate in "${RATES[@]}"; do
    if [ ! -f "$WL/bursty_r${rate}_s${seed}.pkl" ] || [ ! -f "$WL/steady_r${rate}_s${seed}.pkl" ]; then
      python3 exp/scripts/build_paired_workload.py --rate "$rate" --duration "$DURATION" \
        --seed "$seed" --slo-base "$SLO_BASE_FILE" --outdir "$WL" \
        >> "$BASE/logs/workloads.log" 2>&1 || ok=0
    fi
  done; done
  [ "$ok" = 1 ] && mark_done workloads || { echo "FATAL: workload build failed"; exit 1; }
fi

# ------------------------------------------------------ 6. end-to-end runs
run_case() {   # run_case <system> <workload> <rate> <seed>
  local system=$1 workload=$2 rate=$3 seed=$4
  local key="e2e_${system}_${workload}_r${rate}_s${seed}"
  local out="$BASE/raw/$system/$workload/rate_$rate/seed_$seed"
  if [ -f "$out/DONE" ]; then echo "skip $key"; return 0; fi
  if [ -d "$out" ] && find "$out" -mindepth 1 -print -quit | grep -q .; then
    mv "$out" "${out}_failed_attempt_$(date +%Y%m%dT%H%M%S)_$$"
  fi
  mkdir -p "$out"
  mark_state "$key" RUNNING
  local trace="$WL/${workload}_r${rate}_s${seed}.pkl"
  if exp/scripts/run_v4_case.sh "$system" "$workload" "$rate" "$seed" "$trace" "$out" \
       2>&1 | tee "$BASE/logs/${system}_${workload}_r${rate}_s${seed}.log"; then
    if python3 exp/scripts/collect_v2_metrics.py --run-dir "$out" --slo-base "$SLO_BASE_FILE" \
         --trace "$trace" --ttft-scale 5 --tpot-scale 3 --warmup "$WARMUP" \
         --measure "$MEASURE" --label "$system/$workload/$rate/$seed" -o "$out/metrics.json"; then
      touch "$out/DONE"; mark_state "$key" SUCCESS
    else
      mark_state "$key" FAILED; FAILED=1
    fi
  else
    echo "FAILED $key"; mark_state "$key" FAILED; FAILED=1
  fi
  pkill -f 'launch_multi_model[_]server' 2>/dev/null || true
  rm -f /dev/shm/ipc_0_model_*_root 2>/dev/null || true
  sleep 10
}

say "STAGE 6 end-to-end (bursty first: it is the headline comparison)"
for workload in bursty ${V4_EXTRA_WORKLOADS:-steady}; do
  for rate in "${RATES[@]}"; do
    # steady is the regression sanity check only, at the lower rate
    if [ "$workload" = steady ] && [ "$rate" != "${RATES[0]}" ]; then continue; fi
    for seed in "${SEEDS[@]}"; do
      for system in "${SYSTEMS[@]}"; do
        run_case "$system" "$workload" "$rate" "$seed"
      done
    done
  done
done

# ------------------------------------------------------- 7. aggregation
say "STAGE 7 aggregation"
python3 exp/scripts/collect_v4_metrics.py --base "$BASE" \
    > "$BASE/logs/aggregate.log" 2>&1 || { echo "aggregation failed"; FAILED=1; }

say "STAGE 8 report"
python3 exp/scripts/build_report_v4.py --base "$BASE" \
    >> "$BASE/logs/aggregate.log" 2>&1 || { echo "report failed"; FAILED=1; }

say "V4 pipeline finished failed=$FAILED"
if [ "$FAILED" = 0 ]; then
  echo "PIPELINE_DONE $(stamp)" > /workspace/logs/v4-pipeline.status
else
  echo "PIPELINE_FAILED failed=$FAILED $(stamp)" > /workspace/logs/v4-pipeline.status
fi
exit "$FAILED"
