#!/bin/bash
# Resume-safe chain: wait for A -> deploy overlap -> gate -> calibrate -> final C.
set -uo pipefail
ulimit -n 65535

DEV=/workspace/prism-exp
RUN=/workspace/prism-merge
LOG=/workspace/logs
BASE_OUT=$RUN/exp/results/final-prototype-vs-paper-faithful
PIPE_OUT=$RUN/exp/results/final-overlap-pipeline
STATE=$PIPE_OUT/state
WL=$RUN/exp/workloads/final-compare
CAL=$PIPE_OUT/tau-calibration
FREEZE=$PIPE_OUT/FROZEN_TAU.json
FINAL=$BASE_OUT/raw/armC-final-overlap
mkdir -p "$STATE" "$PIPE_OUT/logs" "$CAL" "$WL"

say() { echo "===== $* $(date -Is) ====="; }
status() { cat "$STATE/$1" 2>/dev/null || echo PENDING; }
set_status() { printf '%s\n' "$2" > "$STATE/$1"; }
block() { say "BLOCKED: $*"; printf '%s %s\n' "$(date -Is)" "$*" > "$PIPE_OUT/BLOCKED"; exit 0; }

reap() {
  tmux list-sessions -F '#S' 2>/dev/null | awk '/^v4-paper-faithful-v6-/{print}' |
    while read -r s; do tmux kill-session -t "$s" 2>/dev/null || true; done
  for pattern in 'python3 -m sglang.launch_multi_model_server' 'python3 benchmark.py'; do
    pgrep -f "$pattern" 2>/dev/null | while read -r p; do
      [ "$p" = "$$" ] || kill -TERM "$p" 2>/dev/null || true
    done
  done
  sleep 8
}

say "waiting for the reusable prototype baseline A"
while [ ! -f "$LOG/.tp_final_baseline_done" ]; do
  sleep 30
done
while supervisorctl status tp_final_compare 2>/dev/null | grep -q RUNNING; do
  sleep 15
done

if [ "$(status deploy)" != SUCCESS ]; then
  say "deploying the CPU-validated overlap patch"
  set_status deploy RUNNING
  mkdir -p "$RUN/patches/paper_faithful_v6"
  cp -a "$DEV/patches/paper_faithful_v6/." "$RUN/patches/paper_faithful_v6/"
  for file in run_v4_case.sh collect_v4_metrics.py summarize_v6_overlap_gate.py select_tau_v6.py \
              run_tp_boot.sh tp_probe_requests.py collect_tp2_evidence.py; do
    cp -a "$DEV/exp/scripts/$file" "$RUN/exp/scripts/$file"
  done
  cp -a "$DEV/prism-research/benchmark/multi-model/benchmark.py" \
    "$RUN/prism-research/benchmark/multi-model/benchmark.py"
  "$RUN/prism-venv/bin/python" "$RUN/patches/paper_faithful_v6/apply_v6.py" \
    --repo "$RUN/prism-research" > "$PIPE_OUT/logs/deploy.log" 2>&1 || block "overlap patch deployment failed"
  "$RUN/prism-venv/bin/python" -m py_compile \
    "$RUN/prism-research/python/sglang/multi_model/scheduling/controller_global.py" \
    "$RUN/prism-research/python/sglang/multi_model/scheduling/gpu/worker_pool.py" \
    "$RUN/prism-research/python/sglang/srt/managers/scheduler.py" \
    "$RUN/prism-research/python/sglang/multi_model/model_sevice.py" \
    >> "$PIPE_OUT/logs/deploy.log" 2>&1 || block "deployed source does not compile"
  stash_handlers=$(python3 -c \
    "print(open('$RUN/prism-research/python/sglang/multi_model/model_sevice.py').read().count('if model_key == \"__kv_stash__\"'))")
  [ "$stash_handlers" = 1 ] || block "deployed source has duplicate KV stash handlers"
  set_status deploy SUCCESS
fi

SLO=$RUN/exp/configs/v2/slo_base.json
PREFILL=$RUN/exp/configs/v2/prefill_speed.json
CFG=$RUN/exp/configs/v2/6model_2gpu.json

make_trace() {
  local rate=$1 seed=$2
  [ -f "$WL/bursty_r${rate}_s${seed}.pkl" ] && return 0
  (cd "$RUN" && PRISM_ROOT=$RUN source exp/scripts/env.sh >/dev/null 2>&1 &&
    python3 exp/scripts/build_paired_workload.py --rate "$rate" --duration 420 \
      --seed "$seed" --slo-base "$SLO" --outdir "$WL")
}

run_case() {
  local id=$1 tau=$2 workload=$3 rate=$4 seed=$5 dest=$6 logfile=$7
  [ "$(status "$id")" = SUCCESS ] && return 0
  local trace="$WL/${workload}_r${rate}_s${seed}.pkl"
  [ -f "$trace" ] || return 1
  set_status "$id" RUNNING
  reap
  mkdir -p "$dest"
  (cd "$RUN" && PRISM_ROOT=$RUN NGPU=2 CFG=$CFG MAXMEM=60 KVPR_TAU=$tau \
    SLO_BASE_FILE=$SLO PREFILL_SPEED_FILE=$PREFILL BENCHMARK_TIMEOUT=2100 \
    bash exp/scripts/run_v4_case.sh paper-faithful-v6 "$workload" "$rate" "$seed" "$trace" "$dest") \
      > "$logfile" 2>&1
  local rc=$?
  reap
  if [ "$rc" -eq 0 ]; then touch "$dest/DONE"; set_status "$id" SUCCESS; return 0; fi
  set_status "$id" FAILED
  return "$rc"
}

repair_baseline_case() {
  local workload=$1 rate=$2 seed=$3
  local base_state="$BASE_OUT/state/A_${workload}_r${rate}_s${seed}"
  [ "$(cat "$base_state" 2>/dev/null)" = SUCCESS ] && return 0
  local id="repair_A_${workload}_r${rate}_s${seed}"
  local dest="$BASE_OUT/raw/armA/raw/released-prototype/$workload/rate_$rate/seed_$seed"
  local trace="$WL/${workload}_r${rate}_s${seed}.pkl"
  say "repairing missing baseline $workload r$rate seed$seed"
  if [ -d "$dest" ]; then
    local archive="$PIPE_OUT/failed-baseline-attempts/${workload}_r${rate}_s${seed}_$(date +%s)"
    mkdir -p "$(dirname "$archive")"
    mv "$dest" "$archive"
  fi
  reap
  mkdir -p "$dest"
  (cd /workspace/prism-base && PRISM_ROOT=/workspace/prism-base NGPU=2 CFG=$CFG \
    MAXMEM=67.28 SLO_BASE_FILE=$SLO PREFILL_SPEED_FILE=$PREFILL BENCHMARK_TIMEOUT=2100 \
    bash exp/scripts/run_v4_case.sh released-prototype "$workload" "$rate" "$seed" "$trace" "$dest") \
      > "$PIPE_OUT/logs/${id}.log" 2>&1
  local rc=$?
  reap
  if [ "$rc" -eq 0 ]; then
    touch "$dest/DONE"
    printf 'SUCCESS\n' > "$base_state"
    return 0
  fi
  printf 'FAILED\n' > "$base_state"
  return "$rc"
}

# The baseline driver intentionally continues after an isolated failed case.
# Repair every missing cell before allowing the correctness gate to consume
# the GPUs, so A is exactly the preregistered 24-run grid rather than 23/24.
for workload in bursty steady; do
  rates="2 4 8 14 20"; [ "$workload" = steady ] && rates="4 8 20"
  for rate in $rates; do
    for seed in 1 2 3; do
      repair_baseline_case "$workload" "$rate" "$seed" || \
        block "baseline repair failed: ${workload} r${rate} seed${seed}"
    done
  done
done

# One intentionally busy run. It must demonstrate every requested runtime
# property; a FAIL is terminal and calibration is not permitted to begin.
GATE=$PIPE_OUT/correctness-gate/bursty_r6_s99_attempt7
if [ "$(status correctness_gate)" != SUCCESS ]; then
  say "overlap correctness gate"
  make_trace 6 99 || block "could not build correctness trace"
  run_case gate_bursty_r6_s99_attempt7 0.00035 bursty 6 99 "$GATE" "$PIPE_OUT/logs/gate-r6-attempt7.log" || \
    block "correctness workload failed"
  "$RUN/prism-venv/bin/python" "$RUN/exp/scripts/summarize_v6_overlap_gate.py" --out "$GATE" \
    > "$PIPE_OUT/logs/gate-summary.log" 2>&1 || block "overlap correctness evidence failed"
  set_status correctness_gate SUCCESS
fi

# Regression sanity after the overlap source is deployed.  The earlier 70B
# TP=2 sustained and TP=4 smoke results remain valid evidence, but this short
# run proves the new default phase="full" path did not break large-model TP
# serving before we spend hours on calibration/final evaluation.
if [ "$(status sanity_70b_tp2)" != SUCCESS ]; then
  say "post-overlap 70B TP=2 serving sanity"
  set_status sanity_70b_tp2 RUNNING
  reap
  attempt="$PIPE_OUT/70b-sanity/attempt_$(date +%Y%m%dT%H%M%S)"
  mkdir -p "$attempt"
  (cd "$RUN" && PRISM_ROOT=$RUN TP_MAXMEM=0.8 \
    bash exp/scripts/run_tp_boot.sh "$attempt" 2 2 1 \
      meta-llama/Llama-3.1-70B model_70b) \
      > "$PIPE_OUT/logs/70b-sanity.log" 2>&1
  sanity_rc=$?
  reap
  sanity_verdict=$(python3 -c \
    "import json; print(json.load(open('$attempt/tp2_validation.json')).get('verdict','FAIL'))" \
    2>/dev/null || echo FAIL)
  if [ "$sanity_rc" -ne 0 ] || [ "$sanity_verdict" != PASS ]; then
    set_status sanity_70b_tp2 FAILED
    block "post-overlap 70B TP=2 sanity failed (rc=$sanity_rc verdict=$sanity_verdict)"
  fi
  ln -sfn "$attempt" "$PIPE_OUT/70b-sanity/PASS"
  set_status sanity_70b_tp2 SUCCESS
fi

if [ ! -f "$CAL/CALIBRATION_PROTOCOL.json" ]; then
  "$RUN/prism-venv/bin/python" - "$CAL/CALIBRATION_PROTOCOL.json" <<'PY'
import json,sys
json.dump({
 "status":"PREREGISTERED_BEFORE_RUNS", "workload":"bursty rate=20 duration=420s",
 "held_out_seeds":[0,42], "candidates":["0.00035","0.07","0.10","0.13","0.171086","inf"],
 "primary":"maximum mean Joint-SLO goodput",
 "tie":"within 3% relative: least migration bytes, then count, then larger tau",
 "final_evaluation":"tau frozen; seeds 1,2,3 only"
},open(sys.argv[1],"w"),indent=2)
PY
fi

for seed in 0 42; do make_trace 20 "$seed" || block "could not build held-out trace seed=$seed"; done
for tau in 0.00035 0.07 0.10 0.13 0.171086 inf; do
  label=$(printf '%s' "$tau" | tr '.' 'p')
  for seed in 0 42; do
    id="cal_tau_${label}_s${seed}"
    dest="$CAL/tau_$label/raw/paper-faithful-v6/bursty/rate_20/seed_$seed"
    run_case "$id" "$tau" bursty 20 "$seed" "$dest" "$PIPE_OUT/logs/${id}.log" || \
      block "held-out calibration case failed: $id"
  done
  (cd "$RUN" && PRISM_ROOT=$RUN source exp/scripts/env.sh >/dev/null 2>&1 &&
    python3 exp/scripts/collect_v4_metrics.py --base "$CAL/tau_$label" \
      --slo-base "$SLO" --config "$CFG" --trace-dir "$WL") \
      > "$PIPE_OUT/logs/collect_tau_${label}.log" 2>&1 || block "tau collection failed: $tau"
done

if [ ! -f "$FREEZE" ]; then
  "$RUN/prism-venv/bin/python" "$RUN/exp/scripts/select_tau_v6.py" \
    --base "$CAL" --freeze "$FREEZE" > "$PIPE_OUT/logs/tau-selection.log" 2>&1 || \
    block "tau selection failed"
fi
TAU=$("$RUN/prism-venv/bin/python" -c "import json; print(json.load(open('$FREEZE'))['selected_tau'])")
say "frozen tau=$TAU; starting final C only"

for workload in bursty steady; do
  rates="2 4 8 14 20"; [ "$workload" = steady ] && rates="4 8 20"
  for rate in $rates; do
    for seed in 1 2 3; do
      id="finalC_${workload}_r${rate}_s${seed}"
      dest="$FINAL/raw/paper-faithful-v6/$workload/rate_$rate/seed_$seed"
      run_case "$id" "$TAU" "$workload" "$rate" "$seed" "$dest" "$PIPE_OUT/logs/${id}.log" || \
        block "final C case failed: $id"
    done
  done
done

(cd "$RUN" && PRISM_ROOT=$RUN source exp/scripts/env.sh >/dev/null 2>&1 &&
  python3 exp/scripts/collect_v4_metrics.py --base "$FINAL" --slo-base "$SLO" \
    --config "$CFG" --trace-dir "$WL") > "$PIPE_OUT/logs/collect_finalC.log" 2>&1 || \
  block "final C collection failed"

VIEW=$PIPE_OUT/final-comparison
mkdir -p "$VIEW/raw"
ln -sfn "$BASE_OUT/raw/armA" "$VIEW/raw/armA"
ln -sfn "$BASE_OUT/raw/armB" "$VIEW/raw/armB"
ln -sfn "$FINAL" "$VIEW/raw/armC"
(cd "$RUN" && PRISM_ROOT=$RUN source exp/scripts/env.sh >/dev/null 2>&1 &&
  python3 exp/scripts/compare_arms.py --out "$VIEW") > "$PIPE_OUT/logs/final-compare.log" 2>&1 || \
  block "final comparison report failed"
touch "$PIPE_OUT/COMPLETE"
say "overlap/calibration/final-C pipeline complete"
