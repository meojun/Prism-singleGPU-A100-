#!/bin/bash
# Paper-Faithful Prism v2 -- Shifting-Bursty vs Steady, end to end.
#
#   ./exp/run_paper_faithful_v2.sh [--dry-run] [--resume] [--ablation]
#
# Designed to run unattended in tmux for hours.  --resume skips any run that
# already wrote its DONE marker, so it can be killed and restarted freely.
#
#   0. environment + GPU check, config, profiled c_i / SLO baselines
#   1. build the PAIRED workloads: one request set per (rate, seed), emitted as
#      a bursty trace and a steady trace that differ only in arrival times
#   2. the sweep: systems x {bursty, steady} x rates x seeds
#   3. per-run metric collection -> results.csv
#
# Env overrides: V2_SYSTEMS V2_RATES V2_SEEDS V2_DURATION V2_WARMUP V2_MEASURE
set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/scripts" && pwd)
ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
source "$SCRIPT_DIR/env.sh"

DRY=0; RESUME=0; ABLATION=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY=1 ;;
    --resume) RESUME=1 ;;
    --ablation) ABLATION=1 ;;
    *) echo "unknown option: $arg" >&2; exit 1 ;;
  esac
done

if [ "$ABLATION" = 1 ]; then
  SYSTEMS=(${V2_SYSTEMS:-released-prototype paper-alg1-only paper-alg2-only paper-faithful})
else
  SYSTEMS=(${V2_SYSTEMS:-released-prototype paper-faithful})
fi
RATES=(${V2_RATES:-4 8 14 20})
SEEDS=(${V2_SEEDS:-1})
WORKLOADS=(bursty steady)

DURATION=${V2_DURATION:-420}
WARMUP=${V2_WARMUP:-60}
MEASURE=${V2_MEASURE:-300}
TTFT_SCALE=${V2_TTFT_SCALE:-5}
TPOT_SCALE=${V2_TPOT_SCALE:-3}

BASE=$PRISM_EXP/results/paper-faithful-v2
WL=$PRISM_EXP/workloads/paper-faithful-v2
CFG=$PRISM_EXP/configs/v2/6model_2gpu.json
export SLO_BASE_FILE=${SLO_BASE_FILE:-$PRISM_EXP/configs/v2/slo_base.json}
export PREFILL_SPEED_FILE=${PREFILL_SPEED_FILE:-$PRISM_EXP/configs/v2/prefill_speed.json}
export NGPU=2 CFG TTFT_SCALE TPOT_SCALE
mkdir -p "$BASE"/{raw,processed,logs} "$WL"

say() { echo -e "\n=== $* ===" ; }
run_dir() { echo "$BASE/raw/$1/$2/rate_$3/seed_$4"; }   # system workload rate seed

plan() {
  local n=0
  for seed in "${SEEDS[@]}"; do for rate in "${RATES[@]}"; do
    for wl in "${WORKLOADS[@]}"; do for sys in "${SYSTEMS[@]}"; do
      printf '%-20s | %-6s | rate=%-4s | seed=%s\n' "$sys" "$wl" "$rate" "$seed"; n=$((n+1))
    done; done
  done; done
  echo "Total: $n runs (~12 min each -> ~$((n*12/60))h)"
}
[ "$DRY" = 1 ] && { plan; exit 0; }

say "[0/3] environment"
python3 -c "import torch;assert torch.cuda.device_count()>=2, torch.cuda.device_count()" || exit 1
[ -f "$SLO_BASE_FILE" ] || { echo "FATAL: missing $SLO_BASE_FILE -- run exp/scripts/run_profiling_v2.sh"; exit 1; }
[ -f "$PREFILL_SPEED_FILE" ] || { echo "FATAL: missing $PREFILL_SPEED_FILE"; exit 1; }
[ -f "$CFG" ] || python3 "$SCRIPT_DIR/make_config_v2.py" --num-gpus 2 -o "$CFG"
COMMIT=$(git -C "$ROOT" rev-parse HEAD)
echo "commit=$COMMIT  slo_base=$SLO_BASE_FILE  c_i=$PREFILL_SPEED_FILE"
cat "$PREFILL_SPEED_FILE"

say "[1/3] paired workloads"
for seed in "${SEEDS[@]}"; do for rate in "${RATES[@]}"; do
  tag="r${rate}_s${seed}"
  if [ -f "$WL/bursty_${tag}.pkl" ] && [ -f "$WL/steady_${tag}.pkl" ]; then
    echo "  $tag already built"; continue
  fi
  python3 "$SCRIPT_DIR/build_paired_workload.py" --rate "$rate" --duration "$DURATION" \
    --seed "$seed" --slo-base "$SLO_BASE_FILE" --outdir "$WL" || exit 1
done; done

say "[2/3] sweep"
FAILED=()
for seed in "${SEEDS[@]}"; do for rate in "${RATES[@]}"; do
  for wl in "${WORKLOADS[@]}"; do for sys in "${SYSTEMS[@]}"; do
    d=$(run_dir "$sys" "$wl" "$rate" "$seed")
    if [ "$RESUME" = 1 ] && [ -f "$d/DONE" ]; then echo "skip $sys/$wl/$rate/$seed"; continue; fi
    mkdir -p "$d"
    trace="$WL/${wl}_r${rate}_s${seed}.pkl"
    echo "commit=$COMMIT system=$sys workload=$wl rate=$rate seed=$seed
trace=$trace cfg=$CFG duration=$DURATION warmup=$WARMUP measure=$MEASURE
ttft_slo_scale=$TTFT_SCALE tpot_slo_scale=$TPOT_SCALE
slo_base=$SLO_BASE_FILE prefill_speed=$PREFILL_SPEED_FILE
cmd=exp/scripts/run_v2_case.sh $sys $wl $rate $seed $trace $d" > "$d/META.txt"
    if "$SCRIPT_DIR/run_v2_case.sh" "$sys" "$wl" "$rate" "$seed" "$trace" "$d" \
         2>&1 | tee "$BASE/logs/$(basename "$d")_${sys}_${wl}_${rate}_${seed}.log"; then
      python3 "$SCRIPT_DIR/collect_v2_metrics.py" --run-dir "$d" \
        --slo-base "$SLO_BASE_FILE" --ttft-scale "$TTFT_SCALE" --tpot-scale "$TPOT_SCALE" \
        --warmup "$WARMUP" --measure "$MEASURE" \
        --label "$sys/$wl/$rate/$seed" -o "$d/metrics.json" && touch "$d/DONE"
    else
      echo "!! FAILED $sys/$wl/$rate/$seed"; FAILED+=("$sys/$wl/$rate/$seed")
    fi
    pkill -f "launch_multi_model[_]server" 2>/dev/null || true
    sleep 10
  done; done
done; done

say "[3/3] aggregate"
python3 "$SCRIPT_DIR/aggregate_v2.py" --base "$BASE" -o "$BASE/processed" || true
[ ${#FAILED[@]} -gt 0 ] && { echo "FAILED RUNS:"; printf '  %s\n' "${FAILED[@]}"; }
echo "=== sweep complete ==="
