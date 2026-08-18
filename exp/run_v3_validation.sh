#!/bin/bash
# Resume-safe validation-only extension of the committed Paper-Faithful V3 sweep.
# This script changes no scheduler/runtime code or experiment constants.
set -uo pipefail

cd /workspace/prism-exp
source exp/scripts/env.sh

# Fail before launching a multi-minute server if the reconstructed environment
# is incomplete. bootstrap normally supplies both; these checks also make a
# watchdog restart self-healing after a container/service restart.
if ! redis-cli ping >/dev/null 2>&1; then
  command -v redis-server >/dev/null 2>&1 && redis-server --daemonize yes
fi
redis-cli ping >/dev/null 2>&1 || { echo "FATAL: redis unavailable on localhost:6379"; exit 1; }
grep -q '"kvpr-global-v3"' prism-research/python/sglang/multi_model/multi_model_server_args.py \
  || { echo "FATAL: committed Paper-Faithful V3 patch is not applied"; exit 1; }
if [ ! -s "$SHAREGPT_JSON" ]; then
  mkdir -p "$(dirname "$SHAREGPT_JSON")"
  hf download anon8231489123/ShareGPT_Vicuna_unfiltered \
    ShareGPT_V3_unfiltered_cleaned_split.json --repo-type dataset \
    --local-dir "$(dirname "$SHAREGPT_JSON")" || exit 1
fi

BASE=${V3_VALIDATION_BASE:-exp/results/paper-faithful-v3-validation}
SOURCE_BASE=${V3_SOURCE_BASE:-exp/results/paper-faithful-v3}
WL=${V3_WORKLOAD_DIR:-exp/workloads/paper-faithful-v2}
SYSTEMS=(released-prototype paper-faithful-v3)
WORKLOADS=(steady bursty)
DURATION=420
WARMUP=60
MEASURE=300
export V3_TAU=0.00035 KVPR_TAU=0.00035
export V2_DURATION=$DURATION V2_WARMUP=$WARMUP V2_MEASURE=$MEASURE
export SLO_BASE_FILE=$PRISM_EXP/configs/v2/slo_base.json
export PREFILL_SPEED_FILE=$PRISM_EXP/configs/v2/prefill_speed.json
export NGPU=2 CFG=$PRISM_EXP/configs/v2/6model_2gpu.json

mkdir -p "$BASE"/{raw,logs,figures} "$WL"
printf 'source_branch=exp/paper-faithful-v3\nrates=2 4 8 10 14 20\nthree_seed_rates=4 8 20\nseeds=1 2 3\nduration=420\nwarmup=60\nmeasure=300\nttft_scale=5\ntpot_scale=3\ntau=0.00035\n' > "$BASE/META.txt"

# Reuse only complete, successful committed seed-1 runs. Copying keeps the new
# result namespace self-contained and never modifies the original V3 results.
for rate in 4 8 14 20; do
  for workload in "${WORKLOADS[@]}"; do for system in "${SYSTEMS[@]}"; do
    src="$SOURCE_BASE/raw/$system/$workload/rate_$rate/seed_1"
    dst="$BASE/raw/$system/$workload/rate_$rate/seed_1"
    if [ ! -e "$dst" ] && [ -f "$src/DONE" ] && [ -s "$src/metrics.json" ] \
       && jq -e '.completed > 0 and .failed == 0' "$src/metrics.json" >/dev/null; then
      mkdir -p "$(dirname "$dst")"
      cp -a "$src" "$dst"
      printf 'reused_from=%s\n' "$src" >> "$dst/META.txt"
      src_log="$SOURCE_BASE/logs/${system}_${workload}_r${rate}_s1.log"
      [ -f "$src_log" ] && cp -a "$src_log" "$BASE/logs/reused_${system}_${workload}_r${rate}_s1.log"
    fi
  done; done
done

cases=()
for rate in 2 10 14; do cases+=("$rate:1"); done
for rate in 4 8 20; do for seed in 1 2 3; do cases+=("$rate:$seed"); done; done

for item in "${cases[@]}"; do
  rate=${item%%:*}; seed=${item##*:}
  if [ ! -f "$WL/bursty_r${rate}_s${seed}.pkl" ] || [ ! -f "$WL/steady_r${rate}_s${seed}.pkl" ]; then
    python3 exp/scripts/build_paired_workload.py --rate "$rate" --duration "$DURATION" \
      --seed "$seed" --slo-base "$SLO_BASE_FILE" --outdir "$WL" || exit 1
  fi
done

failed=0
for item in "${cases[@]}"; do
  rate=${item%%:*}; seed=${item##*:}
  for workload in "${WORKLOADS[@]}"; do for system in "${SYSTEMS[@]}"; do
    out="$BASE/raw/$system/$workload/rate_$rate/seed_$seed"
    if [ -f "$out/DONE" ] && jq -e '.completed > 0 and .failed == 0' "$out/metrics.json" >/dev/null 2>&1; then
      echo "skip valid $system/$workload/rate=$rate/seed=$seed"
      continue
    fi
    if [ -d "$out" ] && find "$out" -mindepth 1 -print -quit | grep -q .; then
      mv "$out" "${out}_failed_attempt_$(date +%Y%m%dT%H%M%S)_$$"
    fi
    mkdir -p "$out"
    trace="$WL/${workload}_r${rate}_s${seed}.pkl"
    cat > "$out/META.txt" <<EOF
commit=$(git rev-parse HEAD)
system=$system workload=$workload rate=$rate seed=$seed
trace=$trace cfg=$CFG duration=$DURATION warmup=$WARMUP measure=$MEASURE
ttft_slo_scale=5 tpot_slo_scale=3 tau=$V3_TAU
EOF
    log="$BASE/logs/${system}_${workload}_r${rate}_s${seed}.log"
    if exp/scripts/run_v2_case.sh "$system" "$workload" "$rate" "$seed" "$trace" "$out" \
         2>&1 | tee "$log"; then
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
done

python3 exp/scripts/finalize_v3_validation.py --base "$BASE" || failed=1
echo "validation finished $(date -Is) failed=$failed"
exit "$failed"
