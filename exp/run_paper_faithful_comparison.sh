#!/bin/bash
# Released Prism Prototype vs Paper-Faithful Prism -- full sweep, end to end.
#
#   ./exp/run_paper_faithful_comparison.sh [--dry-run] [--resume] [--skip-smoke]
#
# Runs unattended (designed to be launched inside tmux and left overnight):
#   0. environment + GPU check
#   1. build one ShareGPT trace per (rate, seed)   -- shared by BOTH systems
#   2. profile the no-contention SLO baseline and chunked-prefill speed c_i
#   3. smoke test both systems and PROVE Alg1/Alg2 actually executed
#   4. the sweep: 2 systems x 8 rates x 3 seeds = 48 runs
#   5. aggregate -> results.csv / summary.csv, figures, REPORT.md
#
# --resume skips any run that already wrote its DONE marker, so the script can be
# killed and restarted without losing completed work.
set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/scripts" && pwd)
ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
source "$SCRIPT_DIR/env.sh"

DRY=0; RESUME=0; SKIP_SMOKE=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY=1 ;;
    --resume) RESUME=1 ;;
    --skip-smoke) SKIP_SMOKE=1 ;;
    *) echo "unknown option: $arg" >&2; exit 1 ;;
  esac
done

SYSTEMS=(${PF_SYSTEMS:-released-prototype paper-faithful})
RATES=(${PF_RATES:-2.5 5 7.5 10 15 20 25 30})
SEEDS=(${PF_SEEDS:-1 2 3})

NGPU=2
SLOTS=1,4,5
NSLOTS=3
PHASE_LEN=${PF_PHASE_LEN:-360}     # 60 s warm-up + 300 s measurement
WARMUP=${PF_WARMUP:-60}
MEASURE=${PF_MEASURE:-300}
TTFT_SCALE=${PF_TTFT_SCALE:-5}
TPOT_SCALE=${PF_TPOT_SCALE:-3}
KVPR_TAU=${PF_KVPR_TAU:-0.10}
KVPR_WINDOW=${PF_KVPR_WINDOW:-30}

BASE=$PRISM_EXP/results/paper-faithful-comparison
TRACES=${DATASETS:-/workspace/datasets}/sharegpt/pf
CFG=$PRISM_EXP/configs/llama_2gpu_3x8b.json
export SLO_BASE_FILE=${SLO_BASE_FILE:-$PRISM_EXP/configs/slo_base_pf.json}
export PREFILL_SPEED_FILE=${PREFILL_SPEED_FILE:-$PRISM_EXP/configs/prefill_speed.json}
export NGPU SLOTS CFG TTFT_SCALE TPOT_SCALE KVPR_TAU KVPR_WINDOW

mkdir -p "$BASE"/{metadata,raw,processed,figures,logs} "$TRACES"

say() { echo -e "\n=== $* ===" ; }
run_dir() { echo "$BASE/raw/$1/rate_$2/seed_$3"; }

# ---------------------------------------------------------------- plan / dry-run
plan() {
  local n=0
  for sys in "${SYSTEMS[@]}"; do
    for rate in "${RATES[@]}"; do
      for seed in "${SEEDS[@]}"; do
        printf '%-20s | rate=%-5s | seed=%s\n' "$sys" "$rate" "$seed"
        n=$((n+1))
      done
    done
  done
  echo "Total: $n runs"
}

if [ "$DRY" = 1 ]; then plan; exit 0; fi

say "[0/5] environment"
nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv,noheader
GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
[ "$GPUS" -ge 2 ] || { echo "FATAL: need 2 GPUs, found $GPUS" >&2; exit 1; }
redis-cli ping >/dev/null 2>&1 || { echo "FATAL: redis not answering on :6379" >&2; exit 1; }
python3 -c "import torch,sys; sys.exit(0 if torch.cuda.device_count()>=2 else 1)" \
  || { echo "FATAL: torch sees <2 GPUs" >&2; exit 1; }
echo "redis OK, torch sees $(python3 -c 'import torch;print(torch.cuda.device_count())') GPUs"

# run-wide metadata, captured once
{
  echo "timestamp: $(date -Is)"
  echo "git_branch: $(git -C "$PRISM_ROOT" rev-parse --abbrev-ref HEAD)"
  echo "git_commit: $(git -C "$PRISM_ROOT" rev-parse HEAD)"
  echo "prism_research_commit: $(git -C "$PRISM_REPO" rev-parse HEAD)"
  echo "kvcached_branch: $(git -C "$PRISM_ROOT/kvcached-prism" rev-parse --abbrev-ref HEAD)"
  echo "kvcached_commit: $(git -C "$PRISM_ROOT/kvcached-prism" rev-parse HEAD)"
  echo "cuda_version: $(python3 -c 'import torch;print(torch.version.cuda)')"
  echo "torch_version: $(python3 -c 'import torch;print(torch.__version__)')"
  echo "driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
  echo "gpu_model: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
  echo "gpu_count: $GPUS"
  echo "models: 3x meta-llama/Llama-3.1-8B in slots $SLOTS"
  echo "request_rate_semantics: aggregate over all models (per-model = rate/$NSLOTS)"
  echo "rates: ${RATES[*]}"
  echo "seeds: ${SEEDS[*]}"
  echo "systems: ${SYSTEMS[*]}"
  echo "ttft_slo_scale: $TTFT_SCALE"
  echo "tpot_slo_scale: $TPOT_SCALE"
  echo "kvpr_tau: $KVPR_TAU"
  echo "kvpr_rate_window_s: $KVPR_WINDOW"
  echo "global_scheduler_interval_s: 5 (upstream SCHEDULE_INTERVAL)"
  echo "warmup_s: $WARMUP"
  echo "measure_s: $MEASURE"
  echo "trace_phase_len_s: $PHASE_LEN"
} > "$BASE/metadata/run_metadata.txt"
cat "$BASE/metadata/run_metadata.txt"

say "[1/5] traces (one per rate x seed, shared by both systems)"
SG=${SHAREGPT_JSON}
if [ ! -f "$SG" ]; then
  echo "downloading ShareGPT..."
  hf download anon8231489123/ShareGPT_Vicuna_unfiltered \
     ShareGPT_V3_unfiltered_cleaned_split.json --repo-type dataset \
     --local-dir "$(dirname "$SG")" || { echo "FATAL: ShareGPT download failed" >&2; exit 1; }
fi
for rate in "${RATES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    out=$TRACES/pf_rate${rate}_seed${seed}.pkl
    if [ -f "$out" ]; then echo "  have $(basename "$out")"; continue; fi
    per=$(python3 -c "print(f'{$rate/$NSLOTS:.6f}')")
    python3 "$SCRIPT_DIR/build_sharegpt_trace.py" --variant rate --slots "$SLOTS" \
      --phase-rates "$per,$per,$per" --phase-len "$PHASE_LEN" --cv 1.0 \
      --seed "$seed" --out "$out" >> "$BASE/logs/trace_build.log" 2>&1 \
      || { echo "FATAL: trace build failed for rate=$rate seed=$seed" >&2; exit 1; }
    echo "  built $(basename "$out")  (per-model ${per} req/s)"
  done
done

say "[2/5] no-contention profiling (SLO baseline + chunked-prefill speed c_i)"
if [ -f "$SLO_BASE_FILE" ] && [ -f "$PREFILL_SPEED_FILE" ]; then
  echo "  already profiled:"; cat "$SLO_BASE_FILE"; cat "$PREFILL_SPEED_FILE"
else
  REFDIR=$BASE/raw/_profiling
  REFTRACE=$TRACES/pf_ref_solo.pkl
  [ -f "$REFTRACE" ] || python3 "$SCRIPT_DIR/build_sharegpt_trace.py" --variant rate \
      --slots 1 --phase-rates "4" --phase-len 180 --cv 1.0 --seed 42 --out "$REFTRACE" \
      >> "$BASE/logs/trace_build.log" 2>&1
  mkdir -p "$REFDIR"
  # Solo: one model, one GPU -> no colocation, no queueing.
  NGPU=1 SLOTS=1 CFG=$PRISM_EXP/configs/llama_1gpu_solo8b.json \
  CUDA_VISIBLE_DEVICES=0 \
    bash "$SCRIPT_DIR/run_pf_case.sh" released-prototype 4 42 "$REFTRACE" "$REFDIR" \
    > "$BASE/logs/profiling.log" 2>&1 \
    || { echo "FATAL: profiling run failed, see $BASE/logs/profiling.log" >&2; exit 1; }

  REQF=$(ls -t "$REFDIR"/requests/*_output_requests.json | head -1)
  METF=$(ls -t "$REFDIR"/released-prototype_rate4_seed42_e2e_*rep.json | head -1)
  python3 "$SCRIPT_DIR/derive_slo_baseline.py" --req-file "$REQF" --metrics-file "$METF" \
      --slots model_1 --out "$SLO_BASE_FILE" >> "$BASE/logs/profiling.log" 2>&1 \
    || python3 - "$REQF" "$SLO_BASE_FILE" <<'PY'
# derive_slo_baseline.py expects a --run tag; fall back to computing p95 directly.
import json, sys
import numpy as np
reqs = json.loads(open(sys.argv[1]).read().splitlines()[-1])
ok = [r for r in reqs if r.get("success") and r.get("model") == "model_1"]
ttft = float(np.percentile([r["ttft"] for r in ok], 95))
tpot = float(np.percentile([r["tpot"] for r in ok], 95)) * 1000.0
out = {s: [ttft, tpot] for s in ("model_1", "model_4", "model_5")}
json.dump(out, open(sys.argv[2], "w"), indent=2)
print("derived SLO baseline:", out)
PY
  # Same architecture in all three slots -> one baseline row copied across.
  python3 - "$SLO_BASE_FILE" <<'PY'
import json, sys
p = sys.argv[1]
d = json.load(open(p))
row = d.get("model_1")
for s in ("model_1", "model_4", "model_5"):
    d[s] = row
json.dump(d, open(p, "w"), indent=2)
print("SLO baseline:", d)
PY
  python3 "$SCRIPT_DIR/profile_prefill_speed.py" --req-file "$REQF" \
      --source-slot model_1 --also-slots model_4,model_5 -o "$PREFILL_SPEED_FILE" \
      >> "$BASE/logs/profiling.log" 2>&1 \
    || { echo "FATAL: prefill-speed profiling failed" >&2; exit 1; }
  cat "$PREFILL_SPEED_FILE"
fi
cp "$SLO_BASE_FILE" "$PREFILL_SPEED_FILE" "$BASE/metadata/" 2>/dev/null || true

say "[3/5] smoke test -- both systems, and proof Alg1/Alg2 really ran"
if [ "$SKIP_SMOKE" = 1 ]; then
  echo "  skipped (--skip-smoke)"
else
  SMOKE_TRACE=$TRACES/pf_smoke.pkl
  [ -f "$SMOKE_TRACE" ] || python3 "$SCRIPT_DIR/build_sharegpt_trace.py" --variant rate \
      --slots "$SLOTS" --phase-rates "1,1,1" --phase-len 90 --cv 1.0 --seed 99 \
      --out "$SMOKE_TRACE" >> "$BASE/logs/trace_build.log" 2>&1
  for sys in "${SYSTEMS[@]}"; do
    d=$BASE/raw/_smoke/$sys; mkdir -p "$d"
    echo "  smoke: $sys"
    bash "$SCRIPT_DIR/run_pf_case.sh" "$sys" 3 99 "$SMOKE_TRACE" "$d" \
      > "$BASE/logs/smoke_$sys.log" 2>&1 \
      || { echo "FATAL: smoke failed for $sys, see $BASE/logs/smoke_$sys.log" >&2; exit 1; }
    cat "$d/scheduler_proof.txt"
    if [ "$sys" = "paper-faithful" ]; then
      a1=$(grep -oP 'alg1_log_lines=\K\d+' "$d/scheduler_proof.txt")
      a2=$(grep -oP 'alg2_log_lines=\K\d+' "$d/scheduler_proof.txt")
      [ "${a1:-0}" -gt 0 ] || { echo "FATAL: no [PAPER-ALG1] activity in paper-faithful" >&2; exit 1; }
      [ "${a2:-0}" -gt 0 ] || { echo "FATAL: no [PAPER-ALG2] activity in paper-faithful" >&2; exit 1; }
      echo "  VERIFIED: Algorithm 1 ($a1 log lines) and Algorithm 2 ($a2 log lines) executed"
    fi
    if [ "$sys" = "released-prototype" ]; then
      a1=$(grep -oP 'alg1_log_lines=\K\d+' "$d/scheduler_proof.txt")
      a2=$(grep -oP 'alg2_log_lines=\K\d+' "$d/scheduler_proof.txt")
      [ "${a1:-0}" -eq 0 ] && [ "${a2:-0}" -eq 0 ] \
        || { echo "FATAL: prototype arm shows paper-algorithm activity (a1=$a1 a2=$a2)" >&2; exit 1; }
      echo "  VERIFIED: prototype arm ran with neither paper algorithm"
    fi
  done
fi

say "[4/5] sweep: ${#SYSTEMS[@]} systems x ${#RATES[@]} rates x ${#SEEDS[@]} seeds"
plan
TOTAL=0; DONE=0; FAILED=0
for sys in "${SYSTEMS[@]}"; do
  for rate in "${RATES[@]}"; do
    for seed in "${SEEDS[@]}"; do
      TOTAL=$((TOTAL+1))
      d=$(run_dir "$sys" "$rate" "$seed")
      if [ "$RESUME" = 1 ] && [ -f "$d/DONE" ]; then
        echo "[$TOTAL] SKIP (done)  $sys rate=$rate seed=$seed"
        DONE=$((DONE+1)); continue
      fi
      mkdir -p "$d"
      trace=$TRACES/pf_rate${rate}_seed${seed}.pkl
      echo "[$TOTAL] RUN  $sys rate=$rate seed=$seed  $(date -Is)"
      if bash "$SCRIPT_DIR/run_pf_case.sh" "$sys" "$rate" "$seed" "$trace" "$d" \
           > "$d/run.log" 2>&1; then
        if python3 "$SCRIPT_DIR/collect_pf_metrics.py" --outdir "$d" --system "$sys" \
             --rate "$rate" --seed "$seed" --warmup "$WARMUP" --measure "$MEASURE" \
             --ttft-slo-scale "$TTFT_SCALE" --tpot-slo-scale "$TPOT_SCALE" \
             --slo-base "$SLO_BASE_FILE" --out "$d/metrics.json" >> "$d/run.log" 2>&1; then
          touch "$d/DONE"; DONE=$((DONE+1))
          echo "     -> ok"
        else
          FAILED=$((FAILED+1)); echo "     -> METRICS FAILED (see $d/run.log)"
        fi
      else
        FAILED=$((FAILED+1)); echo "     -> RUN FAILED (see $d/run.log)"
      fi
      # let the GPUs settle between runs
      pkill -f launch_multi_model_server 2>/dev/null || true
      sleep 10
    done
  done
done
echo "sweep finished: $DONE/$TOTAL ok, $FAILED failed"

say "[5/5] aggregate + figures + report"
python3 "$SCRIPT_DIR/aggregate_pf.py" --base "$BASE" >> "$BASE/logs/aggregate.log" 2>&1 \
  && echo "  wrote $BASE/processed/results.csv and summary.csv" \
  || echo "  aggregation failed, see $BASE/logs/aggregate.log"

echo
echo "=== ALL DONE $(date -Is) ==="
echo "results : $BASE/processed/results.csv"
echo "summary : $BASE/processed/summary.csv"
echo "figures : $BASE/figures/"
echo "report  : $BASE/REPORT.md"
