#!/bin/bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
RESULTS="$ROOT/exp/results/final-regression-diagnosis"
PROFILE_RC="/workspace/prism-exp/exp/results/final-regression-diagnosis/profile_pair.rc"
PIPELINE_LOG="$RESULTS/D1/pipeline.log"
mkdir -p "$RESULTS/D1"
trap 'rc=$?; printf "%s\n" "$rc" > "$RESULTS/D1/pipeline.rc"' EXIT

echo "[$(date -u +%FT%TZ)] waiting for representative-model profiling"
for _ in $(seq 1 1200); do
  if [ -s "$PROFILE_RC" ]; then break; fi
  sleep 2
done
if [ ! -s "$PROFILE_RC" ]; then
  echo "profile pipeline timeout" >&2
  exit 124
fi
profile_rc=$(tr -d '[:space:]' < "$PROFILE_RC")
if [ "$profile_rc" != "0" ]; then
  echo "profile pipeline failed rc=$profile_rc" >&2
  exit 1
fi
test -s /workspace/prism-exp/exp/results/final-regression-diagnosis/profiling/model_1.json
test -s /workspace/prism-exp/exp/results/final-regression-diagnosis/profiling/model_6.json
echo "[$(date -u +%FT%TZ)] profiling complete; validating Algorithm 2"

PYTHONPATH="$ROOT/prism-research/python" "$ROOT/prism-venv/bin/python" \
  "$ROOT/exp/tests/test_moore_hodgson.py"

RUN_OUT="$RESULTS/D1/raw/paper-alg2-only/steady/rate_8/seed_1"
mkdir -p "$RUN_OUT"
echo "[$(date -u +%FT%TZ)] starting D1 steady/r8/seed1"
PRISM_ROOT="$ROOT" NGPU=2 \
CFG="$ROOT/exp/configs/v2/6model_2gpu.json" MAXMEM=67.28 \
SLO_BASE_FILE="$ROOT/exp/configs/v2/slo_base.json" \
PREFILL_SPEED_FILE="$ROOT/exp/configs/v2/prefill_speed.json" \
BENCHMARK_TIMEOUT=2100 PRISM_DIAG_ALG2=1 \
bash "$ROOT/exp/scripts/run_v4_case.sh" paper-alg2-only steady 8 1 \
  "$RESULTS/workloads/steady_r8_s1.pkl" "$RUN_OUT"
touch "$RUN_OUT/DONE"

echo "[$(date -u +%FT%TZ)] collecting D1 metrics"
"$ROOT/prism-venv/bin/python" "$ROOT/exp/scripts/collect_v4_metrics.py" \
  --base "$RESULTS/D1" \
  --slo-base "$ROOT/exp/configs/v2/slo_base.json" \
  --config "$ROOT/exp/configs/v2/6model_2gpu.json" \
  --trace-dir "$RESULTS/workloads"
"$ROOT/prism-venv/bin/python" "$ROOT/exp/scripts/finalize_d1_diagnosis.py" \
  --root "$ROOT"
echo "[$(date -u +%FT%TZ)] D1 result complete"
