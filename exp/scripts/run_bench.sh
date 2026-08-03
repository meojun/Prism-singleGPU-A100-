#!/bin/bash
# Run the Prism multi-model benchmark client against a running server.
#
#   ./run_bench.sh <exp-name> <num-models> <port> [extra benchmark.py args...]
#
# Examples:
#   ./run_bench.sh static_2m 2 30000 --req-rate 10 --seed 42
#   ./run_bench.sh elastic_2m 2 30001 --enable-elastic-memory --req-rate 10
#   ./run_bench.sh prism_18m 18 30002 --e2e-benchmark --real-trace ./real_trace.pkl \
#                  --time-scale 1 --replication 1 --num-gpus 1
set -euo pipefail

EXP=${1:?usage: run_bench.sh <exp-name> <num-models> <port> [extra args]}
NMODELS=${2:?missing num-models}
PORT=${3:-30000}
shift 3 2>/dev/null || shift $#

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/env.sh"

MODEL_PATHS=()
for i in $(seq 1 "$NMODELS"); do MODEL_PATHS+=("model_$i"); done

cd "$PRISM_REPO/benchmark/multi-model"
mkdir -p "$PRISM_EXP/results" "$PRISM_EXP/results/requests"

python3 benchmark.py \
  --base-url "http://127.0.0.1:$PORT" \
  --num-models "$NMODELS" \
  --model-paths "${MODEL_PATHS[@]}" \
  --exp-name "$EXP" \
  --results-path "$PRISM_EXP/results" \
  --request-path "$PRISM_EXP/results/requests" \
  --seed 42 \
  "$@"

echo "results -> $PRISM_EXP/results"
