#!/bin/bash
# Sanity-check sweep: 3 Llama colocation cases on 1x A100-80G, full Prism mode.
#
#   A  1x Llama-3.1-8B                (model_1)
#   B  2x Llama-3.1-8B                (model_1 + model_4  -> both 8B SLO slots)
#   C  Llama-3.1-8B + Llama-3.2-3B    (model_1 + model_2)
#
#   ./run_sanity.sh <A|B|C>
set -euo pipefail

CASE=${1:?usage: run_sanity.sh <A|B|C>}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/env.sh"

TTFT_SCALE=${TTFT_SCALE:-5}
TPOT_SCALE=${TPOT_SCALE:-2}

case "$CASE" in
  A) CFG=llama_1x8b.json;  MODELS=(model_1);          PORT=31000
     MAP='{"model_1":"meta-llama/Llama-3.1-8B"}' ;;
  B) CFG=llama_2x8b.json;  MODELS=(model_1 model_4);  PORT=31001
     MAP='{"model_1":"meta-llama/Llama-3.1-8B","model_4":"meta-llama/Llama-3.1-8B"}' ;;
  C) CFG=llama_8b_3b.json; MODELS=(model_1 model_2);  PORT=31002
     MAP='{"model_1":"meta-llama/Llama-3.1-8B","model_2":"meta-llama/Llama-3.2-3B"}' ;;
  *) echo "unknown case $CASE" >&2; exit 1 ;;
esac

N=${#MODELS[@]}
EXP=sanity_${CASE}
LOGDIR=$PRISM_EXP/server-logs/$EXP
mkdir -p "$LOGDIR"
SESSION=prism-sanity-$CASE

echo "### case $CASE : $CFG  (${MODELS[*]})  port=$PORT"

# --- launch full-Prism server -------------------------------------------------
# workers-per-gpu must be >= number of 'on' models on that GPU (README gotcha #5)
ARGS=(
  --model-config-file "$PRISM_EXP/configs/$CFG"
  --host 127.0.0.1 --port "$PORT"
  --disable-cuda-graph --disable-radix-cache
  --log-file "$LOGDIR/server.log"
  --enable-elastic-memory --use-kvcached-v0 --enable-cpu-share-memory
  --max-mem-usage 67.28
  --enable-gpu-scheduler --enable-controller --policy simple-global
  --enable-model-service --enable-worker-pool
  --workers-per-gpu "$N" --num-model-service-workers "$N"
  --num-gpus 1
)

tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" \
  "cd $PRISM_REPO/benchmark/multi-model && source $SCRIPT_DIR/env.sh && \
   python3 -m sglang.launch_multi_model_server ${ARGS[*]} 2>&1 | tee $LOGDIR/stdout.log"

echo -n "waiting for server"
READY=0
for _ in $(seq 1 300); do
    if curl -sf "http://127.0.0.1:$PORT/get_model_names" >/dev/null 2>&1; then READY=1; break; fi
    if ! tmux has-session -t "$SESSION" 2>/dev/null; then
        echo " -> DIED, see $LOGDIR/stdout.log"; exit 1
    fi
    echo -n "."; sleep 2
done
[ "$READY" = 1 ] || { echo " -> timeout, see $LOGDIR/stdout.log"; exit 1; }
echo " -> ready: $(curl -s http://127.0.0.1:$PORT/get_model_names)"

# --- benchmark ----------------------------------------------------------------
cd "$PRISM_REPO/benchmark/multi-model"
mkdir -p "$PRISM_EXP/results/sanity" "$PRISM_EXP/results/sanity/requests"

set +e
python3 benchmark.py \
  --base-url "http://127.0.0.1:$PORT" \
  --num-models "$N" --model-paths "${MODELS[@]}" \
  --exp-name "$EXP" \
  --results-path "$PRISM_EXP/results/sanity" \
  --request-path "$PRISM_EXP/results/sanity/requests" \
  --seed 42 --disable-tqdm \
  --e2e-benchmark --real-trace ./real_trace.pkl \
  --time-scale 1 --replication 1 --num-gpus 1 \
  --enable-elastic-memory \
  --ttft-slo-scale "$TTFT_SCALE" --tpot-slo-scale "$TPOT_SCALE" \
  > "$LOGDIR/bench.log" 2>&1
RC=$?
set -e
tmux kill-session -t "$SESSION" 2>/dev/null || true
[ $RC -eq 0 ] || { echo "benchmark failed rc=$RC, tail:"; tail -30 "$LOGDIR/bench.log"; exit $RC; }

# --- analyze ------------------------------------------------------------------
# time_scale is a float, so the filename carries "1.0x" -- glob rather than guess
REQF=$(ls "$PRISM_EXP"/results/sanity/requests/${EXP}_e2e_1gpu_*_output_requests.json | head -1)
METF=$(ls "$PRISM_EXP"/results/sanity/${EXP}_e2e_1gpu_*rep.json | head -1)
python3 "$SCRIPT_DIR/analyze_slo.py" \
  --req-file "$REQF" \
  --metrics-file "$METF" \
  --ttft-slo-scale "$TTFT_SCALE" --tpot-slo-scale "$TPOT_SCALE" \
  --label "$CASE" --model-map "$MAP" \
  --out "$PRISM_EXP/results/sanity/${EXP}_slo.json" > "$LOGDIR/analyze.log" 2>&1

echo "### case $CASE done -> $PRISM_EXP/results/sanity/${EXP}_slo.json"
