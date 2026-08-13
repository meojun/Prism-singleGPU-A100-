#!/bin/bash
# One Paper-Faithful-comparison run: (system, aggregate rate, seed) -> metrics.
#
#   ./run_pf_case.sh <system> <aggregate_rate> <seed> <trace.pkl> <outdir>
#
# systems (see docs/paper_faithful/design_analysis.md):
#   released-prototype  prototype global (simple-global) + prototype local
#   paper-alg1-only     paper KVPR global               + prototype local
#   paper-alg2-only     prototype global                + paper Moore-Hodgson
#   paper-faithful      paper KVPR global               + paper Moore-Hodgson
#
# Only the scheduler flags differ between systems. GPUs, models, precision,
# kvcached, trace, prompts, output lengths, arrival timestamps, seed, SLO scales,
# warm-up and measurement window are identical by construction -- the trace file is
# passed in and shared by both arms of a (rate, seed) pair.
set -euo pipefail

SYSTEM=${1:?usage: run_pf_case.sh <system> <rate> <seed> <trace> <outdir>}
RATE=${2:?}
SEED=${3:?}
TRACE=${4:?}
OUTDIR=${5:?}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/env.sh"

NGPU=${NGPU:-2}
SLOTS=${SLOTS:-1,4,5}
CFG=${CFG:-$PRISM_EXP/configs/llama_2gpu_3x8b.json}
MAXMEM=${MAXMEM:-67.28}
TTFT_SCALE=${TTFT_SCALE:-5}
TPOT_SCALE=${TPOT_SCALE:-3}
KVPR_TAU=${KVPR_TAU:-0.10}
KVPR_WINDOW=${KVPR_WINDOW:-30}
SLO_BASE=${SLO_BASE_FILE:-$PRISM_EXP/configs/slo_base_3x8b_sharegpt.json}
PREFILL_SPEED=${PREFILL_SPEED_FILE:-$PRISM_EXP/configs/prefill_speed.json}

MODELS=(); for i in ${SLOTS//,/ }; do MODELS+=("model_$i"); done
NMODELS=${#MODELS[@]}

WORKERS=${WORKERS:-$(python3 -c "
import json,collections
c=collections.Counter(p['gpu_ids'][0] for m in json.load(open('$CFG'))
                      for p in m['init_placements'] if p['on'])
print(max(c.values())+1)")}

# Scheduler flags -- the ONLY thing that varies across systems.
CONTROLLER=(--enable-controller)
case "$SYSTEM" in
  released-prototype)
      CONTROLLER+=(--policy simple-global); PORT=33000 ;;
  paper-alg1-only)
      CONTROLLER+=(--policy kvpr-global --kvpr-tau "$KVPR_TAU"
                   --kvpr-rate-window "$KVPR_WINDOW" --slo-base-file "$SLO_BASE"
                   --kvpr-tpot-slo-scale "$TPOT_SCALE"); PORT=33001 ;;
  paper-alg2-only)
      CONTROLLER+=(--policy simple-global --enable-moore-hodgson
                   --prefill-speed-file "$PREFILL_SPEED"); PORT=33002 ;;
  paper-faithful)
      CONTROLLER+=(--policy kvpr-global --kvpr-tau "$KVPR_TAU"
                   --kvpr-rate-window "$KVPR_WINDOW" --slo-base-file "$SLO_BASE"
                   --kvpr-tpot-slo-scale "$TPOT_SCALE"
                   --enable-moore-hodgson --prefill-speed-file "$PREFILL_SPEED")
      PORT=33003 ;;
  *) echo "unknown system: $SYSTEM" >&2; exit 1 ;;
esac

EXP="${SYSTEM}_rate${RATE}_seed${SEED}"
LOGDIR=$OUTDIR/server-logs
mkdir -p "$OUTDIR" "$LOGDIR" "$OUTDIR/requests"
SESSION=$(echo "pf-${SYSTEM}-r${RATE}-s${SEED}" | tr '.' '_')

echo "### $EXP  system=$SYSTEM rate=${RATE}req/s seed=$SEED port=$PORT"
echo "###   trace=$(basename "$TRACE")  cfg=$(basename "$CFG")  workers=$WORKERS"

ARGS=(
  --model-config-file "$CFG"
  --host 127.0.0.1 --port "$PORT"
  --disable-cuda-graph --disable-radix-cache
  --log-file "$LOGDIR/server.log"
  --enable-elastic-memory --use-kvcached-v0 --enable-cpu-share-memory
  --max-mem-usage "$MAXMEM"
  --enable-gpu-scheduler
  "${CONTROLLER[@]}"
  --enable-model-service --enable-worker-pool
  --workers-per-gpu "$WORKERS" --num-model-service-workers "$NMODELS"
  --num-gpus "$NGPU"
)

VISIBLE=$(seq -s, 0 $((NGPU - 1)))

cleanup() {
  kill "${SAMPLER:-0}" 2>/dev/null || true
  tmux kill-session -t "$SESSION" 2>/dev/null || true
  pkill -f "launch_multi_model_server.*--port $PORT" 2>/dev/null || true
  sleep 3
}
trap cleanup EXIT

tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" \
  "export CUDA_VISIBLE_DEVICES=$VISIBLE && cd $PRISM_REPO/benchmark/multi-model && \
   source $SCRIPT_DIR/env.sh && export CUDA_VISIBLE_DEVICES=$VISIBLE && \
   python3 -m sglang.launch_multi_model_server ${ARGS[*]} 2>&1 | tee $LOGDIR/stdout.log"

echo -n "waiting for server"
READY=0
for _ in $(seq 1 450); do
    if curl -sf "http://127.0.0.1:$PORT/get_model_names" >/dev/null 2>&1; then READY=1; break; fi
    if ! tmux has-session -t "$SESSION" 2>/dev/null; then
        echo " -> DIED, see $LOGDIR/stdout.log"; exit 1
    fi
    echo -n "."; sleep 2
done
[ "$READY" = 1 ] || { echo " -> timeout, see $LOGDIR/stdout.log"; exit 1; }
echo " -> ready"

( while true; do
    echo "$(date +%s.%N) $(nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
        --format=csv,noheader,nounits | tr '\n' ';')"
    sleep 2
  done ) > "$LOGDIR/gpu_timeline.txt" 2>/dev/null &
SAMPLER=$!

cd "$PRISM_REPO/benchmark/multi-model"
set +e
python3 benchmark.py \
  --base-url "http://127.0.0.1:$PORT" \
  --num-models "$NMODELS" --model-paths "${MODELS[@]}" \
  --exp-name "$EXP" \
  --results-path "$OUTDIR" \
  --request-path "$OUTDIR/requests" \
  --seed "$SEED" --disable-tqdm \
  --e2e-benchmark --real-trace "$TRACE" \
  --time-scale 1 --replication 1 --num-gpus "$NGPU" \
  --enable-elastic-memory \
  --ttft-slo-scale "$TTFT_SCALE" --tpot-slo-scale "$TPOT_SCALE" \
  > "$LOGDIR/bench.log" 2>&1
RC=$?
set -e
kill "$SAMPLER" 2>/dev/null || true

[ $RC -eq 0 ] || { echo "benchmark failed rc=$RC"; tail -40 "$LOGDIR/bench.log"; exit $RC; }

# Proof that the intended scheduler actually ran, not merely that the flag parsed.
GC="$LOGDIR/server.log.global_controller.log"
GS_GLOB="$LOGDIR/server.log.gpu_scheduler.log"
{
  echo "system=$SYSTEM rate=$RATE seed=$SEED"
  echo "alg1_log_lines=$( [ -f "$GC" ] && grep -c '\[PAPER-ALG1\]' "$GC" || echo 0)"
  echo "alg1_migrations=$( [ -f "$GC" ] && grep -c '\[PAPER-ALG1\] MIGRATE' "$GC" || echo 0)"
  echo "alg2_log_lines=$(cat "$LOGDIR"/*gpu_scheduler*.log 2>/dev/null | grep -c '\[PAPER-ALG2\]' || echo 0)"
  echo "proto_migrations=$( [ -f "$GC" ] && grep -c 'Reason: migrate model' "$GC" || echo 0)"
  echo "activations=$( [ -f "$GC" ] && grep -c 'ACTION: activate' "$GC" || echo 0)"
  echo "deactivations=$( [ -f "$GC" ] && grep -c 'ACTION: deactivate' "$GC" || echo 0)"
  echo "idle_evictions=$( [ -f "$GC" ] && grep -c 'Reason: idle instance eviction' "$GC" || echo 0)"
} > "$OUTDIR/scheduler_proof.txt"

cat "$OUTDIR/scheduler_proof.txt"
echo "### $EXP done"
