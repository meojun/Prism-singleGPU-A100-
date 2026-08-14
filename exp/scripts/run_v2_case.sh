#!/bin/bash
# One paper-faithful-v2 run: (system, workload, rate, seed) -> metrics.
#
#   ./run_v2_case.sh <system> <workload> <rate> <seed> <trace.pkl> <outdir>
#
# systems -- ONLY the scheduler flags differ between them:
#   released-prototype  prototype global (simple-global) + prototype local
#   paper-alg1-only     paper KVPR global               + prototype local
#   paper-alg2-only     prototype global                + paper Moore-Hodgson
#   paper-faithful      paper KVPR global               + paper Moore-Hodgson
#
# workload is a label only (bursty|steady); the trace file decides the arrivals.
# GPUs, models, precision, kvcached, prompts, output lengths, seed, SLO scales,
# warm-up and measurement window are identical by construction.
set -euo pipefail

SYSTEM=${1:?usage: run_v2_case.sh <system> <workload> <rate> <seed> <trace> <outdir>}
WORKLOAD=${2:?}
RATE=${3:?}
SEED=${4:?}
TRACE=${5:?}
OUTDIR=${6:?}
# benchmark.py runs from $PRISM_REPO/benchmark/multi-model, so a relative trace
# path resolves against the wrong directory and dies with FileNotFoundError
# after the server has already loaded all six models.
TRACE=$(readlink -f "$TRACE")
OUTDIR=$(readlink -f "$OUTDIR" 2>/dev/null || { mkdir -p "$OUTDIR"; readlink -f "$OUTDIR"; })

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/env.sh"

NGPU=${NGPU:-2}
CFG=${CFG:-$PRISM_EXP/configs/v2/6model_2gpu.json}
MAXMEM=${MAXMEM:-67.28}
TTFT_SCALE=${TTFT_SCALE:-5}
TPOT_SCALE=${TPOT_SCALE:-3}
KVPR_TAU=${KVPR_TAU:-0.35}
KVPR_WINDOW=${KVPR_WINDOW:-30}
KVPR_COOLDOWN=${KVPR_COOLDOWN:-30}
SLO_BASE=${SLO_BASE_FILE:-$PRISM_EXP/configs/v2/slo_base.json}
PREFILL_SPEED=${PREFILL_SPEED_FILE:-$PRISM_EXP/configs/v2/prefill_speed.json}

MODELS=($(python3 -c "import json;print(' '.join(m['model_name'] for m in json.load(open('$CFG'))))"))
NMODELS=${#MODELS[@]}
WORKERS=${WORKERS:-$(python3 -c "
import json,collections
c=collections.Counter(p['gpu_ids'][0] for m in json.load(open('$CFG'))
                      for p in m['init_placements'] if p['on'])
print(max(c.values())+1)")}

# The port is chosen at RUN TIME, not hard-coded. launch_multi_model_server
# derives one port per model engine from --port and refuses to start if --port
# is taken -- and it probes by binding 0.0.0.0, not 127.0.0.1. The 30000-40000
# range is where editor/tooling servers park dynamic ports, so a fixed base
# eventually collides and the run dies AFTER loading every model.
PORT=$(python3 "$SCRIPT_DIR/find_free_port.py" --from 41000 --span 24) || {
  echo "FATAL: no free port block"; exit 1; }
CONTROLLER=(--enable-controller)
case "$SYSTEM" in
  released-prototype) CONTROLLER+=(--policy simple-global) ;;
  paper-alg1-only)
      CONTROLLER+=(--policy kvpr-global --kvpr-tau "$KVPR_TAU"
                   --kvpr-rate-window "$KVPR_WINDOW" --slo-base-file "$SLO_BASE"
                   --kvpr-migration-cooldown "$KVPR_COOLDOWN"
                   --kvpr-tpot-slo-scale "$TPOT_SCALE") ;;
  paper-alg2-only)
      CONTROLLER+=(--policy simple-global --enable-moore-hodgson
                   --prefill-speed-file "$PREFILL_SPEED") ;;
  paper-faithful)
      CONTROLLER+=(--policy kvpr-global --kvpr-tau "$KVPR_TAU"
                   --kvpr-rate-window "$KVPR_WINDOW" --slo-base-file "$SLO_BASE"
                   --kvpr-migration-cooldown "$KVPR_COOLDOWN"
                   --kvpr-tpot-slo-scale "$TPOT_SCALE"
                   --enable-moore-hodgson --prefill-speed-file "$PREFILL_SPEED")
      ;;
  *) echo "unknown system: $SYSTEM" >&2; exit 1 ;;
esac

EXP="${SYSTEM}_${WORKLOAD}_rate${RATE}_seed${SEED}"
LOGDIR=$OUTDIR/server-logs
mkdir -p "$OUTDIR" "$LOGDIR" "$OUTDIR/requests"
# tmux rewrites '.' to '_' in session names, so a has-session probe on the raw
# name never matches and the readiness loop declares a healthy server DIED --
# orphaning it while it still holds the GPU. Normalise first.
SESSION=$(echo "v2-${SYSTEM}-${WORKLOAD}-r${RATE}-s${SEED}" | tr '.' '_')

echo "### $EXP  port=$PORT trace=$(basename "$TRACE") workers=$WORKERS models=$NMODELS"

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
  # NEVER `kill 0`. If SAMPLER is unset -- which it is on every early exit, i.e.
  # exactly when the server failed to come up -- "${SAMPLER:-0}" expands to 0,
  # and `kill 0` signals the ENTIRE PROCESS GROUP: this script, the sweep
  # driver, the watchdog, everything. Three separate multi-hour runs were lost
  # to this before supervisord's log made it visible ("terminated by SIGTERM;
  # not expected"). Only ever signal a PID we actually started.
  [ -n "${SAMPLER:-}" ] && kill "$SAMPLER" 2>/dev/null || true
  tmux kill-session -t "$SESSION" 2>/dev/null || true
  pkill -f "launch_multi_model_server.*--port $PORT" 2>/dev/null || true
  sleep 3
}
trap cleanup EXIT

tmux kill-session -t "$SESSION" 2>/dev/null || true
# A killed run leaves /dev/shm/ipc_0_model_*_root behind under the same names
# the next launch wants; clear them so a stale segment cannot be inherited.
rm -f /dev/shm/ipc_0_model_*_root 2>/dev/null || true
tmux new-session -d -s "$SESSION" \
  "export CUDA_VISIBLE_DEVICES=$VISIBLE && cd $PRISM_REPO/benchmark/multi-model && \
   source $SCRIPT_DIR/env.sh && export CUDA_VISIBLE_DEVICES=$VISIBLE && \
   python3 -m sglang.launch_multi_model_server ${ARGS[*]} 2>&1 | tee $LOGDIR/stdout.log"

echo -n "waiting for server"
READY=0
for _ in $(seq 1 600); do
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
  --exp-name "$EXP" --results-path "$OUTDIR" --request-path "$OUTDIR/requests" \
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

# Proof the intended scheduler actually ran, not merely that the flag parsed.
# -F is required: the markers are bracketed, and "[PAPER-ALG1]" as a regex is a
# character class whose "R-A" is an invalid range -- grep exits 2 and every
# count silently reads 0, indistinguishable from "the algorithm never ran".
GC="$LOGDIR/server.log.global_controller.log"
count_gc() { local n; n=$(grep -cF -- "$1" "$GC" 2>/dev/null || true); echo "${n:-0}"; }
count_gs() { local n; n=$(cat "$LOGDIR"/*gpu_scheduler*.log 2>/dev/null | grep -cF -- "$1" || true); echo "${n:-0}"; }
{
  echo "system=$SYSTEM workload=$WORKLOAD rate=$RATE seed=$SEED"
  echo "alg1_log_lines=$(count_gc '[PAPER-ALG1]')"
  echo "alg1_migrations=$(count_gc '[PAPER-ALG1] MIGRATE')"
  echo "alg2_log_lines=$(count_gs '[PAPER-ALG2]')"
  echo "alg2_underadmission_warnings=$(count_gs '[PAPER-ALG2-WARN]')"
  echo "proto_migrations=$(count_gc 'Reason: migrate model')"
  echo "activations=$(count_gc 'ACTION: activate')"
  echo "deactivations=$(count_gc 'ACTION: deactivate')"
  echo "idle_evictions=$(count_gc 'Reason: idle instance eviction')"
} > "$OUTDIR/scheduler_proof.txt"
cat "$OUTDIR/scheduler_proof.txt"
echo "### $EXP done"
