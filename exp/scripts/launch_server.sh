#!/bin/bash
# Launch a Prism multi-model server in a tmux session.
#
#   ./launch_server.sh <mode> <config.json> [port] [extra sglang args...]
#
# modes:
#   static   – colocated models, per-model static KV pool (S-Partition baseline)
#   elastic  – colocated models + kvcached elastic memory (Prism §5 only)
#   prism    – full Prism: elastic memory + global placement + GPU-local scheduler
#              + model service / worker pool (Prism §5 + §6)
#
# Session name: prism-<mode>.  Attach with: tmux attach -t prism-<mode>
set -euo pipefail

MODE=${1:?usage: launch_server.sh <static|elastic|prism> <config.json> [port] [extra args]}
CONFIG=${2:?missing model config json}
PORT=${3:-30000}
shift 3 2>/dev/null || shift $#
EXTRA=("$@")

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/env.sh"

CONFIG=$(readlink -f "$CONFIG")
LOGDIR=$PRISM_EXP/server-logs
mkdir -p "$LOGDIR"
SESSION=prism-$MODE

COMMON=(
  --model-config-file "$CONFIG"
  --host 127.0.0.1
  --port "$PORT"
  --disable-cuda-graph
  --disable-radix-cache
  --log-file "$LOGDIR/${MODE}.log"
)

case "$MODE" in
  static)
    ARGS=("${COMMON[@]}")
    ;;
  elastic)
    ARGS=("${COMMON[@]}" --enable-elastic-memory --use-kvcached-v0)
    ;;
  prism)
    # --max-mem-usage is the per-GPU memory budget in GiB that the global
    # scheduler is allowed to hand out (80GB A100 -> leave headroom).
    NUM_GPUS=${NUM_GPUS:-1}
    MAX_MEM=${MAX_MEM:-67.28}
    ARGS=("${COMMON[@]}"
      --enable-elastic-memory --use-kvcached-v0 --enable-cpu-share-memory
      --max-mem-usage "$MAX_MEM"
      --enable-gpu-scheduler --enable-controller --policy simple-global
      --enable-model-service --enable-worker-pool
      --workers-per-gpu "${WORKERS_PER_GPU:-2}"
      --num-model-service-workers "${MODEL_SERVICE_WORKERS:-2}"
      --num-gpus "$NUM_GPUS")
    ;;
  *)
    echo "unknown mode: $MODE" >&2; exit 1 ;;
esac

tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" \
  "cd $PRISM_REPO/benchmark/multi-model && source $SCRIPT_DIR/env.sh && \
   python3 -m sglang.launch_multi_model_server ${ARGS[*]} ${EXTRA[*]} 2>&1 | tee $LOGDIR/${MODE}_stdout.log"

echo "launched tmux session '$SESSION' on port $PORT"
echo "  logs   : $LOGDIR/${MODE}_stdout.log"
echo "  attach : tmux attach -t $SESSION"
echo -n "waiting for server to become ready"
for _ in $(seq 1 180); do
    if curl -sf "http://127.0.0.1:$PORT/get_model_names" >/dev/null 2>&1; then
        echo " -> ready"; curl -s "http://127.0.0.1:$PORT/get_model_names"; echo; exit 0
    fi
    if ! tmux has-session -t "$SESSION" 2>/dev/null; then
        echo " -> server died, see $LOGDIR/${MODE}_stdout.log"; exit 1
    fi
    echo -n "."; sleep 2
done
echo " -> timed out (still starting?); check $LOGDIR/${MODE}_stdout.log"
exit 1
