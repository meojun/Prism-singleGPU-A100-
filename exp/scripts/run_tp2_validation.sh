#!/bin/bash
# Does a TP=2 model actually serve on this 2-GPU allocation?
#
#   ./exp/scripts/run_tp2_validation.sh <outdir>
#
# Checks, all from the run's own logs rather than from the flag parsing:
#   * both TP ranks come up, on DIFFERENT GPUs (the anti-affinity property)
#   * NCCL initialises, and how long it takes
#   * the model answers requests, with TTFT/TPOT/E2E percentiles
#   * no deadlock: startup and the load phase both finish inside their budget
#   * per-GPU memory, so a silently co-located pair would be visible
#
# A TP=2 group is also the one configuration where the prototype turns the
# parallel weight-loading path OFF (model_runner.py: "Tensor parallelism is
# enabled, model service will not be used"), so this run records that too.
set -uo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/env.sh"
OUT=${1:-$PRISM_EXP/results/paper-faithful-v4/tp-validation}
mkdir -p "$OUT"
LOGDIR=$OUT/server-logs
mkdir -p "$LOGDIR"

CFG=$OUT/tp2_config.json
# BOTH models at TP=2, and --tensor-parallel-size 2 to match.
#
# A first attempt mixed one TP=2 model with one TP=1 model and died at
# activation with "Model model_5 not found in shared cpu models".  Worker-pool
# engines are generic -- they are built with the SERVER's tp_size, not the
# model's -- and the CPU weights are keyed by (model_path, tp_size), so an
# engine built at tp_size=1 can never find a model registered at tp_size=2.
# The prototype therefore supports one global TP degree per server, and a
# mixed-TP placement is not expressible.  That is a finding in its own right;
# what this run tests is whether a uniform TP=2 server works at all.
cat > "$CFG" <<JSON
[
  {
    "model_name": "model_5",
    "model_path": "meta-llama/Llama-3.1-8B",
    "tp_size": 2,
    "init_placements": [{"gpu_ids": [0, 1], "on": true, "max_memory_pool_size": 20.0}]
  },
  {
    "model_name": "model_1",
    "model_path": "meta-llama/Llama-3.2-1B",
    "tp_size": 2,
    "init_placements": [{"gpu_ids": [0, 1], "on": true, "max_memory_pool_size": 20.0}]
  }
]
JSON

PORT=$(python3 "$SCRIPT_DIR/find_free_port.py" --from 42000 --span 24) || {
  echo "FATAL: no free port block"; exit 1; }
SESSION=prism-v4-tp2
VISIBLE=0,1

cleanup() {
  [ -n "${SAMPLER:-}" ] && kill "$SAMPLER" 2>/dev/null || true
  tmux kill-session -t "$SESSION" 2>/dev/null || true
  pkill -f "launch_multi_model_server.*--port $PORT" 2>/dev/null || true
  sleep 3
}
trap cleanup EXIT

ARGS=(
  --model-config-file "$CFG"
  --host 127.0.0.1 --port "$PORT"
  --disable-cuda-graph --disable-radix-cache
  --log-file "$LOGDIR/server.log"
  --enable-elastic-memory --use-kvcached-v0 --enable-cpu-share-memory
  --max-mem-usage 67.28
  --enable-gpu-scheduler --enable-controller --policy kvpr-global-v4
  --kvpr-tau "${KVPR_TAU:-0.00035}" --kvpr-rate-window 30
  --slo-base-file "$PRISM_EXP/configs/v2/slo_base.json"
  --kvpr-migration-cooldown 30 --kvpr-tpot-slo-scale 3
  --enable-moore-hodgson --prefill-speed-file "$PRISM_EXP/configs/v2/prefill_speed.json"
  # NO --enable-worker-pool.  Worker-pool engines are created per
  # (gpu_id, worker_id) and bound to a single-GPU list [gpu_id]
  # (multi_model_server.py), so nothing there can form a TP group -- two
  # configurations died at activation with "not found in shared cpu models"
  # before this was read out of the code.  launch_model_engines takes
  # instance_config.gpu_ids whole, so it is the only path that can host TP>1.
  # Prism's own serving mode is the worker-pool one, which is the finding.
  --enable-model-service
  --workers-per-gpu 2 --num-model-service-workers 2
  --num-gpus 2 --tensor-parallel-size 2
)

tmux kill-session -t "$SESSION" 2>/dev/null || true
rm -f /dev/shm/ipc_[0-9]*_root /dev/shm/cuda.shm.* /dev/shm/ipc_0_model_*_root 2>/dev/null || true
T_LAUNCH=$(date +%s.%N)
tmux new-session -d -s "$SESSION" \
  "export CUDA_VISIBLE_DEVICES=$VISIBLE && cd $PRISM_REPO/benchmark/multi-model && \
   source $SCRIPT_DIR/env.sh && export CUDA_VISIBLE_DEVICES=$VISIBLE && \
   python3 -m sglang.launch_multi_model_server ${ARGS[*]} 2>&1 | tee $LOGDIR/stdout.log"

echo -n "waiting for TP=2 server"
READY=0
for _ in $(seq 1 450); do
    if curl -sf "http://127.0.0.1:$PORT/get_model_names" >/dev/null 2>&1; then READY=1; break; fi
    if ! tmux has-session -t "$SESSION" 2>/dev/null; then
        echo " -> DIED"; tail -60 "$LOGDIR/stdout.log" > "$OUT/startup_failure.log"; break
    fi
    echo -n "."; sleep 2
done
T_READY=$(date +%s.%N)
STARTUP=$(python3 -c "print(f'{$T_READY - $T_LAUNCH:.2f}')")
echo " -> ready=$READY in ${STARTUP}s"

( while true; do
    echo "$(date +%s.%N) $(nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
        --format=csv,noheader,nounits | tr '\n' ';')"
    sleep 2
  done ) > "$LOGDIR/gpu_timeline.txt" 2>/dev/null &
SAMPLER=$!

RC_LOAD=1
if [ "$READY" = 1 ]; then
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader > "$OUT/gpu_memory_after_load.txt"
  timeout 900 python3 "$SCRIPT_DIR/tp2_client.py" --url "http://127.0.0.1:$PORT" \
      --models model_5 model_1 --requests "${TP2_REQUESTS:-120}" --concurrency 8 \
      --output-len 64 --out "$OUT/tp2_requests.json" > "$OUT/tp2_client.log" 2>&1
  RC_LOAD=$?
  tail -30 "$OUT/tp2_client.log"
fi
kill "$SAMPLER" 2>/dev/null || true; SAMPLER=

# --- evidence extraction, from the logs the run itself produced -------------
python3 "$SCRIPT_DIR/collect_tp2_evidence.py" \
    --logdir "$LOGDIR" --outdir "$OUT" --startup-seconds "$STARTUP" \
    --ready "$READY" --load-rc "$RC_LOAD" --config "$CFG"
echo "### TP=2 validation written to $OUT"
