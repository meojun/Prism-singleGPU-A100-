#!/bin/bash
# Sustained 70B serving -- Stage 3 of the large-model stability check.
#
# Same launch path as run_tp_serve.sh, but instead of four smoke requests it
# drives open-loop traffic for B70_SUSTAIN_SEC and samples GPU state
# throughout.  Stability is the metric here, not peak throughput: what this
# is looking for is a rank dying, memory creeping, or latency drifting over
# tens of minutes -- none of which a short burst would show.
#
#   ./exp/scripts/run_tp_boot.sh <outdir> [TP] [NGPU] [WORKERS] [MODEL_PATH] [MODEL_NAME]
#
# This is the step-1/1b acceptance test.  The v4 attempt died before any engine
# existed, so the bar here is deliberately concrete:
#
#   * the server comes up with a TP=k model configured;
#   * k scheduler ranks report themselves on k DISTINCT GPUs;
#   * inference against that model succeeds.
#
# Evidence goes to <outdir> in the shape collect_tp2_evidence.py already reads,
# so the verdict is produced the same way v4's was and the two are comparable.
#
# Nothing here is left running: the server is torn down on every exit path.
set -uo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd /workspace/prism-exp
source "$SCRIPT_DIR/env.sh"

OUT=$(mkdir -p "${1:?outdir required}" && cd "$1" && pwd)   # absolute: the
# server cd's into benchmark/multi-model, so a relative --model-config-file
# resolves against the wrong directory and dies before any engine exists.
TP=${2:-2}
NGPU=${3:-2}
WORKERS=${WORKERS:-1}
MODEL_PATH=${4:-meta-llama/Llama-3.1-70B}
MODEL_NAME=${5:-model_70b}
MAXMEM=${TP_MAXMEM:-0.8}

LOGDIR=$OUT/server-logs
mkdir -p "$OUT" "$LOGDIR"
CFG=$OUT/tp_config.json

# GPU 0..TP-1 for the group; the config's init placement names them explicitly
# so the planner and the launcher cannot silently disagree about the group.
GPUS=$(python3 -c "print(list(range($TP)))")
cat > "$CFG" <<JSON
[
  {
    "model_name": "$MODEL_NAME",
    "model_path": "$MODEL_PATH",
    "tp_size": $TP,
    "init_placements": [{"gpu_ids": $GPUS, "on": true, "max_memory_pool_size": 20.0}]
  }
]
JSON
echo "=== TP boot test: tp_size=$TP ngpu=$NGPU workers/gpu=$WORKERS model=$MODEL_PATH"
cat "$CFG"

PORT=$(python3 "$SCRIPT_DIR/find_free_port.py" --from 41000 --span 24) || {
  echo "FATAL: no free port block"; exit 1; }

ARGS=(
  --model-config-file "$CFG"
  --host 127.0.0.1 --port "$PORT"
  --disable-cuda-graph --disable-radix-cache
  --log-file "$LOGDIR/server.log"
  --enable-elastic-memory --use-kvcached-v0 --enable-cpu-share-memory
  --max-mem-usage "$MAXMEM"
  --enable-gpu-scheduler --enable-controller --policy "${TP_POLICY:-simple-global}"
  --enable-model-service --enable-worker-pool
  --enable-tp-worker-pool
  --workers-per-gpu "$WORKERS" --num-model-service-workers 1
  --num-gpus "$NGPU"
)
case "${TP_POLICY:-simple-global}" in
  kvpr-global*)
    # This box's own tau and SLO baselines, not the committed ones -- those are
    # another machine's and are what run_profiling_v2.sh/calibrate_tau_v4.sh
    # exist to replace.
    _TAU=$(python3 -c "import json;print(json.load(open('$PRISM_EXP/results/paper-faithful-tp/calibration/tau.json'))['tau'])" 2>/dev/null || echo 0.171086)
    ARGS+=(--kvpr-tau "$_TAU" --slo-base-file "$PRISM_EXP/configs/v2/slo_base.json")
    echo "  policy=${TP_POLICY} tau=$_TAU (this box)" ;;
esac
[ -n "${TP_MAX_GROUPS:-}" ] && ARGS+=(--tp-max-groups "$TP_MAX_GROUPS")
[ -n "${TP_ANTI_AFFINITY:-}" ] && ARGS+=(--enable-tp-anti-affinity)
[ -n "${TP_ANTI_AFFINITY_STRICT:-}" ] && ARGS+=(--enable-tp-anti-affinity-strict)

VISIBLE=$(seq -s, 0 $((NGPU - 1)))

# Not tmux.  The first two attempts here died 10 s in with a zero-byte
# stdout.log, which reads exactly like a TP crash; tmux was the obvious suspect
# (CLAUDE.md 8.2 records sessions dying on this box).  It was not tmux -- it was
# a relative --model-config-file resolving against the server's own working
# directory.  Kept as a plain background process anyway, because that is what
# captured the traceback that identified it: tmux swallowed it.
rm -f /dev/shm/ipc_* /dev/shm/mp-* 2>/dev/null   # HANDOVER 5.6: ipc_<gpu>_<worker>_*

LOAD_RC=0
(
  cd "$PRISM_REPO/benchmark/multi-model" || exit 1
  export CUDA_VISIBLE_DEVICES=$VISIBLE
  exec python3 -m sglang.launch_multi_model_server "${ARGS[@]}"
) > "$LOGDIR/stdout.log" 2>&1 &
SERVER_PID=$!
echo "server pid=$SERVER_PID  log=$LOGDIR/stdout.log"

cleanup() {
  # Only ever signal a PID we actually started.  Never `kill 0` (CLAUDE.md 8.1).
  if [ -n "${SERVER_PID:-}" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
      kill -TERM "$SERVER_PID" 2>/dev/null
      for _ in $(seq 1 20); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 1; done
      kill -KILL "$SERVER_PID" 2>/dev/null
  fi
  # sglang starts its scheduler workers with the *spawn* method, so they are
  # reparented to init and outlive the server they belong to.  Killing the
  # parent alone leaves them holding GPU memory, which then looks like the next
  # run OOMing.  Match them by the venv interpreter running spawn_main.
  for _p in $(pgrep -f "prism-venv/bin/python3 -c from multiprocessing.spawn import spawn_main" 2>/dev/null); do
      [ "$_p" = "$$" ] && continue
      kill -TERM "$_p" 2>/dev/null
  done
  sleep 5
  for _p in $(pgrep -f "prism-venv/bin/python3 -c from multiprocessing.spawn import spawn_main" 2>/dev/null); do
      [ "$_p" = "$$" ] && continue
      kill -KILL "$_p" 2>/dev/null
  done
  sleep 2
  rm -f /dev/shm/ipc_* /dev/shm/mp-* 2>/dev/null
}
trap cleanup EXIT

echo -n "waiting for server"
T0=$(date +%s)
READY=0
for _ in $(seq 1 450); do
    if curl -sf "http://127.0.0.1:$PORT/get_model_names" >/dev/null 2>&1; then READY=1; break; fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo " -> DIED"; LOAD_RC=1; break
    fi
    echo -n "."; sleep 2
done
if [ "$READY" = 1 ]; then
    echo " -> ready"
else
    echo " -> not ready (see $LOGDIR/stdout.log)"; LOAD_RC=1
fi
STARTUP=$(( $(date +%s) - T0 ))
echo "startup: ${STARTUP}s"


# --- sustained load ----------------------------------------------------------
SECONDS_TO_RUN=${B70_SUSTAIN_SEC:-1800}
RATE=${B70_RATE:-1.0}
echo "=== sustained: ${SECONDS_TO_RUN}s at ~${RATE} req/s"
python3 "$SCRIPT_DIR/sustained_load.py" --port "$PORT" --model "$MODEL_NAME" \
    --seconds "$SECONDS_TO_RUN" --rate "$RATE" --out "$OUT" 2>&1 | tail -40
LOAD_RC=${PIPESTATUS[0]}
echo "=== sustained rc=$LOAD_RC"

# --- what stayed alive -------------------------------------------------------
echo "=== TP ranks seen (tp_size>1 only)"
grep -h "\[PAPER-TP\] engine rank" "$LOGDIR"/*.log 2>/dev/null \
    | sed 's/.*\[PAPER-TP\] //' | grep -v "tp_size=1" | sort -u
echo "=== fatal patterns"
for pat in "CUDA error" "NCCL WARN" "NCCL error" "out of memory" "Traceback"; do
    n=$(grep -cF -- "$pat" "$LOGDIR"/*.log 2>/dev/null | awk -F: '{s+=$2} END{print s+0}')
    echo "  $pat: $n"
done

cleanup
trap - EXIT
exit "$LOAD_RC"
