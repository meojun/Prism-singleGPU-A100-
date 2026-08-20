#!/bin/bash
# Bring a TP>1 model up in the worker-pool path and prove it served.
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

OUT=${1:?outdir required}
TP=${2:-2}
NGPU=${3:-2}
WORKERS=${4:-1}
MODEL_PATH=${5:-meta-llama/Llama-3.2-1B}
MODEL_NAME=${6:-model_1}
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
  --enable-gpu-scheduler --enable-controller --policy simple-global
  --enable-model-service --enable-worker-pool
  --enable-tp-worker-pool
  --workers-per-gpu "$WORKERS" --num-model-service-workers 1
  --num-gpus "$NGPU"
)
[ -n "${TP_MAX_GROUPS:-}" ] && ARGS+=(--tp-max-groups "$TP_MAX_GROUPS")
[ -n "${TP_ANTI_AFFINITY:-}" ] && ARGS+=(--enable-tp-anti-affinity)

VISIBLE=$(seq -s, 0 $((NGPU - 1)))
SESSION=$(echo "tpboot-tp${TP}-g${NGPU}-w${WORKERS}" | tr '.' '_')
tmux kill-session -t "$SESSION" 2>/dev/null

# Stale segments from a dead run make the next one fail in a way that looks
# like a TP bug (CLAUDE.md 8 / HANDOVER 5.6: the names are ipc_<gpu>_<worker>_*).
rm -f /dev/shm/ipc_* /dev/shm/mp-* 2>/dev/null

LOAD_RC=0
tmux new-session -d -s "$SESSION" \
  "export CUDA_VISIBLE_DEVICES=$VISIBLE && cd $PRISM_REPO/benchmark/multi-model && \
   source $SCRIPT_DIR/env.sh && export CUDA_VISIBLE_DEVICES=$VISIBLE && \
   python3 -m sglang.launch_multi_model_server ${ARGS[*]} 2>&1 | tee $LOGDIR/stdout.log"

cleanup() {
  # Only ever signal what we started.  Never `kill 0` (CLAUDE.md 8.1).
  tmux kill-session -t "$SESSION" 2>/dev/null
  sleep 3
  rm -f /dev/shm/ipc_* /dev/shm/mp-* 2>/dev/null
}
trap cleanup EXIT

echo -n "waiting for server"
T0=$(date +%s)
READY=0
for _ in $(seq 1 450); do
    if curl -sf "http://127.0.0.1:$PORT/get_model_names" >/dev/null 2>&1; then READY=1; break; fi
    if ! tmux has-session -t "$SESSION" 2>/dev/null; then
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

# --- inference ---------------------------------------------------------------
REQ_JSON=$OUT/tp2_requests.json
if [ "$READY" = 1 ]; then
    echo "=== sending requests"
    python3 - "$PORT" "$MODEL_NAME" "$REQ_JSON" <<'PY'
import json, sys, time
import urllib.request

port, model, out = sys.argv[1], sys.argv[2], sys.argv[3]
prompts = [
    "The capital of France is",
    "Explain in one sentence why the sky is blue:",
    "List three prime numbers:",
    "Translate to French: good morning.",
]
ok, failed, errors, lat = 0, 0, [], []
for p in prompts:
    body = json.dumps({
        "text": p, "model_name": model,
        "sampling_params": {"max_new_tokens": 24, "temperature": 0.0},
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/generate", data=body,
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            payload = r.read().decode()
        lat.append(time.time() - t0)
        ok += 1
        print(f"  ok  {p[:34]!r} -> {payload[:110]}")
    except Exception as e:
        failed += 1
        errors.append(f"{type(e).__name__}: {e}")
        print(f"  FAIL {p[:34]!r}: {e}")

json.dump({"summary": {
    "requests": len(prompts), "successful": ok, "failed": failed,
    "errors": errors,
    "latency_s": {"mean": (sum(lat)/len(lat)) if lat else None,
                  "all": lat},
}}, open(out, "w"), indent=2)
print(f"=== requests: {ok} ok / {failed} failed")
PY
fi

# --- evidence ----------------------------------------------------------------
cleanup
trap - EXIT
sleep 2

echo "=== rank -> GPU, straight from the engines' own logs"
grep -h "\[PAPER-TP\] engine rank:" "$LOGDIR"/*.log 2>/dev/null | sort -u || echo "  (none found)"
echo "=== slot plan"
grep -h "\[PAPER-TP\] slot plan:" "$LOGDIR"/*.log 2>/dev/null | head -1 || echo "  (none found)"
echo "=== worker pools"
grep -h "\[PAPER-TP\] WorkerPool" "$LOGDIR"/*.log 2>/dev/null | sort -u || echo "  (none found)"

python3 exp/scripts/collect_tp2_evidence.py \
    --logdir "$LOGDIR" --outdir "$OUT" --config "$CFG" \
    --startup-seconds "$STARTUP" \
    --ready "$READY" --load-rc "$LOAD_RC" 2>&1 | tail -25

exit 0
