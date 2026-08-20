#!/bin/bash
# Does a TP group's worker slot actually come back when the model is deactivated?
#
#   ./exp/scripts/run_tp_cycle.sh <outdir> [TP] [NGPU] [WORKERS]
#
# Step 1b's stated acceptance is two-sided: the k ranks must occupy k slots on
# k distinct GPUs *and* both must be returned on deactivation.  Everything so
# far only exercised the first half -- initial placement, then teardown of the
# whole server.  This drives the cycle the scheduler actually performs:
#
#   activate (initial)  ->  serve  ->  deactivate  ->  activate  ->  serve
#
# What is being watched, from the server's own logs:
#   * release_worker returns the group's slot to rank0's free list;
#   * no shadow GPU ever hands that slot to something else;
#   * the second activation lands on the same group and serves again.
#
# A failure here is the interesting kind: it means the static reservation is
# right at startup but the release path is not.
set -uo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd /workspace/prism-exp
source "$SCRIPT_DIR/env.sh"

OUT=$(mkdir -p "${1:?outdir required}" && cd "$1" && pwd)
TP=${2:-2}
NGPU=${3:-4}
WORKERS=${4:-1}
MODEL_PATH=${TP_MODEL_PATH:-meta-llama/Llama-3.2-1B}
MODEL_NAME=${TP_MODEL_NAME:-model_1}

LOGDIR=$OUT/server-logs
mkdir -p "$OUT" "$LOGDIR"
CFG=$OUT/tp_config.json
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
echo "=== TP cycle test: tp_size=$TP ngpu=$NGPU workers/gpu=$WORKERS"

PORT=$(python3 "$SCRIPT_DIR/find_free_port.py" --from 41000 --span 24) || {
  echo "FATAL: no free port block"; exit 1; }

ARGS=(
  --model-config-file "$CFG"
  --host 127.0.0.1 --port "$PORT"
  --disable-cuda-graph --disable-radix-cache
  --log-file "$LOGDIR/server.log"
  --enable-elastic-memory --use-kvcached-v0 --enable-cpu-share-memory
  --max-mem-usage "${TP_MAXMEM:-0.8}"
  --enable-gpu-scheduler --enable-controller --policy "${TP_POLICY:-simple-global}"
  --enable-model-service --enable-worker-pool
  --enable-tp-worker-pool
  --workers-per-gpu "$WORKERS" --num-model-service-workers 1
  --num-gpus "$NGPU"
)
VISIBLE=$(seq -s, 0 $((NGPU - 1)))
rm -f /dev/shm/ipc_* /dev/shm/mp-* 2>/dev/null

(
  cd "$PRISM_REPO/benchmark/multi-model" || exit 1
  export CUDA_VISIBLE_DEVICES=$VISIBLE
  exec python3 -m sglang.launch_multi_model_server "${ARGS[@]}"
) > "$LOGDIR/stdout.log" 2>&1 &
SERVER_PID=$!
echo "server pid=$SERVER_PID"

cleanup() {
  if [ -n "${SERVER_PID:-}" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
      kill -TERM "$SERVER_PID" 2>/dev/null
      for _ in $(seq 1 20); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 1; done
      kill -KILL "$SERVER_PID" 2>/dev/null
  fi
  # spawn workers are reparented to init and outlive the server (see run_tp_boot.sh)
  for _p in $(pgrep -f "prism-venv/bin/python3 -c from multiprocessing.spawn import spawn_main" 2>/dev/null); do
      [ "$_p" = "$$" ] && continue; kill -TERM "$_p" 2>/dev/null; done
  sleep 5
  for _p in $(pgrep -f "prism-venv/bin/python3 -c from multiprocessing.spawn import spawn_main" 2>/dev/null); do
      [ "$_p" = "$$" ] && continue; kill -KILL "$_p" 2>/dev/null; done
  sleep 2; rm -f /dev/shm/ipc_* /dev/shm/mp-* 2>/dev/null
}
trap cleanup EXIT

echo -n "waiting for server"
READY=0
for _ in $(seq 1 450); do
    curl -sf "http://127.0.0.1:$PORT/get_model_names" >/dev/null 2>&1 && { READY=1; break; }
    kill -0 "$SERVER_PID" 2>/dev/null || { echo " -> DIED"; break; }
    echo -n "."; sleep 2
done
[ "$READY" = 1 ] && echo " -> ready" || { echo " -> not ready"; exit 1; }

hit() {  # hit <label>
    echo "--- requests ($1)"
    python3 "$SCRIPT_DIR/tp_probe_requests.py" "$PORT" "$MODEL_NAME" "$OUT/requests_$1.json" 8 \
        2>&1 | tail -6
}
CTL_RC=0
ctl() {  # ctl <activate|deactivate> <owner_gpu>
    python3 - "$PORT" "$1" "$MODEL_NAME" "${2:-0}" <<'PY'
import json, sys, urllib.request
port, what, model, gpu = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
# gpu_id is REQUIRED on ActivateReqInput (io_struct.py:453 -- no default), and
# omitting it is a FastAPI 422 that looks like a server fault.  For a TP group
# it is rank0's GPU, which is exactly what the upstream comment on that field
# says: "For TP case, it's gpu_id for rank0".
common = {"model_name": model, "instance_idx": gpu, "gpu_id": gpu}
if what == "deactivate":
    pload = dict(common, evict_waiting_requests=False,
                 preempt=False, preempt_mode="RECOMPUTE")
else:
    pload = dict(common, memory_pool_size=20.0)
req = urllib.request.Request(f"http://127.0.0.1:{port}/{what}",
                             data=json.dumps(pload).encode(),
                             headers={"Content-Type": "application/json"})
try:
    with urllib.request.urlopen(req, timeout=300) as r:
        body = r.read().decode()
    print(f"  {what}: {body[:220]}")
    sys.exit(0 if '"success":true' in body.replace(" ", "").lower() else 3)
except Exception as e:
    print(f"  {what} FAILED: {type(e).__name__}: {e}")
    sys.exit(2)
PY
    local rc=$?
    [ "$rc" -ne 0 ] && CTL_RC=$rc
    return $rc
}

OWNER=0   # rank0's GPU: the group's control socket is bound there
hit before
echo "=== deactivate"
ctl deactivate "$OWNER"
sleep 10
echo "=== activate again"
ctl activate "$OWNER"
sleep 10
hit after

cleanup
trap - EXIT
sleep 2

echo
echo "=== worker slot accounting, from the server's own logs"
grep -hE "Assign worker|Released worker|release_worker|is not served by any worker|\[PAPER-TP\]" \
     "$LOGDIR"/*.log 2>/dev/null | sed 's/.*\] //' | tail -25
echo
echo "=== verdict"
python3 - "$OUT" "$CTL_RC" "$LOGDIR" <<'PY'
import json, re, sys
from pathlib import Path
out, ctl_rc, logdir = Path(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])

def reqs(name):
    p = out / f"requests_{name}.json"
    if not p.exists():
        return None
    s = json.loads(p.read_text())["summary"]
    return s["successful"], s["failed"]

before, after = reqs("before"), reqs("after")

# The first version of this test reported PASS from before/after alone.  That is
# a false pass: if the deactivate never lands, the model simply stays up and the
# "after" requests succeed for a reason that has nothing to do with the slot
# being released.  So the control calls are checked FIRST, and their failure is
# INVALID (the test did not run), not FAIL (the code is broken).
blob = "\n".join(p.read_text(errors="replace") for p in logdir.glob("*.log"))
released = len(re.findall(r"Released worker|release_worker|_free_workers", blob))
reassigned = len(re.findall(r"Assign worker \d+ to", blob))

if ctl_rc != 0:
    verdict, why = "INVALID", ("the activate/deactivate calls did not succeed, "
                               "so the slot-release path was never exercised")
elif not (before and after):
    verdict, why = "INVALID", "request probes did not produce a result file"
elif before[1] or after[1] or not after[0]:
    verdict, why = "FAIL", "requests failed around the cycle"
else:
    verdict, why = "PASS", "deactivated and reactivated, and served again after"

print(f"  before deactivate : {before}")
print(f"  after  reactivate : {after}")
print(f"  control calls rc  : {ctl_rc}")
print(f"  worker assignments seen in logs: {reassigned}")
print(f"  TP activate/deactivate cycle: {verdict} -- {why}")
json.dump({"verdict": verdict, "why": why, "control_rc": ctl_rc,
           "before": before, "after": after,
           "assign_worker_log_lines": reassigned},
          open(out / "cycle_verdict.json", "w"), indent=2)
PY
exit 0
