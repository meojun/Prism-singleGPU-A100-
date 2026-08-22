#!/bin/bash
# One anti-affinity arm, one seed, under the real trace.
#
#   ./exp/scripts/run_tp_case.sh <arm> <rate> <seed> <trace.pkl> <outdir>
#     arm: off | paper | strict
#
# This is the step-4 measurement, not a smoke test.  What it exists to answer is
# narrow and was chosen because the paper is explicit that the answer is usually
# "no":
#
#   Appendix A.2.2 says the decomposition "increases the LIKELIHOOD that these
#   parts are initially assigned to different GPUs due to rising KVPRs".
#
# So the constraint mostly does not bind, and a run where it never binds is a
# result to report, not a run to discard.  The three arms differ only in what
# happens when it does:
#
#   off     no constraint; the argmin may stack two shards of one model
#   paper   Appendix A.2.2 verbatim -- fall back to the second-lowest KVPR GPU,
#           without re-checking whether that one collides too
#   strict  stronger than the paper -- lowest-KVPR GPU holding no part of this
#           model.  Differs from `paper` only for tp_size >= 3.
#
# Raw data is the point.  Every cycle's decision, its counterfactual, and the
# placement it produced are preserved per run; the aggregate is derived from
# them and never replaces them.
set -uo pipefail
ulimit -n 65535
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd /workspace/prism-exp
source "$SCRIPT_DIR/env.sh"

ARM=${1:?arm required: off|paper|strict}
RATE=${2:?rate required}
SEED=${3:?seed required}
TRACE=${4:?trace .pkl required}
OUTBASE=$(mkdir -p "${5:?outdir required}" && cd "$5" && pwd)

NGPU=${TP_NGPU:-4}
WORKERS=${TP_WORKERS:-2}          # 5 TP=1 models need >4 single-GPU slots
NMODELS=6
CFG=${TP_CFG:-$PRISM_EXP/configs/v2/tp_6model_4gpu.json}
SLO_BASE=$PRISM_EXP/configs/v2/slo_base.json
PREFILL=$PRISM_EXP/configs/v2/prefill_speed.json
TAU=${TP_FORCE_TAU:-$(python3 -c "import json;print(json.load(open('$PRISM_EXP/results/paper-faithful-tp/calibration/tau.json'))['tau'])" 2>/dev/null || echo 0.171086)}

EXP="${TP_TAG:-tp-aa}-${ARM}_r${RATE}_s${SEED}"
OUTDIR=$OUTBASE/$EXP
LOGDIR=$OUTDIR/server-logs
mkdir -p "$OUTDIR" "$LOGDIR" "$OUTDIR/requests"
echo "### $EXP  ngpu=$NGPU workers/gpu=$WORKERS tau=$TAU (this box)"

MODELS=(meta-llama/Llama-3.2-1B Qwen/Qwen2.5-1.5B-Instruct meta-llama/Llama-3.2-3B
        Qwen/Qwen2.5-3B-Instruct meta-llama/Llama-3.1-8B Qwen/Qwen2.5-7B-Instruct)

PORT=$(python3 "$SCRIPT_DIR/find_free_port.py" --from 41000 --span 32) || {
  echo "FATAL: no free port block"; exit 1; }

ARGS=(
  --model-config-file "$CFG"
  --host 127.0.0.1 --port "$PORT"
  --disable-cuda-graph --disable-radix-cache
  --log-file "$LOGDIR/server.log"
  --enable-elastic-memory --use-kvcached-v0 --enable-cpu-share-memory
  --max-mem-usage "${TP_MAXMEM:-0.85}"
  --enable-gpu-scheduler --enable-controller
  --policy kvpr-global-tp
  --kvpr-tau "$TAU" --slo-base-file "$SLO_BASE"
  --kvpr-rate-window 30.0 --kvpr-migration-cooldown 30.0
  --enable-moore-hodgson --prefill-speed-file "$PREFILL"
  --enable-model-service --enable-worker-pool --enable-tp-worker-pool
  --workers-per-gpu "$WORKERS" --num-model-service-workers "$NMODELS"
  --num-gpus "$NGPU"
)
case "$ARM" in
  off)    ;;
  paper)  ARGS+=(--enable-tp-anti-affinity) ;;
  strict) ARGS+=(--enable-tp-anti-affinity --enable-tp-anti-affinity-strict) ;;
  *) echo "FATAL: unknown arm $ARM"; exit 1 ;;
esac

VISIBLE=$(seq -s, 0 $((NGPU - 1)))
rm -f /dev/shm/ipc_* /dev/shm/mp-* 2>/dev/null

(
  cd "$PRISM_REPO/benchmark/multi-model" || exit 1
  export CUDA_VISIBLE_DEVICES=$VISIBLE
  exec python3 -m sglang.launch_multi_model_server "${ARGS[@]}"
) > "$LOGDIR/stdout.log" 2>&1 &
SERVER_PID=$!
SAMPLER=""

cleanup() {
  # Only ever signal a PID we actually started.  Never `kill 0` (CLAUDE.md 8.1):
  # "${SAMPLER:-0}" expands to 0 on every early exit and kills the whole group.
  [ -n "$SAMPLER" ] && kill "$SAMPLER" 2>/dev/null
  if [ -n "${SERVER_PID:-}" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
      kill -TERM "$SERVER_PID" 2>/dev/null
      for _ in $(seq 1 25); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 1; done
      kill -KILL "$SERVER_PID" 2>/dev/null
  fi
  # spawn workers are reparented to init and hold GPU memory after the parent dies
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
for _ in $(seq 1 600); do
    curl -sf "http://127.0.0.1:$PORT/get_model_names" >/dev/null 2>&1 && { READY=1; break; }
    kill -0 "$SERVER_PID" 2>/dev/null || { echo " -> DIED"; break; }
    echo -n "."; sleep 2
done
[ "$READY" = 1 ] && echo " -> ready" || { echo " -> not ready"; tail -25 "$LOGDIR/stdout.log"; exit 1; }

( while true; do
    echo "$(date +%s.%N) $(nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
        --format=csv,noheader,nounits | tr '\n' ';')"
    sleep 2
  done ) > "$LOGDIR/gpu_timeline.txt" 2>/dev/null &
SAMPLER=$!

cd "$PRISM_REPO/benchmark/multi-model"
timeout --signal=TERM --kill-after=30s "${BENCHMARK_TIMEOUT:-1800}" python3 benchmark.py \
  --base-url "http://127.0.0.1:$PORT" \
  --num-models "$NMODELS" --model-paths "${MODELS[@]}" \
  --exp-name "$EXP" --results-path "$OUTDIR" --request-path "$OUTDIR/requests" \
  --seed "$SEED" --disable-tqdm \
  --e2e-benchmark --real-trace "$TRACE" \
  --time-scale 1 --replication 1 --num-gpus "$NGPU" \
  --enable-elastic-memory \
  --ttft-slo-scale "${TTFT_SCALE:-1.0}" --tpot-slo-scale "${TPOT_SCALE:-1.0}" \
  > "$LOGDIR/bench.log" 2>&1
RC=$?
cd /workspace/prism-exp
kill "$SAMPLER" 2>/dev/null; SAMPLER=""

# Proof the intended policy ran, not merely that the flag parsed.  -F is
# required: "[PAPER-ALG1-TP]" as a regex is a character class whose "R-A" is an
# invalid range, so grep exits 2 and every count reads 0 -- indistinguishable
# from "the policy never ran" (CLAUDE.md, learned the hard way in v2).
GC="$LOGDIR/server.log.global_controller.log"
cnt() { local n; n=$(grep -cF -- "$1" "$GC" 2>/dev/null || true); echo "${n:-0}"; }
{
  echo "arm=$ARM rate=$RATE seed=$SEED tau=$TAU benchmark_rc=$RC"
  echo "alg1_tp_lines=$(cnt '[PAPER-ALG1-TP]')"
  echo "alg1_v4_lines=$(cnt '[PAPER-ALG1-V4]')"
  echo "migrations=$(cnt '\"migration_decision\": \"MIGRATE\"')"
  echo "group_moves=$(cnt '[PAPER-ALG1-TP] group moved')"
} | tee "$OUTDIR/run_facts.txt"

cleanup
trap - EXIT
python3 "$SCRIPT_DIR/collect_tp_aa.py" --run "$OUTDIR" --arm "$ARM" \
    --rate "$RATE" --seed "$SEED" --outbase "$OUTBASE"
exit "$RC"
