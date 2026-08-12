#!/bin/bash
# Multi-GPU Prism run: the paper's canonical 8-model mix (§7.2/§7.3) across N GPUs.
#
#   ./run_2gpu.sh <arm> [time_scale]
#
# arms:
#   glob_on   full Prism: ballooning + GPU-local scheduler + GLOBAL controller (§6.1)
#   glob_off  same, minus --enable-controller -> initial placement is frozen
#
# The pair is the §7.3 / Figure 7 ablation ("global model placement on/off").
# Note the GPU-local scheduler cannot be disabled here: the worker-pool request
# handler raises "GPU scheduler must be enabled when using worker pool", so
# --enable-controller is the only clean global-placement knob.
#
# Why this differs from run_sanity.sh (which is 1-GPU only):
#   * --num-gpus is threaded through BOTH the server and the benchmark client
#   * benchmark.py stamps num_gpus into the output filename
#     (f"{exp}_e2e_{num_gpus}gpu_..."), so the result glob is built from
#     $NGPU rather than the literal "1gpu" run_sanity.sh hard-codes
#   * every GPU must own >=1 `on: true` model: launch_multi_model_server only
#     starts a GPU scheduler for gpu_ids present in the initial placement
set -euo pipefail

ARM=${1:?usage: run_2gpu.sh <glob_on|glob_off> [time_scale]}
TS=${2:-1}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/env.sh"

NGPU=${NGPU:-2}
CFG=${CFG:-$PRISM_EXP/configs/llama_2gpu_8model.json}
WORKERS=${WORKERS:-5}          # >= max on:true models per GPU (4), + headroom for migration
MAXMEM=${MAXMEM:-67.28}        # per-GPU GiB budget handed to the global scheduler
TTFT_SCALE=${TTFT_SCALE:-5}
TPOT_SCALE=${TPOT_SCALE:-2}
TRACE=${TRACE:-./real_trace.pkl}
NMODELS=${NMODELS:-8}
TAG=${TAG:-fig7}

case "$ARM" in
  glob_on)  CONTROLLER=(--enable-controller --policy simple-global); PORT=32000 ;;
  glob_off) CONTROLLER=();                                           PORT=32001 ;;
  *) echo "unknown arm: $ARM (want glob_on|glob_off)" >&2; exit 1 ;;
esac

case "$TRACE" in
  ./*|../*) : ;;
  *) TRACE=$(readlink -f "$TRACE") ;;
esac

MODELS=(); for i in $(seq 1 "$NMODELS"); do MODELS+=("model_$i"); done
# slot -> hf path, read straight out of the config so the two can never drift
MAP=$(python3 -c "
import json,sys
d=json.load(open('$CFG'))
print(json.dumps({m['model_name']: m['model_path'] for m in d}))")

EXP=${TAG}_${ARM}_ts${TS}
RESULTS=$PRISM_EXP/results/$TAG
LOGDIR=$PRISM_EXP/server-logs/$EXP
# tmux silently rewrites '.' to '_' in session names, so a name built from a
# float time_scale ("...ts0.5") never matches on has-session and the readiness
# loop reports the server as DIED one iteration in -- while it is in fact still
# starting, and leaks. Sanitise the name the same way tmux does.
SESSION=$(echo "prism-${TAG}-${ARM}-ts${TS}" | tr '.' '_')
mkdir -p "$LOGDIR" "$RESULTS" "$RESULTS/requests"

echo "### $EXP : ${NGPU} GPUs, ${NMODELS} models, arm=$ARM, time_scale=$TS, port=$PORT"
echo "###   config=$(basename "$CFG")  trace=$(basename "$TRACE")  -> results/$TAG/"
python3 -c "
import json,collections
d=json.load(open('$CFG'))
g=collections.defaultdict(list)
for m in d:
    for p in m['init_placements']:
        if p['on']: g[p['gpu_ids'][0]].append(m['model_name'])
for k in sorted(g): print(f'###   GPU {k}: {\" \".join(g[k])}')"

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

# CUDA_VISIBLE_DEVICES is pinned explicitly rather than inherited. tmux keeps ONE
# server process per user and every `new-session` inherits the environment that
# server was first started with -- so a stale CUDA_VISIBLE_DEVICES=0 from an
# earlier 1-GPU run silently follows you here and every GPU>0 worker dies with
# "RuntimeError: CUDA error: invalid device ordinal" during init_torch_distributed.
VISIBLE=$(seq -s, 0 $((NGPU - 1)))

tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" \
  "export CUDA_VISIBLE_DEVICES=$VISIBLE && cd $PRISM_REPO/benchmark/multi-model && \
   source $SCRIPT_DIR/env.sh && export CUDA_VISIBLE_DEVICES=$VISIBLE && \
   python3 -c 'import torch,sys; n=torch.cuda.device_count(); print(f\"visible GPUs: {n}\"); sys.exit(0 if n>=$NGPU else 1)' && \
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
echo " -> ready: $(curl -s http://127.0.0.1:$PORT/get_model_names)"

# sample per-GPU memory for the whole run: Figure 7b is a time series, and
# nothing in the harness records it
( while true; do
    echo "$(date +%s.%N) $(nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
        --format=csv,noheader,nounits | tr '\n' ';')"
    sleep 2
  done ) > "$LOGDIR/gpu_timeline.txt" 2>/dev/null &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null || true' EXIT

cd "$PRISM_REPO/benchmark/multi-model"
set +e
python3 benchmark.py \
  --base-url "http://127.0.0.1:$PORT" \
  --num-models "$NMODELS" --model-paths "${MODELS[@]}" \
  --exp-name "$EXP" \
  --results-path "$RESULTS" \
  --request-path "$RESULTS/requests" \
  --seed 42 --disable-tqdm \
  --e2e-benchmark --real-trace "$TRACE" \
  --time-scale "$TS" --replication 1 --num-gpus "$NGPU" \
  --enable-elastic-memory \
  --ttft-slo-scale "$TTFT_SCALE" --tpot-slo-scale "$TPOT_SCALE" \
  > "$LOGDIR/bench.log" 2>&1
RC=$?
set -e
kill $SAMPLER 2>/dev/null || true
tmux kill-session -t "$SESSION" 2>/dev/null || true
[ $RC -eq 0 ] || { echo "benchmark failed rc=$RC, tail:"; tail -40 "$LOGDIR/bench.log"; exit $RC; }

# num_gpus is stamped into the filename, hence ${NGPU}gpu and not a literal 1gpu
REQF=$(ls -t "$RESULTS"/requests/${EXP}_e2e_${NGPU}gpu_*_output_requests.json | head -1)
METF=$(ls -t "$RESULTS"/${EXP}_e2e_${NGPU}gpu_*rep.json | head -1)
python3 "$SCRIPT_DIR/analyze_slo.py" \
  --req-file "$REQF" --metrics-file "$METF" \
  --ttft-slo-scale "$TTFT_SCALE" --tpot-slo-scale "$TPOT_SCALE" \
  --label "$ARM" --model-map "$MAP" \
  --out "$RESULTS/${EXP}_slo.json" > "$LOGDIR/analyze.log" 2>&1

# What the global controller actually did -- the whole point of the ablation.
# Only the controller log carries these; it does not exist in the glob_off arm.
CL="$LOGDIR/server.log.global_controller.log"
{
  if [ -f "$CL" ]; then
    echo "activations   : $(grep -c 'ACTION: activate' "$CL")"
    echo "deactivations : $(grep -c 'ACTION: deactivate' "$CL")"
    echo "  idle evict  : $(grep -c 'Reason: idle instance eviction' "$CL")"
    echo "migrations    : $(grep -c 'Reason: migrate model' "$CL")"
    echo "migration passes that found nothing: $(grep -c 'No migrations found\|no migrations needed' "$CL")"
  else
    echo "(no global controller -- --enable-controller not passed)"
  fi
} > "$RESULTS/${EXP}_actions.txt"

echo "### $EXP done -> $RESULTS/${EXP}_slo.json"
cat "$RESULTS/${EXP}_actions.txt"
