#!/bin/bash
# No-contention profiling for the heterogeneous 6-model set.
#
#   ./exp/scripts/run_profiling_v2.sh [outdir]
#
# For every model: launch it SOLO on a dedicated GPU (static mode, no other
# tenant, no radix cache) and run exp/scripts/profile_v2.py against it.
# Two models are profiled at a time, one per GPU.  Results land as
# <outdir>/<slot>.json and are folded into
#   exp/configs/v2/slo_base.json        (paper Sec. 7.1 TTFT/TPOT p95)
#   exp/configs/v2/prefill_speed.json   (Algorithm 2's c_i)
set -uo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/env.sh"
OUT=${1:-$PRISM_EXP/results/paper-faithful-v2/sanity/profiling}
mkdir -p "$OUT" "$PRISM_EXP/configs/v2" "$PRISM_EXP/server-logs/profiling"

MODELS=(model_1 model_2 model_3 model_4 model_5 model_6)
declare -A PATHS=(
  [model_1]=meta-llama/Llama-3.2-1B
  [model_2]=Qwen/Qwen2.5-1.5B-Instruct
  [model_3]=meta-llama/Llama-3.2-3B
  [model_4]=Qwen/Qwen2.5-3B-Instruct
  [model_5]=meta-llama/Llama-3.1-8B
  [model_6]=Qwen/Qwen2.5-7B-Instruct
)

launch_solo() {   # launch_solo <slot> <gpu> <port>
  local slot=$1 gpu=$2 port=$3
  local cfg=$PRISM_EXP/configs/v2/solo_${slot}.json
  python3 - "$cfg" "$slot" "${PATHS[$slot]}" <<'PY'
import json, sys
cfg, slot, path = sys.argv[1], sys.argv[2], sys.argv[3]
json.dump([{ "model_name": slot, "model_path": path, "tp_size": 1,
             "init_placements": [{"gpu_ids":[0], "on": True,
                                  "max_memory_pool_size": 40.0}] }],
          open(cfg,"w"), indent=2)
PY
  local log=$PRISM_EXP/server-logs/profiling/${slot}
  tmux kill-session -t "prof-$slot" 2>/dev/null || true
  # A previous attempt can leave engine subprocesses alive holding both the port
  # and GPU memory even after its HTTP app has shut down; the new server then
  # bind-fails and dies while the readiness loop waits out its full timeout.
  # Free the port hard and confirm it is free before launching.
  fuser -k -n tcp "$port" 2>/dev/null || true
  sleep 3
  for _ in $(seq 1 10); do
    ss -ltn 2>/dev/null | grep -q ":$port " || break
    fuser -k -n tcp "$port" 2>/dev/null || true; sleep 2
  done
  tmux new-session -d -s "prof-$slot" \
    "export CUDA_VISIBLE_DEVICES=$gpu && cd $PRISM_REPO/benchmark/multi-model && \
     source $SCRIPT_DIR/env.sh && export CUDA_VISIBLE_DEVICES=$gpu && \
     python3 -m sglang.launch_multi_model_server --model-config-file $cfg \
       --host 127.0.0.1 --port $port --disable-cuda-graph --disable-radix-cache \
       --log-file ${log}.log 2>&1 | tee ${log}_stdout.log"
  for _ in $(seq 1 300); do
    curl -sf "http://127.0.0.1:$port/get_model_names" >/dev/null 2>&1 && return 0
    grep -q "address already in use" "${log}_stdout.log" 2>/dev/null && {
      echo "  $slot BIND CONFLICT on $port"; return 1; }
    tmux has-session -t "prof-$slot" 2>/dev/null || { echo "  $slot DIED (see ${log}_stdout.log)"; return 1; }
    sleep 2
  done
  echo "  $slot TIMEOUT"; return 1
}

profile_one() {   # profile_one <slot> <gpu> <port>
  local slot=$1 gpu=$2 port=$3
  if [ -s "$OUT/$slot.json" ]; then echo "--- $slot already profiled, skipping"; return 0; fi
  echo "--- $slot on GPU$gpu port $port  (${PATHS[$slot]})"
  if launch_solo "$slot" "$gpu" "$port"; then
    python3 "$SCRIPT_DIR/profile_v2.py" --url "http://127.0.0.1:$port" \
      --model "$slot" --model-path "${PATHS[$slot]}" -o "$OUT/$slot.json" \
      2>&1 | stdbuf -oL sed "s/^/  /"
  fi
  tmux kill-session -t "prof-$slot" 2>/dev/null || true
  pkill -f "launch_multi_model_server.*--port $port" 2>/dev/null || true
  sleep 8
}

# Ports are spaced 100 apart, not 1: launch_multi_model_server assigns each model
# engine a port derived from --port (port+1, port+2, ...), so adjacent --port
# values collide -- the second server bind-fails and dies while its GPU sits idle.
for ((i=0; i<${#MODELS[@]}; i+=2)); do
  A=${MODELS[$i]}; B=${MODELS[$((i+1))]:-}
  profile_one "$A" 0 $((31000+i*100)) &
  PA=$!
  if [ -n "$B" ]; then profile_one "$B" 1 $((31050+i*100)) & PB=$!; else PB=; fi
  wait $PA; [ -n "$PB" ] && wait $PB
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
done

echo "=== folding into configs"
python3 - "$OUT" "$PRISM_EXP/configs/v2" <<'PY'
import glob, json, os, sys
out, cfgdir = sys.argv[1], sys.argv[2]
slo, speed, summary = {}, {}, []
for f in sorted(glob.glob(os.path.join(out, "model_*.json"))):
    d = json.load(open(f))
    m = d["model"]
    slo[m] = {"ttft": d["slo_baseline"]["ttft_p95_s"],
              "tpot": d["slo_baseline"]["tpot_p95_s"]}
    e = d["c_i_estimators"]
    # E3-saturated is the paper's definition: the engine's aggregate
    # chunked-prefill token throughput for this model.  Fall back down the
    # chain only if a measurement is missing.
    speed[m] = e["E3_prefill_saturated"] or e["E3_prefill_solo"] or e["E1_ratio_sum_p_over_sum_ttft"]
    summary.append((m, d["model_path"], e, d["slo_baseline"]))
json.dump(slo, open(os.path.join(cfgdir, "slo_base.json"), "w"), indent=2)
json.dump(speed, open(os.path.join(cfgdir, "prefill_speed.json"), "w"), indent=2)
json.dump({m: e for m, _, e, _ in summary},
          open(os.path.join(cfgdir, "prefill_speed_estimators.json"), "w"), indent=2)
print(f"{'slot':9s} {'path':32s} {'E1':>9s} {'E2':>9s} {'E3solo':>9s} {'E3sat':>9s} "
      f"{'TTFTp95ms':>10s} {'TPOTp95ms':>10s}")
for m, p, e, s in summary:
    f = lambda v: f"{v:9.0f}" if v else f"{'-':>9s}"
    print(f"{m:9s} {p:32s} {f(e['E1_ratio_sum_p_over_sum_ttft'])} "
          f"{f(e['E2_regression_slope'])} {f(e['E3_prefill_solo'])} "
          f"{f(e['E3_prefill_saturated'])} {1000*s['ttft_p95_s']:10.1f} {1000*s['tpot_p95_s']:10.2f}")
PY
echo "=== profiling done"
