#!/bin/bash
set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/env.sh"

OUT=${1:-$PRISM_EXP/results/final-regression-diagnosis/profiling}
mkdir -p "$OUT/configs" "$OUT/logs"
STATUS_DIR=$(dirname "$OUT")
trap 'rc=$?; printf "%s\n" "$rc" > "$STATUS_DIR/profile_pair.rc"' EXIT
echo "profile pair started: $(date -u +%FT%TZ)"

launch_and_profile() {
  local slot=$1 model_path=$2 gpu=$3 port=$4 max_pool=$5 sat_concurrency=$6
  local cfg="$OUT/configs/solo_${slot}.json"
  local log="$OUT/logs/${slot}"

  if [ -s "$OUT/${slot}.json" ]; then
    echo "$slot already complete; preserving existing profile"
    return 0
  fi

  python3 - "$cfg" "$slot" "$model_path" "$max_pool" <<'PY'
import json, sys
cfg, slot, path, max_pool = sys.argv[1:]
json.dump([{"model_name": slot, "model_path": path, "tp_size": 1,
            "init_placements": [{"gpu_ids": [0], "on": True,
                                  "max_memory_pool_size": float(max_pool)}]}],
          open(cfg, "w"), indent=2)
PY

  fuser -k -n tcp "$port" 2>/dev/null || true
  tmux kill-session -t "diag-prof-$slot" 2>/dev/null || true
  tmux new-session -d -s "diag-prof-$slot" \
    "export CUDA_VISIBLE_DEVICES=$gpu; cd '$PRISM_REPO/benchmark/multi-model'; source '$SCRIPT_DIR/env.sh'; export CUDA_VISIBLE_DEVICES=$gpu; python3 -m sglang.launch_multi_model_server --model-config-file '$cfg' --host 127.0.0.1 --port '$port' --disable-cuda-graph --disable-radix-cache --log-file '${log}.log' > '${log}_stdout.log' 2>&1"

  local ready=0
  for _ in $(seq 1 300); do
    if curl -sf "http://127.0.0.1:$port/get_model_names" >/dev/null 2>&1; then ready=1; break; fi
    tmux has-session -t "diag-prof-$slot" 2>/dev/null || break
    sleep 2
  done
  if [ "$ready" -ne 1 ]; then
    echo "$slot server failed to become ready" >&2
    tail -80 "${log}_stdout.log" >&2
    return 1
  fi

  python3 "$SCRIPT_DIR/profile_v2.py" --url "http://127.0.0.1:$port" \
    --model "$slot" --model-path "$model_path" --per-bucket 40 \
    --sat-concurrency "$sat_concurrency" --sat-rounds 6 -o "$OUT/${slot}.json" \
    > "$OUT/${slot}_profile.log" 2>&1
  local rc=$?
  tmux kill-session -t "diag-prof-$slot" 2>/dev/null || true
  fuser -k -n tcp "$port" 2>/dev/null || true
  return "$rc"
}

launch_and_profile model_1 meta-llama/Llama-3.2-1B 0 35100 40 48 &
p1=$!

profile_model_6() {
  local concurrency
  for concurrency in 8 4; do
    echo "model_6 profiling attempt with saturated concurrency=$concurrency"
    if launch_and_profile model_6 Qwen/Qwen2.5-7B-Instruct 1 35200 40 "$concurrency"; then
      return 0
    fi
    echo "model_6 attempt failed at concurrency=$concurrency; retrying"
  done
  return 1
}

profile_model_6 &
p6=$!

wait "$p1"; r1=$?
wait "$p6"; r6=$?
printf 'model_1_rc=%s\nmodel_6_rc=%s\n' "$r1" "$r6"
test "$r1" -eq 0 -a "$r6" -eq 0
