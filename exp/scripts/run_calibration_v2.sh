#!/bin/bash
# Locate this box's Low / Medium / High / Near-saturation load levels.
#
#   ./exp/scripts/run_calibration_v2.sh [rates...]
#
# Short STEADY runs of the released prototype only.  Steady is the right probe:
# it has no bursts, so the knee it finds is the cluster's sustained capacity for
# this model set rather than an artefact of one phase.  Both arms and both
# workloads then share the same rate ladder.
set -uo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/env.sh"

RATES=("$@"); [ ${#RATES[@]} -eq 0 ] && RATES=(1 2 3 5 8 10)
DUR=${CAL_DURATION:-200}
WARM=${CAL_WARMUP:-40}
MEAS=${CAL_MEASURE:-140}
SEED=${CAL_SEED:-91}

BASE=$PRISM_EXP/results/paper-faithful-v2/sanity/calibration
WL=$PRISM_EXP/workloads/paper-faithful-v2/calibration
export SLO_BASE_FILE=${SLO_BASE_FILE:-$PRISM_EXP/configs/v2/slo_base.json}
export PREFILL_SPEED_FILE=${PREFILL_SPEED_FILE:-$PRISM_EXP/configs/v2/prefill_speed.json}
export CFG=${CFG:-$PRISM_EXP/configs/v2/6model_2gpu.json}
export NGPU=2
mkdir -p "$BASE" "$WL"

for rate in "${RATES[@]}"; do
  tag="r${rate}_s${SEED}"
  [ -f "$WL/steady_${tag}.pkl" ] || \
    python3 "$SCRIPT_DIR/build_paired_workload.py" --rate "$rate" --duration "$DUR" \
      --seed "$SEED" --slo-base "$SLO_BASE_FILE" --outdir "$WL" || exit 1
done

for rate in "${RATES[@]}"; do
  d="$BASE/rate_$rate"
  [ -f "$d/DONE" ] && { echo "skip rate=$rate"; continue; }
  mkdir -p "$d"
  if "$SCRIPT_DIR/run_v2_case.sh" released-prototype steady "$rate" "$SEED" \
        "$WL/steady_r${rate}_s${SEED}.pkl" "$d" 2>&1 | tee "$d/run.log"; then
    python3 "$SCRIPT_DIR/collect_v2_metrics.py" --run-dir "$d" \
      --slo-base "$SLO_BASE_FILE" --trace "$WL/steady_r${rate}_s${SEED}.pkl" \
      --warmup "$WARM" --measure "$MEAS" \
      --label "calib/$rate" -o "$d/metrics.json" && touch "$d/DONE"
  else
    echo "!! calibration rate=$rate FAILED"
  fi
  pkill -f "launch_multi_model[_]server" 2>/dev/null || true
  sleep 10
done

echo
echo "=== calibration summary ==="
python3 - "$BASE" <<'PY'
import glob, json, os, sys
base = sys.argv[1]
rows = []
for p in sorted(glob.glob(os.path.join(base, "rate_*", "metrics.json")),
                key=lambda p: float(os.path.basename(os.path.dirname(p)).split("_")[1])):
    d = json.load(open(p))
    rows.append((float(os.path.basename(os.path.dirname(p)).split("_")[1]), d))
hdr = f"{'rate':>6s} {'offered':>8s} {'thruput':>8s} {'TTFTp50':>9s} {'TTFTp99':>10s} {'TPOTp50':>8s} {'joint':>6s} {'good':>6s} {'maxQ':>6s} {'util0':>6s} {'util1':>6s}"
print(hdr); print("-"*len(hdr))
for r, d in rows:
    u = d.get("gpu_util_mean", {})
    print(f"{r:6.1f} {d['offered_load_req_s']:8.2f} {d['throughput_req_s']:8.2f} "
          f"{d['ttft_p50_ms']:9.1f} {d['ttft_p99_ms']:10.1f} {d['tpot_p50_ms']:8.1f} "
          f"{d['joint_attainment']:6.3f} {d['goodput_req_s']:6.2f} "
          f"{d.get('max_queue_length',0):6.0f} {u.get('0',float('nan')):6.1f} {u.get('1',float('nan')):6.1f}")
print()
print("pick: Low = highest rate with joint>0.95 | Medium = joint ~0.7-0.9 |")
print("      High = joint ~0.3-0.6 | Near-saturation = throughput stops tracking offered load")
PY
