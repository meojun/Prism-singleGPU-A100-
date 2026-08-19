#!/bin/bash
# Derive Algorithm 1's migration threshold tau on THIS machine.
#
#   ./exp/scripts/calibrate_tau_v4.sh [outdir]
#
# tau is machine-specific for the same reason c_i and the SLO baselines are:
# KVPR carries units of (tokens/s x bytes/token / SLO) / GiB, and both the
# numerator's SLO weighting and the denominator's free memory are properties of
# the box.  Carrying another machine's tau over is what this run exists to
# prevent -- the committed 0.00035 sits two orders of magnitude below this
# box's line-8 deltas, so 29% of decisions clear it and Algorithm 1 thrashes.
#
# The rule is the project's own (docs/paper_faithful/design_analysis.md 5a):
# put the threshold at mean + 2 sd of the measured improvement distribution, so
# migrations fire on real imbalance and not on sampling noise.
#
# Measured with migrations SUPPRESSED (tau huge).  Deriving tau from a run that
# already migrates is circular: those migrations change the placement, which
# changes the very deltas being measured.
set -uo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd /workspace/prism-exp
source "$SCRIPT_DIR/env.sh"

OUT=${1:-/workspace/prism-exp/exp/results/paper-faithful-v4/calibration}
WL=${V4_WORKLOAD_DIR:-/workspace/prism-exp/exp/workloads/paper-faithful-v4}
RATE=${TAU_CAL_RATE:-8}
SEED=${TAU_CAL_SEED:-1}
mkdir -p "$OUT"

export KVPR_TAU=1e9          # suppress every migration
export NGPU=2 CFG=$PRISM_EXP/configs/v2/6model_2gpu.json
export SLO_BASE_FILE=${SLO_BASE_FILE:-$PRISM_EXP/configs/v2/slo_base.json}
export PREFILL_SPEED_FILE=${PREFILL_SPEED_FILE:-$PRISM_EXP/configs/v2/prefill_speed.json}
export V2_DURATION=420 V2_WARMUP=60 V2_MEASURE=300
export CUDA_VISIBLE_DEVICES=0,1

RUN=$OUT/run
rm -rf "$RUN"; mkdir -p "$RUN"
echo "=== tau calibration: v4 arm, migrations suppressed (tau=$KVPR_TAU)"
exp/scripts/run_v4_case.sh paper-faithful-v4 bursty "$RATE" "$SEED" \
    "$WL/bursty_r${RATE}_s${SEED}.pkl" "$RUN" 2>&1 | tee "$OUT/calibration.log"

python3 "$SCRIPT_DIR/derive_tau_v4.py" --run-dir "$RUN" -o "$OUT/tau.json" \
    | tee -a "$OUT/calibration.log"
