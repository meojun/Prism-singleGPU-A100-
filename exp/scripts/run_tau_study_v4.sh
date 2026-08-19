#!/bin/bash
# Follow-on: does a tau calibrated on THIS machine remove the loss it causes?
#
# The main sweep runs tau=0.00035, inherited from the v3 report's machine.
# Measured here, the line-8 deltas have mean 0.0153 and sd 0.0329, so that
# threshold clears on 29% of decisions and Algorithm 1 migrates 13 times per
# 300 s window.  At 8 req/s that costs more than the placement gains; at
# 20 req/s the paper-faithful arms win by a wide margin anyway.
#
# Algorithm 1 already has the knob for this -- line 8 is exactly "do not move
# unless the gain exceeds tau".  Leaving it mis-set and then reporting that
# migration is a net loss would attribute a calibration mistake to the
# algorithm.  So: derive tau here by the project's own rule, then re-run the
# two operating points that matter.
#
#   8 req/s  x 3 seeds   -- does correct tau remove the loss?
#   20 req/s x 1 seed    -- does it keep the gain?
#
# Waits for the main sweep so the two never share the GPUs.
set -uo pipefail
ROOT=/workspace/prism-exp
BASE=$ROOT/exp/results/paper-faithful-v4
OUT=$BASE/tau-study
LOG=/workspace/logs/prism_v4_tau.log
EXPECTED=${V4_EXPECTED_RUNS:-27}
WL=$ROOT/exp/workloads/paper-faithful-v4

exec 9>/workspace/logs/prism_v4_tau.lock
flock -n 9 || exit 0
say() { echo "[$(date -Is)] $*" >> "$LOG"; }
done_runs() { find "$BASE/raw" -path '*/seed_[0-9]*/DONE' -type f 2>/dev/null | wc -l; }

say "tau study waiting for the main sweep ($EXPECTED runs)"
while [ "$(done_runs)" -lt "$EXPECTED" ]; do
    pgrep -f "watchdog_v4.sh" >/dev/null 2>&1 || {
        say "watchdog gone at $(done_runs)/$EXPECTED; proceeding anyway"; break; }
    sleep 120
done
# Let the main pipeline finish its aggregation before taking the GPUs.
while pgrep -f "run_pipeline_v4.sh" >/dev/null 2>&1; do sleep 60; done

cd "$ROOT" || exit 1
source exp/scripts/env.sh >/dev/null 2>&1
mkdir -p "$OUT"
export NGPU=2 CFG=$PRISM_EXP/configs/v2/6model_2gpu.json
export SLO_BASE_FILE=$PRISM_EXP/configs/v2/slo_base.json
export PREFILL_SPEED_FILE=$PRISM_EXP/configs/v2/prefill_speed.json
export V2_DURATION=420 V2_WARMUP=60 V2_MEASURE=300
export CUDA_VISIBLE_DEVICES=0,1

# 1. Calibrate, with migrations suppressed so the estimate is not measured
#    through the very migrations it is meant to govern.
if [ ! -s "$OUT/tau.json" ]; then
    say "calibrating tau"
    ./exp/scripts/calibrate_tau_v4.sh "$OUT/calibration" >> "$LOG" 2>&1
    cp "$OUT/calibration/tau.json" "$OUT/tau.json" 2>/dev/null
fi
TAU=$(python3 -c "import json;print(json.load(open('$OUT/tau.json'))['tau'])" 2>/dev/null)
[ -z "$TAU" ] && { say "FATAL: no tau derived"; exit 1; }
say "tau=$TAU"

run() {  # run <system> <workload> <rate> <seed>
    local sys=$1 wl=$2 rate=$3 seed=$4
    local out="$OUT/raw/$sys/$wl/rate_$rate/seed_$seed"
    [ -f "$out/DONE" ] && { say "skip $sys/$wl/$rate/$seed"; return 0; }
    [ -d "$out" ] && mv "$out" "${out}_failed_$(date +%s)"
    mkdir -p "$out"
    say "running $sys $wl $rate seed $seed at tau=$TAU"
    KVPR_TAU=$TAU ./exp/scripts/run_v4_case.sh "$sys" "$wl" "$rate" "$seed" \
        "$WL/${wl}_r${rate}_s${seed}.pkl" "$out" >> "$LOG" 2>&1 || { say "FAILED $sys/$wl/$rate/$seed"; return 1; }
    python3 exp/scripts/collect_v2_metrics.py --run-dir "$out" \
        --slo-base "$SLO_BASE_FILE" --trace "$WL/${wl}_r${rate}_s${seed}.pkl" \
        --ttft-scale 5 --tpot-scale 3 --warmup 60 --measure 300 \
        --label "tau/$sys/$wl/$rate/$seed" -o "$out/metrics.json" >> "$LOG" 2>&1 \
        && touch "$out/DONE"
    pkill -f 'launch_multi_model[_]server' 2>/dev/null || true
    rm -f /dev/shm/ipc_[0-9]*_root /dev/shm/cuda.shm.* 2>/dev/null || true
    sleep 10
}

# steady first: hot sets do not move there, so a migration is hardest to
# justify, and the main sweep still recorded 13 of them against the
# prototype's 0 -- the clearest place to see whether a correct tau declines
# them.  Then bursty, where migration can genuinely pay.
for seed in 1 2 3; do
    run paper-faithful-v4 steady 8 "$seed"
done
for seed in 1 2 3; do
    run paper-faithful-v4 bursty 8 "$seed"
done
run paper-faithful-v4 bursty 20 1

python3 exp/scripts/collect_v4_metrics.py --base "$OUT" >> "$LOG" 2>&1 || true
git add -A "$OUT" >> "$LOG" 2>&1
git diff --cached --quiet || git commit -q -m "results: tau re-calibrated on this machine

Algorithm 1's line 8 is the rule that declines an unprofitable migration, and
the main sweep runs it with a threshold inherited from another machine that
clears on 29% of decisions. Reporting migration as a net loss under that
setting would blame the algorithm for a calibration mistake. tau re-derived
here with migrations suppressed, then the two operating points re-run." >> "$LOG" 2>&1
for a in 1 2 3; do timeout 300 git push origin exp/paper-faithful-v4 >> "$LOG" 2>&1 && break; sleep 60; done
say "tau study complete"
