#!/bin/bash
# B sweep: does KV migration change anything measurable?
#
# Runs ONLY after the paired validation has shown the mechanism engages.
# Launching this before that is how you spend two hours measuring a no-op.
#
# Design: v4 and v6 differ by exactly one flag, so the pair isolates the KV
# transfer.  Three seeds because the v4 study measured seed-to-seed spread at
# 70-80% of the mean at these rates -- one seed per cell cannot support any
# claim, and two cannot either.
#
#   SWEEP_RATES="8"      2 arms x 2 workloads x 3 seeds = 12 runs, ~2 h
#   SWEEP_RATES="8 20"   adds the saturation point       = 24 runs, ~3.7 h
#
# Resumable: a finished run leaves DONE and is skipped, so a kill or a reboot
# costs one run, not the sweep.
set -uo pipefail
R=/workspace/prism-exp
cd "$R"
set -a; . /workspace/.env; set +a
source "$R/prism-venv/bin/activate"
export PRISM_ROOT=$R PRISM_REPO=$R/prism-research PRISM_EXP=$R/exp
export HF_HOME=/workspace/.hf_home PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1

# This box's own values.  Never the committed ones -- see the v6 provenance
# note on why those are another box's numbers.
export SLO_BASE_FILE=$R/exp/results/paper-faithful-v6/profiling/slo_base_this_box.json
export PREFILL_SPEED_FILE=$R/exp/results/paper-faithful-v6/profiling/prefill_speed_this_box.json
export SLO_BASE=$SLO_BASE_FILE
export PREFILL_SPEED=$PREFILL_SPEED_FILE
# The v4 sweep's tau, so these numbers sit beside that study's.  This box's own
# 0.15992 admits under 1% of decisions -- it would very nearly switch migration
# off, leaving nothing for a KV migration to attach to.
export KVPR_TAU=${KVPR_TAU:-0.00035}
export NGPU=2 CFG=$R/exp/configs/v2/6model_2gpu.json
export V2_DURATION=420 V2_WARMUP=60 V2_MEASURE=300

WL=$R/exp/workloads/paper-faithful-v4
BASE=$R/exp/results/paper-faithful-v6/sweep
ARMS=${SWEEP_ARMS:-"paper-faithful-v4 paper-faithful-v6"}
WORKLOADS=${SWEEP_WORKLOADS:-"bursty steady"}
RATES=${SWEEP_RATES:-"8"}
SEEDS=${SWEEP_SEEDS:-"1 2 3"}

total=0; done_n=0; failed=0
for a in $ARMS; do for w in $WORKLOADS; do for r in $RATES; do for s in $SEEDS; do
    total=$((total+1))
done; done; done; done
echo "=== sweep: $total runs  $(date -Is)"

for arm in $ARMS; do
  for wl in $WORKLOADS; do
    for rate in $RATES; do
      for seed in $SEEDS; do
        # Layout matches what collect_v4_metrics.py globs
        # (base/raw/<system>/<workload>/rate_*/seed_*/DONE), so the sweep
        # produces the same summary.csv and per-request CSVs as the v4 and v5
        # studies and the numbers sit beside theirs instead of needing their
        # own definitions of goodput and SLO attainment.
        OUT=$BASE/raw/$arm/$wl/rate_$rate/seed_$seed
        if [ -f "$OUT/DONE" ]; then
            echo "skip  $arm $wl r$rate s$seed"; done_n=$((done_n+1)); continue
        fi
        trace=$WL/${wl}_r${rate}_s${seed}.pkl
        if [ ! -f "$trace" ]; then
            echo "MISSING TRACE $trace -- build it first"; failed=$((failed+1)); continue
        fi
        rm -rf "$OUT"; mkdir -p "$OUT"
        echo "=== run $((done_n+failed+1))/$total : $arm $wl r$rate s$seed  $(date -Is)"
        ./exp/scripts/run_v4_case.sh "$arm" "$wl" "$rate" "$seed" "$trace" "$OUT" \
            > "$OUT/run.log" 2>&1
        rc=$?
        if [ "$rc" = 0 ]; then touch "$OUT/DONE"; done_n=$((done_n+1));
        else failed=$((failed+1)); fi
        echo "--- $arm $wl r$rate s$seed exit=$rc  $(date -Is)"
        # A stale /dev/shm segment from the run that just ended kills the next
        # server during startup -- measured, not theoretical: the v6 validation
        # died that way after the v4 control finished cleanly beside it.
        /workspace/shm_clean.sh
        sleep 15
      done
    done
  done
done

echo "=== sweep finished: $done_n ok, $failed failed  $(date -Is)"

# Aggregate with the project's own collector, so goodput, SLO attainment and
# the measurement window are the same definitions every earlier study used --
# including the 60 s warmup exclusion, which the raw harness dump cannot express
# because it carries no arrival times.
mkdir -p "$BASE"/raw/{requests,migrations,scheduler,gpu_metrics} "$BASE/logs"
python3 exp/scripts/collect_v4_metrics.py --base "$BASE" \
    --slo-base "$SLO_BASE_FILE" --config "$CFG" --trace-dir "$WL" \
    --warmup 60 --measure 300 > "$BASE/logs/aggregate.log" 2>&1 \
    || echo "WARN: collector failed, see $BASE/logs/aggregate.log"

SUM=$BASE/SUMMARY.txt
python3 "$R/exp/scripts/summarize_v6_sweep.py" --base "$BASE" 2>&1 | tee "$SUM"

# Publish without waiting for a human.  /workspace is not a volume on this
# instance -- a recycle takes the whole sweep with it -- and whoever launched
# this may be long gone by the time it lands.  Results that only exist here are
# results that can vanish.
{
  echo "# v6 sweep -- $(date -Is)"
  echo
  echo "Ran: arms [$ARMS] x workloads [$WORKLOADS] x rates [$RATES] x seeds [$SEEDS]"
  echo "Completed $done_n of $total; $failed failed."
  echo "tau=$KVPR_TAU (the v4 sweep's, not this box's derived 0.15992 -- that one"
  echo "admits under 1% of decisions and would leave nothing to migrate)."
  echo
  echo '```'
  cat "$SUM"
  echo '```'
} > "$BASE/REPORT.md"

cd "$R"
git add -A exp/results/paper-faithful-v6/sweep exp/scripts/summarize_v6_sweep.py 2>/dev/null
if ! git diff --cached --quiet 2>/dev/null; then
    git commit -q -m "v6 sweep: $done_n/$total runs, $failed failed

Automated commit from stage_b_sweep.sh.  Read sweep/SUMMARY.txt for whether the
mechanism engaged before reading any goodput number: the summariser refuses to
compare arms in a cell where no KV moved, because there the two arms are the
same system.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>" \
      && echo "committed"
    git push origin exp/paper-faithful-v6 2>&1 | sed -E 's/github_pat_[A-Za-z0-9_]+/<redacted>/g' | tail -2
else
    echo "nothing to commit"
fi
echo "STAGE_B_SWEEP_DONE $(date -Is)"
