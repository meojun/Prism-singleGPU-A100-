#!/bin/bash
# Wait for the v4 study to finish, then commit and push its results.
#
# Runs as its own supervisor program rather than as a step inside the pipeline:
# the pipeline was already running when this was written, and bash reads a
# script by byte offset, so editing a live one truncates the next command.
#
# Idempotent and safe to restart: it only ever adds result files, and it exits
# once the push has landed.
set -uo pipefail
ROOT=/workspace/prism-exp
BASE=$ROOT/exp/results/paper-faithful-v4
LOG=/workspace/logs/prism_v4_finalize.log
EXPECTED=${V4_EXPECTED_RUNS:-27}
BRANCH=exp/paper-faithful-v4

exec 9>/workspace/logs/prism_v4_finalize.lock
flock -n 9 || exit 0

say() { echo "[$(date -Is)] $*" >> "$LOG"; }
done_runs() { find "$BASE/raw" -path '*/seed_[0-9]*/DONE' -type f 2>/dev/null | wc -l; }

say "finalizer waiting for $EXPECTED runs"
while :; do
    n=$(done_runs)
    if [ "$n" -ge "$EXPECTED" ] && [ -s "$BASE/REPORT.md" ] && [ -s "$BASE/summary.csv" ]; then
        break
    fi
    # The watchdog may have given up; if it is gone and nothing is running,
    # publish whatever completed rather than waiting for a run that will not come.
    if ! pgrep -f "watchdog_v4.sh" >/dev/null 2>&1 \
       && ! pgrep -f "run_pipeline_v4.sh" >/dev/null 2>&1; then
        say "watchdog and pipeline both gone at $n/$EXPECTED runs; publishing partial results"
        break
    fi
    sleep 120
done

cd "$ROOT" || exit 1
say "regenerating aggregates and report from $(done_runs)/$EXPECTED runs"
source exp/scripts/env.sh >/dev/null 2>&1
python3 exp/scripts/collect_v4_metrics.py --base "$BASE" >> "$LOG" 2>&1
python3 exp/scripts/build_report_v4.py  --base "$BASE" >> "$LOG" 2>&1

git add -A exp/results/paper-faithful-v4 exp/workloads/paper-faithful-v4 2>>"$LOG"
if git diff --cached --quiet; then
    say "nothing to commit"
else
    git commit -q -m "results: paper-faithful v4 study ($(done_runs))/$EXPECTED runs

Committed by exp/scripts/finalize_v4.sh once the sweep completed. Raw
per-request, per-migration, per-scheduler-cycle and per-GPU-sample tables are
under raw/; summary.csv and aggregated/ are derived from them, not the other
way round." >> "$LOG" 2>&1
    say "committed"
fi

for attempt in 1 2 3 4 5; do
    if timeout 300 git push origin "$BRANCH" >> "$LOG" 2>&1; then
        say "pushed to $BRANCH"; exit 0
    fi
    say "push attempt $attempt failed; retrying"
    sleep 60
done
say "PUSH FAILED after 5 attempts -- results are committed locally"
exit 1
