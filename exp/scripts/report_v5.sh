#!/bin/bash
# Wait for the V5 sweep, then write the ablation comparison and publish it.
# Separate supervisor program because the pipeline and watchdog are already
# running, and bash reads a script by byte offset -- editing a live one
# truncates its next command.
set -uo pipefail
ROOT=/workspace/prism-exp
V5=$ROOT/exp/results/paper-faithful-v5
V4=$ROOT/exp/results/paper-faithful-v4
LOG=/workspace/logs/prism_v5_report.log
EXPECTED=${V5_EXPECTED_RUNS:-15}
exec 9>/workspace/logs/prism_v5_report.lock
flock -n 9 || exit 0
say() { echo "[$(date -Is)] $*" >> "$LOG"; }
done_runs() { find "$V5/raw" -path '*/seed_[0-9]*/DONE' -type f 2>/dev/null | wc -l; }

say "report waiting for $EXPECTED runs"
while [ "$(done_runs)" -lt "$EXPECTED" ]; do
    pgrep -f "watchdog_v5.sh" >/dev/null 2>&1 || { say "watchdog gone at $(done_runs)"; break; }
    sleep 120
done
while pgrep -f "run_pipeline_v5.sh" >/dev/null 2>&1; do sleep 60; done

cd "$ROOT" || exit 1
source exp/scripts/env.sh >/dev/null 2>&1
python3 exp/scripts/collect_v4_metrics.py --base "$V5" >> "$LOG" 2>&1 || true
python3 exp/scripts/summarize_v5.py --v5 "$V5" --v4 "$V4" > "$V5/REPORT.md" 2>>"$LOG"
git add -A exp/results/paper-faithful-v5 exp/scripts >> "$LOG" 2>&1
git diff --cached --quiet || git commit -q -m "results: v5 ablation + P2P re-validation, with the report" >> "$LOG" 2>&1
for a in 1 2 3 4 5; do
    timeout 300 git push origin exp/paper-faithful-v5 >> "$LOG" 2>&1 && { say "pushed"; break; }
    sleep 60
done
say "report done"
