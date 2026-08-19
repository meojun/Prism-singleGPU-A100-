#!/bin/bash
# Keeps the V5_2 measurement running and publishes it.  Resume-safe: every run
# writes DONE and is skipped if it exists.
set -uo pipefail
ulimit -n 65535 2>/dev/null || true
ROOT=/workspace/prism-exp
BASE=$ROOT/exp/results/paper-faithful-v5_2
LOGDIR=/workspace/logs
MAIN=$LOGDIR/prism_v5_2.log; STATUS=$LOGDIR/prism_v5_2.status; WD=$LOGDIR/prism_v5_2_watchdog.log
EXPECTED=${V52_EXPECTED_RUNS:-9}
mkdir -p "$LOGDIR" "$BASE"
exec 9>"$LOGDIR/prism_v5_2.lock"; flock -n 9 || exit 0
done_runs(){ find "$BASE/raw" -path '*/seed_[0-9]*/DONE' -type f 2>/dev/null | wc -l; }
complete(){ [ "$(done_runs)" -ge "$EXPECTED" ] && [ -s "$BASE/REPORT.md" ]; }

attempt=0
while ! complete; do
  attempt=$((attempt+1))
  printf 'RUNNING attempt=%d runs=%s/%s at=%s\n' "$attempt" "$(done_runs)" "$EXPECTED" "$(date -Is)" > "$STATUS"
  echo "watchdog attempt=$attempt $(date -Is)" | tee -a "$WD" "$MAIN"
  pkill -f 'launch_multi_model[_]server' 2>/dev/null || true
  for s in $(tmux ls -F '#{session_name}' 2>/dev/null | grep -E '^v4-'); do tmux kill-session -t "$s" 2>/dev/null; done
  rm -f /dev/shm/ipc_[0-9]*_root /dev/shm/cuda.shm.* 2>/dev/null || true
  sleep 5
  cd "$ROOT" || exit 1
  ./exp/run_pipeline_v5_2.sh >> "$MAIN" 2>&1
  complete && break
  [ "$attempt" -ge "${V52_MAX_ATTEMPTS:-30}" ] && { echo "GAVE_UP $(date -Is)" > "$STATUS"; break; }
  sleep 30
done
printf 'COMPLETE attempts=%d runs=%s/%s at=%s\n' "$attempt" "$(done_runs)" "$EXPECTED" "$(date -Is)" > "$STATUS"

cd "$ROOT" || exit 1
source exp/scripts/env.sh >/dev/null 2>&1
python3 exp/scripts/collect_v4_metrics.py --base "$BASE" >> "$MAIN" 2>&1 || true
python3 exp/scripts/analyze_v5_2_instr.py --base "$BASE" > "$BASE/REPORT.md" 2>>"$MAIN" || true
git add -A exp/results/paper-faithful-v5_2 exp/run_pipeline_v5_2.sh exp/scripts patches/paper_faithful_v5_2 >> "$MAIN" 2>&1
git diff --cached --quiet || git commit -q -m "results: v5_2 instrumentation -- scheduler-loop cost and deactivation hops" >> "$MAIN" 2>&1
for a in 1 2 3 4 5; do timeout 300 git push origin exp/paper-faithful-v5_2 >> "$MAIN" 2>&1 && break; sleep 60; done
echo "watchdog complete $(date -Is)" | tee -a "$WD" "$MAIN"
