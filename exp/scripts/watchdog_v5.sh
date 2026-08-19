#!/bin/bash
# Keeps the V5 study running until it is finished, and publishes it.
#
# Every stage and run writes a DONE marker and is skipped if it exists, so
# restarting the pipeline always resumes at the first incomplete item and never
# repeats work that succeeded.
#
# Lives under supervisor: on this box tmux sessions and setsid-detached
# processes have been killed mid-sweep, while supervisord has stayed up for
# hours (CLAUDE.md section 8).
set -uo pipefail
ulimit -n 65535 2>/dev/null || true

ROOT=/workspace/prism-exp
BASE=$ROOT/exp/results/paper-faithful-v5
LOGDIR=/workspace/logs
MAIN_LOG=$LOGDIR/prism_v5.log
STATUS=$LOGDIR/prism_v5.status
WD_LOG=$LOGDIR/prism_v5_watchdog.log
EXPECTED=${V5_EXPECTED_RUNS:-15}
BRANCH=exp/paper-faithful-v5
mkdir -p "$LOGDIR" "$BASE"

exec 9>"$LOGDIR/prism_v5.lock"
flock -n 9 || { echo "watchdog already running $(date -Is)" >> "$WD_LOG"; exit 0; }

done_runs() { find "$BASE/raw" -path '*/seed_[0-9]*/DONE' -type f 2>/dev/null | wc -l; }
complete()  { [ "$(done_runs)" -ge "$EXPECTED" ] && [ -s "$BASE/summary.csv" ]; }

attempt=0
while ! complete; do
  attempt=$((attempt + 1))
  printf 'RUNNING attempt=%d runs=%s/%s started=%s\n' \
    "$attempt" "$(done_runs)" "$EXPECTED" "$(date -Is)" > "$STATUS"
  echo "watchdog attempt=$attempt start $(date -Is)" | tee -a "$WD_LOG" "$MAIN_LOG"

  # A dead parent can leave a run's server tmux session alive holding both
  # GPUs; clear that orphan before resuming or the next run fails for a reason
  # that has nothing to do with it.
  pkill -f 'launch_multi_model[_]server' 2>/dev/null || true
  for s in $(tmux ls -F '#{session_name}' 2>/dev/null | grep -E '^v4-'); do
      tmux kill-session -t "$s" 2>/dev/null || true; done
  rm -f /dev/shm/ipc_[0-9]*_root /dev/shm/cuda.shm.* 2>/dev/null || true
  sleep 5

  cd "$ROOT" || exit 1
  ./exp/run_pipeline_v5.sh >> "$MAIN_LOG" 2>&1
  rc=$?
  echo "watchdog attempt=$attempt exit=$rc runs=$(done_runs)/$EXPECTED $(date -Is)" \
    | tee -a "$WD_LOG" "$MAIN_LOG"
  complete && break

  if [ "$attempt" -ge "${V5_MAX_ATTEMPTS:-40}" ]; then
    printf 'GAVE_UP attempts=%d runs=%s/%s at=%s\n' \
      "$attempt" "$(done_runs)" "$EXPECTED" "$(date -Is)" > "$STATUS"
    break
  fi
  printf 'RESTARTING attempt=%d last_rc=%d runs=%s/%s at=%s\n' \
    "$attempt" "$rc" "$(done_runs)" "$EXPECTED" "$(date -Is)" > "$STATUS"
  sleep 30
done

printf 'COMPLETE attempts=%d runs=%s/%s finished=%s\n' \
  "$attempt" "$(done_runs)" "$EXPECTED" "$(date -Is)" > "$STATUS"

# ---- publish, with retries; results are worthless if the box is recycled
cd "$ROOT" || exit 1
source exp/scripts/env.sh >/dev/null 2>&1
python3 exp/scripts/collect_v4_metrics.py --base "$BASE" >> "$MAIN_LOG" 2>&1 || true
git add -A exp/results/paper-faithful-v5 exp/run_pipeline_v5.sh exp/scripts >> "$MAIN_LOG" 2>&1
git diff --cached --quiet || git commit -q -m "results: v5 ablation and P2P re-validation ($(done_runs))/$EXPECTED runs

Committed by exp/scripts/watchdog_v5.sh when the sweep finished." >> "$MAIN_LOG" 2>&1
for a in 1 2 3 4 5; do
    timeout 300 git push origin "$BRANCH" >> "$MAIN_LOG" 2>&1 && { echo "pushed" >> "$WD_LOG"; break; }
    sleep 60
done
echo "watchdog complete attempts=$attempt $(date -Is)" | tee -a "$WD_LOG" "$MAIN_LOG"
