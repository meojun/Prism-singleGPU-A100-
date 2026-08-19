#!/bin/bash
# Keeps the paper-faithful-v4 study running until it is actually finished.
#
# The pipeline marks each stage and each run DONE as it succeeds and skips
# anything already marked, so restarting it is always safe and always resumes
# at the first incomplete item -- never re-running work that succeeded.
#
# This lives under supervisor rather than in a bare tmux session. On this box
# tmux sessions and even setsid-detached processes have been killed mid-sweep,
# while supervisord has stayed up for hours (CLAUDE.md section 8); supervisor's
# autorestart is what makes "it cannot be interrupted" true rather than hoped.
# The model servers themselves still run in tmux, one session per run.
set -uo pipefail

ROOT=/workspace/prism-exp
BASE=$ROOT/exp/results/paper-faithful-v4
LOGDIR=/workspace/logs
MAIN_LOG=$LOGDIR/prism_v4.log
STATUS=$LOGDIR/prism_v4.status
WATCHDOG_LOG=$LOGDIR/prism_v4_watchdog.log
EXPECTED_RUNS=${V4_EXPECTED_RUNS:-27}
mkdir -p "$LOGDIR" "$BASE"

exec 9>"$LOGDIR/prism_v4.lock"
flock -n 9 || { echo "watchdog already running $(date -Is)" >> "$WATCHDOG_LOG"; exit 0; }

done_runs() { find "$BASE/raw" -path '*/seed_[0-9]*/DONE' -type f 2>/dev/null | wc -l; }

complete() {
  [ "$(done_runs)" -ge "$EXPECTED_RUNS" ] || return 1
  [ -s "$BASE/microbench/loading.json" ]   || return 1
  [ -s "$BASE/microbench/migration.json" ] || return 1
  [ -s "$BASE/tp-validation/tp2_validation.json" ] || return 1
  [ -s "$BASE/summary.csv" ] && [ -s "$BASE/REPORT.md" ] \
    && [ -s "$BASE/IMPLEMENTATION_AUDIT.md" ]
}

attempt=0
while ! complete; do
  attempt=$((attempt + 1))
  printf 'RUNNING attempt=%d runs=%s/%s started=%s\n' \
    "$attempt" "$(done_runs)" "$EXPECTED_RUNS" "$(date -Is)" > "$STATUS"
  echo "watchdog attempt=$attempt start $(date -Is)" | tee -a "$WATCHDOG_LOG" "$MAIN_LOG"

  # A dead parent can leave a run's nested server tmux session alive, still
  # holding both GPUs. Clear that orphan before resuming, or the next run
  # contends with it and fails for a reason that has nothing to do with it.
  pkill -f 'launch_multi_model[_]server' 2>/dev/null || true
  rm -f /dev/shm/ipc_[0-9]*_root /dev/shm/cuda.shm.* /dev/shm/ipc_0_model_*_root 2>/dev/null || true
  sleep 5

  cd "$ROOT" || exit 1
  ./exp/run_pipeline_v4.sh >> "$MAIN_LOG" 2>&1
  rc=$?
  echo "watchdog attempt=$attempt exit=$rc runs=$(done_runs)/$EXPECTED_RUNS $(date -Is)" \
    | tee -a "$WATCHDOG_LOG" "$MAIN_LOG"

  complete && break

  if [ "$attempt" -ge "${V4_MAX_ATTEMPTS:-40}" ]; then
    printf 'GAVE_UP attempts=%d runs=%s/%s at=%s\n' \
      "$attempt" "$(done_runs)" "$EXPECTED_RUNS" "$(date -Is)" > "$STATUS"
    echo "watchdog giving up after $attempt attempts" | tee -a "$WATCHDOG_LOG" "$MAIN_LOG"
    exit 1
  fi

  pkill -f 'launch_multi_model[_]server' 2>/dev/null || true
  rm -f /dev/shm/ipc_[0-9]*_root /dev/shm/cuda.shm.* /dev/shm/ipc_0_model_*_root 2>/dev/null || true
  printf 'RESTARTING attempt=%d last_rc=%d runs=%s/%s at=%s\n' \
    "$attempt" "$rc" "$(done_runs)" "$EXPECTED_RUNS" "$(date -Is)" > "$STATUS"
  sleep 30
done

printf 'COMPLETE attempts=%d runs=%s/%s finished=%s\n' \
  "$attempt" "$(done_runs)" "$EXPECTED_RUNS" "$(date -Is)" > "$STATUS"
echo "watchdog complete attempts=$attempt $(date -Is)" | tee -a "$WATCHDOG_LOG" "$MAIN_LOG"
