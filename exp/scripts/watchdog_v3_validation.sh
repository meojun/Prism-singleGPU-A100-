#!/bin/bash
# Persistent, resume-safe supervisor for the Paper-Faithful V3 validation.
# The runner treats a valid DONE+metrics.json as immutable and skips it, so a
# restart executes only missing/invalid cases. Keep this watchdog inside tmux.
set -uo pipefail

ROOT=/workspace/prism-exp
BASE=$ROOT/exp/results/paper-faithful-v3-validation
LOGDIR=/workspace/logs
MAIN_LOG=$LOGDIR/prism_v3_validation.log
STATUS=$LOGDIR/prism_v3_validation.status
WATCHDOG_LOG=$LOGDIR/prism_v3_validation_watchdog.log
mkdir -p "$LOGDIR"

exec 9>"$LOGDIR/prism_v3_validation.lock"
flock -n 9 || {
  echo "watchdog already running $(date -Is)" >> "$WATCHDOG_LOG"
  exit 0
}

complete() {
  [ "$(find "$BASE/raw" -path '*/seed_[0-9]*/DONE' -type f 2>/dev/null | wc -l)" -eq 48 ] || return 1
  [ "$(find "$BASE/figures" -maxdepth 1 -name '*.png' -type f 2>/dev/null | wc -l)" -eq 4 ] || return 1
  [ -s "$BASE/summary.csv" ] && [ -s "$BASE/REPORT.md" ]
}

attempt=0
while ! complete; do
  attempt=$((attempt + 1))
  printf 'RUNNING attempt=%d started=%s\n' "$attempt" "$(date -Is)" > "$STATUS"
  echo "watchdog attempt=$attempt start $(date -Is)" | tee -a "$WATCHDOG_LOG" "$MAIN_LOG"

  # A dead parent can leave the nested server tmux/process alive. Remove that
  # orphan before resuming so the next missing case never contends for GPUs.
  pkill -f 'launch_multi_model[_]server' 2>/dev/null || true
  rm -f /dev/shm/ipc_0_model_*_root 2>/dev/null || true
  cd "$ROOT" || exit 1
  ./exp/run_v3_validation.sh >> "$MAIN_LOG" 2>&1
  rc=$?
  done_count=$(find "$BASE/raw" -path '*/seed_[0-9]*/DONE' -type f 2>/dev/null | wc -l)
  echo "watchdog attempt=$attempt exit=$rc done=$done_count/48 $(date -Is)" \
    | tee -a "$WATCHDOG_LOG" "$MAIN_LOG"

  complete && break

  # Clean only processes and shared-memory segments belonging to a failed
  # benchmark launch. Successfully completed run directories remain untouched.
  pkill -f 'launch_multi_model[_]server' 2>/dev/null || true
  rm -f /dev/shm/ipc_0_model_*_root 2>/dev/null || true
  printf 'RESTARTING attempt=%d last_rc=%d done=%s/48 at=%s\n' \
    "$attempt" "$rc" "$done_count" "$(date -Is)" > "$STATUS"
  sleep 30
done

printf 'COMPLETE attempts=%d finished=%s\n' "$attempt" "$(date -Is)" > "$STATUS"
echo "watchdog complete attempts=$attempt $(date -Is)" | tee -a "$WATCHDOG_LOG" "$MAIN_LOG"
