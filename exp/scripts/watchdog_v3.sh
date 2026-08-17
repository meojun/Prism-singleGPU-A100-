#!/bin/bash
# Restart-safe supervisor. It owns no credentials and is safe to leave in tmux.
set -uo pipefail
LOG=/workspace/logs
mkdir -p "$LOG"
exec 9>"$LOG/v3-watchdog.lock"
flock -n 9 || { echo "watchdog-v3: another supervisor owns the lock" >> "$LOG/v3-watchdog.log"; exit 0; }
for attempt in $(seq 1 12); do
  grep -q PIPELINE_DONE "$LOG/v3-pipeline.status" 2>/dev/null && exit 0
  grep -q PIPELINE_FAILED "$LOG/v3-pipeline.status" 2>/dev/null && rm -f "$LOG/v3-pipeline.status"
  echo "watchdog-v3 attempt=$attempt $(date -Is)" >> "$LOG/v3-watchdog.log"
  /workspace/prism-exp/exp/run_pipeline_v3.sh >> "$LOG/v3-pipeline.log" 2>&1
  rc=$?
  echo "watchdog-v3 exit=$rc $(date -Is)" >> "$LOG/v3-watchdog.log"
  [ "$rc" = 0 ] && exit 0
  pkill -f 'launch_multi_model[_]server' 2>/dev/null || true
  rm -f /dev/shm/ipc_0_model_*_root 2>/dev/null || true
  sleep 30
done
exit 1
