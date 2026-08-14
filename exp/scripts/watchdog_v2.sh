#!/bin/bash
# Restart the pipeline if it dies. Every stage is --resume-safe, so a restart
# picks up where the last one stopped instead of redoing finished runs.
# Stops for good on PIPELINE_DONE or on a sanity-gate failure (which is a
# verdict, not a crash, and must not be retried into the ground).
set -uo pipefail
L=/workspace/logs
for attempt in $(seq 1 12); do
  if grep -q "PIPELINE_DONE" $L/pipeline.status 2>/dev/null; then
    echo "watchdog: pipeline done, exiting"; break
  fi
  if grep -q "PIPELINE_STOPPED_AT=sanity" $L/pipeline.status 2>/dev/null; then
    echo "watchdog: sanity gate failed -- not retrying"; break
  fi
  echo "watchdog: starting pipeline (attempt $attempt) $(date -Is)"
  /workspace/run_pipeline.sh >> $L/pipeline.log 2>&1
  rc=$?
  echo "watchdog: pipeline exited rc=$rc $(date -Is)"
  [ "$rc" = 0 ] && break
  pkill -f "launch_multi_model[_]server" 2>/dev/null || true
  rm -f /dev/shm/ipc_0_model_*_root 2>/dev/null || true
  sleep 30
done
echo "watchdog: finished $(date -Is)"
