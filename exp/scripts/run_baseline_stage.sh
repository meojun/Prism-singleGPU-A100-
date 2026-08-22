#!/bin/bash
set -euo pipefail

STAGE_DIR=${1:?stage dir}
shift
mkdir -p "$STAGE_DIR/monitor"
PIPELINE_LOG="$STAGE_DIR/pipeline.log"
RC_FILE="$STAGE_DIR/pipeline.rc"
trap 'rc=$?; printf "%s\n" "$rc" > "$RC_FILE"' EXIT

echo "[$(date -u +%FT%TZ)] stage command: $*" | tee -a "$PIPELINE_LOG"
"$@" 2>&1 | tee -a "$PIPELINE_LOG"
echo "[$(date -u +%FT%TZ)] stage complete" | tee -a "$PIPELINE_LOG"
