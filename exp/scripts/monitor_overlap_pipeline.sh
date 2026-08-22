#!/bin/bash
# Conservatively resume the overnight overlap pipeline after transient case
# failures. Correctness/sanity/collection/selection failures remain fail-closed.
set -uo pipefail

PIPE=/workspace/prism-merge/exp/results/final-overlap-pipeline
STATE=$PIPE/state
RETRIES=$PIPE/retry-counts
ARCHIVE=$PIPE/failed-run-attempts
mkdir -p "$RETRIES" "$ARCHIVE" "$PIPE/history"

archive_blocked() {
  local stamp=$1
  mv "$PIPE/BLOCKED" "$PIPE/history/BLOCKED_${stamp}" 2>/dev/null || true
}

retry_case() {
  local id=$1 dest=$2 stamp=$3
  local count_file="$RETRIES/$id"
  local count=0
  [ -f "$count_file" ] && read -r count < "$count_file"
  count=$((count + 1))
  if [ "$count" -gt 2 ]; then
    echo "retry limit reached for $id" >&2
    return 1
  fi
  printf '%s\n' "$count" > "$count_file"
  if [ -d "$dest" ]; then
    mv "$dest" "$ARCHIVE/${id}_attempt${count}_${stamp}"
  fi
  if [ -f "$PIPE/logs/${id}.log" ]; then
    cp -a "$PIPE/logs/${id}.log" "$ARCHIVE/${id}_attempt${count}_${stamp}.log"
  fi
  archive_blocked "$stamp"
  echo "retrying $id ($count/2)" >&2
  supervisorctl start tp_overlap_pipeline
}

while true; do
  [ -f "$PIPE/COMPLETE" ] && exit 0
  status=$(supervisorctl status tp_overlap_pipeline 2>/dev/null | awk '{print $2}')
  if [ "$status" = RUNNING ] || [ "$status" = STARTING ]; then
    sleep 30
    continue
  fi

  stamp=$(date -u +%Y%m%dT%H%M%SZ)
  if [ ! -f "$PIPE/BLOCKED" ]; then
    echo "pipeline stopped without BLOCKED; resuming" >&2
    supervisorctl start tp_overlap_pipeline || true
    sleep 30
    continue
  fi

  reason=$(cat "$PIPE/BLOCKED")
  if [[ "$reason" =~ held-out\ calibration\ case\ failed:\ (cal_tau_(.+)_s([0-9]+)) ]]; then
    id=${BASH_REMATCH[1]}
    label=${BASH_REMATCH[2]}
    seed=${BASH_REMATCH[3]}
    dest="$PIPE/tau-calibration/tau_$label/raw/paper-faithful-v6/bursty/rate_20/seed_$seed"
    retry_case "$id" "$dest" "$stamp" || exit 0
  elif [[ "$reason" =~ final\ C\ case\ failed:\ (finalC_(bursty|steady)_r([0-9]+)_s([0-9]+)) ]]; then
    id=${BASH_REMATCH[1]}
    workload=${BASH_REMATCH[2]}
    rate=${BASH_REMATCH[3]}
    seed=${BASH_REMATCH[4]}
    dest=/workspace/prism-merge/exp/results/final-prototype-vs-paper-faithful/raw/armC-final-overlap/raw/paper-faithful-v6/$workload/rate_$rate/seed_$seed
    retry_case "$id" "$dest" "$stamp" || exit 0
  else
    echo "fail-closed: $reason" >&2
    exit 0
  fi
  sleep 30
done
