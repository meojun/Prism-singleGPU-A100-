#!/bin/bash
# One-screen status for the paper-faithful-v4 study.
#   /workspace/prism-exp/exp/scripts/status_v4.sh
BASE=/workspace/prism-exp/exp/results/paper-faithful-v4
EXPECTED=${V4_EXPECTED_RUNS:-27}

echo "=== watchdog ==="
# Judge by output, not exit code: supervisorctl exits non-zero whenever any
# program is not RUNNING, which this image always has (jupyter/pyworker EXITED).
out=$(supervisorctl status prism_v4 2>/dev/null)
[ -n "$out" ] && echo "  $out" || echo "  (supervisor not answering)"
[ -f /workspace/logs/prism_v4.status ] && cat /workspace/logs/prism_v4.status

echo
echo "=== stages ==="
for stage in env_check impl_sanity micro_loading micro_migration tp2 workloads; do
    if [ -f "$BASE/state/$stage.done" ]; then
        printf '  %-16s DONE   %s\n' "$stage" "$(cat "$BASE/state/$stage.done")"
    elif [ -f "$BASE/state/$stage.status" ]; then
        printf '  %-16s %s\n' "$stage" "$(cat "$BASE/state/$stage.status")"
    else
        printf '  %-16s PENDING\n' "$stage"
    fi
done

echo
done_runs=$(find "$BASE/raw" -path '*/seed_[0-9]*/DONE' -type f 2>/dev/null | wc -l)
echo "=== end-to-end runs: $done_runs / $EXPECTED ==="
find "$BASE/raw" -path '*/seed_[0-9]*/DONE' -type f 2>/dev/null | sed "s|$BASE/raw/||; s|/DONE||" | sort | sed 's/^/  SUCCESS  /'
for f in "$BASE"/state/e2e_*.status; do
    [ -e "$f" ] || continue
    grep -q '^SUCCESS' "$f" && continue
    printf '  %-8s %s\n' "$(awk '{print $1}' "$f")" "$(basename "$f" .status | sed 's/^e2e_//')"
done

echo
echo "=== GPUs (2,3 must stay idle) ==="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader

echo
echo "=== last pipeline output ==="
tail -6 /workspace/logs/prism_v4.log 2>/dev/null || echo "  (no log yet)"
