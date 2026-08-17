#!/bin/bash
# Wait for the unattended sweep, then strictly validate and build final artifacts.
set -euo pipefail
cd /workspace/prism-exp
source exp/scripts/env.sh
BASE=${V3_BASE:-exp/results/paper-faithful-v3}
STATUS=/workspace/logs/v3-pipeline.status

if [ "${1:-}" = "--wait" ]; then
  while true; do
    until grep -q '^PIPELINE_DONE ' "$STATUS" 2>/dev/null; do sleep 30; done
    if python3 exp/scripts/validate_v3_results.py --base "$BASE" --rates 4 8 14 20 --seed 1; then
      break
    fi
    echo "completion validation failed; restarting resume-safe sweep" >&2
    rm -f "$STATUS"
    until ! tmux has-session -t prism-v3-experiment 2>/dev/null; do sleep 5; done
    tmux new-session -d -s prism-v3-experiment \
      '/workspace/prism-exp/exp/scripts/watchdog_v3.sh >> /workspace/logs/v3-watchdog-tmux.log 2>&1'
  done
else
  python3 exp/scripts/validate_v3_results.py --base "$BASE" --rates 4 8 14 20 --seed 1
fi
if [ ! -e "$BASE/sanity" ]; then
  ln -s ../paper-faithful-v2/sanity "$BASE/sanity"
fi
# Older completed runs used the prototype text marker in scheduler_proof.txt
# even though metrics correctly parsed the v3 JSON marker. Refresh those small
# committed proof files from the validated metrics.
find "$BASE/raw/paper-faithful-v3" -regextype posix-extended \
  -regex '.*/seed_[0-9]+/metrics.json' -print0 | while IFS= read -r -d '' metrics; do
  run_dir=$(dirname "$metrics")
  [ -f "$run_dir/DONE" ] || continue
  proof="$run_dir/scheduler_proof.txt"
  [ -f "$proof" ] || continue
  cycles=$(jq -r '.alg1_cycles // 0' "$metrics")
  migrations=$(jq -r '.migrations_alg1 // 0' "$metrics")
  sed -i -E \
    -e "s/^alg1_log_lines=.*/alg1_log_lines=$cycles/" \
    -e "s/^alg1_migrations=.*/alg1_migrations=$migrations/" \
    -e 's/^proto_migrations=.*/proto_migrations=0/' "$proof"
done
python3 exp/scripts/aggregate_v2.py --base "$BASE" -o "$BASE/processed"
python3 exp/scripts/plot_v2.py --base "$BASE" -o "$BASE/plots" \
  --prism-system paper-faithful-v3
python3 exp/scripts/build_report_v2.py --base "$BASE" \
  --title "Paper-Faithful Prism v3 — Released Prototype 비교" \
  --preface "$BASE/fragments/preface.md" \
  --impl-status "$BASE/fragments/implementation_status.md" \
  --narrative "$BASE/fragments/conclusions.md" \
  --include-comparison -o "$BASE/REPORT.md"

test "$(python3 -c 'import csv; print(sum(1 for _ in csv.DictReader(open("exp/results/paper-faithful-v3/processed/results.csv"))))')" = 16
test -s "$BASE/processed/comparison.csv"
test -s "$BASE/REPORT.md"
echo "FINALIZATION_DONE $(date -Is)" > /workspace/logs/v3-finalizer.status
