#!/bin/bash
# Publish only after the complete experiment/report chain has succeeded.
set -uo pipefail

DEV=/workspace/prism-exp
RUN=/workspace/prism-merge
PIPE_RUN=$RUN/exp/results/final-overlap-pipeline
PIPE_DEV=$DEV/exp/results/final-overlap-pipeline

while [ ! -f "$PIPE_RUN/COMPLETE" ]; do sleep 60; done
[ -f "$PIPE_DEV/PUBLISHED" ] && exit 0

cd "$DEV"
if [ "$(git branch --show-current)" != exp/tp-v6-merge ]; then
  printf 'wrong branch: %s\n' "$(git branch --show-current)" > "$PIPE_RUN/PUBLISH_BLOCKED"
  exit 0
fi

# The execution worktree owns the final raw data and generated comparison.
# No --delete: preserve reports/evidence that already exist in the dev tree.
mkdir -p "$DEV/exp/results/final-overlap-pipeline" \
         "$DEV/exp/results/final-prototype-vs-paper-faithful" \
         "$DEV/exp/workloads/final-compare"
rsync -a "$RUN/exp/results/final-overlap-pipeline/" \
         "$DEV/exp/results/final-overlap-pipeline/"
rsync -a "$RUN/exp/results/final-prototype-vs-paper-faithful/" \
         "$DEV/exp/results/final-prototype-vs-paper-faithful/"
rsync -a "$RUN/exp/workloads/final-compare/" \
         "$DEV/exp/workloads/final-compare/"

large=$(find \
  exp/results/final-overlap-pipeline \
  exp/results/final-prototype-vs-paper-faithful \
  exp/results/paper-faithful-v6 \
  exp/results/paper-faithful-tp \
  exp/workloads/final-compare \
  exp/scripts exp/tests patches/paper_faithful_v6 \
  -type f -size +95M -print 2>/dev/null | head -n 1)
if [ -n "$large" ]; then
  printf 'file exceeds safe GitHub size: %s\n' "$large" > "$PIPE_DEV/PUBLISH_BLOCKED"
  exit 0
fi

# Intentionally scoped: do not stage venv links, nested dependency trees, or
# unrelated user worktree changes.
git add -- \
  exp/scripts \
  exp/tests \
  patches/paper_faithful_v6 \
  exp/results/final-overlap-pipeline \
  exp/results/final-prototype-vs-paper-faithful \
  exp/results/paper-faithful-v6 \
  exp/results/paper-faithful-tp \
  exp/workloads/final-compare

if ! git diff --cached --quiet; then
  git commit -m "Complete paper-faithful TP overlap evaluation"
fi

for attempt in 1 2 3; do
  if git push origin HEAD:exp/tp-v6-merge; then
    printf 'published_commit=%s\npublished_at=%s\n' \
      "$(git rev-parse HEAD)" "$(date -u -Is)" > "$PIPE_DEV/PUBLISHED"
    exit 0
  fi
  sleep 60
done

printf 'push failed after 3 attempts; local commit=%s\n' \
  "$(git rev-parse HEAD)" > "$PIPE_DEV/PUBLISH_BLOCKED"
exit 0
