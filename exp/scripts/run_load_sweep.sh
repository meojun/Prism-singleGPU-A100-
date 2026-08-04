#!/bin/bash
# Sweep offered load for one colocation case, to find where Prism starts
# missing SLO.  A baseline that already attains 1.000 leaves nothing to improve
# on, so this locates the contended regime before any comparison is run.
#
#   ./run_load_sweep.sh <A|B|C> [time_scale ...]
#
# time_scale multiplies arrival times (trace.py: arrival_time = req_time * ts),
# so SMALLER = more load and a SHORTER run:
#
#   ts 2.0 -> 0.5x load, 1200 s     ts 0.5  -> 2x load, 300 s
#   ts 1.0 -> 1.0x load,  600 s     ts 0.25 -> 4x load, 150 s
#
# Each point gets its own TAG (…_ts<scale>) so results never collide and the
# committed sanity sweep is never touched.
#
#   TRACE=$SHAREGPT_CONTENT ./run_load_sweep.sh B 2.0 1.0 0.5 0.25
set -euo pipefail

CASE=${1:?usage: run_load_sweep.sh <A|B|C> [time_scale ...]}
shift
SCALES=("$@")
[ ${#SCALES[@]} -gt 0 ] || SCALES=(2.0 1.0 0.5 0.25)

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/env.sh"

TRACE=${TRACE:-$SHAREGPT_CONTENT}
BASETAG=${BASETAG:-$(basename "${TRACE%.pkl}")}

echo "load sweep: case $CASE, trace $(basename "$TRACE"), scales ${SCALES[*]}"
est=0
for ts in "${SCALES[@]}"; do
    est=$(python3 -c "print($est + 600*$ts + 60)")
done
echo "예상 소요 약 $(python3 -c "print(round($est/60))")분"
echo

for ts in "${SCALES[@]}"; do
    tag="${BASETAG}_ts${ts}"
    echo "@@@ ts=$ts  tag=$tag  $(date +%H:%M:%S)"
    TRACE="$TRACE" TAG="$tag" TS="$ts" "$SCRIPT_DIR/run_sanity.sh" "$CASE" \
        || echo "@@@ FAIL ts=$ts rc=$?"
done
echo "@@@ SWEEP DONE"
echo
echo "비교:"
for ts in "${SCALES[@]}"; do
    echo "  python exp/scripts/summarize_sanity.py ${BASETAG}_ts${ts}"
done
