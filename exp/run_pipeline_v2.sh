#!/bin/bash
# Full unattended pipeline for paper-faithful-v2.
#
#   calibration -> pick load levels -> SANITY GATE (at the High level)
#   -> main sweep -> ablation -> aggregate -> plots -> REPORT.md -> git push
#
# Calibration runs before the gate on purpose: the gate exists to catch
# Algorithm 2 under-admitting under pressure, and evaluating it at a guessed
# rate that turns out to be far below saturation would let it pass vacuously.
# No run of the main experiment happens until the gate passes.
#
# Results are committed and pushed after every stage, so an instance that
# disappears mid-sweep costs at most one stage rather than the whole night.
set -uo pipefail
cd /workspace/prism-exp
source exp/scripts/env.sh
L=/workspace/logs
BASE=exp/results/paper-faithful-v2
stamp() { date -Is; }

save() {   # save "<message>"
  git add -A >/dev/null 2>&1
  git commit -q -m "$1" >/dev/null 2>&1 && echo "  committed: $1"
  timeout 180 git push -q >/dev/null 2>&1 && echo "  pushed" || echo "  push failed (will retry next stage)"
}

report() {
  python3 exp/scripts/aggregate_v2.py --base $BASE -o $BASE/processed 2>&1 | tail -30
  python3 exp/scripts/plot_v2.py --base $BASE -o $BASE/plots 2>&1 | tail -5
  python3 exp/scripts/answer_questions_v2.py --base $BASE \
      -o $BASE/fragments/questions.md 2>&1 | tail -2
  cat $BASE/fragments/questions.md $BASE/fragments/root_cause.md \
      $BASE/fragments/limitations.md > $BASE/fragments/narrative.md 2>/dev/null \
    || cat $BASE/fragments/questions.md $BASE/fragments/limitations.md \
         > $BASE/fragments/narrative.md 2>/dev/null
  python3 exp/scripts/build_report_v2.py --base $BASE \
      --impl-status $BASE/fragments/implementation_status.md \
      --narrative $BASE/fragments/narrative.md \
      -o $BASE/REPORT.md 2>&1 | tail -2
}

echo "===== PIPELINE START $(stamp)"

echo "===== [1] CALIBRATION $(stamp)"
./exp/scripts/run_calibration_v2.sh 2>&1 | tee $L/calibration.log
save "calibration: 이 장비의 부하 곡선 측정 $(stamp)"

echo "===== [2] PICK LOAD LEVELS $(stamp)"
python3 exp/scripts/pick_rates_v2.py --calib $BASE/sanity/calibration \
  -o $L/chosen_rates.txt 2>&1 | tee $L/pick_rates.log
RATES=$(cat $L/chosen_rates.txt 2>/dev/null)
echo "chosen rates: ${RATES:-<none, using defaults>}"

SANITY_RATE=$(echo "$RATES" | awk '{print ($3 != "" ? $3 : $NF)}')
[ -z "${SANITY_RATE:-}" ] && SANITY_RATE=12

echo "===== [3] SANITY GATE at ${SANITY_RATE} req/s $(stamp)"
./exp/scripts/run_sanity_v2.sh "$SANITY_RATE" 2>&1 | tee $L/sanity.log
SANITY_RC=${PIPESTATUS[0]}
echo "SANITY_RC=$SANITY_RC" | tee -a $L/sanity.log
report
save "sanity gate (rate=$SANITY_RATE, rc=$SANITY_RC) $(stamp)"
if [ "$SANITY_RC" != 0 ]; then
  echo "!!! SANITY GATE FAILED -- not running the main experiment, as the brief requires."
  echo "PIPELINE_STOPPED_AT=sanity rate=$SANITY_RATE $(stamp)" >> $L/pipeline.status
  exit 1
fi

echo "===== [4] MAIN SWEEP $(stamp)"
[ -n "$RATES" ] && export V2_RATES="$RATES"
echo "main sweep rates: ${V2_RATES:-<default>}"
./exp/run_paper_faithful_v2.sh --resume 2>&1 | tee $L/sweep.log
report
save "main sweep: steady vs shifting-bursty $(stamp)"

echo "===== [5] ABLATION $(stamp)"
# Only the middle two rungs: that is where the arms separate, and all four
# would double the sweep for little.
if [ -n "$RATES" ]; then export V2_RATES=$(echo "$RATES" | awk '{print $2, $3}'); fi
echo "ablation rates: ${V2_RATES:-<default>}"
V2_SYSTEMS="paper-alg1-only paper-alg2-only" ./exp/run_paper_faithful_v2.sh --resume 2>&1 | tee $L/ablation.log
report
save "ablation: Algorithm 1 단독 / Algorithm 2 단독 $(stamp)"

echo "PIPELINE_DONE $(stamp)" >> $L/pipeline.status
echo "===== PIPELINE END $(stamp)"
