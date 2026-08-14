#!/bin/bash
# Full unattended pipeline. Runs in tmux; every stage logs under /workspace/logs.
#
#   calibration -> pick load levels -> SANITY GATE at the High level
#   -> main sweep -> ablation -> aggregate -> plots
#
# Calibration runs before the gate on purpose. The gate's whole job is to catch
# Algorithm 2 under-admitting under pressure; evaluating it at a guessed rate
# that turns out to be far below saturation would let it pass vacuously. Nothing
# in the main experiment runs until the gate passes.
set -uo pipefail
cd /workspace/prism-exp
source exp/scripts/env.sh
L=/workspace/logs
stamp() { date -Is; }
echo "===== PIPELINE START $(stamp)"

echo "===== [1] CALIBRATION $(stamp)"
./exp/scripts/run_calibration_v2.sh 2>&1 | tee $L/calibration.log

echo "===== [2] PICK LOAD LEVELS $(stamp)"
python3 exp/scripts/pick_rates_v2.py \
  --calib exp/results/paper-faithful-v2/sanity/calibration \
  -o $L/chosen_rates.txt 2>&1 | tee $L/pick_rates.log
RATES=$(cat $L/chosen_rates.txt 2>/dev/null)
echo "chosen rates: ${RATES:-<none, using defaults>}"

# High is the third rung of the ladder -- the regime where the arms should
# separate and where under-admission, if present, is visible.
SANITY_RATE=$(echo "$RATES" | awk '{print ($3 != "" ? $3 : $NF)}')
[ -z "${SANITY_RATE:-}" ] && SANITY_RATE=12

echo "===== [3] SANITY GATE at ${SANITY_RATE} req/s $(stamp)"
./exp/scripts/run_sanity_v2.sh "$SANITY_RATE" 2>&1 | tee $L/sanity.log
SANITY_RC=${PIPESTATUS[0]}
echo "SANITY_RC=$SANITY_RC" | tee -a $L/sanity.log
if [ "$SANITY_RC" != 0 ]; then
  echo "!!! SANITY GATE FAILED -- not running the main experiment, as the brief requires."
  echo "PIPELINE_STOPPED_AT=sanity rate=$SANITY_RATE $(stamp)" >> $L/pipeline.status
  exit 1
fi

echo "===== [4] MAIN SWEEP $(stamp)"
[ -n "$RATES" ] && export V2_RATES="$RATES"
echo "main sweep rates: ${V2_RATES:-<default>}"
./exp/run_paper_faithful_v2.sh --resume 2>&1 | tee $L/sweep.log

echo "===== [5] ABLATION $(stamp)"
# Only the middle two rungs: that is where the arms separate, and running all
# four would double the sweep for little.
if [ -n "$RATES" ]; then export V2_RATES=$(echo "$RATES" | awk '{print $2, $3}'); fi
echo "ablation rates: ${V2_RATES:-<default>}"
V2_SYSTEMS="paper-alg1-only paper-alg2-only" ./exp/run_paper_faithful_v2.sh --resume 2>&1 | tee $L/ablation.log

echo "===== [6] AGGREGATE + PLOTS $(stamp)"
python3 exp/scripts/aggregate_v2.py --base exp/results/paper-faithful-v2 \
  -o exp/results/paper-faithful-v2/processed 2>&1 | tee $L/aggregate.log
python3 exp/scripts/plot_v2.py --base exp/results/paper-faithful-v2 \
  -o exp/results/paper-faithful-v2/plots 2>&1 | tee -a $L/aggregate.log

echo "PIPELINE_DONE $(stamp)" >> $L/pipeline.status
echo "===== PIPELINE END $(stamp)"
