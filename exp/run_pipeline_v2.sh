#!/bin/bash
# Full unattended pipeline (canonical copy; /workspace/run_pipeline.sh is the runtime one)
# Full unattended pipeline: sanity gate -> calibration -> main sweep -> ablation -> report.
# Runs in tmux; every stage logs to /workspace/logs/.
set -uo pipefail
cd /workspace/prism-exp
source exp/scripts/env.sh
L=/workspace/logs
stamp() { date -Is; }

echo "===== PIPELINE START $(stamp)"

echo "===== [1] SANITY GATE $(stamp)"
./exp/scripts/run_sanity_v2.sh 12 2>&1 | tee $L/sanity.log
SANITY_RC=${PIPESTATUS[0]}
echo "SANITY_RC=$SANITY_RC" | tee -a $L/sanity.log
if [ "$SANITY_RC" != 0 ]; then
  echo "!!! SANITY GATE FAILED -- stopping before the main experiment, as the brief requires."
  echo "PIPELINE_STOPPED_AT=sanity $(stamp)" >> $L/pipeline.status
  exit 1
fi

echo "===== [2] CALIBRATION $(stamp)"
./exp/scripts/run_calibration_v2.sh 2>&1 | tee $L/calibration.log

echo "===== [2b] PICK LOAD LEVELS $(stamp)"
# Derive Low/Medium/High/Near-saturation from the measured calibration curve
# rather than guessing, so the ladder is neither all-easy nor all-collapsed.
python3 exp/scripts/pick_rates_v2.py \
  --calib exp/results/paper-faithful-v2/sanity/calibration \
  -o $L/chosen_rates.txt 2>&1 | tee $L/pick_rates.log

echo "===== [3] MAIN SWEEP $(stamp)"
if [ -s $L/chosen_rates.txt ]; then export V2_RATES=$(cat $L/chosen_rates.txt); fi
echo "main sweep rates: ${V2_RATES:-<default>}"
./exp/run_paper_faithful_v2.sh --resume 2>&1 | tee $L/sweep.log

echo "===== [4] ABLATION $(stamp)"
# Ablation only needs the load points where the arms actually separate: the
# middle two of the ladder. Running all four would double the sweep for little.
if [ -s $L/chosen_rates.txt ]; then
  export V2_RATES=$(awk '{print $2, $3}' $L/chosen_rates.txt)
fi
echo "ablation rates: ${V2_RATES:-<default>}"
V2_SYSTEMS="paper-alg1-only paper-alg2-only" ./exp/run_paper_faithful_v2.sh --resume 2>&1 | tee $L/ablation.log

echo "===== [5] AGGREGATE $(stamp)"
python3 exp/scripts/aggregate_v2.py --base exp/results/paper-faithful-v2 \
  -o exp/results/paper-faithful-v2/processed 2>&1 | tee $L/aggregate.log

echo "PIPELINE_DONE $(stamp)" >> $L/pipeline.status
echo "===== PIPELINE END $(stamp)"
