#!/bin/bash
# Re-run only the v6 arm.  The v4 control already passed (3387 requests, 24
# weight transfers, 7 of them gpu-to-gpu) and re-running it would only burn ten
# minutes and risk leaving another set of shm segments behind.
set -uo pipefail
R=/workspace/prism-exp
cd "$R"
set -a; . /workspace/.env; set +a
source "$R/prism-venv/bin/activate"
export PRISM_ROOT=$R PRISM_REPO=$R/prism-research PRISM_EXP=$R/exp
export HF_HOME=/workspace/.hf_home PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1
export SLO_BASE_FILE=$R/exp/results/paper-faithful-v6/profiling/slo_base_this_box.json
export PREFILL_SPEED_FILE=$R/exp/results/paper-faithful-v6/profiling/prefill_speed_this_box.json
export SLO_BASE=$SLO_BASE_FILE PREFILL_SPEED=$PREFILL_SPEED_FILE
export KVPR_TAU=${KVPR_TAU:-0.00035}
export NGPU=2 CFG=$R/exp/configs/v2/6model_2gpu.json
export V2_DURATION=420 V2_WARMUP=60 V2_MEASURE=300

/workspace/shm_clean.sh
OUT=$R/exp/results/paper-faithful-v6/e2e/paper-faithful-v6
rm -rf "$OUT"; mkdir -p "$OUT"
echo "=== paper-faithful-v6  $(date -Is)"
./exp/scripts/run_v4_case.sh paper-faithful-v6 bursty 8 1 \
    "$R/exp/workloads/paper-faithful-v4/bursty_r8_s1.pkl" "$OUT" 2>&1 | tee "$OUT/run.log"
rc=${PIPESTATUS[0]}
echo "--- exit=$rc  $(date -Is)"
[ "$rc" = 0 ] && touch "$OUT/DONE"

echo "=== did the mechanism engage?"
python3 - <<'PY'
import glob, json, os, re
d = "/workspace/prism-exp/exp/results/paper-faithful-v6/e2e/paper-faithful-v6"
kv = os.path.join(d, "kv_transfers.jsonl")
if os.path.exists(kv):
    recs = [json.loads(l) for l in open(kv)]
    print(f"  kv records   : {len(recs)}")
    print(f"  requests move: {sum(r.get('requests_moved',0) for r in recs)}")
    print(f"  skipped(cap) : {sum(r.get('requests_skipped_over_cap',0) for r in recs)}")
    print(f"  kv bytes     : {sum(r.get('kv_bytes',0) for r in recs)/2**20:.1f} MiB")
    print(f"  paths        : {sorted({r.get('transfer_path') for r in recs})}")
else:
    print("  kv_transfers.jsonl ABSENT -- nothing was transferred")
stash = inject = capfail = 0
for f in glob.glob(os.path.join(d, "server-logs", "*")):
    if not os.path.isfile(f): continue
    t = open(f, errors="ignore").read()
    stash += t.count('"event": "stash"'); inject += t.count('"event": "inject"')
    capfail += len(re.findall(r"capture failed", t))
print(f"  stash={stash} inject={inject} capture_failures={capfail}")
for f in glob.glob(os.path.join(d, "*_e2e_*.json")):
    if f.endswith("_output_requests.json"): continue
    j = json.load(open(f))
    for k in ("completed","num_completed","failed","num_failed","total_requests"):
        if k in j: print(f"  {k:13}: {j[k]}")
    break
PY
echo "STAGE_B_V6_DONE $(date -Is)"
