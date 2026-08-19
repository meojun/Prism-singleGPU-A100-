#!/bin/bash
# Wait for the single V4 validation run, decide on P2P, then resume the sweep.
#
# The validation run exists to answer one question: with ipc_collect() in the
# release path, does GPU-to-GPU migration still exhaust GPU memory?  The answer
# decides whether the sweep runs with it on.  Encoding the decision here means
# the study resumes on its own rather than waiting for someone to look.
#
#   pass  -> keep P2P migration on; v4 is measured with its full mechanism
#   fail  -> pin PRISM_V4_P2P_MIGRATION=0 and run with page-locked loading only,
#            which is still the larger of the two v4 changes; the migration
#            mechanism stands on the microbenchmark, where the NVLink counters
#            prove the path directly
set -uo pipefail
export PATH=/usr/local/bin:/usr/bin:/bin:/usr/local/sbin:/usr/sbin:$PATH
VAL=/tmp/claude-0/-root/84ebfe95-1f29-40e9-800a-5a7337ba7c49/scratchpad/v4valid
LOG=/workspace/logs/prism_v4_resume.log
LIMIT_MIB=${V4_MEM_LIMIT_MIB:-74000}

say() { echo "$(date -Is) $*" >> "$LOG"; }
say "waiting for the validation run"

peak=0
for _ in $(seq 1 400); do
    m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 2>/dev/null | tr -d ' ')
    [ -n "$m" ] && [ "$m" -gt "$peak" ] 2>/dev/null && peak=$m
    grep -q "### .* done" "$VAL/case.log" 2>/dev/null && { verdict=pass; break; }
    grep -qE "benchmark failed|OutOfMemory|-> DIED|-> timeout" "$VAL/case.log" 2>/dev/null && { verdict=fail; break; }
    [ "$peak" -gt "$LIMIT_MIB" ] 2>/dev/null && { verdict=fail; break; }
    sleep 10
done
verdict=${verdict:-fail}

# A run that finishes but drops requests has not passed either.
failed_reqs=$(python3 - "$VAL" <<'PY' 2>/dev/null || echo unknown
import glob, json, sys
c = sorted(glob.glob(sys.argv[1] + "/requests/*_output_requests.json"))
if not c:
    print("unknown"); raise SystemExit
d = json.load(open(c[-1]))
print(sum(1 for r in d if not r.get("success")))
PY
)
say "verdict=$verdict peak_gpu0=${peak}MiB failed_requests=$failed_reqs"
[ "$failed_reqs" != unknown ] && [ "${failed_reqs:-1}" -gt 0 ] 2>/dev/null && verdict=fail

if [ "$verdict" = pass ]; then
    say "P2P migration stays ON for the sweep"
    sed -i '/PRISM_V4_P2P_MIGRATION/d' /workspace/.env
else
    say "P2P migration pinned OFF for the sweep (page-locked loading only)"
    sed -i '/PRISM_V4_P2P_MIGRATION/d' /workspace/.env
    echo 'PRISM_V4_P2P_MIGRATION=0' >> /workspace/.env
fi
printf '%s verdict=%s peak_gpu0=%sMiB failed_requests=%s\n' \
    "$(date -Is)" "$verdict" "$peak" "$failed_reqs" \
    > /workspace/logs/prism_v4_p2p_decision.txt

# Nothing from the validation run may still hold a GPU when the sweep starts.
pkill -f 'launch_multi_model[_]server' 2>/dev/null || true
sleep 8
rm -f /dev/shm/ipc_[0-9]*_root /dev/shm/cuda.shm.* 2>/dev/null || true

say "starting the sweep"
supervisorctl start prism_v4 >> "$LOG" 2>&1
( crontab -l 2>/dev/null | grep -v ensure_v4_running
  echo "* * * * * /workspace/prism-exp/exp/scripts/ensure_v4_running.sh" ) | crontab -
say "cron guard re-armed; done"
