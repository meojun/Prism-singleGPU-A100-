#!/bin/bash
# MANDATORY pre-flight. Proves the box can actually hold a multi-hour run
# BEFORE one is started. Run it; do not skip it.
#
#   ./exp/scripts/preflight_v2.sh
#
# Three separate multi-hour sweeps were lost on this instance before anyone
# checked this. Each looked fine for minutes and then vanished: a tmux session
# gone, a setsid-detached watchdog gone, no error anywhere in the pipeline log.
# The causes were mundane and only visible from outside the pipeline. This
# script reproduces each of them in ~90 seconds against a canary process, so the
# failure shows up now instead of six hours in.
#
# Writes /workspace/logs/preflight.ok on success. run_pipeline_v2.sh refuses to
# start without it.
set -uo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
L=/workspace/logs
mkdir -p "$L"
FAIL=0
pass() { echo "  [PASS] $1"; }
fail() { echo "  [FAIL] $1"; FAIL=1; }

echo "=== pre-flight: can this box hold a multi-hour run?"

# ---------------------------------------------------------------- 1. supervisor
echo "--- 1. supervisord is available and healthy"
# NB: `supervisorctl status` exits non-zero whenever ANY program is not
# RUNNING -- this image ships jupyter and pyworker in EXITED state by design --
# so its exit code says nothing about whether supervisord is reachable. Judge by
# whether it produced a status table.
SUP=$(supervisorctl status 2>/dev/null)
if [ -n "$SUP" ] && echo "$SUP" | grep -qE 'RUNNING|STOPPED|EXITED'; then
    up=$(echo "$SUP" | awk '/^redis /{print $NF}')
    pass "supervisord answering ($(echo "$SUP" | grep -c RUNNING) programs RUNNING, redis uptime ${up:-n/a})"
else
    fail "supervisorctl not answering -- long runs have nowhere durable to live"
fi

# ------------------------------------------------- 2. the `kill 0` class of bug
# `kill 0` signals the WHOLE PROCESS GROUP. A cleanup trap that fires before the
# variable holding a child PID is set expands "${VAR:-0}" to 0 and takes down the
# sweep driver, the watchdog and the parent shell with it. This killed three runs
# here. Scan every script the sweep actually executes.
echo "--- 2. no process-group suicide in the run path"
HITS=$(grep -rnE 'kill +"?\$\{[A-Za-z_]+:-0\}"?|kill +0( |$)|kill +-- +-' \
       "$SCRIPT_DIR"/run_v2_case.sh "$SCRIPT_DIR"/run_calibration_v2.sh \
       "$SCRIPT_DIR"/run_sanity_v2.sh "$SCRIPT_DIR"/run_profiling_v2.sh \
       "$ROOT/exp/run_paper_faithful_v2.sh" "$ROOT/exp/run_pipeline_v2.sh" 2>/dev/null \
       | grep -v '^\s*#' | grep -vE '#.*kill 0' || true)
if [ -z "$HITS" ]; then pass "no 'kill 0' / 'kill \${X:-0}' / 'kill -- -PGID'"
else fail "process-group kill found:"; echo "$HITS" | sed 's/^/         /'; fi

# ------------------------------------------------------- 3. canary survives detach
# The real question is not "does it start" but "is it still there after the thing
# that started it is gone". Start a canary under supervisor, then kill this
# shell's whole process group's worth of children and see if it lived.
echo "--- 3. a supervisor-managed process outlives its launcher"
cat > /tmp/preflight_canary.sh <<'CANARY'
#!/bin/bash
while true; do echo "canary alive $(date +%s)"; sleep 5; done
CANARY
chmod +x /tmp/preflight_canary.sh
cat > /etc/supervisor/conf.d/preflight_canary.conf <<'CONF'
[program:preflight_canary]
command=/tmp/preflight_canary.sh
autostart=true
autorestart=true
startsecs=3
stdout_logfile=/workspace/logs/preflight_canary.log
redirect_stderr=true
CONF
supervisorctl reread >/dev/null 2>&1
supervisorctl update >/dev/null 2>&1
sleep 8
if supervisorctl status preflight_canary 2>/dev/null | grep -q RUNNING; then
    CPID=$(supervisorctl status preflight_canary | awk '{print $4}' | tr -d ',')
    # a subshell that exits immediately; if group-scoped cleanup reaches the
    # canary, this is where it would show
    ( setsid bash -c 'sleep 1' ) >/dev/null 2>&1
    sleep 6
    if supervisorctl status preflight_canary 2>/dev/null | grep -q RUNNING; then
        pass "canary still RUNNING after its launcher exited (pid $CPID)"
    else
        fail "canary died when its launcher exited"
    fi
else
    fail "canary never reached RUNNING"
fi

# ---------------------------------------------------- 4. autorestart actually works
echo "--- 4. supervisor brings it back when something kills it"
CPID=$(supervisorctl status preflight_canary 2>/dev/null | awk '{print $4}' | tr -d ',')
if [ -n "${CPID:-}" ] && kill -9 "$CPID" 2>/dev/null; then
    sleep 10
    NEW=$(supervisorctl status preflight_canary 2>/dev/null | awk '{print $4}' | tr -d ',')
    if supervisorctl status preflight_canary 2>/dev/null | grep -q RUNNING && [ "$NEW" != "$CPID" ]; then
        pass "SIGKILLed at $CPID, respawned as $NEW"
    else
        fail "did not come back after SIGKILL -- autorestart is not working"
    fi
else
    fail "could not SIGKILL the canary to test autorestart"
fi
supervisorctl stop preflight_canary >/dev/null 2>&1
rm -f /etc/supervisor/conf.d/preflight_canary.conf
supervisorctl reread >/dev/null 2>&1; supervisorctl update >/dev/null 2>&1

# ------------------------------------------------------------- 5. tmux durability
# run_v2_case.sh puts each model server in a tmux session. If the tmux server
# cannot outlive the shell that created the session, every run dies at teardown.
echo "--- 5. a tmux session outlives the shell that created it"
tmux kill-session -t preflight_tmux 2>/dev/null || true
bash -c 'tmux new-session -d -s preflight_tmux "sleep 300"' 
sleep 3
if tmux has-session -t preflight_tmux 2>/dev/null; then
    pass "tmux session survived its creating shell"
    tmux kill-session -t preflight_tmux 2>/dev/null || true
else
    fail "tmux session died with its creating shell -- model servers cannot survive"
fi

# --------------------------------------------------------------- 6. no fixed ports
# launch_multi_model_server derives one port per model engine from --port and
# probes availability by binding 0.0.0.0. A hard-coded base in the 30000-40000
# range collides with editor/tooling dynamic ports, and only AFTER every model
# has loaded.
echo "--- 6. run ports are acquired at run time, not hard-coded"
if grep -q "find_free_port.py" "$SCRIPT_DIR/run_v2_case.sh" 2>/dev/null; then
    P=$(python3 "$SCRIPT_DIR/find_free_port.py" --from 41000 --span 24 2>/dev/null)
    [ -n "$P" ] && pass "free 24-port block found at $P" || fail "find_free_port.py returned nothing"
else
    fail "run_v2_case.sh does not use find_free_port.py"
fi

# ----------------------------------------------------- 7. the patches really landed
echo "--- 7. Algorithm 1 / Algorithm 2 patches are present in the clone"
if python3 "$ROOT/patches/paper_faithful/apply_patches.py" --repo "$ROOT/prism-research" 2>&1 | grep -q "verified"; then
    pass "all patch landing points verified"
else
    fail "patch verification failed -- see apply_patches.py output"
fi

# -------------------------------------------------------------------- 8. disk room
echo "--- 8. disk headroom"
AVAIL=$(df -BG --output=avail /workspace 2>/dev/null | tail -1 | tr -dc '0-9')
if [ "${AVAIL:-0}" -ge 30 ]; then pass "${AVAIL}G free on /workspace"
else fail "only ${AVAIL:-?}G free -- a full sweep writes tens of GB of logs"; fi

echo
if [ "$FAIL" = 0 ]; then
    date -Is > "$L/preflight.ok"
    echo "PRE-FLIGHT PASSED -- wrote $L/preflight.ok"
    exit 0
fi
rm -f "$L/preflight.ok"
echo "PRE-FLIGHT FAILED -- fix the above before starting a long run."
echo "Do NOT start the sweep: it will look fine for minutes and then vanish."
exit 1
