#!/bin/bash
# ONE COMMAND to rebuild this entire experiment environment on a fresh GPU box.
#
#   git clone https://github.com/meojun/Prism-singleGPU-A100- /workspace/prism-exp
#   cd /workspace/prism-exp
#   echo 'HF_TOKEN=hf_xxx' > /workspace/.env && chmod 600 /workspace/.env
#   ./setup/quickstart.sh          # ~35 min, unattended, safe to re-run
#
# Does, in order:
#   1. redis under supervisor (the base image has neither)
#   2. bootstrap.sh          -- pinned sources + venv + torch/sglang/vllm/kvcached
#   3. apply_patches.py      -- the Paper-Faithful Algorithm 1 / Algorithm 2 code
#   4. model weights         -- the heterogeneous 6-model set (background)
#   5. unit tests            -- proves both algorithms are wired and correct
#
# Everything long-running goes into tmux, so closing your laptop is fine.
# Re-running is idempotent: completed steps are skipped.
set -uo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
say() { echo -e "\n=== $* ==="; }

say "[1/6] redis"
if ! redis-cli ping >/dev/null 2>&1; then
    apt-get install -y redis-server >/dev/null 2>&1 || true
    mkdir -p /etc/supervisor/conf.d
    cat > /etc/supervisor/conf.d/redis.conf <<'CONF'
[program:redis]
environment=PROC_NAME="%(program_name)s"
command=/usr/bin/redis-server --bind 127.0.0.1 --port 6379 --save ""
autostart=true
autorestart=true
stdout_logfile=/dev/stdout
redirect_stderr=true
stdout_logfile_maxbytes=0
CONF
    supervisorctl reread >/dev/null 2>&1; supervisorctl update >/dev/null 2>&1
    sleep 2
    redis-server --daemonize yes 2>/dev/null || true
fi
redis-cli ping || { echo "FATAL: redis not answering on :6379"; exit 1; }

say "[2/6] model weights (background tmux session 'models')"
mkdir -p /workspace/logs
if ! tmux has-session -t models 2>/dev/null; then
    tmux new-session -d -s models \
      "bash -lc '$ROOT/setup/download_models.sh 2>&1 | tee /workspace/logs/models.log; sleep infinity'"
    echo "  started; follow with: tail -f /workspace/logs/models.log"
else
    echo "  session 'models' already running"
fi

say "[3/6] datasets (background tmux session 'datasets')"
if ! tmux has-session -t datasets 2>/dev/null; then
    tmux new-session -d -s datasets \
      "bash -lc '$ROOT/setup/download_datasets.sh 2>&1 | tee /workspace/logs/datasets.log; sleep infinity'"
    echo "  started; follow with: tail -f /workspace/logs/datasets.log"
else
    echo "  session 'datasets' already running"
fi

say "[4/6] bootstrap (pinned stack) -- this is the ~20 min step"
SKIP_MODELS=1 ./bootstrap.sh 2>&1 | tee /workspace/logs/bootstrap.log
grep -q "BAD" /workspace/logs/bootstrap.log && { echo "FATAL: bootstrap verify failed"; exit 1; }

say "[5/6] Paper-Faithful Algorithm 1 / Algorithm 2 patches"
python3 patches/paper_faithful/apply_patches.py --repo "$ROOT/prism-research"

say "[6/6] unit tests"
source "$ROOT/exp/scripts/env.sh"
python3 exp/tests/test_moore_hodgson.py  || { echo "FATAL: Algorithm 2 tests failed"; exit 1; }
python3 exp/tests/test_kvpr_placement.py || { echo "FATAL: Algorithm 1 tests failed"; exit 1; }

say "supervisor registration"
# Long runs MUST live under supervisor on this box: tmux sessions and even
# setsid-detached processes have been killed mid-sweep here, while supervisord
# has stayed up for hours. See CLAUDE.md section 8.
if [ ! -f /etc/supervisor/conf.d/prism_pipeline.conf ]; then
    cp "$ROOT/setup/prism_pipeline.conf" /etc/supervisor/conf.d/prism_pipeline.conf
    mkdir -p /opt/supervisor-scripts
    cp "$ROOT/setup/prism_pipeline.sh" /opt/supervisor-scripts/prism_pipeline.sh
    chmod +x /opt/supervisor-scripts/prism_pipeline.sh
    supervisorctl reread >/dev/null && supervisorctl update >/dev/null
    supervisorctl stop prism_pipeline >/dev/null 2>&1 || true
    echo "  registered (stopped -- start it when you are ready)"
else
    echo "  already registered"
fi

cat <<DONE

=== quickstart complete ===

Still downloading in the background? Watch them:
    tail -f /workspace/logs/models.log      # ~47 GB of weights
    tail -f /workspace/logs/datasets.log    # ~670 MB ShareGPT

Then, in order:
    source exp/scripts/env.sh

    # 1. per-model c_i and SLO baselines (~40 min, 2 GPUs in parallel).
    #    Machine-specific -- do NOT reuse another box's numbers.
    ./exp/scripts/run_profiling_v2.sh

    # 2. everything else, unattended, under supervisor:
    #    calibration -> load levels -> sanity gate -> sweep -> ablation -> REPORT
    supervisorctl start prism_pipeline
    tail -f /workspace/logs/pipeline.log

Check on it later with:
    supervisorctl status prism_pipeline
    grep -E "^=====" /workspace/logs/pipeline.log
    cat exp/results/paper-faithful-v2/REPORT.md

Read CLAUDE.md section 8 before debugging anything. Every trap in it cost an
hour to find.
DONE
