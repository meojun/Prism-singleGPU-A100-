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

say "[1/5] redis"
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

say "[2/5] model weights (background tmux session 'models')"
mkdir -p /workspace/logs
if ! tmux has-session -t models 2>/dev/null; then
    tmux new-session -d -s models \
      "bash -lc '$ROOT/setup/download_models.sh 2>&1 | tee /workspace/logs/models.log; sleep infinity'"
    echo "  started; follow with: tail -f /workspace/logs/models.log"
else
    echo "  session 'models' already running"
fi

say "[3/5] bootstrap (pinned stack) -- this is the ~20 min step"
SKIP_MODELS=1 ./bootstrap.sh 2>&1 | tee /workspace/logs/bootstrap.log
grep -q "BAD" /workspace/logs/bootstrap.log && { echo "FATAL: bootstrap verify failed"; exit 1; }

say "[4/5] Paper-Faithful Algorithm 1 / Algorithm 2 patches"
python3 patches/paper_faithful/apply_patches.py --repo "$ROOT/prism-research"

say "[5/5] unit tests"
source "$ROOT/exp/scripts/env.sh"
python3 exp/tests/test_moore_hodgson.py  || { echo "FATAL: Algorithm 2 tests failed"; exit 1; }
python3 exp/tests/test_kvpr_placement.py || { echo "FATAL: Algorithm 1 tests failed"; exit 1; }

cat <<DONE

=== quickstart complete ===
  models still downloading?   tail -f /workspace/logs/models.log
  then:                       source exp/scripts/env.sh
                              ./exp/run_paper_faithful_v2.sh --dry-run
DONE
