#!/bin/bash
# Paper-Faithful Prism v2 sweep, run under supervisor.
#
# Supervisor is the only thing on this box that has reliably survived: redis has
# been up under it for hours while both a setsid-detached watchdog and a tmux
# session running the same pipeline were killed mid-run. Whatever reaps those,
# supervisord's children are outside its reach.
#
# The watchdog loop lives here rather than in supervisor's autorestart so that a
# sanity-gate FAILURE (a verdict, per the brief: do not run the main experiment)
# is not retried into the ground, while a crash is.
export HOME=/root
export HF_HOME=/workspace/.hf_home
export PYTHONUNBUFFERED=1
[ -f /workspace/.env ] && { set -a; . /workspace/.env; set +a; }
cd /workspace/prism-exp
exec /workspace/watchdog.sh
