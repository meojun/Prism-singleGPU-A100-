#!/bin/bash
# Restart the v4 study if anything has stopped it.
#
# supervisor's autorestart does not cover an explicit stop, and this image's
# portal re-syncs supervisor configs, which stops managed programs: the sweep
# was SIGTERMed once mid-run that way and sat idle until noticed. The pipeline
# is resume-safe, so simply starting it again continues from the first
# incomplete item.
#
# Installed as a crontab entry running every minute. Does nothing while the
# study is running or once it is complete.
# cron runs with a minimal PATH and cannot find supervisorctl otherwise.
export PATH=/usr/local/bin:/usr/bin:/bin:/usr/local/sbin:/usr/sbin:$PATH
LOG=/workspace/logs/prism_v4_ensure.log
BASE=/workspace/prism-exp/exp/results/paper-faithful-v4

# Finished? Then leave it alone.
if [ -s "$BASE/REPORT.md" ] && [ -s "$BASE/summary.csv" ] \
   && [ "$(find "$BASE/raw" -path '*/seed_[0-9]*/DONE' -type f 2>/dev/null | wc -l)" -ge "${V4_EXPECTED_RUNS:-27}" ]; then
    exit 0
fi

state=$(supervisorctl status prism_v4 2>/dev/null | awk '{print $2}')
case "$state" in
    RUNNING|STARTING) exit 0 ;;
    FATAL|BACKOFF)
        # FATAL usually means a stale lock from a previous run; clear it.
        supervisorctl clear prism_v4 >/dev/null 2>&1
        ;;
esac

echo "$(date -Is) state=${state:-unknown} -- restarting" >> "$LOG"
supervisorctl start prism_v4 >> "$LOG" 2>&1
