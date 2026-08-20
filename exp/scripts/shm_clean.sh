#!/bin/bash
# Between-run cleanup.  A finished server leaves ipc_<gpu>_<worker>_root
# segments and the multiprocessing resource_tracker's own sem.mp-*/mp-* behind;
# the next server then collides with them and dies during startup with
# "resource_tracker: process died unexpectedly" and a KeyError on a segment
# name.  CLAUDE.md records the pattern; what it costs is a whole run, so it is
# done between every run rather than once per sweep.
ps -eo pid,cmd | grep -E "launch_multi_model_server|model_service|benchmark\.py" \
  | grep -v grep | awk '{print $1}' | while read -r p; do kill -9 "$p" 2>/dev/null; done
sleep 3
rm -f /dev/shm/ipc_* /dev/shm/sem.mp-* /dev/shm/mp-* 2>/dev/null
exit 0
