#!/usr/bin/env python3
"""Fail-closed heartbeat/progress watchdog for one baseline-readiness stage."""

import argparse
import json
import os
import re
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def read(path):
    try:
        return path.read_text(errors="replace")
    except OSError:
        return ""


def shell(args):
    return subprocess.run(args, text=True, capture_output=True).stdout.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-dir", type=Path, required=True)
    parser.add_argument("--pipeline-session", required=True)
    parser.add_argument("--inner-session", required=True)
    parser.add_argument("--server-timeout", type=float, default=600)
    parser.add_argument("--benchmark-timeout", type=float, default=600)
    parser.add_argument("--no-progress-timeout", type=float, default=120)
    args = parser.parse_args()

    mon = args.stage_dir / "monitor"
    mon.mkdir(parents=True, exist_ok=True)
    plog = args.stage_dir / "pipeline.log"
    rcfile = args.stage_dir / "pipeline.rc"
    slog = args.stage_dir / "server-logs/stdout.log"
    blog = args.stage_dir / "server-logs/bench.log"
    phase = None
    phase_start = last_progress = time.time()
    last_count = -1
    inner_missing_since = None

    def fail(reason, status):
        status.update(state="FAIL", failure_reason=reason)
        write(status)
        (mon / "FAIL").write_text(reason + "\n")
        for session in (args.inner_session, args.pipeline_session):
            subprocess.run(
                ["tmux", "kill-session", "-t", session],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )

    def write(status):
        tmp = mon / "status.json.tmp"
        tmp.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
        os.replace(tmp, mon / "status.json")
        with (mon / "heartbeat.jsonl").open("a") as f:
            f.write(json.dumps(status, sort_keys=True) + "\n")

    while True:
        now = time.time()
        pipeline = read(plog)
        server = read(slog)
        bench = read(blog)
        rc = read(rcfile).strip()
        if rc:
            new_phase = "complete" if rc == "0" else "failed"
            count = len(re.findall(r"Completed requests:|stage complete", pipeline + bench))
            event = f"pipeline rc={rc}"
            log_path = plog
        elif " -> ready" in pipeline:
            new_phase = "benchmark"
            responses = len(re.findall(r'POST /generate HTTP/1\.1" 200 OK', server))
            arrivals = len(re.findall(r"^Request .* arrives", bench, re.MULTILINE))
            count = responses
            event = f"completed_responses={responses}, arrivals={arrivals}"
            log_path = slog if responses else blog
        else:
            load = re.findall(
                r"Loading safetensors checkpoint shards: 100%[^\n]*|"
                r"Model service worker \d+ started|Controller process started|"
                r"Started page preallocation thread",
                server,
            )
            new_phase = "model_load" if load else "server_startup"
            count = len(load)
            event = load[-1][-300:] if load else "launcher started"
            log_path = slog if slog.exists() else plog

        if new_phase != phase:
            phase = new_phase
            phase_start = last_progress = now
            last_count = -1
        if count > last_count:
            last_count = count
            last_progress = now
        try:
            log_mtime = log_path.stat().st_mtime
        except OSError:
            log_mtime = None

        ps = shell(["ps", "-eo", "pid=,args="])
        pids = [
            row.strip() for row in ps.splitlines()
            if str(args.stage_dir) in row
            and "monitor_baseline_stage.py" not in row
        ]
        gpu = shell([
            "nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
            "--format=csv,noheader",
        ]).splitlines()
        gpu_processes = shell([
            "nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory",
            "--format=csv,noheader",
        ]).splitlines()
        timeout = (
            args.benchmark_timeout if phase == "benchmark"
            else args.server_timeout
        )
        status = {
            "state": "COMPLETE" if phase == "complete" else "RUNNING",
            "current_phase": phase,
            "heartbeat_timestamp": now,
            "heartbeat_iso_utc": datetime.fromtimestamp(
                now, timezone.utc
            ).isoformat(),
            "last_progress_timestamp": last_progress,
            "last_progress_iso_utc": datetime.fromtimestamp(
                last_progress, timezone.utc
            ).isoformat(),
            "last_actual_progress_event": event,
            "progress_count": count,
            "last_log_timestamp": log_mtime,
            "pids": pids,
            "gpu": gpu,
            "gpu_processes": gpu_processes,
            "phase_elapsed_s": round(now - phase_start, 3),
            "phase_hard_timeout_s": timeout,
            "no_progress_timeout_s": args.no_progress_timeout,
        }
        write(status)
        if phase == "complete":
            return 0
        if phase == "failed":
            fail(f"pipeline failed with rc={rc}", status)
            return 1
        inner_alive = subprocess.run(
            ["tmux", "has-session", "-t", args.inner_session],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode == 0
        if phase == "benchmark" and not inner_alive:
            inner_missing_since = inner_missing_since or now
            # Give the wrapper a short interval to publish pipeline.rc after a
            # normal benchmark exit.  A dead server/session with no result is
            # otherwise a hard failure even if the outer tmux session survives.
            if now - inner_missing_since > 10:
                fail("benchmark: inner server session exited without result", status)
                return 1
        else:
            inner_missing_since = None
        if now - phase_start > timeout:
            fail(f"{phase}: hard timeout exceeded", status)
            return 1
        if now - last_progress > args.no_progress_timeout:
            fail(f"{phase}: no actual progress event", status)
            return 1
        time.sleep(5)


if __name__ == "__main__":
    raise SystemExit(main())
