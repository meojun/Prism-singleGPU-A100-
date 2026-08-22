#!/usr/bin/env python3
"""Fail-closed progress watchdog for final-regression D1."""
import json
import os
import re
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/root/Prism-final-regression-diagnosis")
BASE = ROOT / "exp/results/final-regression-diagnosis"
D1 = BASE / "D1"
MON = D1 / "monitor"
PLOG = D1 / "pipeline.log"
RUN = D1 / "raw/paper-alg2-only/steady/rate_8/seed_1"
SLOG = RUN / "server-logs/stdout.log"
BLOG = RUN / "server-logs/bench.log"
PROFILE_RC = Path("/workspace/prism-exp/exp/results/final-regression-diagnosis/profile_pair.rc")
PROFILE_LOG = Path("/workspace/prism-exp/exp/results/final-regression-diagnosis/profiling/model_6_profile.log")
RESULT = D1 / "result.json"

TIMEOUTS = {
    "profile_wait": 900,
    "server_startup": 180,
    "model_load": 600,
    "benchmark": 900,
    "benchmark_progress": 120,
    "result_generation": 120,
}
PROFILE_HARD_DEADLINE = 1787377800  # 2026-08-22 14:50:00 KST


def text(path):
    try:
        return path.read_text(errors="replace")
    except OSError:
        return ""


def shell(args):
    return subprocess.run(args, text=True, capture_output=True).stdout.strip()


def pids():
    rows = shell(["ps", "-eo", "pid=,args="]).splitlines()
    needles = (
        "run_d1_diagnosis_pipeline.sh",
        "run_v4_case.sh paper-alg2-only",
        "sglang.launch_multi_model_server",
        "python3 benchmark.py",
    )
    return [r.strip() for r in rows if any(n in r for n in needles)
            and "monitor_d1_pipeline.py" not in r]


def gpu_state():
    gpu = shell(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
                 "--format=csv,noheader"])
    apps = shell(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory",
                  "--format=csv,noheader"])
    return gpu.splitlines(), apps.splitlines()


def phase_and_progress():
    plog, slog, blog = text(PLOG), text(SLOG), text(BLOG)
    if RESULT.exists():
        return "complete", 1, "D1/result.json written", RESULT
    if "collecting D1 metrics" in plog:
        events = re.findall(r"collected |wrote |D1 result complete", plog)
        return "result_generation", len(events), events[-1] if events else "collecting D1 metrics", PLOG
    proc = "\n".join(pids())
    if "python3 benchmark.py" in proc:
        posts = re.findall(r'POST /generate HTTP/1\.1" 200 OK', slog)
        event = f"completed HTTP generation response #{len(posts)}" if posts else "benchmark process started"
        return "benchmark", len(posts), event, SLOG if posts else BLOG
    if "starting D1 steady/r8/seed1" in plog:
        load_events = re.findall(
            r"Preparing engine[^\n]*|Loading safetensors checkpoint shards: 100%[^\n]*|"
            r"Started server process[^\n]*|Application startup complete[^\n]*", slog)
        if load_events:
            return "model_load", len(load_events), load_events[-1][-500:], SLOG
        return "server_startup", 0, "D1 launcher started; no model-load event yet", SLOG
    prc = text(PROFILE_RC).strip()
    pev = re.findall(r"\[model_6\][^\n]*|POST /generate HTTP/1\.1\" 200 OK", text(PROFILE_LOG))
    event = pev[-1] if pev else "waiting for first profiling benchmark event"
    return "profile_wait", len(pev), event, PROFILE_LOG


def fail(reason, status):
    status["state"] = "FAIL"
    status["failure_reason"] = reason
    status["heartbeat_timestamp"] = time.time()
    write_status(status)
    (MON / "FAIL").write_text(reason + "\n")
    for row in pids():
        try:
            pid = int(row.split(None, 1)[0])
            os.kill(pid, signal.SIGTERM)
        except (ValueError, ProcessLookupError, PermissionError):
            pass
    subprocess.run(["tmux", "kill-session", "-t", "v4-paper-alg2-only-steady-r8-s1"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["tmux", "kill-session", "-t", "prism_d1_pipeline"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def write_status(status):
    MON.mkdir(parents=True, exist_ok=True)
    tmp = MON / "status.json.tmp"
    tmp.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, MON / "status.json")
    with (MON / "heartbeat.jsonl").open("a") as f:
        f.write(json.dumps(status, sort_keys=True) + "\n")


def main():
    MON.mkdir(parents=True, exist_ok=True)
    phase = None
    phase_started = time.time()
    last_count = -1
    last_progress = time.time()
    while True:
        now = time.time()
        new_phase, count, event, log_path = phase_and_progress()
        if new_phase != phase:
            phase = new_phase
            phase_started = now
            last_count = -1
            last_progress = now
        if count > last_count:
            last_count = count
            last_progress = now
        try:
            log_mtime = log_path.stat().st_mtime
        except OSError:
            log_mtime = None
        gpu, apps = gpu_state()
        status = {
            "state": "RUNNING" if phase != "complete" else "COMPLETE",
            "current_phase": phase,
            "heartbeat_timestamp": now,
            "heartbeat_iso_utc": datetime.fromtimestamp(now, timezone.utc).isoformat(),
            "heartbeat_age_s": 0,
            "last_progress_timestamp": last_progress,
            "last_progress_iso_utc": datetime.fromtimestamp(last_progress, timezone.utc).isoformat(),
            "last_actual_progress_event": event,
            "progress_count": count,
            "last_log_timestamp": log_mtime,
            "last_log_iso_utc": (datetime.fromtimestamp(log_mtime, timezone.utc).isoformat()
                                  if log_mtime else None),
            "pids": pids(),
            "gpu": gpu,
            "gpu_processes": apps,
            "phase_elapsed_s": round(now - phase_started, 3),
            "phase_hard_timeout_s": TIMEOUTS.get(phase),
            "benchmark_progress_timeout_s": TIMEOUTS["benchmark_progress"],
            "profile_hard_deadline_epoch": PROFILE_HARD_DEADLINE,
        }
        write_status(status)
        if phase == "complete":
            return 0
        if phase == "profile_wait" and now >= PROFILE_HARD_DEADLINE and count == 0:
            fail("profile: no benchmark progress event by 2026-08-22 14:50:00 KST", status)
            return 1
        if now - phase_started > TIMEOUTS[phase]:
            fail(f"{phase}: phase hard timeout exceeded", status)
            return 1
        if phase == "benchmark" and now - last_progress > TIMEOUTS["benchmark_progress"]:
            fail("benchmark: no completed-request progress event for 120 seconds", status)
            return 1
        time.sleep(5)


if __name__ == "__main__":
    raise SystemExit(main())
