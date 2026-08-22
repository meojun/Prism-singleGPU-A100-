#!/usr/bin/env python3
"""Paired short D0/D1 run with audit writes disabled and fail-closed progress."""
import json
import os
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/root/Prism-final-regression-diagnosis")
BASE = ROOT / "exp/results/final-regression-diagnosis/alg2-root-cause/sanity"
TRACE = BASE / "steady_r8_s1_120s.pkl"
STATUS = BASE / "status.json"
HEARTBEAT = BASE / "heartbeat.jsonl"
SERVER_TIMEOUT = 600
BENCH_TIMEOUT = 600
PROGRESS_TIMEOUT = 120
RESULT_TIMEOUT = 120


def gpu():
    a = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
         "--format=csv,noheader"], text=True, capture_output=True).stdout.strip()
    p = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory",
         "--format=csv,noheader"], text=True, capture_output=True).stdout.strip()
    return a.splitlines(), p.splitlines()


def count_posts(path):
    try:
        return path.read_text(errors="replace").count('POST /generate HTTP/1.1" 200 OK')
    except OSError:
        return 0


def benchmark_pid():
    out = subprocess.run(["ps", "-eo", "pid=,args="], text=True,
                         capture_output=True).stdout.splitlines()
    rows = [x.strip() for x in out if x.strip().split(None, 1)[1].startswith("python3 benchmark.py")]
    return rows


def emit(arm, phase, phase_start, last_progress, count, event, proc):
    now = time.time()
    g, gp = gpu()
    rec = {
        "arm": arm, "current_phase": phase, "state": "RUNNING",
        "heartbeat_timestamp": now,
        "heartbeat_iso_utc": datetime.fromtimestamp(now, timezone.utc).isoformat(),
        "last_progress_timestamp": last_progress,
        "last_progress_iso_utc": datetime.fromtimestamp(last_progress, timezone.utc).isoformat(),
        "last_actual_progress_event": event, "progress_count": count,
        "pid": proc.pid if proc else None, "benchmark_pids": benchmark_pid(),
        "gpu": g, "gpu_processes": gp,
        "phase_elapsed_s": round(now - phase_start, 3),
        "phase_hard_timeout_s": SERVER_TIMEOUT if phase == "server_startup" else BENCH_TIMEOUT,
        "progress_timeout_s": PROGRESS_TIMEOUT,
    }
    tmp = STATUS.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, indent=2) + "\n")
    os.replace(tmp, STATUS)
    with HEARTBEAT.open("a") as f:
        f.write(json.dumps(rec, sort_keys=True) + "\n")
    return rec


def terminate(proc, session):
    if proc.poll() is None:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
    subprocess.run(["tmux", "kill-session", "-t", session],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def run_arm(label, system):
    out = BASE / label / "raw" / system / "steady" / "rate_8" / "seed_1"
    out.mkdir(parents=True, exist_ok=True)
    log_path = BASE / label / "run.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({
        "PRISM_ROOT": str(ROOT), "NGPU": "2", "MAXMEM": "67.28",
        "CFG": str(ROOT / "exp/configs/v2/6model_2gpu.json"),
        "SLO_BASE_FILE": str(ROOT / "exp/configs/v2/slo_base.json"),
        "PREFILL_SPEED_FILE": str(ROOT / "exp/configs/v2/prefill_speed.json"),
        "BENCHMARK_TIMEOUT": str(BENCH_TIMEOUT),
    })
    env.pop("PRISM_DIAG_ALG2", None)
    cmd = ["bash", str(ROOT / "exp/scripts/run_v4_case.sh"), system,
           "steady", "8", "1", str(TRACE), str(out)]
    with log_path.open("w") as log:
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log,
                                stderr=subprocess.STDOUT)
        phase = "server_startup"
        phase_start = last_progress = time.time()
        last_count = 0
        event = "launcher started"
        session = f"v4-{system}-steady-r8-s1"
        while proc.poll() is None:
            server_log = out / "server-logs/stdout.log"
            count = count_posts(server_log)
            bp = benchmark_pid()
            new_phase = "benchmark" if bp else "server_startup"
            if new_phase != phase:
                phase = new_phase
                phase_start = last_progress = time.time()
                last_count = count
                event = "benchmark process started"
            if count > last_count:
                last_count = count
                last_progress = time.time()
                event = f"completed HTTP generation response #{count}"
            rec = emit(label, phase, phase_start, last_progress, count, event, proc)
            elapsed = time.time() - phase_start
            if phase == "server_startup" and elapsed > SERVER_TIMEOUT:
                terminate(proc, session)
                raise RuntimeError(f"{label}: server startup hard timeout")
            if phase == "benchmark" and elapsed > BENCH_TIMEOUT:
                terminate(proc, session)
                raise RuntimeError(f"{label}: benchmark hard timeout")
            if phase == "benchmark" and time.time() - last_progress > PROGRESS_TIMEOUT:
                terminate(proc, session)
                raise RuntimeError(f"{label}: benchmark progress stalled")
            time.sleep(5)
        if proc.returncode != 0:
            raise RuntimeError(f"{label}: run failed rc={proc.returncode}")
    (out / "DONE").touch()


def main():
    BASE.mkdir(parents=True, exist_ok=True)
    try:
        run_arm("D0", "released-prototype")
        run_arm("D1-audit-off", "paper-alg2-only")
    except Exception as e:
        (BASE / "FAIL").write_text(str(e) + "\n")
        raise
    (BASE / "DONE").touch()
    rec = json.loads(STATUS.read_text())
    rec.update({"state": "COMPLETE", "current_phase": "complete",
                "last_actual_progress_event": "paired sanity complete"})
    STATUS.write_text(json.dumps(rec, indent=2) + "\n")


if __name__ == "__main__":
    main()
