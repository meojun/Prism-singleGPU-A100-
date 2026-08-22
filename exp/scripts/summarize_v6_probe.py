#!/usr/bin/env python3
"""Summarize the instrumented V6 KV hand-off end-to-end validation."""

import argparse
import json
import re
from pathlib import Path


PROBE = re.compile(
    r"\[KV-PROBE (service|engine)\]\s+key=(\S+)\s+have=(\[[^\n]*?\])\s+id=(\S+)"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ns = ap.parse_args()
    out = Path(ns.out).resolve()

    # The launcher writes engine records to server.log and service relay
    # records to server.log.model_service.log.  stdout.log contains HTTP output
    # but not these logger records; the other split logs can duplicate them.
    log_paths = (sorted(out.rglob("server-logs/server.log")) +
                 sorted(out.rglob("server-logs/server.log.model_service.log")))
    if not log_paths:
        log_paths = sorted(path for path in out.rglob("*")
                           if path.is_file() and path.suffix in {".log", ".txt"})
    texts = []
    for path in log_paths:
        try:
            texts.append(path.read_text(errors="replace"))
        except OSError:
            pass
    blob = "\n".join(texts)

    records = []
    for side, key, have, ident in PROBE.findall(blob):
        try:
            have_value = json.loads(have)
        except json.JSONDecodeError:
            have_value = have
        record = {"side": side, "key": key, "have": have_value, "id": ident}
        if record not in records:
            records.append(record)

    sides = {side: sorted({r["key"] for r in records if r["side"] == side})
             for side in ("service", "engine")}
    keys_match = bool(sides["service"] and sides["engine"] and
                      set(sides["service"]) & set(sides["engine"]))
    # Engine events are JSON after the marker; relay events are plain text.
    # Count both formats explicitly instead of assuming the event name follows
    # the marker directly.
    json_events = []
    for line in blob.splitlines():
        if "[PAPER-KV-V6] {" not in line:
            continue
        try:
            json_events.append(json.loads(line.split("[PAPER-KV-V6] ", 1)[1]))
        except (IndexError, json.JSONDecodeError):
            pass
    events = {name: sum(r.get("event") == name for r in json_events)
              for name in ("stash", "inject", "resume", "recompute")}
    events.update({
        "fetch": blob.count("[PAPER-KV-V6] fetch "),
        "relay_clone": blob.count("[PAPER-KV-V6] relay-clone "),
        "capture_failures": sum(r.get("capture_failures", 0) for r in json_events),
    })
    requests_injected = sum(r.get("requests_injected", 0) for r in json_events
                            if r.get("event") == "inject")
    transfer = out / "kv_transfers.jsonl"
    transfers = []
    if transfer.exists():
        for line in transfer.read_text(errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                transfers.append(json.loads(line))
            except json.JSONDecodeError:
                transfers.append({"parse_error": line})
    transfer_records = len(transfers)
    p2p_transfer_records = sum(
        record.get("transfer_path") == "gpu-to-gpu-p2p"
        for record in transfers
    )
    kv_bytes = sum(int(record.get("kv_bytes", 0)) for record in transfers)
    benchmarks = []
    for path in sorted(out.glob("*_e2e_*.json")):
        try:
            result = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(result, dict) and "completed" in result:
            benchmarks.append({
                "file": path.name,
                "completed": int(result.get("completed", 0)),
                "aborted": int(result.get("aborted", 0)),
            })
    completed = sum(r["completed"] for r in benchmarks)
    aborted = sum(r["aborted"] for r in benchmarks)
    fetch_timeouts = blob.count("fetch timed out")
    fatal_patterns = (
        "resume failed", "inject failed", "inject stage failed", "rebuild failed",
        "Attempted to send CUDA tensor received from another process",
        "CUDA out of memory", "OutOfMemoryError",
    )
    fatal_errors = {pattern: blob.count(pattern) for pattern in fatal_patterns}
    failures = []
    if not keys_match:
        failures.append("service/engine queue keys did not match")
    if not benchmarks or completed <= 0 or aborted != 0:
        failures.append("benchmark did not complete with zero aborts")
    for name in ("stash", "fetch", "relay_clone", "inject", "resume"):
        if events[name] <= 0:
            failures.append(f"missing {name} event")
    if requests_injected != events["resume"]:
        failures.append(
            f"injected/resumed mismatch {requests_injected}/{events['resume']}")
    if fetch_timeouts:
        failures.append(f"fetch timeouts: {fetch_timeouts}")
    if transfer_records <= 0:
        failures.append("no KV transfer records")
    elif p2p_transfer_records != transfer_records:
        failures.append(
            f"non-P2P KV transfer records: "
            f"{transfer_records - p2p_transfer_records}/{transfer_records}")
    if any(fatal_errors.values()):
        failures.append("fatal V6 errors observed")

    summary = {
        "status": "PASS" if not failures else "FAIL",
        "probe_records": records,
        "service_keys": sides["service"],
        "engine_keys": sides["engine"],
        "service_engine_key_intersection": sorted(
            set(sides["service"]) & set(sides["engine"])),
        "keys_match_observed": keys_match,
        "events": events,
        "requests_injected": requests_injected,
        "benchmarks": benchmarks,
        "completed": completed,
        "aborted": aborted,
        "fetch_timeouts": fetch_timeouts,
        "fatal_errors": fatal_errors,
        "kv_transfer_records": transfer_records,
        "p2p_transfer_records": p2p_transfer_records,
        "kv_bytes": kv_bytes,
        "failures": failures,
    }
    (out / "PROBE_SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")

    lines = ["# V6 KV hand-off E2E validation", "",
             "수정된 KV stash/inject/request-resume 경로를 계측한 실제 워크로드 결과다. "
             "로그와 raw JSON 에서 확인되지 않은 내용은 포함하지 않는다.", "",
             f"* service keys: `{sides['service']}`",
             f"* engine keys: `{sides['engine']}`",
             f"* observed key intersection: `{summary['service_engine_key_intersection']}`",
             f"* V6 events: `{events}`",
             f"* injected/resumed requests: `{requests_injected}/{events['resume']}`",
             f"* completed/aborted: `{completed}/{aborted}`",
             f"* fetch timeouts: {summary['fetch_timeouts']}",
             f"* KV transfer records: {transfer_records} "
             f"(P2P {p2p_transfer_records}, {kv_bytes / 2**30:.3f} GiB)",
             f"* status: **{summary['status']}**",
             f"* failures: `{failures}`", "", "## Probe records", "",
             "```json", json.dumps(records, indent=2), "```", ""]
    (out / "PROBE_SUMMARY.md").write_text("\n".join(lines))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
