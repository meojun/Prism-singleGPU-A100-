#!/usr/bin/env python3
"""Fail closed unless one run proves the V6 overlap/KV protocol end to end."""

import argparse
import json
import statistics
from pathlib import Path


def marked(blob, marker):
    records = []
    for line in blob.splitlines():
        if marker not in line:
            continue
        try:
            records.append(json.loads(line.split(marker, 1)[1]))
        except (IndexError, json.JSONDecodeError):
            pass
    return records


def percentile(values, q):
    if not values:
        return None
    values = sorted(values)
    pos = (len(values) - 1) * q / 100
    lo, hi = int(pos), min(int(pos) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ns = ap.parse_args()
    out = Path(ns.out).resolve()
    paths = sorted((out / "server-logs").glob("*.log"))
    blob = "\n".join(p.read_text(errors="replace") for p in paths)
    overlap = marked(blob, "[PAPER-OVERLAP-V6] ")
    actions = marked(blob, "[PAPER-ACTION-V4] ")
    kv = marked(blob, "[PAPER-KV-V6] ")

    # A migration's engine events use distinct request ids, so pair by model
    # and chronology: ready -> next serving-source quiesce -> next commit.
    sequences = []
    for ready in sorted((e for e in overlap if e.get("event") == "target_ready"),
                        key=lambda e: e.get("ready", 0)):
        model, t_ready = ready.get("model"), ready.get("ready", 0)
        quiesce = next((e for e in sorted(overlap, key=lambda x: x.get("time", 0))
                         if e.get("event") == "source_quiesce"
                         and e.get("model") == model and e.get("time", 0) >= t_ready
                         and e.get("was_serving") is True), None)
        if quiesce is None:
            continue
        commit = next((e for e in sorted(overlap, key=lambda x: x.get("ready", 0))
                        if e.get("event") == "target_commit"
                        and e.get("model") == model
                        and e.get("ready", 0) >= quiesce["time"]), None)
        if commit is None:
            continue
        sequences.append({
            "model": model, "target_ready": t_ready,
            "source_quiesce": quiesce["time"], "target_commit": commit["ready"],
            "source_was_serving": True,
            "exposed_downtime_s": commit["ready"] - quiesce["time"],
        })

    prepare = [a for a in actions if a.get("phase") == "prepare" and a.get("success")]
    serving_during_prepare = sum(bool(a.get("source_completed_during_prepare"))
                                 for a in prepare)
    resumes = [e for e in kv if e.get("event") == "resume"]
    injects = [e for e in kv if e.get("event") == "inject"]
    injected = sum(int(e.get("requests_injected", 0)) for e in injects)
    continuation = sum(int(e.get("output_tokens", 0)) > 0
                       and int(e.get("resume_extend_len", -1)) == 1 for e in resumes)

    transfers = []
    transfer_file = out / "kv_transfers.jsonl"
    if transfer_file.exists():
        for line in transfer_file.read_text(errors="replace").splitlines():
            try:
                transfers.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    p2p = sum(e.get("transfer_path") == "gpu-to-gpu-p2p" for e in transfers)

    completed = aborted = 0
    benchmark_files = []
    for path in sorted(out.glob("*_e2e_*.json")):
        try:
            value = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(value, list):
            completed += sum(bool(x.get("success", True)) for x in value if isinstance(x, dict))
            aborted += sum(not bool(x.get("success", True)) for x in value if isinstance(x, dict))
        elif isinstance(value, dict):
            completed += int(value.get("completed", 0))
            aborted += int(value.get("aborted", 0))
        benchmark_files.append(path.name)

    fatal_patterns = (
        "CUDA out of memory", "OutOfMemoryError", "NCCL error", "CUDA error",
        "Segmentation fault", "Aborted", "capture failed", "fetch timed out",
        "stash readiness timed out", "resume failed", "inject failed",
        "inject stage failed", "rebuild failed", "target rollback failed",
        "commit without matching prepare",
    )
    fatal = {p: blob.count(p) for p in fatal_patterns if blob.count(p)}
    failures = []
    if not sequences:
        failures.append("no ready -> serving-source quiesce -> commit sequence")
    if not prepare:
        failures.append("no successful prepare action")
    if serving_during_prepare <= 0:
        failures.append("no source request completed while target was preparing")
    if not resumes or injected != len(resumes):
        failures.append(f"KV inject/resume mismatch {injected}/{len(resumes)}")
    if continuation <= 0:
        failures.append("no migrated decode continuation with preserved output")
    if not transfers or p2p != len(transfers):
        failures.append(f"KV transfers are not all GPU P2P: {p2p}/{len(transfers)}")
    if completed <= 0 or aborted:
        failures.append(f"benchmark completed/aborted={completed}/{aborted}")
    if fatal:
        failures.append("fatal runtime errors observed")

    downtime = [s["exposed_downtime_s"] for s in sequences]
    summary = {
        "status": "PASS" if not failures else "FAIL",
        "benchmark_files": benchmark_files,
        "completed": completed, "aborted": aborted,
        "successful_prepare_actions": len(prepare),
        "source_completed_during_prepare": serving_during_prepare,
        "sequences": sequences,
        "exposed_downtime_s": {
            "count": len(downtime),
            "mean": statistics.mean(downtime) if downtime else None,
            "p50": percentile(downtime, 50), "p95": percentile(downtime, 95),
            "max": max(downtime) if downtime else None,
        },
        "resume_events": len(resumes), "continuation_events": continuation,
        "requests_injected": injected, "kv_transfer_records": len(transfers),
        "p2p_transfer_records": p2p, "fatal_errors": fatal,
        "failures": failures,
    }
    (out / "OVERLAP_GATE.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "OVERLAP_GATE.md").write_text(
        "# V6 overlap correctness gate\n\n"
        f"Status: **{summary['status']}**\n\n"
        f"- complete overlap sequences: {len(sequences)}\n"
        f"- source completions during prepare: {serving_during_prepare}\n"
        f"- resumed continuations: {continuation}/{len(resumes)}\n"
        f"- KV P2P transfers: {p2p}/{len(transfers)}\n"
        f"- completed/aborted: {completed}/{aborted}\n"
        f"- exposed downtime: `{summary['exposed_downtime_s']}`\n"
        f"- failures: `{failures}`\n")
    print(json.dumps(summary, indent=2))
    raise SystemExit(0 if not failures else 1)


if __name__ == "__main__":
    main()
