#!/usr/bin/env python3
"""Build one row per D2 migration and phase-duration statistics."""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def marked(path, marker):
    records = []
    if not path.exists():
        return records
    for line in path.read_text(errors="replace").splitlines():
        if marker not in line:
            continue
        try:
            records.append(json.loads(line.split(marker, 1)[1]))
        except (IndexError, json.JSONDecodeError):
            pass
    return records


def jsonl(path):
    records = []
    if not path.exists():
        return records
    for line in path.read_text(errors="replace").splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return records


def first(records, predicate, timestamp):
    candidates = [r for r in records if predicate(r) and timestamp(r) is not None]
    return min(candidates, key=timestamp) if candidates else None


def duration(end, start):
    if end is None or start is None:
        return None
    value = end - start
    return value if value >= -1e-6 else None


def stats(values):
    values = [v for v in values if v is not None and math.isfinite(v)]
    if not values:
        return {"n": 0, "mean": None, "p50": None, "p95": None,
                "p99": None, "max": None}
    return {
        "n": len(values),
        "mean": float(np.mean(values)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--summary", type=Path, required=True)
    args = ap.parse_args()

    logs = args.run / "server-logs"
    gc = logs / "server.log.global_controller.log"
    server = logs / "server.log"
    alg1 = marked(gc, "[PAPER-ALG1-V4] ")
    actions = marked(gc, "[PAPER-ACTION-V4] ")
    gc_timeline = marked(gc, "[PAPER-MIGRATION-TIMELINE] ")
    overlap = marked(server, "[PAPER-OVERLAP-V6] ")
    kv_events = marked(server, "[PAPER-KV-V6] ")
    engine_timeline = marked(server, "[PAPER-MIGRATION-TIMELINE] ")
    weights = jsonl(args.run / "weight_transfers.jsonl")
    kv_transfers = jsonl(args.run / "kv_transfers.jsonl")

    decisions = sorted(
        (r for r in alg1 if r.get("migration_decision") == "MIGRATE"),
        key=lambda r: r["timestamp"],
    )
    rows = []
    for index, decision in enumerate(decisions, 1):
        model = decision["candidate"]["model"]
        source_gpu = decision["candidate"]["from"]
        target_gpu = decision["candidate"]["to"]
        decision_ts = decision["timestamp"]

        prepare = first(
            actions,
            lambda r: r.get("action") == "ActivateAction"
            and r.get("phase") == "prepare" and r.get("model") == model
            and r.get("start", 0) >= decision_ts,
            lambda r: r.get("start"),
        )
        prepare_start = prepare.get("start") if prepare else None
        prepare_end = prepare.get("end") if prepare else None
        ready = first(
            overlap,
            lambda r: r.get("event") == "target_ready" and r.get("model") == model
            and r.get("ready", 0) >= (prepare_start or decision_ts),
            lambda r: r.get("ready"),
        )
        target_ready = ready.get("ready") if ready else None

        quiesce_action = first(
            actions,
            lambda r: r.get("action") == "DeactivateAction"
            and r.get("model") == model
            and r.get("start", 0) >= (prepare_end or decision_ts),
            lambda r: r.get("start"),
        )
        quiesce_request = quiesce_action.get("start") if quiesce_action else None
        quiesce = first(
            overlap,
            lambda r: r.get("event") == "source_quiesce" and r.get("model") == model
            and r.get("time", 0) >= (quiesce_request or decision_ts),
            lambda r: r.get("time"),
        )
        quiesce_begin = quiesce.get("time") if quiesce else None
        drain = first(
            engine_timeline,
            lambda r: r.get("event") == "request_drain" and r.get("model") == model
            and r.get("start", 0) >= (quiesce_begin or decision_ts),
            lambda r: r.get("start"),
        )
        stash = first(
            kv_events,
            lambda r: r.get("event") == "stash"
            and r.get("logical_model") == model
            and r.get("start", 0) >= (quiesce_begin or decision_ts),
            lambda r: r.get("start"),
        )
        release = first(
            engine_timeline,
            lambda r: r.get("event") == "source_release" and r.get("model") == model
            and r.get("time", 0) >= (quiesce_begin or decision_ts),
            lambda r: r.get("time"),
        )
        weight = first(
            weights,
            lambda r: r.get("source") == "gpu"
            and r.get("target_gpu") == target_gpu
            and r.get("start_time", 0) >= (prepare_start or decision_ts),
            lambda r: r.get("start_time"),
        )
        commit = first(
            actions,
            lambda r: r.get("action") == "ActivateAction"
            and r.get("phase") == "commit" and r.get("model") == model
            and r.get("start", 0) >= (quiesce_begin or decision_ts),
            lambda r: r.get("start"),
        )
        commit_start = commit.get("start") if commit else None
        inject = first(
            engine_timeline,
            lambda r: r.get("event") == "target_inject" and r.get("model") == model
            and r.get("start", 0) >= (commit_start or decision_ts),
            lambda r: r.get("start"),
        )
        kv_transfer = first(
            kv_transfers,
            lambda r: str(r.get("tag", "")).startswith(model + "|")
            and r.get("target_gpu") == target_gpu
            and r.get("start_time", 0) >= (commit_start or decision_ts),
            lambda r: r.get("start_time"),
        )
        routing = first(
            gc_timeline,
            lambda r: r.get("event") == "routing_switch" and r.get("model") == model
            and r.get("start", 0) >= (commit_start or decision_ts),
            lambda r: r.get("start"),
        )
        first_request = first(
            engine_timeline,
            lambda r: r.get("event") == "first_request_on_target"
            and r.get("model") == model
            and r.get("time", 0) >= ((routing or {}).get("end") or commit_start or decision_ts),
            lambda r: r.get("time"),
        )
        first_decode = first(
            engine_timeline,
            lambda r: r.get("event") == "first_decode_on_target"
            and r.get("model") == model
            and r.get("time", 0) >= ((first_request or {}).get("time") or commit_start or decision_ts),
            lambda r: r.get("time"),
        )

        row = {
            "migration_id": index,
            "model": model,
            "source_gpu": source_gpu,
            "target_gpu": target_gpu,
            "migration_decision": decision_ts,
            "target_prepare_start": prepare_start,
            "target_prepare_end": prepare_end,
            "target_ready": target_ready,
            "source_quiesce_request": quiesce_request,
            "source_quiesce_begin": quiesce_begin,
            "request_drain_start": (drain or {}).get("start"),
            "request_drain_end": (drain or {}).get("end"),
            "kv_stash_start": (stash or {}).get("start"),
            "kv_stash_end": (stash or {}).get("end"),
            "weight_transfer_start": (weight or {}).get("start_time"),
            "weight_transfer_end": (weight or {}).get("end_time"),
            "kv_transfer_start": (kv_transfer or {}).get("start_time"),
            "kv_transfer_end": (kv_transfer or {}).get("end_time"),
            "target_inject_start": (inject or {}).get("start"),
            "target_inject_end": (inject or {}).get("end"),
            "routing_switch_start": (routing or {}).get("start"),
            "routing_switch_end": (routing or {}).get("end"),
            "source_release": (release or {}).get("time"),
            "first_request_on_target": (first_request or {}).get("time"),
            "first_decode_on_target": (first_decode or {}).get("time"),
        }
        inject_total = duration(row["target_inject_end"], row["target_inject_start"])
        kv_transfer_duration = duration(row["kv_transfer_end"], row["kv_transfer_start"])
        inject_exclusive = (
            max(0.0, inject_total - kv_transfer_duration)
            if inject_total is not None and kv_transfer_duration is not None
            else inject_total
        )
        row.update({
            "target_prepare_s": duration(row["target_ready"], row["target_prepare_start"]),
            "target_ready_to_quiesce_s": duration(row["source_quiesce_begin"], row["target_ready"]),
            "quiesce_control_s": duration(row["source_quiesce_begin"], row["source_quiesce_request"]),
            "request_drain_s": duration(row["request_drain_end"], row["request_drain_start"]),
            "kv_stash_s": duration(row["kv_stash_end"], row["kv_stash_start"]),
            "weight_transfer_s": duration(row["weight_transfer_end"], row["weight_transfer_start"]),
            "kv_transfer_s": kv_transfer_duration,
            "target_inject_total_s": inject_total,
            "kv_inject_s": inject_exclusive,
            "routing_switch_s": duration(row["routing_switch_end"], row["routing_switch_start"]),
            "routing_to_first_request_s": duration(row["first_request_on_target"], row["routing_switch_end"]),
            "first_request_to_first_decode_s": duration(row["first_decode_on_target"], row["first_request_on_target"]),
            "exposed_downtime_s": duration(row["first_request_on_target"], row["source_quiesce_begin"]),
            "total_migration_wall_s": duration(row["first_decode_on_target"], row["migration_decision"]),
        })
        rows.append(row)

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else [
        "migration_id", "model", "source_gpu", "target_gpu",
        "migration_decision", "target_prepare_start", "target_prepare_end",
        "target_ready", "source_quiesce_request", "source_quiesce_begin",
        "request_drain_start", "request_drain_end", "kv_stash_start",
        "kv_stash_end", "weight_transfer_start", "weight_transfer_end",
        "kv_transfer_start", "kv_transfer_end", "target_inject_start",
        "target_inject_end", "routing_switch_start", "routing_switch_end",
        "source_release", "first_request_on_target", "first_decode_on_target",
    ]
    with args.csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    duration_fields = [
        "target_prepare_s", "target_ready_to_quiesce_s", "quiesce_control_s",
        "request_drain_s", "kv_stash_s", "weight_transfer_s", "kv_transfer_s",
        "target_inject_total_s", "kv_inject_s",
        "routing_switch_s", "routing_to_first_request_s",
        "first_request_to_first_decode_s", "exposed_downtime_s",
        "total_migration_wall_s",
    ]
    summary = {
        "number_of_migrations": len(rows),
        "phase_stats_seconds": {
            key: stats([row.get(key) for row in rows]) for key in duration_fields
        },
        "missing_timestamps": {
            key: sum(row.get(key) is None for row in rows)
            for key in fieldnames
            if key not in {"migration_id", "model", "source_gpu", "target_gpu"}
            and not key.endswith("_s")
        },
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
