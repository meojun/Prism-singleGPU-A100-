#!/usr/bin/env python3
"""Compare fixed D1 with D0 in the standard warmup-excluded sanity window."""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_sharegpt_trace import _Unpickler


def percentile(values):
    a = np.asarray([v for v in values if v is not None and math.isfinite(v)])
    if not len(a):
        return {"n": 0}
    return {
        "n": int(len(a)),
        "mean_ms": float(np.mean(a) * 1000),
        "p50_ms": float(np.percentile(a, 50) * 1000),
        "p95_ms": float(np.percentile(a, 95) * 1000),
        "p99_ms": float(np.percentile(a, 99) * 1000),
    }


def request_file(path):
    files = list((path / "requests").glob("*_output_requests.json"))
    if len(files) != 1:
        raise RuntimeError(f"expected one request dump under {path}: {files}")
    return files[0]


def metrics_file(path):
    files = list(path.glob("*_e2e_*rep.json"))
    if len(files) != 1:
        raise RuntimeError(f"expected one metrics dump under {path}: {files}")
    return files[0]


def delta(end, start):
    if end is None or start is None:
        return None
    value = end - start
    return value if value >= -1e-6 else None


def analyze_arm(path, trace, warmup, measure):
    outputs = json.load(request_file(path).open())
    raw_metrics = json.load(metrics_file(path).open())
    if len(outputs) != len(trace):
        raise RuntimeError(f"output/trace mismatch: {len(outputs)} != {len(trace)}")
    window = [
        out for req, out in zip(trace, outputs)
        if warmup <= float(req.arrival_time) < warmup + measure
    ]
    completed = [r for r in window if r.get("success")]
    ttft_hit = [r["ttft"] <= r["slo_ttft"] for r in completed]
    tpot_hit = [r["tpot"] <= r["slo_tpot"] for r in completed]
    joint = [a and b for a, b in zip(ttft_hit, tpot_hit)]

    stages = {
        "frontend_wait": [],
        "local_scheduler_wait": [],
        "engine_prefill_wait": [],
        "actual_prefill_service": [],
        "prefill_to_first_decode_wait": [],
        "ttft": [],
        "tpot": [],
        "e2e": [],
    }
    for r in completed:
        decodes = r.get("decode_timestamps") or []
        values = {
            "frontend_wait": delta(
                r.get("gpu_scheduler_queue_time"), r.get("arrival_time")
            ),
            "local_scheduler_wait": delta(
                r.get("gpu_scheduler_dispatch_time"),
                r.get("gpu_scheduler_queue_time"),
            ),
            "engine_prefill_wait": delta(
                r.get("out_queue_time"), r.get("gpu_scheduler_dispatch_time")
            ),
            "actual_prefill_service": delta(
                r.get("prefill_finish_time"), r.get("out_queue_time")
            ),
            "prefill_to_first_decode_wait": delta(
                decodes[0], r.get("prefill_finish_time")
            ) if decodes else None,
            "ttft": r.get("ttft"),
            "tpot": r.get("tpot"),
            "e2e": r.get("latency_server") or r.get("latency"),
        }
        for key, value in values.items():
            if value is not None:
                stages[key].append(value)

    return {
        "offered": len(window),
        "completed": len(completed),
        "unfinished": len(window) - len(completed),
        "rejected": 0,
        "aborted": len(window) - len(completed),
        "achieved_throughput_req_s": len(completed) / measure,
        "benchmark_drain_inclusive_throughput_req_s": raw_metrics[
            "request_throughput"
        ],
        "goodput_req_s": sum(joint) / measure,
        "ttft_slo": float(np.mean(ttft_hit)),
        "tpot_slo": float(np.mean(tpot_hit)),
        "joint_slo": float(np.mean(joint)),
        "latency_and_wait": {k: percentile(v) for k, v in stages.items()},
        "full_trace": {
            "offered": len(outputs),
            "completed": sum(bool(r.get("success")) for r in outputs),
            "aborted": raw_metrics.get("aborted"),
        },
    }


def runtime_audit(d1, csv_path=None):
    path = d1 / "server-logs/server.log.gpu_scheduler.log"
    by_gpu = {}
    for line in path.read_text(errors="replace").splitlines():
        marker = "[PAPER-ALG2-RUNTIME] "
        if marker not in line:
            continue
        rec = json.loads(line.split(marker, 1)[1])
        by_gpu.setdefault(rec["gpu_id"], []).append(rec)

    result = {}
    csv_rows = []
    for gpu, events in sorted(by_gpu.items()):
        dispatch = [
            (e["alg2_seq"], e["actual_rids"][0], e["actual_model"])
            for e in events if e["event"] == "dispatch"
        ]
        admissions = [
            (e["alg2_seq"], e["actual_rids"][-1], e["actual_model"])
            for e in events if e["event"] == "backend_admit"
        ]
        starts = [
            (e["alg2_seq"], e["actual_rids"][-1], e["actual_model"])
            for e in events if e["event"] == "prefill_start"
        ]
        active = set()
        max_outstanding = 0
        for event in events:
            if event["event"] == "dispatch":
                active.update(event["actual_rids"])
            elif event["event"] == "prefill_complete":
                active.difference_update(event["actual_rids"])
            max_outstanding = max(max_outstanding, len(active))
        result[str(gpu)] = {
            "dispatches": len(dispatch),
            "backend_admissions": len(admissions),
            "initial_prefill_starts": len(starts),
            "prefill_completions": sum(
                len(e["actual_rids"])
                for e in events if e["event"] == "prefill_complete"
            ),
            "dispatch_equals_backend_admission_order": dispatch == admissions,
            "dispatch_equals_prefill_start_order": dispatch == starts,
            "all_order_ok": all(e["order_ok"] for e in events),
            "max_outstanding_prefills": max_outstanding,
            "remaining_outstanding": len(active),
        }
        by_request = {}
        for event in events:
            for rid in event["actual_rids"]:
                row = by_request.setdefault(
                    rid,
                    {
                        "request_id": rid,
                        "model": event["actual_model"],
                        "gpu_id": gpu,
                        "alg2_seq": event["alg2_seq"],
                        "dispatch_ts": "",
                        "backend_admit_ts": "",
                        "prefill_start_ts": "",
                        "prefill_complete_ts": "",
                        "order_ok": True,
                    },
                )
                timestamp_field = {
                    "dispatch": "dispatch_ts",
                    "backend_admit": "backend_admit_ts",
                    "prefill_start": "prefill_start_ts",
                    "prefill_complete": "prefill_complete_ts",
                }.get(event["event"])
                if timestamp_field:
                    row[timestamp_field] = event["event_time"]
                row["order_ok"] = row["order_ok"] and event["order_ok"]
        csv_rows.extend(by_request.values())
    if csv_path is not None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "request_id", "model", "gpu_id", "alg2_seq", "dispatch_ts",
            "backend_admit_ts", "prefill_start_ts", "prefill_complete_ts",
            "order_ok",
        ]
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(sorted(csv_rows, key=lambda r: (r["gpu_id"], r["alg2_seq"])))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--d0", type=Path, required=True)
    parser.add_argument("--d1", type=Path, required=True)
    parser.add_argument("--warmup", type=float, default=60.0)
    parser.add_argument("--measure", type=float, default=60.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime-csv", type=Path)
    args = parser.parse_args()
    with args.trace.open("rb") as f:
        _, trace = _Unpickler(f).load()

    arms = {
        "D0": analyze_arm(args.d0, trace, args.warmup, args.measure),
        "D1_fixed": analyze_arm(args.d1, trace, args.warmup, args.measure),
    }
    result = {
        "window": {"warmup_s": args.warmup, "measure_s": args.measure},
        "arms": arms,
        "delta_D1_minus_D0": {
            key: arms["D1_fixed"][key] - arms["D0"][key]
            for key in (
                "achieved_throughput_req_s", "goodput_req_s", "ttft_slo",
                "tpot_slo", "joint_slo",
            )
        },
        "runtime_order_audit": runtime_audit(args.d1, args.runtime_csv),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
