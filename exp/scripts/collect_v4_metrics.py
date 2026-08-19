#!/usr/bin/env python3
"""Turn the v4 study's run directories into raw tables and summaries.

Raw first: every request, every control action, every weight transfer and every
GPU sample is written out per run before anything is averaged, because an
aggregate cannot be un-aggregated later.  The summary tables are derived from
those files, not the other way round.
"""
import argparse
import csv
import glob
import json
import math
import os
import re
import shutil
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from collect_v2_metrics import join_trace, load_requests   # noqa: E402


def pct(xs, q):
    return float(np.percentile(xs, q)) if len(xs) else float("nan")


def stats(xs, scale=1.0):
    xs = [x * scale for x in xs if x is not None and not math.isnan(x)]
    if not xs:
        return {k: float("nan") for k in ("mean", "p50", "p95", "p99", "max")}
    return {"mean": float(np.mean(xs)), "p50": pct(xs, 50), "p95": pct(xs, 95),
            "p99": pct(xs, 99), "max": float(max(xs))}


# ----------------------------------------------------------------- raw pulls
def read_actions(run_dir):
    """Per-action timing records emitted by the patched controller."""
    out = []
    for path in glob.glob(os.path.join(run_dir, "server-logs", "*global_controller*")):
        with open(path, errors="replace") as fh:
            for line in fh:
                idx = line.find("[PAPER-ACTION-V4] ")
                if idx < 0:
                    continue
                try:
                    out.append(json.loads(line[idx + len("[PAPER-ACTION-V4] "):]))
                except json.JSONDecodeError:
                    pass
    return sorted(out, key=lambda r: r.get("start", 0))


def read_alg1(run_dir):
    """Algorithm 1 decisions, whichever marker this arm emits."""
    out = []
    for path in glob.glob(os.path.join(run_dir, "server-logs", "*global_controller*")):
        with open(path, errors="replace") as fh:
            for line in fh:
                for marker in ("[PAPER-ALG1-V4] ", "[PAPER-ALG1-V3] "):
                    idx = line.find(marker)
                    if idx < 0:
                        continue
                    try:
                        out.append(json.loads(line[idx + len(marker):]))
                    except json.JSONDecodeError:
                        pass
                    break
    return out


def read_transfers(run_dir):
    path = os.path.join(run_dir, "weight_transfers.jsonl")
    if not os.path.exists(path):
        return []
    out = []
    with open(path, errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def pair_migrations(actions, transfers, alg1):
    """Reconstruct migrations from the action stream.

    A migration is an Activate of a model on one GPU followed by a Deactivate
    of the same model on another.  Ordering decides the two headline numbers:
    with target-first the source is still serving while the target loads, so
    downtime is zero and latency is the load; with the prototype's
    deactivate-first ordering the model is resident nowhere in between and the
    gap between the two actions is real downtime.
    """
    by_model = defaultdict(list)
    for act in actions:
        if act.get("model"):
            by_model[act["model"]].append(act)

    weight_by_model = defaultdict(list)
    for rec in transfers:
        tag = rec.get("tag") or ""
        model = tag.split("|")[0] if "|" in tag else None
        weight_by_model[model].append(rec)

    decisions = [a for a in alg1 if a.get("migration_decision") == "MIGRATE"]
    migrations = []
    for model, acts in by_model.items():
        activates = [a for a in acts if a["action"] == "ActivateAction"]
        deactivates = [a for a in acts if a["action"] == "DeactivateAction"]
        for act in activates:
            # The Deactivate that retires a *different* GPU's copy, nearest in time.
            partner = None
            for dea in deactivates:
                if dea.get("gpu_id") == act.get("gpu_id"):
                    continue
                if abs(dea["start"] - act["start"]) > 900:
                    continue
                if partner is None or abs(dea["start"] - act["start"]) < abs(partner["start"] - act["start"]):
                    partner = dea
            if partner is None:
                continue
            target_first = partner["start"] >= act["start"]
            start = min(act["start"], partner["start"])
            end = max(act["end"], partner["end"])
            downtime = 0.0 if target_first else max(0.0, act["end"] - partner["start"])
            # Weight transfer for this activation, if one was recorded.
            wrec = None
            best = None
            for rec in weight_by_model.get(model, []):
                if rec.get("target_gpu") != act.get("gpu_id"):
                    continue
                if best is None:
                    best, wrec = 0, rec
            migrations.append({
                "model": model,
                "source_gpu": partner.get("gpu_id"),
                "target_gpu": act.get("gpu_id"),
                "migration_start": start,
                "target_ready_time": act["end"],
                "migration_end": end,
                "migration_latency_s": act["end"] - start,
                "service_downtime_s": downtime,
                "migration_total_s": end - start,
                "ordering": "target-first" if target_first else "source-first",
                "weight_bytes": (wrec or {}).get("payload_bytes", 0),
                "kv_bytes": 0,
                "total_bytes": (wrec or {}).get("payload_bytes", 0),
                "transfer_path": (wrec or {}).get("transfer_path"),
                "effective_gbps": (wrec or {}).get("payload_gbps"),
                "success": bool(act.get("success")) and bool(partner.get("success")),
            })
    migrations.sort(key=lambda m: m["migration_start"])
    for i, mig in enumerate(migrations):
        mig["migration_id"] = i
    # Attach the KVPR the planner saw for the nearest decision.
    for mig in migrations:
        near = min(decisions, key=lambda d: abs(d.get("timestamp", 0) - mig["migration_start"]),
                   default=None) if decisions else None
        if near:
            mig["peak_kvpr_before"] = near.get("peak_kvpr")
            mig["peak_kvpr_after"] = (near.get("candidate") or {}).get("peak_kvpr_after")
            mig["tau"] = near.get("tau")
            mig["alg1_cycle"] = near.get("cycle")
    return migrations


def read_gpu_timeline(run_dir):
    path = os.path.join(run_dir, "server-logs", "gpu_timeline.txt")
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, errors="replace") as fh:
        for line in fh:
            parts = line.strip().split(" ", 1)
            if len(parts) != 2:
                continue
            try:
                ts = float(parts[0])
            except ValueError:
                continue
            for entry in parts[1].split(";"):
                entry = entry.strip()
                if not entry:
                    continue
                fields = [f.strip() for f in entry.split(",")]
                if len(fields) < 3:
                    continue
                try:
                    rows.append({"timestamp": ts, "gpu_id": int(fields[0]),
                                 "memory_used_mib": int(fields[1]),
                                 "gpu_utilization": int(fields[2])})
                except ValueError:
                    pass
    return rows


# ------------------------------------------------------------------ per run
def process_run(run_dir, base, system, workload, rate, seed, slo_base,
                ttft_scale, tpot_scale, warmup, measure, trace_dir):
    metrics_path = os.path.join(run_dir, "metrics.json")
    metrics = json.load(open(metrics_path)) if os.path.exists(metrics_path) else {}

    trace = os.path.join(trace_dir, f"{workload}_r{rate}_s{seed}.pkl")
    reqs = load_requests(run_dir)
    if os.path.exists(trace):
        reqs = join_trace(reqs, trace)

    base_slo = json.load(open(slo_base))
    slo_ttft = {m: v["ttft"] * ttft_scale for m, v in base_slo.items()}
    slo_tpot = {m: v["tpot"] * tpot_scale for m, v in base_slo.items()}

    arrivals = [r["arrival_time"] for r in reqs if r.get("arrival_time") is not None]
    t0 = min(arrivals) if arrivals else 0.0
    lo, hi = t0 + warmup, t0 + warmup + measure

    tag = f"{system}_{workload}_r{rate}_s{seed}"
    # ---- per-request raw CSV
    req_path = Path(base) / "raw/requests" / f"{tag}.csv"
    req_path.parent.mkdir(parents=True, exist_ok=True)
    window = []
    with open(req_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["request_id", "model", "arrival_time", "completion_time",
                         "prompt_tokens", "output_tokens", "ttft_s", "tpot_s", "e2e_s",
                         "ttft_slo_s", "tpot_slo_s", "ttft_slo_met", "tpot_slo_met",
                         "joint_slo_met", "success", "in_measurement_window"])
        for r in reqs:
            arrival = r.get("arrival_time")
            e2e = r.get("latency")
            in_win = arrival is not None and lo <= arrival < hi
            ttft, tpot = r.get("ttft"), r.get("tpot")
            st, sp = slo_ttft.get(r.get("model")), slo_tpot.get(r.get("model"))
            ttft_ok = ttft is not None and st is not None and ttft <= st
            tpot_ok = tpot is not None and sp is not None and tpot <= sp
            writer.writerow([
                r.get("request_id"), r.get("model"), arrival,
                (arrival + e2e) if (arrival is not None and e2e is not None) else None,
                r.get("prompt_len"), r.get("output_len"), ttft, tpot, e2e, st, sp,
                int(bool(ttft_ok)), int(bool(tpot_ok)),
                int(bool(ttft_ok and tpot_ok)), int(bool(r.get("success"))), int(in_win)])
            if in_win:
                window.append(r)

    ok = [r for r in window if r.get("success")]
    lat = {
        "ttft": stats([r.get("ttft") for r in ok], 1000.0),
        "tpot": stats([r.get("tpot") for r in ok], 1000.0),
        "e2e": stats([r.get("latency") for r in ok], 1000.0),
    }
    joint = [r for r in ok
             if r.get("ttft") is not None and r.get("tpot") is not None
             and r["ttft"] <= slo_ttft.get(r["model"], float("inf"))
             and r["tpot"] <= slo_tpot.get(r["model"], float("inf"))]
    ttft_hit = [r for r in ok if r.get("ttft") is not None
                and r["ttft"] <= slo_ttft.get(r["model"], float("inf"))]
    tpot_hit = [r for r in ok if r.get("tpot") is not None
                and r["tpot"] <= slo_tpot.get(r["model"], float("inf"))]

    # ---- migrations
    actions = read_actions(run_dir)
    alg1 = read_alg1(run_dir)
    transfers = read_transfers(run_dir)
    migrations = pair_migrations(actions, transfers, alg1)
    mig_path = Path(base) / "raw/migrations" / f"{tag}.csv"
    mig_path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["migration_id", "model", "source_gpu", "target_gpu", "migration_start",
            "target_ready_time", "migration_end", "migration_latency_s",
            "service_downtime_s", "migration_total_s", "ordering", "weight_bytes",
            "kv_bytes", "total_bytes", "transfer_path", "effective_gbps", "success",
            "peak_kvpr_before", "peak_kvpr_after", "tau", "alg1_cycle"]
    with open(mig_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(migrations)

    # ---- scheduler / Algorithm 1 decisions
    sch_path = Path(base) / "raw/scheduler" / f"{tag}_alg1.jsonl"
    sch_path.parent.mkdir(parents=True, exist_ok=True)
    with open(sch_path, "w") as fh:
        for rec in alg1:
            fh.write(json.dumps(rec) + "\n")
    act_path = Path(base) / "raw/scheduler" / f"{tag}_actions.jsonl"
    with open(act_path, "w") as fh:
        for rec in actions:
            fh.write(json.dumps(rec) + "\n")
    if transfers:
        tr_path = Path(base) / "raw/migrations" / f"{tag}_weight_transfers.jsonl"
        with open(tr_path, "w") as fh:
            for rec in transfers:
                fh.write(json.dumps(rec) + "\n")

    # ---- gpu time series
    gpu_rows = read_gpu_timeline(run_dir)
    gpu_path = Path(base) / "raw/gpu_metrics" / f"{tag}.csv"
    gpu_path.parent.mkdir(parents=True, exist_ok=True)
    with open(gpu_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["timestamp", "gpu_id", "gpu_utilization", "memory_used_mib"])
        for row in gpu_rows:
            writer.writerow([row["timestamp"], row["gpu_id"],
                             row["gpu_utilization"], row["memory_used_mib"]])

    ok_migs = [m for m in migrations if m["success"]]
    sched = metrics.get("scheduler", {}) if isinstance(metrics.get("scheduler"), dict) else metrics
    row = {
        "implementation": system, "workload": workload,
        "request_rate": rate, "seed": seed,
        "offered_rate": len(window) / measure,
        "achieved_throughput": len(ok) / measure,
        "goodput": len(joint) / measure,
        "total_requests": len(window),
        "completed_requests": len(ok),
        "failed_requests": len(window) - len(ok),
        "joint_slo": len(joint) / len(window) if window else float("nan"),
        "ttft_slo": len(ttft_hit) / len(window) if window else float("nan"),
        "tpot_slo": len(tpot_hit) / len(window) if window else float("nan"),
    }
    for name in ("ttft", "tpot", "e2e"):
        for key in ("mean", "p50", "p95", "p99", "max"):
            row[f"{name}_{key}"] = lat[name][key]
    lat_ms = [m["migration_latency_s"] * 1000 for m in ok_migs]
    down_ms = [m["service_downtime_s"] * 1000 for m in ok_migs]
    bw = [m["effective_gbps"] for m in ok_migs if m.get("effective_gbps")]
    mstats, dstats = stats(lat_ms), stats(down_ms)
    row.update({
        "migration_count": len(migrations),
        "migration_success": len(ok_migs),
        "migration_failed": len(migrations) - len(ok_migs),
        "migration_latency_mean": mstats["mean"], "migration_latency_p50": mstats["p50"],
        "migration_latency_p95": mstats["p95"], "migration_latency_p99": mstats["p99"],
        "migration_latency_max": mstats["max"],
        "migration_total_time": sum(m["migration_total_s"] for m in ok_migs),
        "migration_weight_bytes": sum(m["weight_bytes"] for m in ok_migs),
        "migration_kv_bytes": sum(m["kv_bytes"] for m in ok_migs),
        "migration_total_bytes": sum(m["total_bytes"] for m in ok_migs),
        "migration_bandwidth": float(np.mean(bw)) if bw else float("nan"),
        "migration_bandwidth_p50": pct(bw, 50) if bw else float("nan"),
        "migration_paths": ";".join(sorted({m["transfer_path"] for m in ok_migs
                                            if m.get("transfer_path")})) or "",
        "service_downtime_mean": dstats["mean"], "service_downtime_p50": dstats["p50"],
        "service_downtime_p95": dstats["p95"], "service_downtime_p99": dstats["p99"],
        "admitted_requests": sched.get("alg2_selected", ""),
        "deferred_requests": sched.get("alg2_deferred", ""),
        "requeued_requests": sched.get("alg2_requeued", ""),
        "shed_requests": sched.get("alg2_late_dispatched", ""),
        "deadline_violations": sched.get("alg2_late_dispatched", ""),
        "alg2_rounds": sched.get("alg2_rounds", ""),
        "alg2_eligible": sched.get("alg2_eligible", ""),
        "alg2_pathological_rounds": sched.get("alg2_pathological_rounds", ""),
        "queue_length_mean": sched.get("mean_queue_length", ""),
        "queue_length_max": sched.get("max_queue_length", ""),
        "alg1_cycles": len(alg1),
        "alg1_placement_decisions": sum(len(a.get("line8", [])) for a in alg1),
        "alg1_migrate_decisions": sum(1 for a in alg1
                                      if a.get("migration_decision") == "MIGRATE"),
        "alg1_suppressed_by_tau": (alg1[-1].get("audit_totals", {}) or {}).get(
            "suppressed_by_tau", "") if alg1 else "",
        "alg1_rejected_by_memory": (alg1[-1].get("audit_totals", {}) or {}).get(
            "rejected_by_memory", "") if alg1 else "",
        "alg1_convergence_gap_mean": float(np.mean(
            [a["convergence_gap"] for a in alg1 if "convergence_gap" in a]))
            if any("convergence_gap" in a for a in alg1) else "",
        "weight_transfers": len(transfers),
        "p2p_weight_transfers": sum(1 for t in transfers
                                    if t.get("transfer_path") == "gpu-to-gpu-p2p"),
    })
    per_model = {}
    for model in sorted({r["model"] for r in window}):
        sub = [r for r in window if r["model"] == model]
        sok = [r for r in sub if r.get("success")]
        sj = [r for r in sok
              if r.get("ttft") is not None and r.get("tpot") is not None
              and r["ttft"] <= slo_ttft.get(model, float("inf"))
              and r["tpot"] <= slo_tpot.get(model, float("inf"))]
        per_model[model] = {
            "requests": len(sub), "completed": len(sok),
            "goodput": len(sj) / measure,
            "joint_slo": len(sj) / len(sub) if sub else float("nan"),
            "ttft_p50_ms": pct([r["ttft"] * 1000 for r in sok if r.get("ttft")], 50),
            "ttft_p95_ms": pct([r["ttft"] * 1000 for r in sok if r.get("ttft")], 95),
            "ttft_p99_ms": pct([r["ttft"] * 1000 for r in sok if r.get("ttft")], 99),
            "tpot_p50_ms": pct([r["tpot"] * 1000 for r in sok if r.get("tpot")], 50),
            "tpot_p95_ms": pct([r["tpot"] * 1000 for r in sok if r.get("tpot")], 95),
            "tpot_p99_ms": pct([r["tpot"] * 1000 for r in sok if r.get("tpot")], 99),
            "e2e_p50_ms": pct([r["latency"] * 1000 for r in sok if r.get("latency")], 50),
            "e2e_p95_ms": pct([r["latency"] * 1000 for r in sok if r.get("latency")], 95),
            "e2e_p99_ms": pct([r["latency"] * 1000 for r in sok if r.get("latency")], 99),
        }
    return row, per_model, migrations


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--slo-base", default=None)
    ap.add_argument("--trace-dir", default="/workspace/prism-exp/exp/workloads/paper-faithful-v4")
    ap.add_argument("--ttft-scale", type=float, default=5.0)
    ap.add_argument("--tpot-scale", type=float, default=3.0)
    ap.add_argument("--warmup", type=float, default=60.0)
    ap.add_argument("--measure", type=float, default=300.0)
    a = ap.parse_args()
    base = Path(a.base)
    slo_base = a.slo_base or "/workspace/prism-exp/exp/configs/v2/slo_base.json"

    rows, model_rows, all_migs = [], [], []
    pattern = str(base / "raw/*/*/rate_*/seed_*/DONE")
    for done in sorted(glob.glob(pattern)):
        run_dir = os.path.dirname(done)
        parts = Path(run_dir).parts
        seed = int(parts[-1].split("_")[1])
        rate = int(parts[-2].split("_")[1])
        workload = parts[-3]
        system = parts[-4]
        try:
            row, per_model, migs = process_run(
                run_dir, base, system, workload, rate, seed, slo_base,
                a.ttft_scale, a.tpot_scale, a.warmup, a.measure, a.trace_dir)
        except (Exception, SystemExit) as exc:                  # noqa: BLE001
            # load_requests raises SystemExit when a dump is missing, and
            # SystemExit is not an Exception -- without naming it here one bad
            # run directory would take the whole collection down.
            print(f"WARN {run_dir}: {type(exc).__name__}: {exc}")
            continue
        rows.append(row)
        all_migs.extend({**m, "implementation": system, "workload": workload,
                         "request_rate": rate, "seed": seed} for m in migs)
        for model, stat in per_model.items():
            model_rows.append({"implementation": system, "workload": workload,
                               "request_rate": rate, "seed": seed,
                               "model": model, **stat})
        print(f"collected {system}/{workload}/{rate}/{seed}: "
              f"goodput={row['goodput']:.3f} joint={row['joint_slo']:.3f} "
              f"migrations={row['migration_count']}")

    if not rows:
        print("no completed runs yet")
        return

    def write_csv(path, records, cols=None):
        cols = cols or list(records[0].keys())
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(records)

    rows.sort(key=lambda r: (r["implementation"], r["workload"], r["request_rate"], r["seed"]))
    write_csv(base / "summary.csv", rows)
    write_csv(base / "latency_summary.csv", rows, cols=[
        "implementation", "workload", "request_rate", "seed",
        *[f"{n}_{k}" for n in ("ttft", "tpot", "e2e")
          for k in ("mean", "p50", "p95", "p99", "max")]])
    write_csv(base / "migration_summary.csv", rows, cols=[
        "implementation", "workload", "request_rate", "seed", "migration_count",
        "migration_success", "migration_failed",
        "migration_latency_mean", "migration_latency_p50", "migration_latency_p95",
        "migration_latency_p99", "migration_latency_max", "migration_total_time",
        "migration_weight_bytes", "migration_kv_bytes", "migration_total_bytes",
        "migration_bandwidth", "migration_bandwidth_p50", "migration_paths",
        "service_downtime_mean", "service_downtime_p50", "service_downtime_p95",
        "service_downtime_p99", "weight_transfers", "p2p_weight_transfers"])
    if model_rows:
        write_csv(base / "model_summary.csv", model_rows)
    if all_migs:
        write_csv(base / "raw/migrations/all_migrations.csv", all_migs)

    # ---- 3-seed aggregation: mean and standard deviation
    agg = []
    groups = defaultdict(list)
    for row in rows:
        groups[(row["implementation"], row["workload"], row["request_rate"])].append(row)
    numeric = [k for k, v in rows[0].items()
               if isinstance(v, (int, float)) and k not in ("seed", "request_rate")]
    for (system, workload, rate), members in sorted(groups.items()):
        entry = {"implementation": system, "workload": workload,
                 "request_rate": rate, "seeds": len(members),
                 "seed_list": ";".join(str(m["seed"]) for m in sorted(members, key=lambda x: x["seed"]))}
        for key in numeric:
            vals = [m[key] for m in members
                    if isinstance(m.get(key), (int, float)) and not math.isnan(m[key])]
            entry[f"{key}_mean"] = float(np.mean(vals)) if vals else float("nan")
            entry[f"{key}_std"] = float(statistics.stdev(vals)) if len(vals) > 1 else 0.0
        agg.append(entry)
    write_csv(base / "aggregated/summary_by_condition.csv", agg)
    print(f"\nwrote summary.csv ({len(rows)} runs), aggregated over "
          f"{len(agg)} conditions")


if __name__ == "__main__":
    main()
