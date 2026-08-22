#!/usr/bin/env python3
"""Turn the v4 study's run directories into raw tables and summaries.

Raw first.  Every request, every control action, every weight transfer and
every GPU sample is written out per run before anything is averaged, because an
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
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from collect_v2_metrics import join_trace, load_requests   # noqa: E402

PAPER_ARMS = ("paper-faithful-v3", "paper-faithful-v4", "paper-faithful-v6")


def pct(xs, q):
    return float(np.percentile(xs, q)) if len(xs) else float("nan")


def stats(xs, scale=1.0):
    xs = [x * scale for x in xs if x is not None and not math.isnan(x)]
    if not xs:
        return {k: float("nan") for k in ("mean", "p50", "p95", "p99", "max")}
    return {"mean": float(np.mean(xs)), "p50": pct(xs, 50), "p95": pct(xs, 95),
            "p99": pct(xs, 99), "max": float(max(xs))}


# ----------------------------------------------------------------- raw pulls
def _controller_logs(run_dir):
    return glob.glob(os.path.join(run_dir, "server-logs", "*global_controller*"))


def read_json_markers(run_dir, markers):
    out = []
    for path in _controller_logs(run_dir):
        with open(path, errors="replace") as fh:
            for line in fh:
                for marker in markers:
                    idx = line.find(marker)
                    if idx < 0:
                        continue
                    try:
                        out.append(json.loads(line[idx + len(marker):]))
                    except json.JSONDecodeError:
                        pass
                    break
    return out


def read_actions(run_dir):
    """Per-action timing records emitted by the patched controller."""
    return sorted(read_json_markers(run_dir, ["[PAPER-ACTION-V4] "]),
                  key=lambda r: r.get("start", 0))


def read_alg1(run_dir):
    """Algorithm 1 decisions, whichever marker this arm emits."""
    return read_json_markers(run_dir, ["[PAPER-ALG1-V4] ", "[PAPER-ALG1-V3] "])


def read_transfers(run_dir):
    path = os.path.join(run_dir, "weight_transfers.jsonl")
    out = []
    if not os.path.exists(path):
        return out
    with open(path, errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def read_kv_transfers(run_dir):
    path = os.path.join(run_dir, "kv_transfers.jsonl")
    if not os.path.exists(path):
        return []
    out = []
    with open(path, errors="replace") as fh:
        for line in fh:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


PROTO_MIGRATION_RE = re.compile(
    r"ACTION: deactivate (\S+) on GPU (\d+) and activate \S+ on GPU (\d+)\. "
    r"Reason: migrate model")


def read_migration_decisions(run_dir, alg1):
    """The migrations the controller actually decided on, from its own log.

    Counting Activate/Deactivate pairs instead over-counts badly: an idle
    eviction on one GPU and an unrelated activation on the other look exactly
    like a move.  On one prototype run the controller decided 2 migrations and
    logged 92 idle evictions, and pair-counting reported 4 -- two of them with
    25 s and 30 s "latencies" that were really evict-then-reactivate.  The
    decision log is the authority for *whether* a migration happened; the
    action records supply *how long* it took.
    """
    out = []
    for a in alg1:
        if a.get("migration_decision") != "MIGRATE":
            continue
        cand = a.get("candidate") or {}
        if cand.get("model") is None:
            continue
        out.append({"timestamp": a.get("timestamp"), "model": cand["model"],
                    "source_gpu": cand.get("from"), "target_gpu": cand.get("to"),
                    "peak_kvpr_before": a.get("peak_kvpr"),
                    "peak_kvpr_after": cand.get("peak_kvpr_after"),
                    "tau": a.get("tau"), "alg1_cycle": a.get("cycle")})
    for path in _controller_logs(run_dir):
        with open(path, errors="replace") as fh:
            for line in fh:
                m = PROTO_MIGRATION_RE.search(line)
                if m:
                    out.append({"timestamp": None, "model": m.group(1),
                                "source_gpu": int(m.group(2)),
                                "target_gpu": int(m.group(3))})
    return out


def pair_migrations(actions, transfers, kv_transfers, alg1, run_dir, slot_to_path, overlap,
                    window_s=120.0):
    """Attach action timings and byte counts to each decided migration.

    ``overlap`` says whether this arm ran with the readiness barrier.  Without
    it the control handler returns as soon as the request has been *submitted*,
    so an action lasts ~15 ms no matter how long the weights actually take.
    Every row carries that flag, so a submission time is never read as a
    migration latency.
    """
    decisions = read_migration_decisions(run_dir, alg1)
    acts = defaultdict(list)
    for a in actions:
        if a.get("model"):
            acts[(a["model"], a["action"], a.get("gpu_id"))].append(a)
    for v in acts.values():
        v.sort(key=lambda a: a["start"])

    by_path = defaultdict(list)
    for rec in transfers:
        tag = rec.get("tag") or ""
        by_path[tag.split("|")[0] if "|" in tag else None].append(rec)

    used, migrations = set(), []
    for dec in decisions:
        model, src, dst = dec["model"], dec["source_gpu"], dec["target_gpu"]
        cand_a = [a for a in acts.get((model, "ActivateAction", dst), []) if id(a) not in used]
        cand_d = [a for a in acts.get((model, "DeactivateAction", src), []) if id(a) not in used]
        if not cand_a or not cand_d:
            continue
        ref = dec["timestamp"] or cand_a[0]["start"]
        prepares = [a for a in cand_a if a.get("phase") == "prepare"]
        act = min(prepares or cand_a, key=lambda a: abs(a["start"] - ref))
        dea = min(cand_d, key=lambda a: abs(a["start"] - act["start"]))
        if abs(dea["start"] - act["start"]) > window_s:
            continue
        used.add(id(act))
        used.add(id(dea))

        commits = [a for a in cand_a if a.get("phase") == "commit"
                   and a.get("start", 0) >= dea.get("start", 0)]
        commit = min(commits, key=lambda a: a["start"]) if commits else None
        if commit is not None:
            used.add(id(commit))

        target_first = dea["start"] >= act["start"]
        begin = min(act["start"], dea["start"])
        end = max(act["end"], dea["end"], commit["end"] if commit else 0)
        downtime = (max(0.0, commit["end"] - dea["start"])
                    if commit is not None else
                    (0.0 if target_first else max(0.0, act["end"] - dea["start"])))
        path = slot_to_path.get(model)
        wrec = next((r for r in by_path.get(path, []) if r.get("target_gpu") == dst), None)
        kv_for_model = [r for r in kv_transfers
                        if model in str(r.get("tag", ""))]
        kv_bytes = sum(int(r.get("kv_bytes", 0)) for r in kv_for_model)
        migrations.append({
            "model": model, "model_path": path,
            "source_gpu": src, "target_gpu": dst,
            "migration_start": begin,
            "target_ready_time": act["end"],
            "migration_end": end,
            "migration_latency_s": act["end"] - begin,
            "service_downtime_s": downtime,
            "migration_total_s": end - begin,
            "ordering": ("prepare-quiesce-commit" if commit is not None else
                         ("target-first" if target_first else "source-first")),
            "readiness_barrier": int(bool(overlap)),
            "latency_is_submission_only": int(not overlap),
            "weight_bytes": (wrec or {}).get("payload_bytes", 0),
            "kv_bytes": kv_bytes,
            "total_bytes": (wrec or {}).get("payload_bytes", 0) + kv_bytes,
            "transfer_path": (wrec or {}).get("transfer_path"),
            "effective_gbps": (wrec or {}).get("payload_gbps"),
            "success": (bool(act.get("success")) and bool(dea.get("success"))
                        and (commit is None or bool(commit.get("success")))),
            "peak_kvpr_before": dec.get("peak_kvpr_before"),
            "peak_kvpr_after": dec.get("peak_kvpr_after"),
            "tau": dec.get("tau"), "alg1_cycle": dec.get("alg1_cycle"),
        })

    migrations.sort(key=lambda m: m["migration_start"])
    for i, mig in enumerate(migrations):
        mig["migration_id"] = i
    return migrations, len(decisions)


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
                ttft_scale, tpot_scale, warmup, measure, trace_dir, cfg_path):
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

    # ---- per-request raw CSV (every request, in and out of the window)
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
            arrival, e2e = r.get("arrival_time"), r.get("latency")
            in_win = arrival is not None and lo <= arrival < hi
            ttft, tpot = r.get("ttft"), r.get("tpot")
            st, sp = slo_ttft.get(r.get("model")), slo_tpot.get(r.get("model"))
            ttft_ok = ttft is not None and st is not None and ttft <= st
            tpot_ok = tpot is not None and sp is not None and tpot <= sp
            writer.writerow([
                r.get("request_id"), r.get("model"), arrival,
                (arrival + e2e) if (arrival is not None and e2e is not None) else None,
                r.get("prompt_len"), r.get("output_len"), ttft, tpot, e2e, st, sp,
                int(bool(ttft_ok)), int(bool(tpot_ok)), int(bool(ttft_ok and tpot_ok)),
                int(bool(r.get("success"))), int(in_win)])
            if in_win:
                window.append(r)

    ok = [r for r in window if r.get("success")]
    lat = {"ttft": stats([r.get("ttft") for r in ok], 1000.0),
           "tpot": stats([r.get("tpot") for r in ok], 1000.0),
           "e2e": stats([r.get("latency") for r in ok], 1000.0)}
    joint = [r for r in ok if r.get("ttft") is not None and r.get("tpot") is not None
             and r["ttft"] <= slo_ttft.get(r["model"], float("inf"))
             and r["tpot"] <= slo_tpot.get(r["model"], float("inf"))]
    ttft_hit = [r for r in ok if r.get("ttft") is not None
                and r["ttft"] <= slo_ttft.get(r["model"], float("inf"))]
    tpot_hit = [r for r in ok if r.get("tpot") is not None
                and r["tpot"] <= slo_tpot.get(r["model"], float("inf"))]

    # ---- migrations
    actions, alg1, transfers = read_actions(run_dir), read_alg1(run_dir), read_transfers(run_dir)
    kv_transfers = read_kv_transfers(run_dir)
    slot_to_path = {}
    if os.path.exists(cfg_path):
        slot_to_path = {m["model_name"]: m["model_path"] for m in json.load(open(cfg_path))}
    overlap = system in PAPER_ARMS
    migrations, decided = pair_migrations(actions, transfers, kv_transfers, alg1, run_dir,
                                          slot_to_path, overlap)

    mig_cols = ["migration_id", "model", "model_path", "source_gpu", "target_gpu",
                "migration_start", "target_ready_time", "migration_end",
                "migration_latency_s", "service_downtime_s", "migration_total_s",
                "ordering", "readiness_barrier", "latency_is_submission_only",
                "weight_bytes", "kv_bytes", "total_bytes", "transfer_path",
                "effective_gbps", "success", "peak_kvpr_before", "peak_kvpr_after",
                "tau", "alg1_cycle"]
    mig_path = Path(base) / "raw/migrations" / f"{tag}.csv"
    mig_path.parent.mkdir(parents=True, exist_ok=True)
    with open(mig_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=mig_cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(migrations)

    # ---- scheduler / Algorithm 1 raw
    sch_dir = Path(base) / "raw/scheduler"
    sch_dir.mkdir(parents=True, exist_ok=True)
    with open(sch_dir / f"{tag}_alg1.jsonl", "w") as fh:
        for rec in alg1:
            fh.write(json.dumps(rec) + "\n")
    with open(sch_dir / f"{tag}_actions.jsonl", "w") as fh:
        for rec in actions:
            fh.write(json.dumps(rec) + "\n")
    if transfers:
        with open(Path(base) / "raw/migrations" / f"{tag}_weight_transfers.jsonl", "w") as fh:
            for rec in transfers:
                fh.write(json.dumps(rec) + "\n")

    # ---- gpu time series (all four GPUs, so the idle pair is evidence)
    gpu_path = Path(base) / "raw/gpu_metrics" / f"{tag}.csv"
    gpu_path.parent.mkdir(parents=True, exist_ok=True)
    with open(gpu_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["timestamp", "gpu_id", "gpu_utilization", "memory_used_mib"])
        for row in read_gpu_timeline(run_dir):
            writer.writerow([row["timestamp"], row["gpu_id"],
                             row["gpu_utilization"], row["memory_used_mib"]])

    ok_migs = [m for m in migrations if m["success"]]
    lat_ms = [m["migration_latency_s"] * 1000 for m in ok_migs]
    down_ms = [m["service_downtime_s"] * 1000 for m in ok_migs]
    bw = [m["effective_gbps"] for m in ok_migs if m.get("effective_gbps")]
    mstats, dstats = stats(lat_ms), stats(down_ms)
    sched = metrics

    row = {
        "implementation": system, "workload": workload, "request_rate": rate, "seed": seed,
        "offered_rate": len(window) / measure,
        "achieved_throughput": len(ok) / measure,
        "goodput": len(joint) / measure,
        "total_requests": len(window), "completed_requests": len(ok),
        "failed_requests": len(window) - len(ok),
        "joint_slo": len(joint) / len(window) if window else float("nan"),
        "ttft_slo": len(ttft_hit) / len(window) if window else float("nan"),
        "tpot_slo": len(tpot_hit) / len(window) if window else float("nan"),
    }
    for name in ("ttft", "tpot", "e2e"):
        for key in ("mean", "p50", "p95", "p99", "max"):
            row[f"{name}_{key}"] = lat[name][key]
    row.update({
        "migration_count": len(migrations),
        "migration_decisions_logged": decided,
        "migration_success": len(ok_migs),
        "migration_failed": len(migrations) - len(ok_migs),
        "migration_latency_is_submission_only": int(not overlap),
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
        "activations": sched.get("activations", ""),
        "deactivations": sched.get("deactivations", ""),
        "idle_evictions": sched.get("idle_evictions", ""),
        "alg1_cycles": len(alg1),
        "alg1_placement_decisions": sum(len(a.get("line8", [])) for a in alg1),
        "alg1_migrate_decisions": sum(1 for a in alg1
                                      if a.get("migration_decision") == "MIGRATE"),
        # Where a migration's wall time actually goes.  v4 optimises the
        # transfer; whether the transfer is the expensive part is a question
        # the run can answer rather than an assumption.
        "activation_count": sum(1 for a in actions if a["action"] == "ActivateAction"),
        "activation_total_s": sum(a["duration_s"] for a in actions
                                  if a["action"] == "ActivateAction"),
        "deactivation_count": sum(1 for a in actions if a["action"] == "DeactivateAction"),
        "deactivation_total_s": sum(a["duration_s"] for a in actions
                                    if a["action"] == "DeactivateAction"),
        "weight_transfer_total_s": sum(t.get("seconds", 0.0) for t in transfers),
        "weight_transfer_mean_gbps": (
            float(np.mean([t["payload_gbps"] for t in transfers])) if transfers else float("nan")),
        "weight_transfers": len(transfers),
        "p2p_weight_transfers": sum(1 for t in transfers
                                    if t.get("transfer_path") == "gpu-to-gpu-p2p"),
        "page_locked_transfers": sum(1 for t in transfers if t.get("host_registered")),
    })
    audit = (alg1[-1].get("audit_totals") or {}) if alg1 else {}
    row["alg1_suppressed_by_tau"] = audit.get("suppressed_by_tau", "")
    row["alg1_rejected_by_memory"] = audit.get("rejected_by_memory", "")
    gaps = [a["convergence_gap"] for a in alg1 if "convergence_gap" in a]
    row["alg1_convergence_gap_mean"] = float(np.mean(gaps)) if gaps else ""

    per_model = {}
    for model in sorted({r["model"] for r in window}):
        sub = [r for r in window if r["model"] == model]
        sok = [r for r in sub if r.get("success")]
        sj = [r for r in sok if r.get("ttft") is not None and r.get("tpot") is not None
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


def write_csv(path, records, cols=None):
    if not records:
        return
    cols = cols or list(records[0].keys())
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--slo-base", default="/workspace/prism-exp/exp/configs/v2/slo_base.json")
    ap.add_argument("--config", default="/workspace/prism-exp/exp/configs/v2/6model_2gpu.json")
    ap.add_argument("--trace-dir", default="/workspace/prism-exp/exp/workloads/paper-faithful-v4")
    ap.add_argument("--ttft-scale", type=float, default=5.0)
    ap.add_argument("--tpot-scale", type=float, default=3.0)
    ap.add_argument("--warmup", type=float, default=60.0)
    ap.add_argument("--measure", type=float, default=300.0)
    a = ap.parse_args()
    base = Path(a.base)

    rows, model_rows, all_migs = [], [], []
    for done in sorted(glob.glob(str(base / "raw/*/*/rate_*/seed_*/DONE"))):
        run_dir = os.path.dirname(done)
        parts = Path(run_dir).parts
        seed, rate = int(parts[-1].split("_")[1]), int(parts[-2].split("_")[1])
        workload, system = parts[-3], parts[-4]
        try:
            row, per_model, migs = process_run(
                run_dir, base, system, workload, rate, seed, a.slo_base,
                a.ttft_scale, a.tpot_scale, a.warmup, a.measure, a.trace_dir, a.config)
        except (Exception, SystemExit) as exc:               # noqa: BLE001
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
                               "request_rate": rate, "seed": seed, "model": model, **stat})
        print(f"collected {system}/{workload}/{rate}/{seed}: "
              f"goodput={row['goodput']:.3f} joint={row['joint_slo']:.3f} "
              f"migrations={row['migration_count']} (decided {row['migration_decisions_logged']})")

    if not rows:
        print("no completed runs yet")
        return

    rows.sort(key=lambda r: (r["implementation"], r["workload"], r["request_rate"], r["seed"]))
    write_csv(base / "summary.csv", rows)
    write_csv(base / "latency_summary.csv", rows, cols=[
        "implementation", "workload", "request_rate", "seed",
        *[f"{n}_{k}" for n in ("ttft", "tpot", "e2e")
          for k in ("mean", "p50", "p95", "p99", "max")]])
    write_csv(base / "migration_summary.csv", rows, cols=[
        "implementation", "workload", "request_rate", "seed", "migration_count",
        "migration_decisions_logged", "migration_success", "migration_failed",
        "migration_latency_is_submission_only",
        "migration_latency_mean", "migration_latency_p50", "migration_latency_p95",
        "migration_latency_p99", "migration_latency_max", "migration_total_time",
        "migration_weight_bytes", "migration_kv_bytes", "migration_total_bytes",
        "migration_bandwidth", "migration_bandwidth_p50", "migration_paths",
        "service_downtime_mean", "service_downtime_p50", "service_downtime_p95",
        "service_downtime_p99", "weight_transfers", "p2p_weight_transfers",
        "page_locked_transfers"])
    write_csv(base / "model_summary.csv", model_rows)
    write_csv(base / "raw/migrations/all_migrations.csv", all_migs)

    # ---- 3-seed aggregation: mean and standard deviation
    groups = defaultdict(list)
    for row in rows:
        groups[(row["implementation"], row["workload"], row["request_rate"])].append(row)
    numeric = [k for k, v in rows[0].items()
               if isinstance(v, (int, float)) and k not in ("seed", "request_rate")]
    agg = []
    for (system, workload, rate), members in sorted(groups.items()):
        entry = {"implementation": system, "workload": workload, "request_rate": rate,
                 "seeds": len(members),
                 "seed_list": ";".join(str(m["seed"]) for m in sorted(members, key=lambda x: x["seed"]))}
        for key in numeric:
            vals = [m[key] for m in members
                    if isinstance(m.get(key), (int, float)) and not math.isnan(m[key])]
            entry[f"{key}_mean"] = float(np.mean(vals)) if vals else float("nan")
            entry[f"{key}_std"] = float(statistics.stdev(vals)) if len(vals) > 1 else 0.0
        agg.append(entry)
    write_csv(base / "aggregated/summary_by_condition.csv", agg)
    print(f"\nwrote summary.csv ({len(rows)} runs), aggregated over {len(agg)} conditions")


if __name__ == "__main__":
    main()
