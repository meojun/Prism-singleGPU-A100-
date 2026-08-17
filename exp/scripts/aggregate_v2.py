#!/usr/bin/env python3
"""Aggregate every paper-faithful-v2 run into results.csv / summary tables.

    python aggregate_v2.py --base exp/results/paper-faithful-v2 -o <outdir>

Reads <base>/raw/<system>/<workload>/rate_<R>/seed_<S>/metrics.json.
Emits results.csv (one row per run), summary.csv (seed-averaged), and
comparison.csv (Prism vs baseline, per workload and load).
"""
import argparse
import csv
import glob
import json
import os
import re
import statistics
from collections import defaultdict

FIELDS = [
    "system", "workload", "rate", "seed",
    "requests_in_window", "completed", "failed", "offered_load_req_s",
    "ttft_mean_ms", "ttft_p50_ms", "ttft_p95_ms", "ttft_p99_ms",
    "tpot_mean_ms", "tpot_p50_ms", "tpot_p95_ms", "tpot_p99_ms",
    "ttft_attainment", "tpot_attainment", "joint_attainment",
    "throughput_req_s", "goodput_req_s",
    "output_token_throughput", "prompt_token_throughput",
    "migrations_alg1", "migrations_proto", "activations", "deactivations",
    "idle_evictions", "alg1_cycles",
    "kvpr_peak_mean", "kvpr_peak_cv", "kvpr_improvement_mean",
    "kvpr_improvement_std", "kvpr_improvement_max",
    "alg2_rounds", "alg2_eligible", "alg2_selected", "alg2_deferred",
    "alg2_requeued", "alg2_late_dispatched", "alg2_selected_ratio",
    "alg2_pathological_rounds", "alg2_max_zero_streak",
    "alg2_underadmission_warnings",
    "mean_queue_length", "max_queue_length",
    "failure_attempt", "failure_reproduced",
]


def collect(base):
    rows = []
    for p in sorted(glob.glob(os.path.join(base, "raw", "*", "*", "rate_*", "seed_*", "metrics.json"))):
        parts = p.split(os.sep)
        # Retry archives intentionally retain their metrics for diagnosis but
        # are not independent seeds and must never enter the final aggregate.
        if re.fullmatch(r"seed_\d+", parts[-2]) is None:
            continue
        if not os.path.isfile(os.path.join(os.path.dirname(p), "DONE")):
            continue
        d = json.load(open(p))
        row = {
            "system": parts[-5], "workload": parts[-4],
            "rate": parts[-3].replace("rate_", ""), "seed": parts[-2].replace("seed_", ""),
        }
        for k in FIELDS:
            if k in row:
                continue
            v = d.get(k)
            row[k] = "" if v is None else v
        row["_per_model"] = d.get("per_model", {})
        row["_gpu"] = {k: d.get(k) for k in ("gpu_util_mean", "gpu_mem_mean_mib", "gpu_mem_max_mib")}
        rows.append(row)
    return rows


def num(v):
    try:
        f = float(v)
        return f if f == f else None          # drop NaN
    except (TypeError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    rows = collect(a.base)
    if not rows:
        print("no metrics.json found yet")
        return
    with open(os.path.join(a.out, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=FIELDS, extrasaction="ignore", lineterminator="\n"
        )
        w.writeheader()
        w.writerows(rows)
    json.dump([{k: v for k, v in r.items()} for r in rows],
              open(os.path.join(a.out, "results_full.json"), "w"), indent=1, default=str)

    # ---- seed average
    groups = defaultdict(list)
    for r in rows:
        groups[(r["system"], r["workload"], r["rate"])].append(r)
    summ = []
    for (sys_, wl, rate), rs in sorted(groups.items(), key=lambda kv: (kv[0][2], kv[0][1], kv[0][0])):
        out = {"system": sys_, "workload": wl, "rate": rate, "n_seeds": len(rs)}
        for k in FIELDS[4:]:
            vals = [num(r[k]) for r in rs]
            vals = [v for v in vals if v is not None]
            out[k] = statistics.fmean(vals) if vals else ""
            if len(vals) > 1:
                out[k + "_sd"] = statistics.stdev(vals)
        summ.append(out)
    keys = list(dict.fromkeys(k for s in summ for k in s))
    with open(os.path.join(a.out, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=keys, extrasaction="ignore", lineterminator="\n"
        )
        w.writeheader()
        w.writerows(summ)

    # ---- Prism vs baseline, per (workload, rate)
    idx = {(s["system"], s["workload"], s["rate"]): s for s in summ}
    comp = []
    for (sys_, wl, rate), s in idx.items():
        if sys_ == "released-prototype":
            continue
        b = idx.get(("released-prototype", wl, rate))
        if not b:
            continue
        rel = lambda k, lower_better: (
            ((num(b[k]) - num(s[k])) / num(b[k])) if lower_better else
            ((num(s[k]) - num(b[k])) / num(b[k]))
        ) if num(b.get(k)) not in (None, 0) and num(s.get(k)) is not None else ""
        comp.append({
            "system": sys_, "workload": wl, "rate": rate,
            "joint_attainment_base": b["joint_attainment"],
            "joint_attainment_prism": s["joint_attainment"],
            "joint_attainment_pp": (num(s["joint_attainment"]) - num(b["joint_attainment"]))
                if num(s.get("joint_attainment")) is not None and num(b.get("joint_attainment")) is not None else "",
            "joint_attainment_rel": rel("joint_attainment", False),
            "goodput_rel": rel("goodput_req_s", False),
            "ttft_p99_rel": rel("ttft_p99_ms", True),
            "tpot_p99_rel": rel("tpot_p99_ms", True),
            "throughput_rel": rel("throughput_req_s", False),
        })
    if comp:
        with open(os.path.join(a.out, "comparison.csv"), "w", newline="") as f:
            w = csv.DictWriter(
                f, fieldnames=list(comp[0]), lineterminator="\n"
            )
            w.writeheader()
            w.writerows(comp)

    print(f"{len(rows)} runs -> {a.out}/results.csv, summary.csv, comparison.csv")
    hdr = f"{'system':20s} {'wl':7s} {'rate':>5s} {'TTFTp50':>8s} {'TTFTp99':>9s} {'TPOTp50':>8s} {'joint':>6s} {'good':>6s} {'mig':>4s} {'evict':>5s} {'act':>4s}"
    print(hdr); print("-" * len(hdr))
    for s in summ:
        g = lambda k, f="{:.1f}": (f.format(num(s[k])) if num(s.get(k)) is not None else "-")
        print(f"{s['system']:20s} {s['workload']:7s} {s['rate']:>5s} {g('ttft_p50_ms'):>8s} "
              f"{g('ttft_p99_ms'):>9s} {g('tpot_p50_ms'):>8s} {g('joint_attainment','{:.3f}'):>6s} "
              f"{g('goodput_req_s','{:.2f}'):>6s} "
              f"{g('migrations_alg1','{:.0f}') if num(s.get('migrations_alg1')) else g('migrations_proto','{:.0f}'):>4s} "
              f"{g('idle_evictions','{:.0f}'):>5s} {g('activations','{:.0f}'):>4s}")


if __name__ == "__main__":
    main()
