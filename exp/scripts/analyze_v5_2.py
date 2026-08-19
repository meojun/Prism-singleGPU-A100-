#!/usr/bin/env python3
"""V5_2: what actually costs the paper-faithful arms their goodput?

Two questions, both answered from the v4 study's raw data with no new runs.

1. Deactivation dominates the control-action wall time, but is it on the
   critical path for request latency?  v4 already halved the weight transfer
   and goodput did not move, so "large in seconds" is not evidence by itself.
   Here: do requests arriving near a migration actually suffer?

2. The paper-faithful arms lose to the released prototype at moderate load and
   the tau contrast ruled out migration as the cause.  Where does the loss
   come from instead?
"""
import argparse
import csv
import glob
import json
import math
import os
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

import numpy as np


def read_csv(p):
    try:
        with open(p) as fh:
            return list(csv.DictReader(fh))
    except FileNotFoundError:
        return []


def num(x, default=float("nan")):
    try:
        v = float(x)
        return default if math.isnan(v) else v
    except (TypeError, ValueError):
        return default


def trace_epoch_anchor(base, tag):
    """Requests carry trace-relative arrivals; migrations carry epoch time.

    The GPU sampler is started immediately before the benchmark, so its first
    sample is the closest anchor the run recorded for "trace t0 in epoch".
    Good to a second or two, which is fine for the +-10 s windows below.
    """
    rows = read_csv(base / "raw/gpu_metrics" / f"{tag}.csv")
    ts = [num(r["timestamp"]) for r in rows if r.get("timestamp")]
    return min(ts) if ts else None


def q1_migration_on_critical_path(base, out):
    print("\n" + "=" * 78)
    print("Q1  Is the deactivation / migration cost on the critical path?")
    print("=" * 78)
    print("For each run: latency of requests arriving within +-W s of a migration,")
    print("against requests arriving outside that window, in the same run.\n")
    W = 10.0
    rows_out = []
    print(f"{'arm':20} {'wl':7} {'rate':>4} {'sd':>3} {'mig':>4} "
          f"{'near n':>7} {'far n':>7} {'TTFT near/far':>18} {'TPOT near/far':>18}")
    for req_path in sorted(glob.glob(str(base / "raw/requests/*.csv"))):
        tag = Path(req_path).stem
        m = re.match(r"(.+)_(bursty|steady)_r(\d+)_s(\d+)$", tag)
        if not m:
            continue
        arm, wl, rate, seed = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
        migs = read_csv(base / "raw/migrations" / f"{tag}.csv")
        if not migs:
            continue
        anchor = trace_epoch_anchor(base, tag)
        if anchor is None:
            continue
        mig_rel = [num(x["migration_start"]) - anchor for x in migs]
        reqs = [r for r in read_csv(req_path)
                if r.get("in_measurement_window") == "1" and r.get("success") == "1"]
        near, far = [], []
        for r in reqs:
            a = num(r["arrival_time"])
            (near if any(abs(a - t) <= W for t in mig_rel) else far).append(r)

        def med(rs, k):
            v = [num(x[k]) * 1000 for x in rs if x.get(k)]
            return float(np.median(v)) if v else float("nan")

        tn, tf = med(near, "ttft_s"), med(far, "ttft_s")
        pn, pf = med(near, "tpot_s"), med(far, "tpot_s")
        print(f"{arm:20} {wl:7} {rate:>4} {seed:>3} {len(migs):>4} "
              f"{len(near):>7} {len(far):>7} "
              f"{tn:>8.1f}/{tf:<8.1f} {pn:>8.1f}/{pf:<8.1f}")
        rows_out.append({"arm": arm, "workload": wl, "rate": rate, "seed": seed,
                         "migrations": len(migs), "near_n": len(near), "far_n": len(far),
                         "ttft_near_ms": tn, "ttft_far_ms": tf,
                         "tpot_near_ms": pn, "tpot_far_ms": pf,
                         "ttft_ratio": tn / tf if tf else float("nan"),
                         "tpot_ratio": pn / pf if pf else float("nan")})
    if rows_out:
        tr = [r["ttft_ratio"] for r in rows_out if not math.isnan(r["ttft_ratio"])]
        pr = [r["tpot_ratio"] for r in rows_out if not math.isnan(r["tpot_ratio"])]
        print(f"\n  across {len(rows_out)} runs: TTFT near/far ratio "
              f"median {np.median(tr):.2f}, TPOT {np.median(pr):.2f}")
        print("  ratio ~1.0 means requests near a migration are no worse ->")
        print("  the migration cost is NOT on the request critical path.")
        with open(out / "q1_migration_critical_path.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows_out[0]))
            w.writeheader(); w.writerows(rows_out)
    return rows_out


def q2_where_the_deficit_comes_from(base, out):
    print("\n" + "=" * 78)
    print("Q2  Where do the paper-faithful arms lose at moderate load?")
    print("=" * 78)
    rows = read_csv(base / "summary.csv")
    g = defaultdict(list)
    for r in rows:
        g[(r["workload"], int(r["request_rate"]), r["implementation"])].append(r)

    def agg(v, k, d=1):
        xs = [num(x.get(k)) for x in v]
        xs = [x for x in xs if not math.isnan(x)]
        if not xs:
            return "—"
        return f"{st.fmean(xs):.{d}f}" + (f" ± {st.stdev(xs):.{d}f}" if len(xs) > 1 else "")

    order = ["released-prototype", "paper-faithful-v3", "paper-faithful-v4"]
    for wl, rate in sorted({(k[0], k[1]) for k in g}):
        print(f"\n--- {wl} {rate} req/s")
        print(f"{'arm':20} {'goodput':>13} {'thruput':>10} {'TTFT p50':>10} {'TPOT p50':>10} "
              f"{'alg2 sel/elig':>15} {'queue max':>10} {'deferred':>9} {'late':>7}")
        for arm in order:
            v = g.get((wl, rate, arm))
            if not v:
                continue
            sel = [num(x.get("admitted_requests")) for x in v]
            elig = [num(x.get("alg2_eligible")) for x in v]
            ratio = "—"
            if all(not math.isnan(s) for s in sel) and sum(e for e in elig if not math.isnan(e)) > 0:
                ratio = f"{sum(sel)/sum(elig):.3f}"
            print(f"{arm:20} {agg(v,'goodput',2):>13} {agg(v,'achieved_throughput',2):>10} "
                  f"{agg(v,'ttft_p50'):>10} {agg(v,'tpot_p50'):>10} {ratio:>15} "
                  f"{agg(v,'queue_length_max'):>10} {agg(v,'deferred_requests'):>9} "
                  f"{agg(v,'shed_requests'):>7}")
    print("\n  Throughput identical across arms means nothing is being dropped;")
    print("  a goodput gap with equal throughput is entirely an SLO-attainment gap.")


def q3_kv_resident_at_migration(base, out):
    print("\n" + "=" * 78)
    print("Q3  How much KV would a KV-migration actually have to move?")
    print("=" * 78)
    print("The engine releases the KV pool on deactivate without draining, so")
    print("whatever was resident is discarded.  Requests still in flight for the")
    print("migrated model at that moment are the ones a KV migration would save.\n")
    W = 2.0
    print(f"{'arm':20} {'wl':7} {'rate':>4} {'sd':>3} {'mig':>4} "
          f"{'in-flight at migration (mean/max)':>34}")
    totals = []
    for req_path in sorted(glob.glob(str(base / "raw/requests/*.csv"))):
        tag = Path(req_path).stem
        m = re.match(r"(.+)_(bursty|steady)_r(\d+)_s(\d+)$", tag)
        if not m:
            continue
        arm, wl, rate, seed = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
        migs = read_csv(base / "raw/migrations" / f"{tag}.csv")
        anchor = trace_epoch_anchor(base, tag)
        if not migs or anchor is None:
            continue
        reqs = read_csv(req_path)
        by_model = defaultdict(list)
        for r in reqs:
            a, c = num(r["arrival_time"]), num(r["completion_time"])
            if not math.isnan(a) and not math.isnan(c):
                by_model[r["model"]].append((a, c))
        counts = []
        for mig in migs:
            t = num(mig["migration_start"]) - anchor
            model = mig["model"]
            counts.append(sum(1 for a, c in by_model.get(model, []) if a <= t <= c))
        if counts:
            print(f"{arm:20} {wl:7} {rate:>4} {seed:>3} {len(migs):>4} "
                  f"{st.fmean(counts):>18.1f} / {max(counts):<12}")
            totals.extend(counts)
    if totals:
        print(f"\n  across {len(totals)} migrations: mean {st.fmean(totals):.1f}, "
              f"median {np.median(totals):.0f}, max {max(totals)}, "
              f"zero on {100*sum(1 for c in totals if c==0)/len(totals):.0f}% of them")
        print("  A migration with no in-flight request has no KV worth moving.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="/workspace/prism-exp/exp/results/paper-faithful-v4")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    base = Path(a.base)
    out = Path(a.out or (base / "v5_2-analysis"))
    out.mkdir(parents=True, exist_ok=True)
    q1_migration_on_critical_path(base, out)
    q2_where_the_deficit_comes_from(base, out)
    q3_kv_resident_at_migration(base, out)
    print(f"\nwrote {out}/")


if __name__ == "__main__":
    main()
