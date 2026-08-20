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
    print("Requests arriving within +-10 s of a migration, against requests")
    print("arriving within +-10 s of a SHAM time that has no migration near it.")
    print("A plain 'everything else' control is unusable when migrations are")
    print("frequent -- it covers 87% of the run and leaves only the quiet gaps.\n")
    W = 10.0
    rows_out = []
    print(f"{'arm':20} {'wl':7} {'rate':>4} {'sd':>3} {'mig':>4} "
          f"{'near n':>7} {'sham n':>7} {'TTFT near/far':>18} {'TPOT near/far':>18}")
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
        # "far" is not a usable control when migrations are frequent: with 13 of
        # them in 300 s, +-10 s windows cover ~87% of the run and the control is
        # a thin sliver taken from the quietest moments.  Use sham times instead
        # -- one per migration, placed away from any real migration -- so both
        # groups are drawn the same way and only the migration differs.
        arrivals = [num(r["arrival_time"]) for r in reqs]
        lo_t, hi_t = (min(arrivals), max(arrivals)) if arrivals else (0.0, 0.0)
        step = (hi_t - lo_t) / (len(mig_rel) + 1) if mig_rel else 0.0
        sham = [t for t in (lo_t + i * step for i in range(1, len(mig_rel) + 1))
                if all(abs(t - m) > 2 * W for m in mig_rel)]
        near, far = [], []
        for r in reqs:
            a = num(r["arrival_time"])
            if any(abs(a - t) <= W for t in mig_rel):
                near.append(r)
            elif any(abs(a - t) <= W for t in sham):
                far.append(r)

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
        # Only runs that actually got a sham window can be compared.  When
        # migrations are frequent there is no quiet stretch left to sham
        # against, and far_n comes back 0 -- those runs carry no evidence
        # either way, so say how many were usable instead of quoting the
        # median as if it covered every run.
        usable = [r for r in rows_out if not math.isnan(r["ttft_ratio"])]
        tr = [r["ttft_ratio"] for r in usable]
        pr = [r["tpot_ratio"] for r in rows_out if not math.isnan(r["tpot_ratio"])]
        print(f"\n  {len(usable)} of {len(rows_out)} runs had a sham control "
              f"(the rest ran migrations too often to leave a quiet window)")
        print(f"  pooled: TTFT near/far ratio median {np.median(tr):.2f}, "
              f"TPOT {np.median(pr):.2f}")

        # Split by arm.  The prototype migrates source-first -- the source is
        # torn down before the target is up, so the transfer window IS a
        # service gap -- while v3/v4 migrate target-first with zero downtime.
        # Those are different mechanisms, and pooling them lets the prototype's
        # stop-the-world cost speak for arms that do not have it.
        print("  by arm (the two orderings are different mechanisms):")
        for arm in sorted({r["arm"] for r in usable}):
            v = [r["ttft_ratio"] for r in usable if r["arm"] == arm]
            print(f"    {arm:22} n={len(v):>2}  TTFT ratio median {np.median(v):.2f}  "
                  f"[{min(v):.2f}, {max(v):.2f}]")

        # The verdict follows the number rather than being asserted.
        m = float(np.median(tr))
        if m < 1.10:
            print(f"  -> median {m:.2f}: requests near a migration are no worse; "
                  "the cost is NOT on the request critical path.")
        else:
            print(f"  -> median {m:.2f}: requests near a migration are measurably "
                  "worse, so the cost IS on the critical path -- but read the "
                  "per-arm split above before attributing it to any one arm.")
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


def q4_reprefill_cost_of_a_discarded_kv(base, out):
    """Does losing the KV on migration actually cost the affected requests?

    The engine releases the KV pool on deactivate without draining, so a
    request of the migrated model that was mid-flight loses its cache.  If a
    KV migration is worth building, those requests should be measurably worse
    than requests of the SAME model that did not span a migration -- same
    model, same run, so load and model size are controlled.
    """
    print("\n" + "=" * 78)
    print("Q4  What does discarding the KV cost the requests it happens to?")
    print("=" * 78)
    rows_out = []
    print("Control = requests spanning a SHAM time with no migration near it, so")
    print("both groups are equally biased towards long requests.\n")
    print(f"{'arm':22} {'wl':7} {'rate':>4} {'sd':>3} {'affected':>9} {'placebo':>8} "
          f"{'TPOT aff/ctl (ms)':>21} {'E2E aff/ctl (ms)':>22}")
    for req_path in sorted(glob.glob(str(base / "raw/requests/*.csv"))):
        tag = Path(req_path).stem
        m = re.match(r"(.+)_(bursty|steady)_r(\d+)_s(\d+)$", tag)
        if not m:
            continue
        arm, wl, rate, seed = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
        migs = read_csv(base / "raw/migrations" / f"{tag}.csv")
        anchor_t = trace_epoch_anchor(base, tag)
        if not migs or anchor_t is None:
            continue
        mig_by_model = defaultdict(list)
        for x in migs:
            mig_by_model[x["model"]].append(num(x["migration_start"]) - anchor_t)
        # Selecting requests that SPAN a migration selects long requests, so a
        # plain comparison measures duration, not the migration -- the first
        # cut of this showed E2E 2.2x higher and TPOT *lower*, both artefacts.
        # Placebo control: for every real migration time, a matched sham time
        # in the same run with no migration within 20 s.  Spanning a sham
        # selects long requests in exactly the same way, so what survives the
        # difference is the migration itself.
        all_t = sorted(t for ts in mig_by_model.values() for t in ts)
        arrivals = [num(r["arrival_time"]) for r in read_csv(req_path)
                    if not math.isnan(num(r["arrival_time"]))]
        lo_t, hi_t = (min(arrivals), max(arrivals)) if arrivals else (0.0, 0.0)
        sham_by_model = {}
        for model, ts in mig_by_model.items():
            shams, step = [], (hi_t - lo_t) / (len(ts) + 1 or 1)
            for i in range(1, len(ts) + 1):
                cand = lo_t + i * step
                if all(abs(cand - t) > 20.0 for t in all_t):
                    shams.append(cand)
            sham_by_model[model] = shams

        aff, ctl = [], []
        for r in read_csv(req_path):
            if r.get("success") != "1" or r.get("in_measurement_window") != "1":
                continue
            times = mig_by_model.get(r["model"])
            if not times:
                continue
            a, c = num(r["arrival_time"]), num(r["completion_time"])
            if math.isnan(a) or math.isnan(c):
                continue
            if any(a <= t <= c for t in times):
                aff.append(r)
            elif any(a <= t <= c for t in sham_by_model.get(r["model"], [])):
                ctl.append(r)

        def med(rs, k):
            v = [num(x[k]) * 1000 for x in rs if x.get(k)]
            return float(np.median(v)) if v else float("nan")

        if not aff or not ctl:
            continue
        pa, pc = med(aff, "tpot_s"), med(ctl, "tpot_s")
        ea, ec = med(aff, "e2e_s"), med(ctl, "e2e_s")
        print(f"{arm:22} {wl:7} {rate:>4} {seed:>3} {len(aff):>9} {len(ctl):>8} "
              f"{pa:>9.1f}/{pc:<10.1f} {ea:>10.1f}/{ec:<11.1f}")
        rows_out.append({"arm": arm, "workload": wl, "rate": rate, "seed": seed,
                         "affected_n": len(aff), "control_n": len(ctl),
                         "tpot_affected_ms": pa, "tpot_control_ms": pc,
                         "e2e_affected_ms": ea, "e2e_control_ms": ec,
                         "tpot_ratio": pa / pc if pc else float("nan"),
                         "e2e_ratio": ea / ec if ec else float("nan")})
    if rows_out:
        tr = [r["tpot_ratio"] for r in rows_out if not math.isnan(r["tpot_ratio"])]
        er = [r["e2e_ratio"] for r in rows_out if not math.isnan(r["e2e_ratio"])]
        print(f"\n  across {len(rows_out)} runs: TPOT affected/control median "
              f"{np.median(tr):.2f}, E2E {np.median(er):.2f}")
        # The median alone reads far firmer than this evidence is.  Print the
        # spread too: with a handful of runs and ratios straddling 1 in both
        # directions, "near 1.0" is a weak signal, not a settled result.
        print(f"  spread: TPOT [{min(tr):.2f}, {max(tr):.2f}], "
              f"E2E [{min(er):.2f}, {max(er):.2f}] over n={len(tr)} runs")
        print("  A ratio near 1.0 means losing the KV cost these requests little,")
        print("  and a KV migration would have little to recover -- but with this")
        print("  n and this spread, treat it as weak evidence, not a rejection.")
        with open(out / "q4_reprefill_cost.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows_out[0]))
            w.writeheader(); w.writerows(rows_out)
    return rows_out


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
    q4_reprefill_cost_of_a_discarded_kv(base, out)
    print(f"\nwrote {out}/")


if __name__ == "__main__":
    main()
