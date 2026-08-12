#!/usr/bin/env python3
"""SLO analysis where the SLO is a SLOWDOWN against the same request run
unloaded on this A100 (TABLE VI), not an absolute latency bound.

    TTFT   P50 2x    P90 3x    P99 6x
    TBT    P50 1.25x P90 1.5x  P99 5x
    E2E    P50 1.25x P90 1.5x  P99 5x

Why per-request and not percentile-vs-percentile: the harness replays the same
trace in the same order every run, so the dedicated (no-contention) run and the
colocated run contain the SAME requests at the same indices -- verified on
output_len.  That lets us divide request i's latency by request i's own
unloaded latency, which is what "slowdown compared to a request running under
no contention" actually says.  Comparing p99-to-p99 would instead compare two
different requests.

The table gives a different factor per percentile, so "attainment" needs a
choice.  Both readings are reported and neither is hidden:

  percentile compliance  the table read literally -- is the observed p50/p90/p99
                         of the slowdown distribution within 2x/3x/6x?  This is
                         a property of the run, PASS/FAIL per cell.
  per-request attainment fraction of requests whose slowdown is within the
                         factor for a given tier, for all three metrics at once.
                         Reported at each tier; goodput and violation rate
                         follow from it.

Usage:
    python exp/scripts/analyze_slowdown.py \
        --baseline base --measurement exp --case B --slots model_1 model_4 \
        --out exp/results/2-colocation/exp_B_slowdown.json
"""
import argparse
import glob
import json
import os

import numpy as np

RESULTS = "/workspace/prism-exp/exp/results"

# TABLE VI -- metric -> {percentile: allowed slowdown}
SLO_TABLE = {
    "TTFT": {50: 2.0, 90: 3.0, 99: 6.0},
    "TBT":  {50: 1.25, 90: 1.5, 99: 5.0},
    "E2E":  {50: 1.25, 90: 1.5, 99: 5.0},
}
# request-dump field carrying each metric
FIELD = {"TTFT": "ttft", "TBT": "tpot", "E2E": "latency"}


def load(tag, case, slot):
    pat = os.path.join(RESULTS, tag, "requests", f"{tag}_{case}_e2e_1gpu_*_output_requests.json")
    hits = sorted(glob.glob(pat), key=os.path.getmtime, reverse=True)
    if not hits:
        raise SystemExit(f"missing request dump: {pat}")
    rows = json.load(open(hits[0]))
    return [r for r in rows if r.get("model") == slot]


def pair(base_rows, meas_rows, slot):
    """Match by index, refusing to guess if the two runs disagree on the trace."""
    if len(base_rows) != len(meas_rows):
        raise SystemExit(
            f"{slot}: baseline has {len(base_rows)} requests, measurement has "
            f"{len(meas_rows)} -- not the same trace, cannot pair")
    bad = [i for i, (b, m) in enumerate(zip(base_rows, meas_rows))
           if b["output_len"] != m["output_len"]]
    if bad:
        raise SystemExit(
            f"{slot}: output_len differs at {len(bad)} indices (first {bad[:5]}) "
            f"-- request order is not aligned, cannot pair")
    return [(b, m) for b, m in zip(base_rows, meas_rows)
            if b.get("success") and m.get("success")]


def analyse_slot(base_rows, meas_rows, slot, duration, warmup=0, min_slowdown=None):
    """warmup     drop the first N requests of the slot.  In a colocated run a
                  model is loaded when its first request arrives, while in its
                  dedicated baseline it was already resident -- so request 0 of
                  model_4 measured 5969 ms against a 68 ms baseline (88x). That
                  is Prism's model-load path, not steady-state contention, and
                  it moved TTFT p99 from 3.79x (pass) to 63.58x (fail).

       min_slowdown  flag/drop pairs whose slowdown is far BELOW 1. Those are
                  not contention going negative; they are requests whose
                  BASELINE was the anomaly. This trace bursts, and a burst that
                  fires in the baseline run but not in the measurement run
                  leaves an inflated denominator: 5 of model_1's baseline
                  requests took 938-3467 ms where the measurement took 31-54 ms,
                  giving slowdown 0.01-0.03. Those requests then pass any SLO
                  unconditionally, inflating attainment by up to 1.7pp.
    """
    if warmup:
        base_rows = base_rows[warmup:]
        meas_rows = meas_rows[warmup:]
    pairs = pair(base_rows, meas_rows, slot)
    n_total = len(meas_rows)

    contaminated = []
    if min_slowdown is not None:
        keep = []
        for b, m in pairs:
            r = [m[f] / b[f] for f in ("ttft", "tpot", "latency")
                 if b.get(f) and m.get(f) and b[f] > 0]
            (contaminated if r and min(r) < min_slowdown else keep).append((b, m))
        pairs = keep

    n_ok = len(pairs)

    slow = {}
    for metric, field in FIELD.items():
        vals = []
        for b, m in pairs:
            bv, mv = b.get(field), m.get(field)
            if not bv or not mv or bv <= 0:
                continue
            vals.append(mv / bv)
        slow[metric] = np.array(vals)

    out = {
        "slot": slot,
        "requests": n_total,
        "completed": n_ok,
        "failed": n_total - n_ok - len(contaminated),
        "warmup_dropped": warmup,
        "baseline_contaminated_dropped": len(contaminated),
        "duration_s": duration,
        "throughput_rps": n_ok / duration if duration else None,
        "output_tok_throughput": sum(m["output_len"] for _, m in pairs) / duration if duration else None,
    }

    # absolute latencies of the measurement run (what the user asked to see)
    for metric, field in FIELD.items():
        v = np.array([m[field] for _, m in pairs if m.get(field)])
        scale = 1000.0  # s -> ms
        out[f"{metric.lower()}_p50_ms"] = float(np.percentile(v, 50) * scale) if len(v) else None
        out[f"{metric.lower()}_p99_ms"] = float(np.percentile(v, 99) * scale) if len(v) else None

    # Two ways to score the table, reported side by side because they answer
    # different questions and disagree in the tail:
    #
    #   per-request   percentile OF the per-request slowdown distribution.
    #                 Faithful to "a request ... under no contention", and the
    #                 one to trust: numerator and denominator are the same
    #                 request, so a burst that queues in both runs cancels.
    #   ratio         measured pX / baseline pX. Reported only as a cross-check.
    #
    # The ratio view breaks in the tail and must not be read as a slowdown
    # there. This trace bursts (8 requests in 3.5 s around t=424 s), so the
    # no-contention baseline's own p99 is set by a handful of queued requests.
    # If that burst resolves less badly in the measurement run, the ratio goes
    # BELOW 1 -- measured TTFT p99 / baseline TTFT p99 came out at 0.11 on real
    # data, which would read as "colocation made it 9x faster". It did not; the
    # denominator is just a rare event. Values < 1 here mean the baseline tail
    # is event-dominated, not that contention helped.
    compliance, ratio = {}, {}
    for metric, tiers in SLO_TABLE.items():
        s = slow[metric]
        if not len(s):
            continue
        field = FIELD[metric]
        bvals = np.array([b[field] for b, _ in pairs if b.get(field)])
        mvals = np.array([m[field] for _, m in pairs if m.get(field)])
        for p, factor in tiers.items():
            obs = float(np.percentile(s, p))
            compliance[f"{metric}_p{p}"] = {
                "observed_slowdown": obs,
                "allowed": factor,
                "pass": bool(obs <= factor),
            }
            if len(bvals) and len(mvals):
                r = float(np.percentile(mvals, p) / np.percentile(bvals, p))
                ratio[f"{metric}_p{p}"] = {
                    "observed_slowdown": r,
                    "allowed": factor,
                    "pass": bool(r <= factor),
                }
    out["percentile_compliance"] = compliance
    out["percentile_ratio"] = ratio

    # per-request attainment at each tier (all three metrics must hold)
    attain = {}
    for p in (50, 90, 99):
        ok = np.ones(len(pairs), dtype=bool)
        for metric in SLO_TABLE:
            s = slow[metric]
            if len(s) != len(ok):      # a metric missing on some requests
                ok = ok[: len(s)] if len(s) < len(ok) else ok
                s = s[: len(ok)]
            ok &= s <= SLO_TABLE[metric][p]
        frac = float(ok.mean()) if len(ok) else None
        good_tok = sum(m["output_len"] for (_, m), k in zip(pairs, ok) if k)
        attain[f"tier_p{p}"] = {
            "attainment": frac,
            "violation_rate": 1 - frac if frac is not None else None,
            "goodput_rps": float(ok.sum()) / duration if duration else None,
            "goodput_tok_s": good_tok / duration if duration else None,
        }
    out["attainment"] = attain

    out["slowdown"] = {
        m: {"p50": float(np.percentile(s, 50)), "p90": float(np.percentile(s, 90)),
            "p99": float(np.percentile(s, 99))}
        for m, s in slow.items() if len(s)
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, help="TAG of the no-contention runs")
    ap.add_argument("--measurement", required=True, help="TAG of the colocated runs")
    ap.add_argument("--case", required=True)
    ap.add_argument("--slots", nargs="+", required=True)
    ap.add_argument("--base-case", nargs="+", default=None,
                    help="baseline case per slot (default M<n> from the slot name)")
    ap.add_argument("--duration", type=float, default=None)
    ap.add_argument("--warmup", type=int, default=0,
                    help="drop the first N requests per slot (model-load transient)")
    ap.add_argument("--min-slowdown", type=float, default=None,
                    help="drop pairs with slowdown below this (inflated baseline, "
                         "e.g. a burst that fired only in the baseline run)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    base_cases = args.base_case or [f"M{s.split('_')[1]}" for s in args.slots]
    if len(base_cases) != len(args.slots):
        raise SystemExit("--base-case must have one entry per slot")

    duration = args.duration
    if duration is None:
        pat = os.path.join(RESULTS, args.measurement, f"{args.measurement}_{args.case}_e2e_1gpu_*rep.json")
        hits = sorted(glob.glob(pat), key=os.path.getmtime, reverse=True)
        if hits:
            m = json.load(open(hits[0]))
            duration = m.get("duration") or m.get("benchmark_duration") or 600.0
        else:
            duration = 600.0

    res = {"case": args.case, "baseline_tag": args.baseline,
           "measurement_tag": args.measurement, "duration_s": duration,
           "slo_table": {k: {str(p): v for p, v in t.items()} for k, t in SLO_TABLE.items()},
           "warmup": args.warmup, "min_slowdown": args.min_slowdown,
           "per_slot": {}}

    for slot, bc in zip(args.slots, base_cases):
        b = load(args.baseline, bc, slot)
        m = load(args.measurement, args.case, slot)
        res["per_slot"][slot] = analyse_slot(b, m, slot, duration, args.warmup, args.min_slowdown)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
