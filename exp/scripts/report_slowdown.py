#!/usr/bin/env python3
"""Render the slowdown-SLO results as the requested table.

    python exp/scripts/report_slowdown.py --measurement exp [--tier 99]

Columns: throughput, attainment, goodput, TTFT p50/p99, TPOT(TBT) p50/p99,
E2E p50/p99, violation rate.

Two blocks are printed.  The first is TABLE VI read literally -- for each
metric and percentile, the observed slowdown against the allowed factor.  The
second is the per-request view at one tier, which is where attainment, goodput
and violation rate come from; --tier selects it (default 99, the tail bound).
Absolute latencies are the measured values, not slowdowns.
"""
import argparse
import json
import os

RESULTS = "/workspace/prism-exp/exp/results"
SUFFIX = ""
CASES = [("A", "1x Llama-3.1-8B"), ("B", "2x Llama-3.1-8B"),
         ("C", "Llama-3.2-3B + Llama-3.1-8B")]


def load(tag, case):
    p = os.path.join(RESULTS, tag, f"{tag}_{case}_slowdown{SUFFIX}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--measurement", default="exp")
    ap.add_argument("--suffix", default="", help="e.g. _clean for the warmup-excluded analysis")
    # P50 tier (2x / 1.25x / 1.25x) is the discriminating one. The P99 tier is
    # 6x / 5x / 5x, loose enough that attainment saturates at 1.000 and hides
    # the decode contention entirely -- all three are printed regardless.
    ap.add_argument("--tier", type=int, default=50, choices=[50, 90, 99])
    ap.add_argument("--view", default="percentile_compliance",
                    choices=["percentile_compliance", "percentile_ratio"],
                    help="per-request slowdown (default, trustworthy) or "
                         "measured-pX/baseline-pX (cross-check only; unreliable "
                         "in the tail when the baseline p99 is burst-dominated)")
    args = ap.parse_args()
    global SUFFIX
    SUFFIX = args.suffix

    docs = [(c, name, load(args.measurement, c)) for c, name in CASES]
    docs = [(c, n, d) for c, n, d in docs if d]
    if not docs:
        print("no results yet")
        return

    vlabel = ("요청별 slowdown" if args.view == "percentile_compliance"
              else "백분위 비율 (측정 pX / baseline pX)")
    print("\n" + "=" * 104)
    print(f"1) TABLE VI 준수 여부 — 무경합 A100 대비 slowdown (관측 / 허용) · {vlabel}")
    print("=" * 104)
    hdr = f"{'case / slot':26s}"
    for m in ("TTFT", "TBT", "E2E"):
        for p in (50, 90, 99):
            hdr += f" {m+' p'+str(p):>12s}"
    print(hdr)
    print("-" * 104)
    for c, name, d in docs:
        for slot, s in d["per_slot"].items():
            line = f"{c+'  '+slot:26s}"
            for m in ("TTFT", "TBT", "E2E"):
                for p in (50, 90, 99):
                    cell = s[args.view].get(f"{m}_p{p}")
                    if not cell:
                        line += f" {'-':>12s}"
                        continue
                    mark = "OK" if cell["pass"] else "X"
                    line += f" {cell['observed_slowdown']:7.2f}/{cell['allowed']:.2g}{mark:>3s}"
            print(line)
        print("-" * 104)

    print(f"\n{'=' * 104}")
    print(f"2) 측정값 — attainment/goodput/위반율은 tier P{args.tier} 기준 "
          f"(TTFT·TBT·E2E 세 조건 동시 충족)")
    print("=" * 104)
    cols = (f"{'case / slot':26s} {'tput r/s':>9s} {'tput tok/s':>11s} {'attain':>8s} "
            f"{'goodput r/s':>12s} {'goodput t/s':>12s} {'violation':>10s}")
    print(cols)
    print("-" * 104)
    for c, name, d in docs:
        for slot, s in d["per_slot"].items():
            a = s["attainment"][f"tier_p{args.tier}"]
            print(f"{c+'  '+slot:26s} {s['throughput_rps']:9.3f} {s['output_tok_throughput']:11.1f} "
                  f"{a['attainment']:8.3f} {a['goodput_rps']:12.3f} {a['goodput_tok_s']:12.1f} "
                  f"{a['violation_rate']:10.3f}")
        print("-" * 104)

    print(f"\n{'case / slot':26s} {'TTFT p50':>10s} {'TTFT p99':>10s} {'TPOT p50':>10s} "
          f"{'TPOT p99':>10s} {'E2E p50':>10s} {'E2E p99':>10s}   (ms)")
    print("-" * 104)
    for c, name, d in docs:
        for slot, s in d["per_slot"].items():
            print(f"{c+'  '+slot:26s} {s['ttft_p50_ms']:10.1f} {s['ttft_p99_ms']:10.1f} "
                  f"{s['tbt_p50_ms']:10.2f} {s['tbt_p99_ms']:10.2f} "
                  f"{s['e2e_p50_ms']:10.1f} {s['e2e_p99_ms']:10.1f}")
        print("-" * 104)

    print("\n다른 tier:")
    for c, name, d in docs:
        for slot, s in d["per_slot"].items():
            t = s["attainment"]
            print(f"  {c} {slot:9s} attain  P50 {t['tier_p50']['attainment']:.3f}   "
                  f"P90 {t['tier_p90']['attainment']:.3f}   P99 {t['tier_p99']['attainment']:.3f}")
    print()
    for c, name in CASES:
        d = load(args.measurement, c)
        if d:
            n = sum(s["requests"] for s in d["per_slot"].values())
            f = sum(s["failed"] for s in d["per_slot"].values())
            print(f"  {c} = {name}: {n} reqs, {f} failed, duration {d['duration_s']:.1f}s")


if __name__ == "__main__":
    main()
