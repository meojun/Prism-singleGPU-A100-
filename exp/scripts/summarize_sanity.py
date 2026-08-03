#!/usr/bin/env python3
"""Print the sanity-check sweep as one comparison table."""
import glob
import json
import os

CASES = {
    "A": "1x Llama-3.1-8B",
    "B": "2x Llama-3.1-8B",
    "C": "Llama-3.1-8B + Llama-3.2-3B",
}

ROOT = "/workspace/prism-exp/exp/results/sanity"

COLS = [
    ("reqs", "requests", "{:d}"),
    ("done", "completed", "{:d}"),
    ("SLO TTFT (s)", "slo_ttft_s", "{:.3f}"),
    ("SLO TPOT (ms)", "slo_tpot_ms", "{:.1f}"),
    ("tput (req/s)", "req_throughput_rps", "{:.3f}"),
    ("tput (tok/s)", "output_tok_throughput", "{:.1f}"),
    ("attain TTFT", "attain_ttft", "{:.3f}"),
    ("attain TPOT", "attain_tpot", "{:.3f}"),
    ("attain both", "attain_both", "{:.3f}"),
    ("violation", "violation_rate", "{:.3f}"),
    ("goodput r/s", "goodput_rps", "{:.3f}"),
    ("goodput t/s", "goodput_tok_s", "{:.1f}"),
    ("TTFT p50 ms", "ttft_p50_ms", "{:.1f}"),
    ("TTFT p99 ms", "ttft_p99_ms", "{:.1f}"),
    ("TPOT p50 ms", "tpot_p50_ms", "{:.1f}"),
    ("TPOT p99 ms", "tpot_p99_ms", "{:.1f}"),
]


def fmt(v, f):
    if v is None:
        return "-"
    try:
        return f.format(v)
    except (ValueError, TypeError):
        return str(v)


def main():
    rows = []
    for c in "ABC":
        p = os.path.join(ROOT, f"sanity_{c}_slo.json")
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        for name, stats in d["per_model"].items():
            mp = d["models"].get(name, "")
            label = f"{c}  {name}" + (f" ({mp.split('/')[-1]})" if mp else " (ALL)")
            rows.append((label, d, stats))

    if not rows:
        print("no results yet")
        return

    hdr = ["case / model"] + [c[0] for c in COLS]
    widths = [max(len(hdr[0]), max(len(r[0]) for r in rows))] + [
        max(len(c[0]), 11) for c in COLS
    ]

    def line(cells):
        return "  ".join(str(c).ljust(w) for c, w in zip(cells, widths))

    d0 = rows[0][1]
    print(f"\nPrism full mode | 1x A100-80G | real_trace.pkl (time_scale=1, replication=1)")
    print(f"SLO = paper p95 baseline x scale   (TTFT x{d0['ttft_slo_scale']:g}, TPOT x{d0['tpot_slo_scale']:g})\n")
    print(line(hdr))
    print("-" * (sum(widths) + 2 * len(widths)))
    prev = None
    for label, d, s in rows:
        if prev and label[0] != prev:
            print("-" * (sum(widths) + 2 * len(widths)))
        prev = label[0]
        print(line([label] + [fmt(s.get(k), f) for _, k, f in COLS]))
    print()
    for c in "ABC":
        p = os.path.join(ROOT, f"sanity_{c}_slo.json")
        if os.path.exists(p):
            d = json.load(open(p))
            print(f"  {c} = {CASES[c]}: {d['total_requests']} reqs, "
                  f"{d['failed']} failed, duration {d['duration_s']:.1f}s")


if __name__ == "__main__":
    main()
