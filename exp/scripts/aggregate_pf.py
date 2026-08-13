#!/usr/bin/env python3
"""Aggregate the Paper-Faithful sweep: results.csv, summary.csv, 8 figures, REPORT.md."""
import argparse
import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np

COLUMNS = [
    "system", "request_rate", "seed",
    "ttft_p50_ms", "ttft_p99_ms", "tpot_p50_ms", "tpot_p99_ms",
    "ttft_slo_attainment", "tpot_slo_attainment", "joint_slo_attainment",
    "throughput_req_s", "joint_slo_goodput_req_s",
    "num_completed", "num_failed",
    "num_migrations", "num_evictions", "num_activations", "num_mh_deferred",
    "ttft_p95_ms", "tpot_p95_ms", "requests_in_window",
    "alg1_log_lines", "alg2_log_lines",
]

METRICS = [
    ("ttft_p50_ms", "TTFT p50 (ms)", "lower"),
    ("ttft_p99_ms", "TTFT p99 (ms)", "lower"),
    ("tpot_p50_ms", "TPOT p50 (ms)", "lower"),
    ("tpot_p99_ms", "TPOT p99 (ms)", "lower"),
    ("ttft_slo_attainment", "TTFT SLO attainment", "higher"),
    ("tpot_slo_attainment", "TPOT SLO attainment", "higher"),
    ("joint_slo_attainment", "Joint SLO attainment", "higher"),
    ("joint_slo_goodput_req_s", "Joint-SLO goodput (req/s)", "higher"),
]

PROTO, PAPER = "released-prototype", "paper-faithful"


def load_rows(base):
    rows = []
    for path in sorted(glob.glob(os.path.join(base, "raw", "*", "rate_*", "seed_*", "metrics.json"))):
        with open(path) as fh:
            rows.append(json.load(fh))
    return rows


def write_results(base, rows):
    out = os.path.join(base, "processed", "results.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    rows = sorted(rows, key=lambda r: (r["system"], r["request_rate"], r["seed"]))
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return out


def summarize(base, rows):
    """mean/std per (system, rate). std is the sample std over seeds."""
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["system"], r["request_rate"])].append(r)

    metric_keys = [c for c in COLUMNS if c not in ("system", "request_rate", "seed")]
    out = os.path.join(base, "processed", "summary.csv")
    fields = ["system", "request_rate", "n_runs"]
    for k in metric_keys:
        fields += [f"{k}_mean", f"{k}_std"]

    summary = {}
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for (sys_, rate) in sorted(grouped, key=lambda x: (x[0], x[1])):
            group = grouped[(sys_, rate)]
            row = {"system": sys_, "request_rate": rate, "n_runs": len(group)}
            for k in metric_keys:
                vals = [g[k] for g in group if isinstance(g.get(k), (int, float))
                        and not (isinstance(g[k], float) and np.isnan(g[k]))]
                row[f"{k}_mean"] = float(np.mean(vals)) if vals else float("nan")
                row[f"{k}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            w.writerow(row)
            summary[(sys_, rate)] = row
    return out, summary


def figures(base, summary):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"matplotlib unavailable ({e}); skipping figures")
        return []

    figdir = os.path.join(base, "figures")
    os.makedirs(figdir, exist_ok=True)
    written = []
    systems = sorted({s for s, _ in summary})

    for i, (key, label, _) in enumerate(METRICS, start=1):
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        drew = False
        for sys_ in systems:
            pts = sorted((r for (s, r) in summary if s == sys_))
            xs, ys, es = [], [], []
            for rate in pts:
                row = summary[(sys_, rate)]
                mean = row.get(f"{key}_mean")
                if mean is None or (isinstance(mean, float) and np.isnan(mean)):
                    continue
                xs.append(rate); ys.append(mean); es.append(row.get(f"{key}_std", 0.0))
            if xs:
                # Error bars are the seed-to-seed std (n=3); they are a spread
                # indicator, not a confidence interval.
                ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3, label=sys_)
                drew = True
        if not drew:
            plt.close(fig); continue
        ax.set_xlabel("Offered aggregate request rate (req/s)")
        ax.set_ylabel(label)
        ax.set_title(f"Figure {i}: Request rate vs {label}")
        ax.grid(alpha=0.3)
        ax.legend()
        if "attainment" in key:
            ax.set_ylim(-0.02, 1.02)
        fig.tight_layout()
        path = os.path.join(figdir, f"fig{i}_{key}.png")
        fig.savefig(path, dpi=140)
        plt.close(fig)
        written.append(path)
    return written


def improvements(summary):
    """Paper vs prototype at each rate. Latency: lower is better."""
    rates = sorted({r for (s, r) in summary if s == PROTO} & {r for (s, r) in summary if s == PAPER})
    out = []
    for rate in rates:
        p, q = summary[(PROTO, rate)], summary[(PAPER, rate)]
        entry = {"rate": rate}
        for key, label, direction in METRICS:
            a, b = p.get(f"{key}_mean"), q.get(f"{key}_mean")
            if a is None or b is None or np.isnan(a) or np.isnan(b) or a == 0:
                entry[key] = None
                continue
            rel = ((a - b) / a * 100) if direction == "lower" else ((b - a) / a * 100)
            entry[key] = {"proto": a, "paper": b, "rel_pct": rel, "abs_pp": (b - a) * 100
                          if "attainment" in key else None}
        out.append(entry)
    return out


def write_report(base, rows, summary, imps, figs):
    meta = ""
    mpath = os.path.join(base, "metadata", "run_metadata.txt")
    if os.path.exists(mpath):
        meta = open(mpath).read()

    rates = sorted({r["request_rate"] for r in rows})
    lines = []
    A = lines.append
    A("# Released Prism Prototype vs Paper-Faithful Prism\n")
    A("> Auto-generated by `exp/scripts/aggregate_pf.py`. Numbers are means over seeds "
      "with the seed-to-seed standard deviation; with n=3 these are a spread indicator, "
      "not a significance claim.\n")

    A("## 1. Research question\n")
    A("As offered request rate rises, does a faithful implementation of the paper's "
      "Algorithm 1 (KVPR global placement) and Algorithm 2 (Moore-Hodgson GPU-local "
      "scheduling) outperform the released prism-research prototype on TTFT, TPOT, "
      "SLO attainment and joint-SLO goodput?\n")

    A("## 2. Prototype vs paper — what actually differs\n")
    A("See `docs/paper_faithful/design_analysis.md` for the audited, line-referenced "
      "comparison. In short: the prototype balances a smoothed *request count* per byte "
      "of free memory with a 15x ratio threshold, and its local scheduler is a "
      "`deadline - exec` priority heap that admits everything "
      "(`net_available = float('inf')`). Neither the KVPR metric nor the "
      "drop-the-longest step of Moore-Hodgson exists upstream.\n")

    A("## 3-6. Hardware, software, models, workload\n")
    A("```\n" + meta + "```\n")
    A("Workload: ShareGPT text, gamma(cv=1)=Poisson arrivals, one trace per "
      "(rate, seed) **shared by both systems**, so prompts, lengths, routing and "
      "arrival timestamps are identical across arms.\n")

    A("## 11-14. Results\n")
    hdr = ("| Rate | System | TTFT p50 | TTFT p99 | TPOT p50 | TPOT p99 | "
           "TTFT att. | TPOT att. | Joint att. | Goodput |")
    A(hdr)
    A("| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for rate in rates:
        for sys_ in (PROTO, PAPER):
            row = summary.get((sys_, rate))
            if not row:
                continue
            def g(k, mul=1.0, fmt="{:.1f}"):
                v = row.get(f"{k}_mean")
                return "n/a" if v is None or np.isnan(v) else fmt.format(v * mul)
            A(f"| {rate:g} | {sys_} | {g('ttft_p50_ms')} | {g('ttft_p99_ms')} | "
              f"{g('tpot_p50_ms')} | {g('tpot_p99_ms')} | "
              f"{g('ttft_slo_attainment', 1, '{:.3f}')} | "
              f"{g('tpot_slo_attainment', 1, '{:.3f}')} | "
              f"{g('joint_slo_attainment', 1, '{:.3f}')} | "
              f"{g('joint_slo_goodput_req_s', 1, '{:.2f}')} |")
    A("")

    A("## 17-18. Paper-Faithful vs Prototype, by rate\n")
    A("Latency improvement = (prototype − paper) / prototype; goodput and attainment "
      "improvement = (paper − prototype) / prototype. Positive is always better for "
      "Paper-Faithful.\n")
    A("| Rate | TTFT p99 | TPOT p99 | Joint att. (pp) | Joint att. (rel) | Goodput |")
    A("| ---: | ---: | ---: | ---: | ---: | ---: |")
    for e in imps:
        def f(k, field="rel_pct"):
            v = e.get(k)
            return "n/a" if not v or v.get(field) is None else f"{v[field]:+.1f}%"
        ja = e.get("joint_slo_attainment")
        pp = "n/a" if not ja or ja.get("abs_pp") is None else f"{ja['abs_pp']:+.1f}pp"
        A(f"| {e['rate']:g} | {f('ttft_p99_ms')} | {f('tpot_p99_ms')} | {pp} | "
          f"{f('joint_slo_attainment')} | {f('joint_slo_goodput_req_s')} |")
    A("")

    A("## 15-16. Diagnostics\n")
    A("| Rate | System | migrations | evictions | activations | MH deferred | completed | failed |")
    A("| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for rate in rates:
        for sys_ in (PROTO, PAPER):
            row = summary.get((sys_, rate))
            if not row:
                continue
            def g(k):
                v = row.get(f"{k}_mean")
                return "n/a" if v is None or np.isnan(v) else f"{v:.1f}"
            A(f"| {rate:g} | {sys_} | {g('num_migrations')} | {g('num_evictions')} | "
              f"{g('num_activations')} | {g('num_mh_deferred')} | {g('num_completed')} | "
              f"{g('num_failed')} |")
    A("")

    if figs:
        A("## Figures\n")
        for p in figs:
            A(f"![{os.path.basename(p)}](figures/{os.path.basename(p)})")
        A("")

    A("## 19. Limitations\n")
    A("- n=3 seeds per point: enough for aggregates, thin for p99. Where the seed-to-seed "
      "std in `summary.csv` is comparable to the difference between arms, the difference "
      "is not resolved by this data.\n"
      "- `tau`, the token-rate window and the fate of Moore-Hodgson-deferred requests are "
      "not specified in the paper; the values used are recorded in the metadata and in "
      "`design_analysis.md`. No per-rate tuning was applied to either arm.\n"
      "- 3 models on 2 GPUs is necessarily a 1+2 split, so no placement policy can "
      "equalise load; this bounds how much Algorithm 1 can achieve here.\n"
      "- The prototype is a simplified research release, not the paper's artifact. A "
      "difference measured here is prototype-vs-paper-algorithm, not "
      "authors'-implementation-vs-paper.\n")

    A("## 20. Reproduction\n")
    A("```bash\nsource exp/scripts/env.sh\n"
      "./exp/run_paper_faithful_comparison.sh --dry-run   # print the 48-run plan\n"
      "./exp/run_paper_faithful_comparison.sh --resume    # run / resume the sweep\n"
      "python exp/tests/test_moore_hodgson.py             # Algorithm 2 unit tests\n"
      "python exp/tests/test_kvpr_placement.py            # Algorithm 1 unit tests\n```\n")

    path = os.path.join(base, "REPORT.md")
    open(path, "w").write("\n".join(lines))
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    a = ap.parse_args()

    rows = load_rows(a.base)
    if not rows:
        raise SystemExit(f"no metrics.json found under {a.base}/raw")
    res = write_results(a.base, rows)
    summ, summary = summarize(a.base, rows)
    figs = figures(a.base, summary)
    imps = improvements(summary)
    rep = write_report(a.base, rows, summary, imps, figs)
    print(f"rows={len(rows)}\nresults={res}\nsummary={summ}\nfigures={len(figs)}\nreport={rep}")


if __name__ == "__main__":
    main()
