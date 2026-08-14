#!/usr/bin/env python3
"""Four plots, no more: the comparison this study exists to make.

    python plot_v2.py --base exp/results/paper-faithful-v2 -o <base>/plots

  fig1  joint SLO attainment vs offered load, one panel per workload
  fig2  joint-SLO goodput vs offered load, one panel per workload
  fig3  TTFT p99 vs offered load (log y), one panel per workload
  fig4  Prism's relative joint-attainment gain over the baseline,
        bursty vs steady on the same axes -- the headline contrast
"""
import argparse
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WORKLOADS = ["steady", "bursty"]
COLORS = {"released-prototype": "#4C6EF5", "paper-faithful": "#E8590C",
          "paper-alg1-only": "#2F9E44", "paper-alg2-only": "#9C36B5"}


def num(v):
    try:
        f = float(v)
        return f if f == f else None
    except (TypeError, ValueError):
        return None


def load(base):
    p = os.path.join(base, "processed", "summary.csv")
    if not os.path.exists(p):
        return []
    with open(p) as f:
        return list(csv.DictReader(f))


def series(rows, workload, metric):
    out = defaultdict(list)
    for r in rows:
        if r["workload"] != workload:
            continue
        v, x = num(r[metric]), num(r["rate"])
        if v is not None and x is not None:
            out[r["system"]].append((x, v))
    return {k: sorted(v) for k, v in out.items()}


def panel(rows, metric, title, ylabel, path, logy=False):
    fig, axes = plt.subplots(1, len(WORKLOADS), figsize=(11, 4.2), sharey=True)
    for ax, wl in zip(axes, WORKLOADS):
        for sysname, pts in sorted(series(rows, wl, metric).items()):
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-",
                    label=sysname, color=COLORS.get(sysname), lw=2, ms=5)
        ax.set_title(wl)
        ax.set_xlabel("offered load (req/s)")
        ax.grid(alpha=0.3)
        if logy:
            ax.set_yscale("log")
    axes[0].set_ylabel(ylabel)
    axes[-1].legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print("wrote", path)


def contrast(rows, out):
    base = {(r["workload"], r["rate"]): num(r["joint_attainment"])
            for r in rows if r["system"] == "released-prototype"}
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    style = {"steady": ("-", "o"), "bursty": ("--", "s")}
    plotted = False
    for wl in WORKLOADS:
        pts = []
        for r in rows:
            if r["workload"] != wl or r["system"] != "paper-faithful":
                continue
            b = base.get((wl, r["rate"]))
            v, x = num(r["joint_attainment"]), num(r["rate"])
            if b and v is not None and x is not None:
                pts.append((x, 100.0 * (v - b) / b))
        if pts:
            pts.sort()
            ls, mk = style[wl]
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker=mk, ls=ls,
                    lw=2, ms=6, label=f"{wl}")
            plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.axhline(0, color="k", lw=1)
    ax.set_xlabel("offered load (req/s)")
    ax.set_ylabel("paper-faithful vs baseline\njoint attainment (% relative)")
    ax.set_title("Same request set, same offered load — only arrival timing differs")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = load(a.base)
    if not rows:
        print("no summary.csv yet")
        return
    panel(rows, "joint_attainment", "Joint SLO attainment", "joint attainment",
          os.path.join(a.out, "fig1_joint_attainment.png"))
    panel(rows, "goodput_req_s", "Joint-SLO goodput", "goodput (req/s)",
          os.path.join(a.out, "fig2_goodput.png"))
    panel(rows, "ttft_p99_ms", "TTFT p99", "TTFT p99 (ms)",
          os.path.join(a.out, "fig3_ttft_p99.png"), logy=True)
    contrast(rows, os.path.join(a.out, "fig4_bursty_vs_steady.png"))


if __name__ == "__main__":
    main()
