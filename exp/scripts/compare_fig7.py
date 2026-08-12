#!/usr/bin/env python3
"""Side-by-side for the §7.3 / Figure 7 ablation: global placement on vs off.

Figure 7a is SLO attainment with the global scheduler enabled/disabled; 7b is
the resulting per-GPU load imbalance. The harness records neither, so:
  * 7a comes from analyze_slo.py's recomputed attainment (results/<tag>/*_slo.json)
  * 7b comes from the nvidia-smi timeline run_2gpu.sh samples during the run
    (server-logs/<exp>/gpu_timeline.txt) -- a proxy for the paper's
    "available KV memory per request per GPU", which the engines do not export.

  python compare_fig7.py [--tag fig7] [--ts 1]
"""
import argparse
import glob
import json
import os
import statistics as st

ROOT = os.environ.get("PRISM_EXP", "/workspace/prism-exp/exp")
ARMS = [("glob_on", "w/ global"), ("glob_off", "w/o global")]


def load_arm(tag, arm, ts):
    f = os.path.join(ROOT, "results", tag, f"{tag}_{arm}_ts{ts}_slo.json")
    return json.load(open(f)) if os.path.exists(f) else None


def load_timeline(tag, arm, ts):
    """gpu_timeline.txt lines: '<epoch> 0, <MiB>, <util>;1, <MiB>, <util>;'"""
    f = os.path.join(ROOT, "server-logs", f"{tag}_{arm}_ts{ts}", "gpu_timeline.txt")
    if not os.path.exists(f):
        return None
    per_gpu = {}
    for line in open(f):
        parts = line.strip().split(" ", 1)
        if len(parts) != 2:
            continue
        for entry in parts[1].split(";"):
            entry = entry.strip()
            if not entry:
                continue
            fields = [x.strip() for x in entry.split(",")]
            if len(fields) < 3:
                continue
            try:
                idx, mem, util = int(fields[0]), float(fields[1]), float(fields[2])
            except ValueError:
                continue
            per_gpu.setdefault(idx, {"mem": [], "util": []})
            per_gpu[idx]["mem"].append(mem)
            per_gpu[idx]["util"].append(util)
    return per_gpu or None


def actions(tag, arm, ts):
    f = os.path.join(ROOT, "results", tag, f"{tag}_{arm}_ts{ts}_actions.txt")
    return open(f).read().strip() if os.path.exists(f) else "(none recorded)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="fig7")
    ap.add_argument("--ts", default="1")
    a = ap.parse_args()

    data = {arm: load_arm(a.tag, arm, a.ts) for arm, _ in ARMS}
    have = [arm for arm, _ in ARMS if data[arm]]
    if not have:
        raise SystemExit(f"no results under {ROOT}/results/{a.tag} for ts={a.ts}")

    print(f"=== §7.3 Figure 7 ablation — tag={a.tag}, time_scale={a.ts} ===\n")

    # --- 7a: attainment ------------------------------------------------------
    hdr = f"{'model':<9}{'reqs':>6}"
    for _, lbl in ARMS:
        hdr += f"{lbl+' TTFT':>16}{lbl+' TPOT':>16}{lbl+' both':>16}"
    print(hdr)
    print("-" * len(hdr))
    names = sorted(
        {n for arm in have for n in data[arm]["per_model"] if n != "ALL"},
        key=lambda s: int(s.split("_")[1]),
    ) + ["ALL"]
    for n in names:
        any_row = next(data[arm]["per_model"][n] for arm in have if n in data[arm]["per_model"])
        line = f"{n:<9}{any_row['requests']:>6}"
        for arm, _ in ARMS:
            r = data[arm]["per_model"].get(n) if data[arm] else None
            if r is None:
                line += f"{'-':>16}{'-':>16}{'-':>16}"
            else:
                line += f"{r['attain_ttft']:>16.3f}{r['attain_tpot']:>16.3f}{r['attain_both']:>16.3f}"
        print(line)

    print()
    for arm, lbl in ARMS:
        d = data[arm]
        if not d:
            continue
        allr = d["per_model"]["ALL"]
        print(f"{lbl:<12} completed {d['completed']}/{d['total_requests']}  "
              f"dur {d['duration_s']:.0f}s  goodput {allr['goodput_rps']:.2f} req/s "
              f"({allr['goodput_tok_s']:.0f} tok/s)  "
              f"TTFT p99 {allr['ttft_p99_ms']:.0f}ms  TPOT p99 {allr['tpot_p99_ms']:.1f}ms")

    # --- 7b: per-GPU imbalance ----------------------------------------------
    print("\n--- per-GPU load (nvidia-smi sampled @2s) ---")
    for arm, lbl in ARMS:
        tl = load_timeline(a.tag, arm, a.ts)
        if not tl:
            print(f"{lbl:<12} (no timeline)")
            continue
        parts, means = [], []
        for g in sorted(tl):
            m, u = tl[g]["mem"], tl[g]["util"]
            means.append(st.mean(m))
            parts.append(f"GPU{g} mem avg {st.mean(m)/1024:.1f}GiB max {max(m)/1024:.1f}GiB "
                         f"util avg {st.mean(u):.0f}%")
        imb = (max(means) / min(means)) if min(means) > 0 else float("inf")
        print(f"{lbl:<12} " + " | ".join(parts) + f"  -> mem imbalance {imb:.2f}x")

    # --- what the controller did --------------------------------------------
    print("\n--- global controller actions ---")
    for arm, lbl in ARMS:
        print(f"{lbl}:")
        for ln in actions(a.tag, arm, a.ts).splitlines():
            print(f"    {ln}")


if __name__ == "__main__":
    main()
