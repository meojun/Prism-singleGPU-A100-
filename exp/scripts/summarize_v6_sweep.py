#!/usr/bin/env python3
"""Summarise the v4-vs-v6 sweep: did KV migration engage, and did it matter?

Two questions, kept apart on purpose.

Engagement is a yes/no the raw data answers directly -- capsules stashed,
capsules injected, KV bytes moved, requests failed.  If those are zero then the
two arms were the same system in that cell, and a goodput table there would
read as a result while being nothing of the kind.  So it is not printed.

Effect is a comparison, and only reportable with the spread beside it.  The v4
study measured seed-to-seed spread at 70-80% of the mean at these rates, so a
difference in means smaller than the spread is not a finding.  Performance
numbers come from summary.csv, written by the project's own
collect_v4_metrics.py, so goodput, SLO attainment and the 60 s warmup exclusion
carry the same definitions every earlier study used.
"""
import argparse
import csv
import glob
import json
import os
import re
import statistics as st
from collections import defaultdict


def num(x, d=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return d


def engagement(run_dir):
    """Did the mechanism fire in this run?  Read from its own raw records."""
    out = dict(kv_records=0, kv_bytes=0, kv_moved=0, kv_skipped=0,
               stash=0, inject=0, capture_failures=0, paths=set())
    kv = os.path.join(run_dir, "kv_transfers.jsonl")
    if os.path.exists(kv):
        for line in open(kv):
            try:
                r = json.loads(line)
            except Exception:
                continue
            out["kv_records"] += 1
            out["kv_bytes"] += int(num(r.get("kv_bytes")))
            out["kv_moved"] += int(num(r.get("requests_moved")))
            out["kv_skipped"] += int(num(r.get("requests_skipped_over_cap")))
            if r.get("transfer_path"):
                out["paths"].add(r["transfer_path"])
    for f in glob.glob(os.path.join(run_dir, "server-logs", "*")):
        if not os.path.isfile(f):
            continue
        try:
            t = open(f, errors="ignore").read()
        except OSError:
            continue
        out["stash"] += t.count('"event": "stash"')
        out["inject"] += t.count('"event": "inject"')
        out["capture_failures"] += len(re.findall(r"capture failed", t))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    a = ap.parse_args()
    base = a.base

    # ---------------------------------------------------------- engagement
    eng = defaultdict(lambda: defaultdict(int))
    eng_paths = defaultdict(set)
    runs = sorted(glob.glob(os.path.join(base, "raw/*/*/rate_*/seed_*/DONE")))
    for done in runs:
        d = os.path.dirname(done)
        parts = d.split(os.sep)
        arm, wl, rate = parts[-4], parts[-3], parts[-2].replace("rate_", "")
        e = engagement(d)
        key = (wl, rate, arm)
        for k in ("kv_records", "kv_bytes", "kv_moved", "kv_skipped",
                  "stash", "inject", "capture_failures"):
            eng[key][k] += e[k]
        eng[key]["runs"] += 1
        eng_paths[key] |= e["paths"]

    print("=" * 82)
    print("1. Did the mechanism engage?   (v4 is the control -- it must read zero)")
    print("=" * 82)
    if not runs:
        print("no finished runs under", base)
        return 1
    print(f"{'workload':9} {'rate':>5} {'arm':22} {'n':>2} {'stash':>6} {'inject':>7} "
          f"{'reqs':>6} {'KV MiB':>9} {'capfail':>8}  paths")
    for key in sorted(eng):
        wl, rate, arm = key
        e = eng[key]
        print(f"{wl:9} {rate:>5} {arm:22} {e['runs']:>2} {e['stash']:>6} "
              f"{e['inject']:>7} {e['kv_moved']:>6} {e['kv_bytes']/2**20:>9.1f} "
              f"{e['capture_failures']:>8}  {','.join(sorted(eng_paths[key])) or '-'}")

    # ------------------------------------------------------------ summary.csv
    scsv = os.path.join(base, "summary.csv")
    if not os.path.exists(scsv):
        print(f"\nNo summary.csv -- collect_v4_metrics.py did not run or failed.")
        print("Engagement above is still valid; performance cannot be reported.")
        return 1
    rows = list(csv.DictReader(open(scsv)))

    fails = defaultdict(int)
    for r in rows:
        fails[(r["workload"], r["request_rate"], r["implementation"])] += \
            int(num(r.get("failed_requests")))

    print()
    verdict_ok = {}
    for key in sorted(eng):
        wl, rate, arm = key
        if "v6" not in arm:
            continue
        e = eng[key]
        f = fails.get(key, 0)
        moved, mib = e["kv_moved"], e["kv_bytes"] / 2 ** 20
        if moved == 0 or mib == 0:
            verdict_ok[(wl, rate)] = False
            print(f"  {wl} r{rate}: NOT ENGAGED -- no KV moved. Below, the two arms "
                  f"are the same system here; no comparison is printed.")
            if e["stash"] == 0:
                print("      stash=0: nothing was captured. Either the migration did "
                      "not preempt, or the retract path was not taken.")
            elif e["inject"] == 0:
                print("      stash>0 but inject=0: capsules were handed over and never "
                      "collected -- look at the model service routing.")
            if e["capture_failures"]:
                print(f"      {e['capture_failures']} capture failures -- grep the "
                      f"server logs for 'capture failed'.")
        elif f:
            verdict_ok[(wl, rate)] = False
            print(f"  {wl} r{rate}: engaged ({moved} requests, {mib:.1f} MiB) but "
                  f"{f} requests FAILED. Not a pass.")
        else:
            verdict_ok[(wl, rate)] = True
            note = f", {e['capture_failures']} capture failures" if e["capture_failures"] else ""
            print(f"  {wl} r{rate}: engaged -- {moved} requests, {mib:.1f} MiB, "
                  f"0 failed requests{note}")

    # ---------------------------------------------------------------- effect
    print()
    print("=" * 82)
    print("2. Did it change anything?   (from summary.csv -- the project's own"
          " definitions)")
    print("=" * 82)
    cells = defaultdict(list)
    for r in rows:
        cells[(r["workload"], r["request_rate"], r["implementation"])].append(r)

    print(f"{'workload':9} {'rate':>5} {'arm':22} {'n':>2} {'goodput':>17}  "
          f"{'joint SLO':>16}  per-seed goodput")
    means = {}
    for key in sorted(cells):
        wl, rate, arm = key
        g = [num(r.get("goodput")) for r in cells[key]]
        j = [num(r.get("joint_slo")) for r in cells[key]]
        gm, gsd = st.mean(g), (st.pstdev(g) if len(g) > 1 else 0.0)
        jm, jsd = st.mean(j), (st.pstdev(j) if len(j) > 1 else 0.0)
        means[key] = (gm, gsd, g)
        print(f"{wl:9} {rate:>5} {arm:22} {len(g):>2} {gm:>8.3f} +- {gsd:<6.3f} "
              f"{jm:>7.3f} +- {jsd:<6.3f}  " + ", ".join(f"{v:.2f}" for v in g))

    print()
    print("Verdict per condition -- a gap inside the combined spread is not a finding:")
    for (wl, rate) in sorted({(w, r) for (w, r, _) in means}):
        ctl = next((v for (w, r, arm), v in means.items()
                    if (w, r) == (wl, rate) and "v6" not in arm), None)
        trt = next((v for (w, r, arm), v in means.items()
                    if (w, r) == (wl, rate) and "v6" in arm), None)
        if not ctl or not trt:
            continue
        if not verdict_ok.get((wl, rate)):
            print(f"  {wl} r{rate}: mechanism did not engage (or requests failed) "
                  f"-- no comparison.")
            continue
        (cm, csd, cv), (tm, tsd, tv) = ctl, trt
        spread = csd + tsd
        if abs(tm - cm) <= spread:
            print(f"  {wl} r{rate}: v6 {tm:.3f} vs v4 {cm:.3f} -- INDISTINGUISHABLE "
                  f"(gap {abs(tm-cm):.3f} <= spread {spread:.3f}, n={len(tv)}/arm)")
        else:
            d = "higher" if tm > cm else "lower"
            print(f"  {wl} r{rate}: v6 {d} -- {tm:.3f} vs {cm:.3f}, gap "
                  f"{abs(tm-cm):.3f} > spread {spread:.3f}, n={len(tv)}/arm. "
                  f"Provisional: this n cannot settle it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
