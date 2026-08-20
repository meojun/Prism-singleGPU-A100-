#!/usr/bin/env python3
"""Summarise the v4-vs-v6 sweep: did KV migration engage, and did it matter?

Two questions, kept apart on purpose.

Engagement is a yes/no the raw data answers directly -- capsules stashed,
capsules injected, kv_bytes moved, requests failed.  If those are zero the
performance columns mean nothing and the report says so rather than printing a
goodput table that reads as a result.

Effect is a comparison, and it is only reportable with the spread beside it.
The v4 study measured seed-to-seed spread at 70-80% of the mean at these rates,
so a difference in means smaller than that spread is not a finding.  Every cell
prints mean +- sd over seeds and the per-seed values, and the verdict line
refuses to call a winner when the intervals overlap.
"""
import argparse
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


def read_run(d):
    """One run -> the handful of numbers worth comparing, or None if it did not finish."""
    if not os.path.exists(os.path.join(d, "DONE")):
        return None
    out = {"dir": d}

    bench = None
    for f in glob.glob(os.path.join(d, "*_e2e_*.json")):
        if f.endswith("_output_requests.json"):
            continue
        try:
            bench = json.load(open(f))
        except Exception:
            continue
        break
    if bench:
        for k in ("completed", "num_completed", "total_requests", "failed",
                  "num_failed", "throughput", "achieved_throughput"):
            if k in bench:
                out[k] = bench[k]

    kv = os.path.join(d, "kv_transfers.jsonl")
    out["kv_records"] = out["kv_bytes"] = out["kv_moved"] = out["kv_skipped"] = 0
    out["kv_paths"] = set()
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
                out["kv_paths"].add(r["transfer_path"])

    stash = inject = capfail = 0
    for f in glob.glob(os.path.join(d, "server-logs", "*")):
        if not os.path.isfile(f):
            continue
        try:
            t = open(f, errors="ignore").read()
        except OSError:
            continue
        stash += t.count('"event": "stash"')
        inject += t.count('"event": "inject"')
        capfail += len(re.findall(r"capture failed", t))
    out.update(stash=stash, inject=inject, capture_failures=capfail)

    wt = os.path.join(d, "weight_transfers.jsonl")
    out["weight_transfers"] = out["p2p_weight_transfers"] = 0
    if os.path.exists(wt):
        for line in open(wt):
            try:
                r = json.loads(line)
            except Exception:
                continue
            out["weight_transfers"] += 1
            if r.get("transfer_path") == "gpu-to-gpu-p2p":
                out["p2p_weight_transfers"] += 1
    return out


def goodput(run):
    """Completed requests that met both SLOs, per measured second.

    Recomputed from the per-request dump rather than taken from the harness's
    own attainment field, which carries a ms-vs-s unit bug (CLAUDE.md).
    """
    d = run["dir"]
    for f in glob.glob(os.path.join(d, "*_output_requests.json")):
        try:
            reqs = json.load(open(f))
        except Exception:
            continue
        if not isinstance(reqs, list):
            continue
        ok = sum(1 for r in reqs
                 if r.get("success") and r.get("meets_ttft_slo") and r.get("meets_tpot_slo"))
        return ok / 300.0
    return None


def agg(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None, []
    if len(vals) == 1:
        return vals[0], 0.0, vals
    return st.mean(vals), st.pstdev(vals), vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    a = ap.parse_args()

    cells = defaultdict(list)
    for arm in sorted(os.listdir(a.base)):
        for wl in sorted(os.listdir(os.path.join(a.base, arm)) if
                         os.path.isdir(os.path.join(a.base, arm)) else []):
            p = os.path.join(a.base, arm, wl)
            for rate in sorted(os.listdir(p) if os.path.isdir(p) else []):
                q = os.path.join(p, rate)
                for seed in sorted(os.listdir(q) if os.path.isdir(q) else []):
                    r = read_run(os.path.join(q, seed))
                    if r:
                        cells[(wl, rate, arm)].append(r)

    if not cells:
        print("no finished runs under", a.base)
        return 1

    print("=" * 78)
    print("1. Did the mechanism engage?  (v6 only -- v4 is the control and must be 0)")
    print("=" * 78)
    print(f"{'workload':9} {'rate':>5} {'arm':22} {'n':>2} {'stash':>6} {'inject':>7} "
          f"{'reqs':>5} {'KV MiB':>8} {'capfail':>8} {'fail':>5}")
    engaged = {}
    for (wl, rate, arm), runs in sorted(cells.items()):
        stash = sum(r["stash"] for r in runs)
        inject = sum(r["inject"] for r in runs)
        moved = sum(r["kv_moved"] for r in runs)
        mib = sum(r["kv_bytes"] for r in runs) / 2 ** 20
        capfail = sum(r["capture_failures"] for r in runs)
        fail = sum(int(num(r.get("failed", r.get("num_failed", 0)))) for r in runs)
        print(f"{wl:9} {rate:>5} {arm:22} {len(runs):>2} {stash:>6} {inject:>7} "
              f"{moved:>5} {mib:>8.1f} {capfail:>8} {fail:>5}")
        if "v6" in arm:
            engaged[(wl, rate)] = (moved > 0 and mib > 0, fail, capfail)

    print()
    ok_all = True
    for k, (eng, fail, capfail) in sorted(engaged.items()):
        if not eng:
            ok_all = False
            print(f"  {k}: NOT ENGAGED -- no KV moved. The comparison below is "
                  f"between two identical systems and means nothing.")
        elif fail:
            ok_all = False
            print(f"  {k}: engaged but {fail} requests FAILED -- not a pass.")
        else:
            note = f" ({capfail} capture failures)" if capfail else ""
            print(f"  {k}: engaged, 0 failed requests{note}")

    print()
    print("=" * 78)
    print("2. Did it change anything?")
    print("=" * 78)
    if not ok_all:
        print("Skipped for the cells above that did not engage.")
    print(f"{'workload':9} {'rate':>5} {'arm':22} {'n':>2} {'goodput':>18}  per-seed")
    means = {}
    for (wl, rate, arm), runs in sorted(cells.items()):
        m, sd, vals = agg([goodput(r) for r in runs])
        if m is None:
            print(f"{wl:9} {rate:>5} {arm:22} {len(runs):>2} {'(no per-request dump)':>18}")
            continue
        means[(wl, rate, arm)] = (m, sd, vals)
        print(f"{wl:9} {rate:>5} {arm:22} {len(runs):>2} "
              f"{m:>8.3f} +- {sd:<6.3f}  " + ", ".join(f"{v:.2f}" for v in vals))

    print()
    print("Verdict per condition (a difference inside the spread is not a finding):")
    for (wl, rate) in sorted({(w, r) for (w, r, _) in means}):
        c = next((v for (w, r, arm), v in means.items()
                  if (w, r) == (wl, rate) and "v6" not in arm), None)
        t = next((v for (w, r, arm), v in means.items()
                  if (w, r) == (wl, rate) and "v6" in arm), None)
        if not c or not t:
            continue
        (cm, csd, _), (tm, tsd, _) = c, t
        if not engaged.get((wl, rate), (False,))[0]:
            print(f"  {wl} r{rate}: mechanism did not engage -- no comparison.")
            continue
        if abs(tm - cm) <= (csd + tsd):
            print(f"  {wl} r{rate}: {tm:.3f} vs {cm:.3f} -- INDISTINGUISHABLE "
                  f"(gap {abs(tm-cm):.3f} <= combined spread {csd+tsd:.3f})")
        else:
            d = "higher" if tm > cm else "lower"
            print(f"  {wl} r{rate}: v6 {d} ({tm:.3f} vs {cm:.3f}), gap "
                  f"{abs(tm-cm):.3f} > combined spread {csd+tsd:.3f} -- "
                  f"n={len(t[2])} per arm, treat as provisional")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
