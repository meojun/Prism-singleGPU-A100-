#!/usr/bin/env python3
"""Join Algorithm 1's per-cycle log to the bursty phase schedule.

Answers Q4/Q5 of the brief with evidence rather than assertion:
  - does the hot set actually move, and does KVPR follow it?
  - do placement changes / migrations / evictions / activations rise under
    shifting-bursty relative to steady?
  - when Prism wins, is the win traceable to KVPR balancing and memory headroom?

    python analyze_alg1_v2.py --run-dir <run> --phases <phases_*.json> -o out.json
"""
import argparse
import glob
import json
import os
import re
import statistics


def alg1_cycles(run_dir):
    out = []
    p = os.path.join(run_dir, "server-logs", "server.log.global_controller.log")
    if not os.path.exists(p):
        return out
    with open(p, errors="ignore") as f:
        for line in f:
            if "[PAPER-ALG1]" in line and " MIGRATE " not in line:
                m = re.search(r"\[PAPER-ALG1\] (\{.*\})\s*$", line)
                if m:
                    try:
                        out.append(json.loads(m.group(1)))
                    except json.JSONDecodeError:
                        pass
    return sorted(out, key=lambda c: c["t"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--phases", default=None)
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args()

    cyc = alg1_cycles(a.run_dir)
    if not cyc:
        json.dump({"error": "no [PAPER-ALG1] cycles (prototype arm?)"}, open(a.out, "w"), indent=2)
        print("no Algorithm 1 log in", a.run_dir)
        return

    t0 = cyc[0]["t"]
    phases = json.load(open(a.phases))["phases"] if a.phases and os.path.exists(a.phases) else []

    rows = []
    for c in cyc:
        rel = c["t"] - t0
        ph = next((i for i, p in enumerate(phases) if p["start"] <= rel < p["end"]), None)
        kv = {int(k): v for k, v in (c.get("kvpr") or {}).items()}
        vals = [v for v in kv.values() if v]
        rows.append({
            "t_rel": round(rel, 1), "phase": ph,
            "hot": phases[ph]["hot"] if ph is not None else None,
            "peak_kvpr": c.get("peak_kvpr"),
            "kvpr_spread": ((max(vals) - min(vals)) / max(vals)) if len(vals) > 1 and max(vals) else None,
            "shared_kv": c.get("shared_kv"),
            "improvement": c.get("improvement"),
            "decision": c.get("migration_decision"),
            "reason": c.get("migration_reason"),
            "top_model_by_w": max(c["models"], key=lambda m: m["w_token_rate"])["model"] if c.get("models") else None,
            "gpu_of_top": next((m["current_gpu"] for m in sorted(
                c["models"], key=lambda m: -m["w_token_rate"])), None) if c.get("models") else None,
        })

    peaks = [r["peak_kvpr"] for r in rows if r["peak_kvpr"]]
    spreads = [r["kvpr_spread"] for r in rows if r["kvpr_spread"] is not None]
    imps = [r["improvement"] for r in rows if r["improvement"] is not None]
    reasons = {}
    for r in rows:
        reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
    # does the model carrying the highest weighted token rate change over time?
    tops = [r["top_model_by_w"] for r in rows if r["top_model_by_w"]]
    switches = sum(1 for i in range(1, len(tops)) if tops[i] != tops[i - 1])

    summary = {
        "run_dir": a.run_dir,
        "cycles": len(rows),
        "peak_kvpr_mean": statistics.fmean(peaks) if peaks else None,
        "peak_kvpr_cv": (statistics.stdev(peaks) / statistics.fmean(peaks))
                        if len(peaks) > 1 and statistics.fmean(peaks) else None,
        "kvpr_spread_mean": statistics.fmean(spreads) if spreads else None,
        "kvpr_spread_max": max(spreads) if spreads else None,
        "improvement_mean": statistics.fmean(imps) if imps else None,
        "improvement_std": statistics.stdev(imps) if len(imps) > 1 else None,
        "improvement_max": max(imps) if imps else None,
        "migrations": sum(1 for r in rows if r["decision"] == "MIGRATE"),
        "decision_reasons": reasons,
        "hottest_model_switches": switches,
        "distinct_hottest_models": len(set(tops)),
    }
    json.dump({"summary": summary, "cycles": rows}, open(a.out, "w"), indent=1)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
