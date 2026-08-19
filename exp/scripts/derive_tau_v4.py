#!/usr/bin/env python3
"""Derive Algorithm 1's tau from a migration-suppressed run on this machine.

The rule is the one this project already used (docs/paper_faithful/
design_analysis.md 5a): put the threshold at mean + 2 sd of the measured
line-8 improvement distribution, so a migration fires on real imbalance rather
than on sampling noise in the rate estimator.

Only cycles where a move was actually available count.  Algorithm 1 evaluates
every model every cycle, and a model that is already on its best GPU
contributes a delta of exactly 0; those zeros are not observations of the
noise floor, and letting them into the mean would drag the threshold down
towards a value that lets noise through.
"""
import argparse
import glob
import json
import os

import numpy as np


def read_alg1(run_dir):
    out = []
    for path in glob.glob(os.path.join(run_dir, "server-logs", "*global_controller*")):
        with open(path, errors="replace") as fh:
            for line in fh:
                for marker in ("[PAPER-ALG1-V4] ", "[PAPER-ALG1-V3] "):
                    idx = line.find(marker)
                    if idx < 0:
                        continue
                    try:
                        out.append(json.loads(line[idx + len(marker):]))
                    except json.JSONDecodeError:
                        pass
                    break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args()

    recs = read_alg1(a.run_dir)
    if not recs:
        raise SystemExit(f"no Algorithm 1 records under {a.run_dir}")

    all_deltas, moves = [], []
    for rec in recs:
        for row in rec.get("line8", []):
            d = row.get("absolute_delta")
            if d is None or not np.isfinite(d):
                continue
            all_deltas.append(d)
            if row.get("best_gpu") != row.get("current_gpu"):
                moves.append(d)

    arr = np.array(moves if moves else all_deltas, dtype=float)
    tau = float(arr.mean() + 2 * arr.std())
    peaks = [r["peak_kvpr"] for r in recs if r.get("peak_kvpr")]

    result = {
        "tau": tau,
        "rule": "mean + 2 sd of line-8 absolute delta, over cycles where a "
                "different GPU was actually the argmin",
        "cycles": len(recs),
        "decisions_total": len(all_deltas),
        "decisions_with_a_move_available": len(moves),
        "delta_mean": float(arr.mean()),
        "delta_sd": float(arr.std()),
        "delta_median": float(np.median(arr)),
        "delta_p90": float(np.percentile(arr, 90)),
        "delta_p99": float(np.percentile(arr, 99)),
        "delta_max": float(arr.max()),
        "fraction_of_all_decisions_above_tau":
            float((np.array(all_deltas) > tau).mean()) if all_deltas else 0.0,
        "peak_kvpr_median": float(np.median(peaks)) if peaks else None,
        "migrations_during_calibration": sum(
            1 for r in recs if r.get("migration_decision") == "MIGRATE"),
    }
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(result, fh, indent=2)

    print(f"\ntau = {tau:.6g}")
    print(f"  from {result['decisions_with_a_move_available']} decisions where a move "
          f"was available (of {result['decisions_total']} total, {result['cycles']} cycles)")
    print(f"  delta mean={result['delta_mean']:.6g} sd={result['delta_sd']:.6g} "
          f"p90={result['delta_p90']:.6g} max={result['delta_max']:.6g}")
    print(f"  this tau would admit {result['fraction_of_all_decisions_above_tau']:.1%} "
          f"of all line-8 decisions")
    if result["migrations_during_calibration"]:
        print(f"  WARNING: {result['migrations_during_calibration']} migrations fired "
              f"during calibration -- the estimate is not from a static placement")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
