#!/usr/bin/env python3
"""Apply the pre-registered held-out tau rule and write an immutable freeze file."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def number(value):
    return float(value) if value not in (None, "", "nan") else math.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--freeze", required=True)
    ns = ap.parse_args()
    base, freeze = Path(ns.base), Path(ns.freeze)
    if freeze.exists():
        print(f"tau already frozen: {freeze}")
        return

    rows = []
    for protocol in sorted(base.glob("tau_*/summary.csv")):
        tau = protocol.parent.name.removeprefix("tau_").replace("p", ".")
        with protocol.open() as fh:
            samples = list(csv.DictReader(fh))
        if len(samples) != 2 or {int(r["seed"]) for r in samples} != {0, 42}:
            raise SystemExit(f"incomplete held-out samples for {tau}: {len(samples)}")
        rows.append({
            "tau": tau,
            "mean_joint_slo_goodput": sum(number(r["goodput"]) for r in samples) / 2,
            "migration_count": sum(int(float(r["migration_count"])) for r in samples),
            "migration_total_bytes": sum(number(r["migration_total_bytes"]) for r in samples),
            "summary_sha256": hashlib.sha256(protocol.read_bytes()).hexdigest(),
        })
    if len(rows) != 6:
        raise SystemExit(f"expected six tau candidates, found {len(rows)}")
    best = max(r["mean_joint_slo_goodput"] for r in rows)
    eligible = [r for r in rows if r["mean_joint_slo_goodput"] >= 0.97 * best]

    def tau_rank(value):
        return math.inf if value == "inf" else float(value)

    # Within 3% of the best goodput, minimize moved bytes, then migration
    # count, then choose the more conservative (larger) tau.
    selected = min(eligible, key=lambda r: (
        r["migration_total_bytes"], r["migration_count"], -tau_rank(r["tau"])))
    payload = {
        "status": "FROZEN", "selected_tau": selected["tau"],
        "objective": "mean Joint-SLO goodput over held-out seeds 0 and 42",
        "eligibility": "within 3% relative of maximum mean goodput",
        "tie_break": "least migration bytes, then count, then larger tau",
        "candidates": rows,
    }
    freeze.parent.mkdir(parents=True, exist_ok=True)
    freeze.write_text(json.dumps(payload, indent=2) + "\n")
    (base / "SELECTION.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
