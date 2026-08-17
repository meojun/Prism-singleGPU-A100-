#!/usr/bin/env python3
"""Strict completion gate for the 2 systems x 2 workloads x 4 rates v3 sweep."""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--rates", nargs="+", default=["4", "8", "14", "20"])
    ap.add_argument("--seed", default="1")
    a = ap.parse_args()
    base = Path(a.base)
    systems = ("released-prototype", "paper-faithful-v3")
    workloads = ("steady", "bursty")
    errors = []
    checked = []
    for system in systems:
        for workload in workloads:
            for rate in a.rates:
                run = base / "raw" / system / workload / f"rate_{rate}" / f"seed_{a.seed}"
                done, metrics = run / "DONE", run / "metrics.json"
                if not done.is_file():
                    errors.append(f"missing DONE: {run}")
                    continue
                if not metrics.is_file():
                    errors.append(f"missing metrics.json: {run}")
                    continue
                try:
                    d = json.loads(metrics.read_text())
                except Exception as exc:
                    errors.append(f"invalid metrics.json: {run}: {exc}")
                    continue
                required = ("requests_in_window", "completed", "failed",
                            "joint_attainment", "goodput_req_s", "ttft_p99_ms", "tpot_p99_ms")
                missing = [k for k in required if d.get(k) is None]
                if missing:
                    errors.append(f"missing metrics {missing}: {run}")
                elif d["requests_in_window"] <= 0 or d["completed"] <= 0:
                    errors.append(f"empty measurement window: {run}")
                elif d.get("failed", 0) > 0 and not (run / "REPRODUCED_REQUEST_FAILURE").is_file():
                    errors.append(f"request failures were not retried/reproduced: {run}")
                else:
                    checked.append(str(run))
    out = {"expected": 16, "valid": len(checked), "errors": errors, "runs": checked}
    (base / "VALIDATION.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"expected": 16, "valid": len(checked), "errors": errors}, indent=2))
    raise SystemExit(1 if errors or len(checked) != 16 else 0)


if __name__ == "__main__":
    main()
