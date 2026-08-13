#!/usr/bin/env python3
"""Measure per-model chunked-prefill speed c_i (tokens/s) for Algorithm 2's e_i = p_i / c_i.

The brief forbids an arbitrary constant, and there is no prefill-speed table anywhere
in prism-research, model_info.json or kvcached. So c_i is fitted from an UNCONTENDED
solo run: on a GPU running one model with no queueing, TTFT is essentially the prefill
time, so

    ttft(p) ~= intercept + p / c

is fitted by least squares over the run's (prompt_len, ttft) pairs and c recovered as
1/slope. Requests are trimmed to the central TTFT quantiles first, because a handful of
scheduler-delayed outliers would otherwise dominate an ordinary least-squares slope.

The result is written once to exp/configs/prefill_speed.json and read by the server at
start-up, so c_i cannot drift between runs of the sweep.

    python exp/scripts/profile_prefill_speed.py --req-file <dump.json> \
        --source-slot model_1 --also-slots model_4,model_5 -o exp/configs/prefill_speed.json
"""
import argparse
import glob
import json
import os

import numpy as np


def load(req_file):
    with open(req_file) as fh:
        lines = [ln for ln in fh.read().splitlines() if ln.strip()]
    return json.loads(lines[-1])


def fit_speed(reqs, slot, lo_q=5, hi_q=80):
    pts = [
        (r["prompt_len"], r["ttft"])
        for r in reqs
        if r.get("success") and r.get("model") == slot and r.get("prompt_len")
    ]
    if len(pts) < 30:
        raise SystemExit(f"{slot}: only {len(pts)} usable samples, need >= 30")

    p = np.array([x for x, _ in pts], dtype=float)
    t = np.array([y for _, y in pts], dtype=float)

    # Trim TTFT outliers: queueing delay is additive noise that biases the slope.
    lo, hi = np.percentile(t, lo_q), np.percentile(t, hi_q)
    keep = (t >= lo) & (t <= hi)
    p, t = p[keep], t[keep]

    slope, intercept = np.polyfit(p, t, 1)
    if slope <= 0:
        raise SystemExit(f"{slot}: non-positive slope {slope:.3g}; run was not uncontended")
    speed = 1.0 / slope
    resid = t - (slope * p + intercept)
    return {
        "tokens_per_second": float(speed),
        "intercept_s": float(intercept),
        "n_samples": int(keep.sum()),
        "rmse_s": float(np.sqrt(np.mean(resid ** 2))),
        "prompt_len_p50": float(np.percentile(p, 50)),
        "prompt_len_p99": float(np.percentile(p, 99)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--req-file", help="raw *_output_requests.json from a solo run")
    ap.add_argument("--req-glob", help="glob; newest match is used")
    ap.add_argument("--source-slot", default="model_1")
    ap.add_argument("--also-slots", default="",
                    help="comma-separated slots that run the SAME model architecture "
                         "and therefore share the measured speed")
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args()

    req_file = a.req_file
    if not req_file:
        files = sorted(glob.glob(a.req_glob), key=os.path.getmtime)
        if not files:
            raise SystemExit(f"no file matching {a.req_glob}")
        req_file = files[-1]

    reqs = load(req_file)
    stats = fit_speed(reqs, a.source_slot)
    print(f"# fitted from {req_file}")
    print(f"# {a.source_slot}: {json.dumps(stats, indent=2)}")

    speeds = {a.source_slot: round(stats["tokens_per_second"], 3)}
    for slot in [s for s in a.also_slots.split(",") if s]:
        # Same architecture -> same prefill kernel and same cell_size, so the fitted
        # speed carries over. This mirrors how EXPERIMENT.md copies the SLO baseline
        # across the three Llama-3.1-8B slots.
        speeds[slot] = speeds[a.source_slot]

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(speeds, fh, indent=2)
    with open(a.out.replace(".json", "_detail.json"), "w") as fh:
        json.dump({"source": req_file, "source_slot": a.source_slot, "fit": stats}, fh, indent=2)
    print(f"wrote {a.out}: {speeds}")


if __name__ == "__main__":
    main()
