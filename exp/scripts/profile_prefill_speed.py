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
import sys

import numpy as np


def load(req_file):
    """The `--model-paths` + `--real-trace` path writes ONE pretty-printed array."""
    with open(req_file) as fh:
        return json.load(fh)


def load_trace(path):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from build_sharegpt_trace import _Unpickler
    with open(path, "rb") as fh:
        _, trace = _Unpickler(fh).load()
    return trace


def fit_speed(reqs, trace, slot, lo_q=5, hi_q=80):
    """Join dump to trace BY INDEX to recover prompt_len.

    run_tp_mode's dump carries only {success, latency, ttft, tpot, output_len,
    model, error} -- no prompt_len and no arrival time. benchmark.py writes the
    dump in trace order and never reorders it, so request i in the dump is
    request i in the pkl (same property collect_metrics.py relies on).
    """
    if len(reqs) != len(trace):
        raise SystemExit(f"dump has {len(reqs)} records but trace has {len(trace)}; "
                         "cannot join by index")
    pts = [
        (tr.prompt_len, r["ttft"])
        for r, tr in zip(reqs, trace)
        if r.get("success") and r.get("model") == slot and tr.prompt_len
    ]
    if len(pts) < 30:
        raise SystemExit(f"{slot}: only {len(pts)} usable samples, need >= 30")

    p = np.array([x for x, _ in pts], dtype=float)
    t = np.array([y for _, y in pts], dtype=float)

    # Trim TTFT outliers: queueing delay is additive noise on an otherwise
    # uncontended measurement.
    lo, hi = np.percentile(t, lo_q), np.percentile(t, hi_q)
    keep = (t >= lo) & (t <= hi)
    p, t = p[keep], t[keep]

    # RATIO estimator, deliberately not the regression slope.
    #
    # The paper's execution estimate is e_i = p_i / c_i with NO intercept term, so
    # c_i has to be the speed that reproduces the WHOLE prefill time, not the
    # marginal cost of one extra token. Fitting ttft = a + p/c on this box gives
    # a ~= 29 ms with 1/slope ~= 20,800 tok/s: the fixed overhead dominates at these
    # prompt lengths (p50 = 70 tokens). Feeding that slope into e_i = p/c would put
    # every e_i in the 3-30 ms range -- an order of magnitude under the real prefill
    # time -- and Moore-Hodgson's feasibility test would essentially never fire.
    # total_tokens / total_time is the estimator consistent with the paper's formula.
    # The slope fit is still reported below as a diagnostic.
    speed = float(p.sum() / t.sum())
    slope, intercept = np.polyfit(p, t, 1)
    per_req = p / t
    return {
        "tokens_per_second": speed,
        "estimator": "ratio: sum(prompt_len) / sum(ttft), matches e_i = p_i / c_i",
        "n_samples": int(keep.sum()),
        "per_request_speed_p50": float(np.percentile(per_req, 50)),
        "per_request_speed_p90": float(np.percentile(per_req, 90)),
        "diagnostic_slope_tokens_per_second": float(1.0 / slope) if slope > 0 else None,
        "diagnostic_intercept_s": float(intercept),
        "ttft_p50_s": float(np.percentile(t, 50)),
        "prompt_len_p50": float(np.percentile(p, 50)),
        "prompt_len_p99": float(np.percentile(p, 99)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--req-file", help="raw *_output_requests.json from a solo run")
    ap.add_argument("--req-glob", help="glob; newest match is used")
    ap.add_argument("--trace", required=True,
                    help="the .pkl replayed by that run; supplies prompt_len, which "
                         "the TP-mode dump does not record")
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
    trace = load_trace(a.trace)
    stats = fit_speed(reqs, trace, a.source_slot)
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
