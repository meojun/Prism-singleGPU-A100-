#!/usr/bin/env python3
"""Per-run metrics for the Paper-Faithful comparison.

Reads one run's raw per-request dump plus its scheduler logs and emits a single
CSV row. Warm-up requests are excluded by ARRIVAL TIME, not by index, so both arms
of a (rate, seed) pair drop exactly the same requests.

TPOT is the harness' own definition, taken verbatim from the dump:
    tpot = (finish_time - prefill_finish_time) / (output tokens - 1)
It is NOT recomputed as mean-ITL or as e2e/output_len. The harness' own
`average_attainment_tpot` field is unusable (ms-vs-s bug); everything here is
recomputed from raw records.
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np

# Fallback per-slot baselines, verbatim from trace.py (authors' hardware).
BUILTIN = {
    "model_1": (0.04286971092, 11.46289839),
    "model_2": (0.04077239752, 9.607242716),
    "model_3": (0.02226704121, 6.097079518),
    "model_4": (0.04649914742, 11.23110303),
    "model_5": (0.04855853081, 11.08433425),
    "model_6": (0.03988536596, 8.795845509),
    "model_7": (0.03698700905, 6.114510298),
    "model_8": (0.02187654972, 5.484097324),
}


def pct(xs, q):
    return float(np.percentile(xs, q)) if len(xs) else float("nan")


def load_requests(outdir, exp):
    """The dump is ONE pretty-printed JSON array (same as analyze_slo.py reads)."""
    pat = os.path.join(outdir, "requests", f"{exp}_e2e_*_output_requests.json")
    files = sorted(glob.glob(pat), key=os.path.getmtime)
    if not files:
        raise SystemExit(f"no request dump matching {pat}")
    with open(files[-1]) as fh:
        return json.load(fh)


def load_trace(path):
    """Trace records, which carry req_time and prompt_len.

    The `--model-paths` + `--real-trace` client path (run_tp_mode) dumps only
    {success, latency, ttft, tpot, output_len, model, error}; there is no arrival
    time in it. benchmark.py writes the dump in trace order and never reorders it,
    so request i in the dump is request i in the pkl -- the same index join
    collect_metrics.py uses -- and that is where the arrival time comes from.
    """
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from build_sharegpt_trace import _Unpickler
    with open(path, "rb") as fh:
        _, trace = _Unpickler(fh).load()
    return trace


def count_in_logs(logdir, pattern, globpat="*.log"):
    n = 0
    for path in glob.glob(os.path.join(logdir, globpat)):
        try:
            with open(path, errors="ignore") as fh:
                for line in fh:
                    if pattern in line:
                        n += 1
        except OSError:
            pass
    return n


def max_cumulative_deferred(logdir):
    """Peak deferred-request counter across the GPU-scheduler logs.

    Each GPU scheduler keeps its own counter, so the per-GPU peaks are summed.

    Two spellings must both be matched. The livelock fix changed the counter from
    per-round ("cumulative deferred=") to per-request ("distinct deferred="), but
    only on the branch that logs a deferral; the other branch still prints the old
    wording. Matching just one spelling reads whichever branch happens to log last
    and undercounts badly -- observed 56 against an actual 4338.
    """
    total = 0
    rx = re.compile(r"(?:cumulative|distinct) deferred=(\d+)")
    for path in glob.glob(os.path.join(logdir, "*gpu_scheduler*.log")):
        best = 0
        try:
            with open(path, errors="ignore") as fh:
                for line in fh:
                    m = rx.search(line)
                    if m:
                        best = max(best, int(m.group(1)))
        except OSError:
            pass
        total += best
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--trace", required=True,
                    help="the .pkl replayed by this run; supplies req_time (and "
                         "prompt_len), which the TP-mode dump does not record")
    ap.add_argument("--system", required=True)
    ap.add_argument("--rate", type=float, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--warmup", type=float, default=60.0)
    ap.add_argument("--measure", type=float, default=300.0)
    ap.add_argument("--ttft-slo-scale", type=float, default=5.0)
    ap.add_argument("--tpot-slo-scale", type=float, default=3.0)
    ap.add_argument("--slo-base", default=os.environ.get("SLO_BASE_FILE"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    base = dict(BUILTIN)
    slo_source = "builtin"
    if a.slo_base and os.path.exists(a.slo_base):
        base.update({k: tuple(v) for k, v in json.load(open(a.slo_base)).items()})
        slo_source = a.slo_base

    exp = f"{a.system}_rate{a.rate:g}_seed{a.seed}"
    reqs = load_requests(a.outdir, exp)
    trace = load_trace(a.trace)
    if len(reqs) != len(trace):
        raise SystemExit(f"{exp}: dump has {len(reqs)} records but trace has "
                         f"{len(trace)}; cannot join by index")

    # Warm-up exclusion by TRACE arrival time (req_time is relative to trace start),
    # so both arms of a (rate, seed) pair drop exactly the same requests.
    lo, hi = a.warmup, a.warmup + a.measure
    win = [r for r, tr in zip(reqs, trace) if lo <= tr.req_time < hi]
    if not win:
        raise SystemExit(f"{exp}: no requests in the measurement window")

    ok = [r for r in win if r.get("success")]
    ttfts = [r["ttft"] for r in ok]
    tpots = [r["tpot"] for r in ok]

    hit_ttft = [r["ttft"] <= base[r["model"]][0] * a.ttft_slo_scale for r in ok]
    hit_tpot = [r["tpot"] <= base[r["model"]][1] / 1000.0 * a.tpot_slo_scale for r in ok]
    joint = [x and y for x, y in zip(hit_ttft, hit_tpot)]

    # Denominator is completed requests in the window, per the study definition.
    n_ok = len(ok)
    logdir = os.path.join(a.outdir, "server-logs")

    row = {
        "system": a.system,
        "request_rate": a.rate,
        "seed": a.seed,
        "ttft_p50_ms": pct(ttfts, 50) * 1000,
        "ttft_p99_ms": pct(ttfts, 99) * 1000,
        "tpot_p50_ms": pct(tpots, 50) * 1000,
        "tpot_p99_ms": pct(tpots, 99) * 1000,
        "ttft_slo_attainment": (sum(hit_ttft) / n_ok) if n_ok else float("nan"),
        "tpot_slo_attainment": (sum(hit_tpot) / n_ok) if n_ok else float("nan"),
        "joint_slo_attainment": (sum(joint) / n_ok) if n_ok else float("nan"),
        "throughput_req_s": n_ok / a.measure,
        "joint_slo_goodput_req_s": sum(joint) / a.measure,
        "num_completed": n_ok,
        "num_failed": len(win) - n_ok,
        "num_migrations": count_in_logs(logdir, "[PAPER-ALG1] MIGRATE")
        + count_in_logs(logdir, "Reason: migrate model"),
        "num_evictions": count_in_logs(logdir, "Reason: idle instance eviction"),
        "num_activations": count_in_logs(logdir, "ACTION: activate"),
        "num_mh_deferred": max_cumulative_deferred(logdir),
        # diagnostics
        "ttft_p95_ms": pct(ttfts, 95) * 1000,
        "tpot_p95_ms": pct(tpots, 95) * 1000,
        "requests_in_window": len(win),
        "measure_s": a.measure,
        "warmup_s": a.warmup,
        "slo_base_source": slo_source,
        "alg1_log_lines": count_in_logs(logdir, "[PAPER-ALG1]"),
        "alg2_log_lines": count_in_logs(logdir, "[PAPER-ALG2]"),
    }

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(row, fh, indent=2)
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
