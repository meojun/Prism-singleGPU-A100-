#!/usr/bin/env python3
"""Compute SLO attainment / goodput / violation rate for a Prism e2e run.

The released benchmark client's TP-mode path (which is what `--real-trace` +
`--model-paths` takes) dumps only raw per-request records and no SLO stats, so
we recompute everything here from the paper's own per-model SLO baselines in
`benchmark/multi-model/trace.py::generate_e2e_benchmark_reqs`.

NOTE on units: trace.py stores the TPOT baselines in **milliseconds** but
benchmark.py compares them against `output.tpot`, which is in **seconds**
(`attainment_tpot.append(1 if outputs[i].tpot < outputs[i].slo_tpot ...)`).
That comparison is therefore always true. We convert to seconds here so the
TPOT numbers are real.
"""
import argparse
import json
import os

import numpy as np

# meta-llama slot -> (p95 TTFT baseline [s], p95 TPOT baseline [ms])
# verbatim from trace.py::generate_e2e_benchmark_reqs
SLO_BASE = {
    "model_1": (0.04286971092, 11.46289839),   # Llama-3.1-8B
    "model_2": (0.04077239752, 9.607242716),   # Llama-3.2-3B
    "model_3": (0.02226704121, 6.097079518),   # Llama-3.2-1B
    "model_4": (0.04649914742, 11.23110303),   # Llama-3.1-8B
    "model_5": (0.04855853081, 11.08433425),   # Llama-3.1-8B
    "model_6": (0.03988536596, 8.795845509),   # Llama-3.2-1B
    "model_7": (0.03698700905, 6.114510298),   # Llama-3.2-1B
    "model_8": (0.02187654972, 5.484097324),   # Llama-3.2-1B
}


def pct(xs, q):
    return float(np.percentile(xs, q)) if len(xs) else float("nan")


def analyze(req_file, metrics_file, ttft_scale, tpot_scale, label, model_map):
    reqs = json.load(open(req_file))
    m = json.load(open(metrics_file))

    duration = m["completed"] / m["request_throughput"] if m["request_throughput"] else float("nan")
    total = len(reqs)
    ok = [r for r in reqs if r["success"]]

    rows = []
    for name in sorted({r["model"] for r in reqs}, key=lambda s: int(s.split("_")[1])):
        sub = [r for r in reqs if r["model"] == name]
        rows.append((name, sub))
    rows.append(("ALL", reqs))

    out = {
        "case": label,
        "models": model_map,
        "ttft_slo_scale": ttft_scale,
        "tpot_slo_scale": tpot_scale,
        "duration_s": duration,
        "total_requests": total,
        "completed": len(ok),
        "failed": total - len(ok),
        "per_model": {},
    }

    for name, sub in rows:
        good = [r for r in sub if r["success"]]
        if name == "ALL":
            # per-request SLO, mixed models
            hit_ttft = [r["ttft"] <= SLO_BASE[r["model"]][0] * ttft_scale for r in good]
            hit_tpot = [r["tpot"] <= SLO_BASE[r["model"]][1] / 1000.0 * tpot_scale for r in good]
            slo_ttft = slo_tpot = None
        else:
            slo_ttft = SLO_BASE[name][0] * ttft_scale
            slo_tpot = SLO_BASE[name][1] / 1000.0 * tpot_scale
            hit_ttft = [r["ttft"] <= slo_ttft for r in good]
            hit_tpot = [r["tpot"] <= slo_tpot for r in good]

        n = len(sub)
        both = [a and b for a, b in zip(hit_ttft, hit_tpot)]
        # failed requests count as violations of everything
        att_ttft = sum(hit_ttft) / n if n else 0.0
        att_tpot = sum(hit_tpot) / n if n else 0.0
        att_both = sum(both) / n if n else 0.0

        ttfts = [r["ttft"] for r in good]
        tpots = [r["tpot"] for r in good]
        out_toks = sum(r["output_len"] for r in good)
        good_toks = sum(r["output_len"] for r, b in zip(good, both) if b)

        out["per_model"][name] = {
            "requests": n,
            "completed": len(good),
            "slo_ttft_s": slo_ttft,
            "slo_tpot_ms": slo_tpot * 1000 if slo_tpot else None,
            "req_throughput_rps": len(good) / duration,
            "output_tok_throughput": out_toks / duration,
            "attain_ttft": att_ttft,
            "attain_tpot": att_tpot,
            "attain_both": att_both,
            "violation_rate": 1 - att_both,
            "goodput_rps": sum(both) / duration,
            "goodput_tok_s": good_toks / duration,
            "ttft_p50_ms": pct(ttfts, 50) * 1000,
            "ttft_p99_ms": pct(ttfts, 99) * 1000,
            "tpot_p50_ms": pct(tpots, 50) * 1000,
            "tpot_p99_ms": pct(tpots, 99) * 1000,
        }
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--req-file", required=True)
    ap.add_argument("--metrics-file", required=True)
    ap.add_argument("--ttft-slo-scale", type=float, required=True)
    ap.add_argument("--tpot-slo-scale", type=float, required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--model-map", default="{}")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    res = analyze(a.req_file, a.metrics_file, a.ttft_slo_scale, a.tpot_slo_scale,
                  a.label, json.loads(a.model_map))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=2)
    print(json.dumps(res, indent=2))
