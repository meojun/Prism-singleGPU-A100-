#!/usr/bin/env python3
"""Assemble the single REPORT.md for paper-faithful-v2.

    python build_report_v2.py --base exp/results/paper-faithful-v2 \
        -o exp/results/paper-faithful-v2/REPORT.md

Structure follows Section 22 of the brief exactly.  Numbers and tables only --
narrative sections are emitted as placeholders the author fills in, so nothing
in this file can silently invent a conclusion.
"""
import argparse
import csv
import glob
import json
import os
import platform
import subprocess
from collections import defaultdict

MODELS = [
    ("model_1", "meta-llama/Llama-3.2-1B", "1.24B", 2.28, 32768),
    ("model_2", "Qwen/Qwen2.5-1.5B-Instruct", "1.54B", 3.01, 28672),
    ("model_3", "meta-llama/Llama-3.2-3B", "3.21B", 6.00, 114688),
    ("model_4", "Qwen/Qwen2.5-3B-Instruct", "3.09B", 5.84, 36864),
    ("model_5", "meta-llama/Llama-3.1-8B", "8.03B", 15.08, 131072),
    ("model_6", "Qwen/Qwen2.5-7B-Instruct", "7.62B", 14.28, 57344),
]


def sh(cmd, default="n/a"):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                              timeout=30).stdout.strip() or default
    except Exception:
        return default


def num(v):
    try:
        f = float(v)
        return f if f == f else None
    except (TypeError, ValueError):
        return None


def fmt(v, spec="{:.1f}"):
    n = num(v)
    return spec.format(n) if n is not None else "-"


def read_csv(p):
    if not os.path.exists(p):
        return []
    with open(p) as f:
        return list(csv.DictReader(f))


def section_env(base, root):
    gpu = sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1")
    cnt = sh("nvidia-smi --list-gpus | wc -l", "?")
    cpu = sh("lscpu | grep 'Model name' | head -1 | cut -d: -f2- | xargs")
    ram = sh("free -g | awk 'NR==2{print $2\" GiB\"}'")
    drv = sh("nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1")
    def ver(mod):
        return sh(f"{root}/prism-venv/bin/python -c "
                  f"'import {mod};print({mod}.__version__)'")
    rows = [
        ("GPU", f"{cnt} x {gpu}"),
        ("Driver", drv),
        ("CPU", cpu), ("RAM", ram),
        ("OS", f"{platform.system()} {platform.release()}"),
        ("CUDA (torch)", sh(f"{root}/prism-venv/bin/python -c 'import torch;print(torch.version.cuda)'")),
        ("torch", ver("torch")),
        ("SGLang (prism-research fork)", ver("sglang")),
        ("prism-research commit", sh(f"git -C {root}/prism-research rev-parse HEAD")),
        ("kvcached commit (prism/shm)", sh(f"git -C {root}/kvcached-prism rev-parse HEAD")),
        ("Prism harness branch", sh(f"git -C {root} rev-parse --abbrev-ref HEAD")),
        ("Prism harness commit", sh(f"git -C {root} rev-parse HEAD")),
    ]
    out = ["## 1. Experimental Environment", "", "| Item | Value |", "| --- | --- |"]
    out += [f"| {k} | {v} |" for k, v in rows]
    return "\n".join(out) + "\n"


def section_models(base):
    prof = {}
    for p in glob.glob(os.path.join(base, "sanity", "profiling", "model_*.json")):
        d = json.load(open(p))
        prof[d["model"]] = d
    out = ["## 2. Models", "",
           "| Slot | Model | Params | dtype | Weights (GiB) | KV cell (B/token) | "
           "TTFT p95 (ms) | TPOT p95 (ms) |",
           "| --- | --- | --- | --- | ---: | ---: | ---: | ---: |"]
    for slot, path, params, w, cell in MODELS:
        b = prof.get(slot, {}).get("slo_baseline", {})
        out.append(f"| {slot} | `{path}` | {params} | bf16 | {w:.2f} | {cell} | "
                   f"{fmt(1000*b['ttft_p95_s'] if b.get('ttft_p95_s') else None)} | "
                   f"{fmt(1000*b['tpot_p95_s'] if b.get('tpot_p95_s') else None, '{:.2f}')} |")
    out += ["",
            "KV cell size is deliberately **not** monotone in parameter count: "
            "`model_3` carries 3.1x `model_4`'s KV per token at the same size, and "
            "`model_5` 2.3x `model_6`'s. Without that, KVPR degenerates into a "
            "relabelling of resident weight and Algorithm 1's objective goes flat -- "
            "which is what happened in the v1 study's 3 x Llama-3.1-8B setup.",
            "TTFT/TPOT p95 are no-contention solo measurements on this box "
            "(paper Sec. 7.1 method); the SLOs used below are these times the "
            "scale factors."]
    return "\n".join(out) + "\n"


def section_ci(base):
    rows = []
    for p in sorted(glob.glob(os.path.join(base, "sanity", "profiling", "model_*.json"))):
        d = json.load(open(p))
        e = d["c_i_estimators"]
        rows.append((d["model"], e))
    out = ["### c_i estimators (tokens/s)", "",
           "| Slot | E1 ratio Sp/Sttft | E2 regression slope | E2 intercept (ms) | "
           "E3 measured prefill, solo | E3 measured prefill, saturated | **used** |",
           "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for m, e in rows:
        used = e.get("E3_prefill_saturated") or e.get("E3_prefill_solo") or e.get("E1_ratio_sum_p_over_sum_ttft")
        out.append(f"| {m} | {fmt(e['E1_ratio_sum_p_over_sum_ttft'], '{:.0f}')} | "
                   f"{fmt(e['E2_regression_slope'], '{:.0f}')} | "
                   f"{fmt(1000*e['E2_intercept_s'] if e.get('E2_intercept_s') else None, '{:.1f}')} | "
                   f"{fmt(e['E3_prefill_solo'], '{:.0f}')} | "
                   f"{fmt(e['E3_prefill_saturated'], '{:.0f}')} | "
                   f"**{fmt(used, '{:.0f}')}** |")
    return "\n".join(out) + "\n"


def section_sanity(base):
    p = os.path.join(base, "sanity", "sanity_gate.json")
    out = ["## 4. Sanity Check", "", "| Test | Result | Pass/Fail |", "| --- | --- | --- |"]
    if not os.path.exists(p):
        out.append("| (not run) | | |")
        return "\n".join(out) + "\n"
    d = json.load(open(p))
    for r in d["results"]:
        tag = "PASS" if r["pass"] else ("**FAIL**" if r["hard"] else "WARN")
        out.append(f"| {r['check']} | {r['detail']} | {tag} |")
    out.append("")
    out.append(f"Hard failures: **{d['hard_failures']}**"
               + ("" if d["hard_failures"] else " -- gate passed, main experiment ran."))
    return "\n".join(out) + "\n"


def section_calibration(base):
    rows = []
    for p in sorted(glob.glob(os.path.join(base, "sanity", "calibration", "rate_*", "metrics.json")),
                    key=lambda q: float(os.path.basename(os.path.dirname(q)).split("_")[1])):
        d = json.load(open(p))
        rows.append((float(os.path.basename(os.path.dirname(p)).split("_")[1]), d))
    if not rows:
        return ""
    out = ["### Load calibration (released prototype, steady, short runs)", "",
           "| Offered (req/s) | Throughput | TTFT p50 (ms) | TTFT p99 (ms) | "
           "TPOT p50 (ms) | Joint attainment | Goodput | max queue |",
           "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r, d in rows:
        out.append(f"| {r:g} | {fmt(d['throughput_req_s'],'{:.2f}')} | {fmt(d['ttft_p50_ms'])} | "
                   f"{fmt(d['ttft_p99_ms'])} | {fmt(d['tpot_p50_ms'])} | "
                   f"{fmt(d['joint_attainment'],'{:.3f}')} | {fmt(d['goodput_req_s'],'{:.2f}')} | "
                   f"{fmt(d.get('max_queue_length'),'{:.0f}')} |")
    return "\n".join(out) + "\n"


def section_workloads(base, root):
    wl = os.path.join(root, "exp/workloads/paper-faithful-v2")
    metas = sorted(glob.glob(os.path.join(wl, "paired_requests_*.json")))
    out = ["## 5. Workloads", ""]
    if metas:
        out += ["| Rate (req/s) | Duration (s) | Seed | Total requests | Avg offered load | "
                + " | ".join(m[0] for m in MODELS) + " |",
                "| ---: | ---: | ---: | ---: | ---: |" + " ---: |" * len(MODELS)]
        for p in metas:
            d = json.load(open(p))
            pm = d["per_model_requests"]
            out.append(f"| {d['rate']:g} | {d['duration']:g} | {d['seed']} | "
                       f"{d['total_requests']} | {d['average_offered_load_req_s']:.2f} | "
                       + " | ".join(str(pm.get(m[0], 0)) for m in MODELS) + " |")
    ph = sorted(glob.glob(os.path.join(wl, "phases_*.json")))
    if ph:
        d = json.load(open(ph[0]))
        out += ["", "### Shifting Bursty", "",
                f"- phase duration range: {d['phase_len_range'][0]:g}-{d['phase_len_range'][1]:g} s "
                f"(random, fixed seed)",
                f"- hot-set size: 1-3 models, redrawn every phase",
                f"- rate multipliers: HOT {d['hot_multiplier_range']}, "
                f"MEDIUM {d['medium_multiplier_range']}, LOW {d['low_multiplier_range']}, IDLE = 0",
                f"- base share: {d['base_share']}",
                f"- seed: {d['seed']}",
                "- **the aggregate arrival rate is renormalised to the same constant in "
                "every phase**, so the cluster's total offered load never moves; only "
                "*which* model is hot does. That removes 'the cluster got busier' as a "
                "confound and isolates the effect Prism claims to exploit.",
                "", "### Steady", "",
                "- per-model request counts taken verbatim from the bursty trace",
                "- each model's arrivals drawn uniformly at random over the full duration "
                "(N uniform points = a homogeneous Poisson process conditioned on N)",
                "", "### What is held equal", "",
                "| Property | Bursty | Steady |", "| --- | --- | --- |",
                "| Request set | Same | Same |", "| Prompts | Same | Same |",
                "| Model assignment | Same | Same |", "| Output lengths | Same | Same |",
                "| Per-model request count | Same | Same |", "| Total requests | Same | Same |",
                "| Duration | Same | Same |", "| Average offered load | Same | Same |",
                "| Random seed | Same | Same |",
                "| **Arrival timing** | **Bursty** | **Uniform** |"]
    return "\n".join(out) + "\n"


def section_results(base):
    summ = read_csv(os.path.join(base, "processed", "summary.csv"))
    if not summ:
        return "## 6. Results\n\n_(no runs aggregated yet)_\n"
    out = ["## 6. Results", ""]
    by_rate = defaultdict(list)
    for r in summ:
        by_rate[r["rate"]].append(r)
    for rate in sorted(by_rate, key=lambda x: float(x)):
        out += [f"### Offered load {rate} req/s", "",
                "| System | Workload | TTFT p50 | TTFT p95 | TTFT p99 | TPOT p50 | TPOT p95 | "
                "TPOT p99 | TTFT att | TPOT att | Joint att | Throughput | Goodput | "
                "Mig | Act | Evict | max Q |",
                "| --- | --- |" + " ---: |" * 15]
        for r in sorted(by_rate[rate], key=lambda r: (r["workload"], r["system"])):
            mig = num(r.get("migrations_alg1")) or 0
            mig += num(r.get("migrations_proto")) or 0
            out.append(
                f"| {r['system']} | {r['workload']} | "
                f"{fmt(r['ttft_p50_ms'])} | {fmt(r['ttft_p95_ms'])} | {fmt(r['ttft_p99_ms'])} | "
                f"{fmt(r['tpot_p50_ms'])} | {fmt(r['tpot_p95_ms'])} | {fmt(r['tpot_p99_ms'])} | "
                f"{fmt(r['ttft_attainment'],'{:.3f}')} | {fmt(r['tpot_attainment'],'{:.3f}')} | "
                f"{fmt(r['joint_attainment'],'{:.3f}')} | {fmt(r['throughput_req_s'],'{:.2f}')} | "
                f"{fmt(r['goodput_req_s'],'{:.2f}')} | {mig:.0f} | "
                f"{fmt(r['activations'],'{:.0f}')} | {fmt(r['idle_evictions'],'{:.0f}')} | "
                f"{fmt(r['max_queue_length'],'{:.0f}')} |")
        out.append("")
    out += ["Latency in ms, throughput and goodput in req/s. "
            "Joint attainment = fraction of window requests meeting BOTH TTFT and TPOT SLO. "
            "Goodput = those requests / measurement window.", ""]
    return "\n".join(out) + "\n"


def section_comparison(base):
    comp = read_csv(os.path.join(base, "processed", "comparison.csv"))
    if not comp:
        return "## 7. Steady vs Bursty Comparison\n\n_(pending)_\n"
    out = ["## 7. Steady vs Bursty Comparison", "",
           "Relative improvement of each paper arm over `released-prototype` at the "
           "same workload and load. Positive = the paper arm wins.", "",
           "| System | Workload | Rate | Joint att (base) | Joint att (paper) | dpp | "
           "Joint rel | Goodput rel | TTFT p99 rel | TPOT p99 rel |",
           "| --- | --- | ---: |" + " ---: |" * 7]
    for r in sorted(comp, key=lambda r: (r["system"], float(r["rate"]), r["workload"])):
        out.append(f"| {r['system']} | {r['workload']} | {r['rate']} | "
                   f"{fmt(r['joint_attainment_base'],'{:.3f}')} | "
                   f"{fmt(r['joint_attainment_prism'],'{:.3f}')} | "
                   f"{fmt(num(r['joint_attainment_pp'])*100 if num(r['joint_attainment_pp']) is not None else None,'{:+.1f}')} | "
                   f"{fmt(num(r['joint_attainment_rel'])*100 if num(r['joint_attainment_rel']) is not None else None,'{:+.1f}%')} | "
                   f"{fmt(num(r['goodput_rel'])*100 if num(r['goodput_rel']) is not None else None,'{:+.1f}%')} | "
                   f"{fmt(num(r['ttft_p99_rel'])*100 if num(r['ttft_p99_rel']) is not None else None,'{:+.1f}%')} | "
                   f"{fmt(num(r['tpot_p99_rel'])*100 if num(r['tpot_p99_rel']) is not None else None,'{:+.1f}%')} |")
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--root", default="/workspace/prism-exp")
    ap.add_argument("--impl-status", default=None,
                    help="markdown fragment for Section 3 (implementation status)")
    ap.add_argument("--narrative", default=None,
                    help="markdown fragment for Sections 8-10")
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args()

    parts = ["# Paper-Faithful Prism v2 — Shifting-Bursty vs Steady", "",
             f"_Generated by `exp/scripts/build_report_v2.py`. "
             f"Harness commit `{sh(f'git -C {a.root} rev-parse --short HEAD')}`._", ""]
    parts.append(section_env(a.base, a.root))
    parts.append(section_models(a.base))
    if a.impl_status and os.path.exists(a.impl_status):
        parts.append(open(a.impl_status).read())
    parts.append("## 4. Sanity Check\n".replace("## 4. Sanity Check\n", ""))
    parts.append(section_ci(a.base))
    parts.append(section_sanity(a.base))
    parts.append(section_calibration(a.base))
    parts.append(section_workloads(a.base, a.root))
    parts.append(section_results(a.base))
    parts.append(section_comparison(a.base))
    if a.narrative and os.path.exists(a.narrative):
        parts.append(open(a.narrative).read())
    with open(a.out, "w") as f:
        f.write("\n".join(parts))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
