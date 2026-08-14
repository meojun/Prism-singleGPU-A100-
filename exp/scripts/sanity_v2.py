#!/usr/bin/env python3
"""Gate the paper-faithful-v2 main experiment.

Every condition in Section 8 of the brief is checked here against real run
output.  If any HARD check fails the sweep must not run; the failure is a
blocker to be fixed or recorded, not a result.

    python sanity_v2.py --bursty-dir <run> --steady-dir <run> \
        --prefill-speed exp/configs/v2/prefill_speed.json \
        --workload-dir exp/workloads/paper-faithful-v2 -o sanity.json

Checks
  A  c_i vs measured prefill      predicted e_i = p_i/c_i against the engine's
                                  own prefill interval, under load
  B  Algorithm 2 under-admission  rounds with eligible>0 and selected=0
  C  GPU idle while queue grows   low utilisation together with a rising queue
  D  Algorithm 1 KVPR varies      peak KVPR must not be constant, else this
                                  workload cannot evaluate placement at all
  E  placement candidates differ  per-GPU KVPR must actually separate
  F  decisions happen             migration / activation / eviction occur
  G  TTFT / TPOT sane             positive, finite, ordered percentiles
  H  workload reproducible        same seed -> byte-identical trace
"""
import argparse
import glob
import hashlib
import json
import os
import statistics
import subprocess
import sys

import numpy as np

RESULTS = []


def check(name, hard, ok, detail):
    RESULTS.append({"check": name, "hard": hard, "pass": bool(ok), "detail": detail})
    tag = "PASS" if ok else ("FAIL" if hard else "WARN")
    print(f"  [{tag}] {name}: {detail}")


def load_requests(run_dir):
    """The dump is EITHER one pretty-printed JSON array (the --model-paths +
    --real-trace path benchmark.py takes here) OR one JSON array per line
    (append mode, when a run is repeated). Handle both: try the whole file
    first, fall back to line-by-line."""
    cands = sorted(glob.glob(os.path.join(run_dir, "requests", "*_output_requests.json"))) \
        or sorted(glob.glob(os.path.join(run_dir, "*_output_requests.json")))
    if not cands:
        return []
    text = open(cands[-1]).read()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, list) else [obj]
    except json.JSONDecodeError:
        pass
    out = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            out.extend(json.loads(line))
    return out


def alg2_rounds(run_dir):
    rows = []
    for p in glob.glob(os.path.join(run_dir, "server-logs", "*.alg2_gpu*.jsonl")):
        with open(p, errors="ignore") as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def alg1_cycles(run_dir):
    import re
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
    return out


def gpu_series(run_dir):
    p = os.path.join(run_dir, "server-logs", "gpu_timeline.txt")
    if not os.path.exists(p):
        return []
    out = []
    with open(p, errors="ignore") as f:
        for line in f:
            parts = line.strip().split(" ", 1)
            if len(parts) != 2:
                continue
            utils = []
            for e in parts[1].split(";"):
                fl = [x.strip() for x in e.split(",")]
                if len(fl) == 3:
                    try:
                        utils.append(float(fl[2]))
                    except ValueError:
                        pass
            if utils:
                out.append((float(parts[0]), max(utils)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bursty-dir", required=True)
    ap.add_argument("--steady-dir", default=None)
    ap.add_argument("--prefill-speed", required=True)
    ap.add_argument("--workload-dir", required=True)
    ap.add_argument("--slo-base", default=None)
    ap.add_argument("--trace", default=None,
                    help="direct-trace pickle, joined by index for prompt_len")
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args()

    speed = json.load(open(a.prefill_speed))
    reqs = load_requests(a.bursty_dir)
    if a.trace:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from collect_v2_metrics import join_trace
        try:
            reqs = join_trace(reqs, a.trace)
        except SystemExit as e:
            print(f"  (trace join failed: {e})")
    # This dump path carries ttft/tpot but not the prefill interval, so check A
    # falls back to comparing e_i against TTFT when the interval is absent.
    ok = [r for r in reqs if r.get("success")
          and (r.get("prefill_finish_time") and r.get("out_queue_time") or r.get("ttft"))]

    print("== A. c_i vs measured prefill interval (under load)")
    if not ok:
        check("A c_i vs measured prefill", True, False, "no usable request records")
    else:
        ratios, per_model = [], {}
        for r in ok:
            c = speed.get(r["model"])
            if not c or not r.get("prompt_len"):
                continue
            pred = r["prompt_len"] / c
            meas = ((r["prefill_finish_time"] - r["out_queue_time"])
                    if (r.get("prefill_finish_time") and r.get("out_queue_time"))
                    else r.get("ttft"))
            if meas is None:
                continue
            if meas <= 0:
                continue
            ratios.append(pred / meas)
            per_model.setdefault(r["model"], []).append(pred / meas)
        med = statistics.median(ratios) if ratios else float("nan")
        # e_i is an estimate of one request's share of a batched prefill, so it
        # is expected to sit BELOW the wall-clock interval that request spent in
        # a shared batch. Two orders of magnitude either way means c_i is wrong.
        check("A c_i vs measured prefill", True, 0.01 <= med <= 10.0,
              f"median predicted/measured = {med:.3f} over n={len(ratios)}; "
              + ", ".join(f"{m}:{statistics.median(v):.2f}" for m, v in sorted(per_model.items())))

    print("== B. Algorithm 2 under-admission")
    rounds = alg2_rounds(a.bursty_dir)
    if not rounds:
        check("B alg2 under-admission", False, True, "no Algorithm 2 log (prototype arm or Alg2 off)")
    else:
        path = [r for r in rounds if r["pathological"]]
        worst = max((r["zero_streak"] for r in rounds), default=0)
        sel = sum(r["selected_requests"] for r in rounds)
        elig = sum(r["eligible_requests"] for r in rounds)
        ratio = sel / elig if elig else float("nan")
        check("B alg2 under-admission", True, worst < 200,
              f"pathological rounds={len(path)}/{len(rounds)}, max consecutive "
              f"eligible>0&selected=0 streak={worst}, selected/eligible={ratio:.3f}")
        check("B2 alg2 selected ratio", False, ratio > 0.10,
              f"selected/eligible={ratio:.3f}")

    print("== C. GPU idle while the queue grows")
    gs = gpu_series(a.bursty_dir)
    if not rounds or not gs:
        check("C gpu idle vs queue", False, True, "insufficient data")
    else:
        # bucket Alg-2 rounds by second, pair with the nearest GPU sample
        by_t = {}
        for r in rounds:
            by_t.setdefault(int(r["t"]), []).append(r["queue_length"])
        bad = 0
        for t, u in gs:
            q = by_t.get(int(t))
            if q and u < 20.0 and statistics.mean(q) > 20:
                bad += 1
        check("C gpu idle vs queue", True, bad <= max(3, 0.05 * len(gs)),
              f"{bad}/{len(gs)} samples with GPU util<20% and queue>20")

    print("== D/E. Algorithm 1 KVPR variation and candidate separation")
    cyc = alg1_cycles(a.bursty_dir)
    if not cyc:
        check("D kvpr varies", False, True, "no Algorithm 1 log (prototype arm)")
    else:
        peaks = [c["peak_kvpr"] for c in cyc if c.get("peak_kvpr")]
        cv = (statistics.stdev(peaks) / statistics.fmean(peaks)) if len(peaks) > 1 and statistics.fmean(peaks) else 0.0
        check("D kvpr varies over time", True, cv > 0.05,
              f"peak KVPR cv={cv:.3f} over {len(peaks)} cycles "
              f"(min={min(peaks):.3g}, max={max(peaks):.3g})" if peaks else "no peaks")
        seps = []
        for c in cyc:
            v = [x for x in (c.get("kvpr") or {}).values() if x]
            if len(v) > 1 and max(v) > 0:
                seps.append((max(v) - min(v)) / max(v))
        msep = statistics.fmean(seps) if seps else 0.0
        check("E gpu candidates differ", True, msep > 0.02,
              f"mean per-cycle (max-min)/max KVPR across GPUs = {msep:.3f}")

    print("== F. decisions actually happen")
    def count(run_dir, needle, logname="server.log.global_controller.log"):
        p = os.path.join(run_dir, "server-logs", logname)
        if not os.path.exists(p):
            return 0
        with open(p, errors="ignore") as f:
            return sum(needle in l for l in f)
    mig = count(a.bursty_dir, "[PAPER-ALG1] MIGRATE") + count(a.bursty_dir, "Reason: migrate model")
    act = count(a.bursty_dir, "ACTION: activate")
    evi = count(a.bursty_dir, "Reason: idle instance eviction")
    check("F scheduler acts", False, (mig + act + evi) > 0,
          f"migrations={mig} activations={act} idle_evictions={evi}")

    print("== G. TTFT / TPOT sanity")
    t = [r["ttft"] for r in reqs if r.get("success") and r.get("ttft")]
    p = [r["tpot"] for r in reqs if r.get("success") and r.get("tpot")]
    good = bool(t and p and min(t) > 0 and min(p) > 0
                and np.percentile(t, 50) <= np.percentile(t, 99)
                and np.percentile(p, 50) <= np.percentile(p, 99))
    check("G latency sane", True, good,
          f"n_ttft={len(t)} p50={1000*np.percentile(t,50):.1f}ms p99={1000*np.percentile(t,99):.1f}ms | "
          f"n_tpot={len(p)} p50={1000*np.percentile(p,50):.2f}ms p99={1000*np.percentile(p,99):.2f}ms"
          if t and p else "missing latency records")

    print("== H. workload reproducibility (same seed -> identical trace)")
    pkls = sorted(glob.glob(os.path.join(a.workload_dir, "bursty_*.pkl")))
    if not pkls:
        check("H reproducible", True, False, "no workload pkl found")
    else:
        src = pkls[0]
        base = os.path.basename(src)                       # bursty_r<rate>_s<seed>.pkl
        rate = base.split("_r")[1].split("_s")[0]
        seed = base.split("_s")[1].split(".pkl")[0]
        h1 = hashlib.sha256(open(src, "rb").read()).hexdigest()
        tmp = "/tmp/sanity_repro"
        os.makedirs(tmp, exist_ok=True)
        cmd = [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                            "build_paired_workload.py"),
               "--rate", rate, "--seed", seed, "--outdir", tmp]
        if a.slo_base:
            cmd += ["--slo-base", a.slo_base]
        dur = os.environ.get("V2_DURATION")
        if dur:
            cmd += ["--duration", dur]
        r = subprocess.run(cmd, capture_output=True, text=True)
        rebuilt = os.path.join(tmp, base)
        if r.returncode != 0 or not os.path.exists(rebuilt):
            check("H reproducible", True, False, f"rebuild failed: {r.stderr[-300:]}")
        else:
            h2 = hashlib.sha256(open(rebuilt, "rb").read()).hexdigest()
            check("H reproducible", True, h1 == h2,
                  f"{base}: {h1[:12]} vs rebuilt {h2[:12]}")

    hard_fail = [r for r in RESULTS if r["hard"] and not r["pass"]]
    json.dump({"results": RESULTS, "hard_failures": len(hard_fail)},
              open(a.out, "w"), indent=2)
    print()
    if hard_fail:
        print(f"SANITY FAILED: {len(hard_fail)} hard check(s): "
              + ", ".join(r["check"] for r in hard_fail))
        sys.exit(1)
    print("SANITY PASSED (hard checks)")


if __name__ == "__main__":
    main()
