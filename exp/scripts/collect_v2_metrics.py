#!/usr/bin/env python3
"""Collect every metric the paper-faithful-v2 brief asks for, from one run dir.

    python collect_v2_metrics.py --run-dir <dir> --slo-base exp/configs/v2/slo_base.json \
        --ttft-scale 5 --tpot-scale 3 --warmup 60 --measure 300 -o <dir>/metrics.json

Definitions, and why:

  TTFT   prefill_finish_timestamp - arrival_timestamp        (benchmark.py:140)
  TPOT   (finish_timestamp - prefill_finish_timestamp) / output_len  (benchmark.py:141)
         Both are the harness's own server-side definitions; we do NOT
         substitute mean-ITL or e2e/output_len.
  SLO    per-model no-contention p95 from --slo-base, times the scale flags.
         benchmark.py's own `average_attainment_tpot` field is unusable: it
         compares a millisecond baseline against a second-valued measurement
         and is therefore always 1.0.  Everything here is recomputed from the
         per-request dump.
  Joint-SLO goodput   |{completed AND ttft<=slo_ttft AND tpot<=slo_tpot}| / measure_s
         Not a metric the paper defines; defined here.

Requests are assigned to the warm-up / measurement window by their arrival
time relative to the first arrival, identically for every arm.
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np


def pct(xs, q):
    return float(np.percentile(xs, q)) if len(xs) else float("nan")


def load_requests(run_dir):
    """The dump is EITHER one pretty-printed JSON array (the --model-paths +
    --real-trace path benchmark.py takes here) OR one JSON array per line
    (append mode, when a run is repeated). Handle both: try the whole file
    first, fall back to line-by-line."""
    cands = sorted(glob.glob(os.path.join(run_dir, "requests", "*_output_requests.json"))) \
        or sorted(glob.glob(os.path.join(run_dir, "*_output_requests.json")))
    if not cands:
        raise SystemExit(f"no request dump under {run_dir}")
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


def load_trace(path):
    """Read a direct-trace pickle without importing trace.py (which needs the
    sglang package on the path)."""
    import pickle

    class Request:  # structural stand-in; the pickle records this class name
        pass

    class _U(pickle.Unpickler):
        def find_class(self, module, name):
            if name == "Request":
                return Request
            return super().find_class(module, name)

    with open(path, "rb") as f:
        _adapters, reqs = _U(f).load()
    reqs.sort(key=lambda r: r.arrival_time)
    return reqs


def join_trace(reqs, trace_path):
    """The dump benchmark.py writes on this path carries only
    {success, latency, ttft, tpot, output_len, model, error} -- no arrival time
    and no prompt length. It is written in TRACE ORDER and never reordered, so
    request i in the dump is request i in the (arrival-sorted) trace. Join by
    index to recover arrival_time / prompt_len / request_id, and verify the
    model sequence matches so a silent misalignment cannot pass as data."""
    trace = load_trace(trace_path)
    if len(trace) != len(reqs):
        raise SystemExit(f"trace has {len(trace)} requests, dump has {len(reqs)} "
                         "-- cannot join by index")
    mism = sum(1 for t, r in zip(trace, reqs) if r.get("model") != t.model)
    if mism:
        raise SystemExit(f"index join is misaligned: {mism} model mismatches")
    for t, r in zip(trace, reqs):
        r["arrival_time"] = t.arrival_time
        r["prompt_len"] = t.prompt_len
        r["request_id"] = t.req_id
        r.setdefault("output_len", t.output_len)
    return reqs


def scheduler_stats(run_dir):
    """Algorithm 1 / Algorithm 2 activity + prototype controller actions."""
    log = os.path.join(run_dir, "server-logs")
    st = {
        "alg1_cycles": 0, "migrations_alg1": 0, "migrations_proto": 0,
        "activations": 0, "deactivations": 0, "idle_evictions": 0,
        "alg2_rounds": 0, "alg2_eligible": 0, "alg2_selected": 0,
        "alg2_deferred": 0, "alg2_requeued": 0, "alg2_late_dispatched": 0,
        "alg2_pathological_rounds": 0, "alg2_max_zero_streak": 0,
        "alg2_underadmission_warnings": 0, "max_queue_length": 0,
        "mean_queue_length": float("nan"), "kvpr_samples": [],
    }
    gc = os.path.join(log, "server.log.global_controller.log")
    if os.path.exists(gc):
        with open(gc, errors="ignore") as f:
            for line in f:
                if "[PAPER-ALG1]" in line:
                    if " MIGRATE " in line:
                        st["migrations_alg1"] += 1
                        continue
                    st["alg1_cycles"] += 1
                    m = re.search(r'\[PAPER-ALG1\] (\{.*\})\s*$', line)
                    if m:
                        try:
                            st["kvpr_samples"].append(json.loads(m.group(1)))
                        except json.JSONDecodeError:
                            pass
                if "Reason: migrate model" in line:
                    st["migrations_proto"] += 1
                if "ACTION: activate" in line:
                    st["activations"] += 1
                if "ACTION: deactivate" in line:
                    st["deactivations"] += 1
                if "Reason: idle instance eviction" in line:
                    st["idle_evictions"] += 1

    qlens = []
    for p in glob.glob(os.path.join(log, "*.alg2_gpu*.jsonl")):
        with open(p, errors="ignore") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                st["alg2_rounds"] += 1
                st["alg2_eligible"] += r["eligible_requests"]
                st["alg2_selected"] += r["selected_requests"]
                st["alg2_deferred"] += r["deferred_requests"]
                st["alg2_requeued"] += r["requeued"]
                st["alg2_late_dispatched"] += r["late_dispatched"]
                st["alg2_pathological_rounds"] += int(r["pathological"])
                st["alg2_max_zero_streak"] = max(st["alg2_max_zero_streak"], r["zero_streak"])
                qlens.append(r["queue_length"])
    for p in glob.glob(os.path.join(log, "*gpu_scheduler*.log")):
        with open(p, errors="ignore") as f:
            st["alg2_underadmission_warnings"] += sum("[PAPER-ALG2-WARN]" in l for l in f)
    # The prototype arm has no Alg-2 log; fall back to its own queue_len prints.
    if not qlens:
        for p in glob.glob(os.path.join(log, "*gpu_scheduler*.log")):
            with open(p, errors="ignore") as f:
                for line in f:
                    m = re.search(r"queue_len: (\d+)", line)
                    if m:
                        qlens.append(int(m.group(1)))
    if qlens:
        st["max_queue_length"] = max(qlens)
        st["mean_queue_length"] = float(np.mean(qlens))
    st["alg2_selected_ratio"] = (st["alg2_selected"] / st["alg2_eligible"]
                                 if st["alg2_eligible"] else float("nan"))
    kv = st.pop("kvpr_samples")
    if kv:
        peaks = [s["peak_kvpr"] for s in kv if s.get("peak_kvpr")]
        imps = [s["improvement"] for s in kv if s.get("improvement") is not None]
        st["kvpr_peak_mean"] = float(np.mean(peaks)) if peaks else float("nan")
        st["kvpr_peak_cv"] = float(np.std(peaks) / np.mean(peaks)) if peaks and np.mean(peaks) else float("nan")
        st["kvpr_improvement_mean"] = float(np.mean(imps)) if imps else float("nan")
        st["kvpr_improvement_std"] = float(np.std(imps)) if imps else float("nan")
        st["kvpr_improvement_max"] = float(np.max(imps)) if imps else float("nan")
    return st


def gpu_stats(run_dir):
    p = os.path.join(run_dir, "server-logs", "gpu_timeline.txt")
    if not os.path.exists(p):
        return {}
    util, mem = defaultdict(list), defaultdict(list)
    with open(p, errors="ignore") as f:
        for line in f:
            parts = line.strip().split(" ", 1)
            if len(parts) != 2:
                continue
            for entry in parts[1].split(";"):
                fields = [x.strip() for x in entry.split(",")]
                if len(fields) == 3:
                    try:
                        g, mu, u = int(fields[0]), float(fields[1]), float(fields[2])
                    except ValueError:
                        continue
                    mem[g].append(mu)
                    util[g].append(u)
    return {
        "gpu_util_mean": {str(g): float(np.mean(v)) for g, v in util.items()},
        "gpu_mem_mean_mib": {str(g): float(np.mean(v)) for g, v in mem.items()},
        "gpu_mem_max_mib": {str(g): float(np.max(v)) for g, v in mem.items()},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--slo-base", required=True)
    ap.add_argument("--ttft-scale", type=float, default=5.0)
    ap.add_argument("--tpot-scale", type=float, default=3.0)
    ap.add_argument("--warmup", type=float, default=60.0)
    ap.add_argument("--measure", type=float, default=300.0)
    ap.add_argument("--trace", default=None,
                    help="direct-trace pickle; joined by index to recover "
                         "arrival_time / prompt_len the dump omits")
    ap.add_argument("--label", default="")
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args()

    base = json.load(open(a.slo_base))
    slo_ttft = {m: v["ttft"] * a.ttft_scale for m, v in base.items()}
    slo_tpot = {m: v["tpot"] * a.tpot_scale for m, v in base.items()}

    reqs = load_requests(a.run_dir)
    if a.trace:
        reqs = join_trace(reqs, a.trace)
    arrivals = [r["arrival_time"] for r in reqs if r.get("arrival_time") is not None]
    if not arrivals:
        raise SystemExit("no arrival timestamps -- pass --trace to join by index")
    t0 = min(arrivals)
    lo, hi = t0 + a.warmup, t0 + a.warmup + a.measure

    win = [r for r in reqs if r.get("arrival_time") is not None
           and lo <= r["arrival_time"] < hi]
    ok = [r for r in win if r.get("success")]
    ttfts = [r["ttft"] for r in ok if r.get("ttft")]
    tpots = [r["tpot"] for r in ok if r.get("tpot")]

    ttft_hit = [r for r in ok if r.get("ttft") is not None and r["ttft"] <= slo_ttft.get(r["model"], float("inf"))]
    tpot_hit = [r for r in ok if r.get("tpot") is not None and r["tpot"] <= slo_tpot.get(r["model"], float("inf"))]
    joint = [r for r in ok
             if r.get("ttft") is not None and r.get("tpot") is not None
             and r["ttft"] <= slo_ttft.get(r["model"], float("inf"))
             and r["tpot"] <= slo_tpot.get(r["model"], float("inf"))]

    n = len(win)
    per_model = {}
    for m in sorted({r["model"] for r in win}):
        sub = [r for r in win if r["model"] == m]
        sok = [r for r in sub if r.get("success")]
        sj = [r for r in sok
              if r.get("ttft") is not None and r.get("tpot") is not None
              and r["ttft"] <= slo_ttft.get(m, float("inf"))
              and r["tpot"] <= slo_tpot.get(m, float("inf"))]
        per_model[m] = {
            "requests": len(sub), "completed": len(sok),
            "ttft_p50_ms": 1000 * pct([r["ttft"] for r in sok if r.get("ttft")], 50),
            "ttft_p95_ms": 1000 * pct([r["ttft"] for r in sok if r.get("ttft")], 95),
            "tpot_p50_ms": 1000 * pct([r["tpot"] for r in sok if r.get("tpot")], 50),
            "tpot_p95_ms": 1000 * pct([r["tpot"] for r in sok if r.get("tpot")], 95),
            "joint_attainment": len(sj) / len(sub) if sub else float("nan"),
            "slo_ttft_ms": 1000 * slo_ttft.get(m, float("nan")),
            "slo_tpot_ms": 1000 * slo_tpot.get(m, float("nan")),
        }

    res = {
        "label": a.label, "run_dir": a.run_dir,
        "warmup_s": a.warmup, "measure_s": a.measure,
        "ttft_slo_scale": a.ttft_scale, "tpot_slo_scale": a.tpot_scale,
        "requests_in_window": n, "completed": len(ok),
        "failed": n - len(ok),
        "ttft_mean_ms": 1000 * float(np.mean(ttfts)) if ttfts else float("nan"),
        "ttft_p50_ms": 1000 * pct(ttfts, 50), "ttft_p95_ms": 1000 * pct(ttfts, 95),
        "ttft_p99_ms": 1000 * pct(ttfts, 99),
        "tpot_mean_ms": 1000 * float(np.mean(tpots)) if tpots else float("nan"),
        "tpot_p50_ms": 1000 * pct(tpots, 50), "tpot_p95_ms": 1000 * pct(tpots, 95),
        "tpot_p99_ms": 1000 * pct(tpots, 99),
        "ttft_attainment": len(ttft_hit) / n if n else float("nan"),
        "tpot_attainment": len(tpot_hit) / n if n else float("nan"),
        "joint_attainment": len(joint) / n if n else float("nan"),
        "throughput_req_s": len(ok) / a.measure,
        "goodput_req_s": len(joint) / a.measure,
        "output_token_throughput": sum(r.get("output_len") or 0 for r in ok) / a.measure,
        "prompt_token_throughput": sum(r.get("prompt_len") or 0 for r in ok) / a.measure,
        "offered_load_req_s": n / a.measure,
        "per_model": per_model,
    }
    res.update(scheduler_stats(a.run_dir))
    res.update(gpu_stats(a.run_dir))
    with open(a.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"{a.label or a.run_dir}: n={n} ok={len(ok)} "
          f"TTFTp50={res['ttft_p50_ms']:.0f}ms p99={res['ttft_p99_ms']:.0f}ms "
          f"TPOTp50={res['tpot_p50_ms']:.1f}ms joint={res['joint_attainment']:.3f} "
          f"goodput={res['goodput_req_s']:.2f} mig={res['migrations_alg1']+res['migrations_proto']}")


if __name__ == "__main__":
    main()
