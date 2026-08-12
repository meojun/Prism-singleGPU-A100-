#!/usr/bin/env python3
"""Turn one Prism run's logs into post-processable CSV.

Nothing here instruments Prism. Every quantity below is already written by the
stock engine/scheduler/controller loggers; this only parses and joins them, so
the experiment adds zero code to the serving path.

Sources and what each one uniquely provides:

  server.log                 per-worker batch lines, tagged with the model:
      "GPU=0 Worker 0 (model_1) TP=0] Decode batch. #running-req: 1,
       #token: 5191, token usage: 0.03, gen throughput (token/s): 62.51,
       #queue-req: 0"
    -> per-model KV occupancy (#token, token usage), running requests,
       per-model queue depth, decode throughput. Prefill lines additionally
       give #new-seq, which is the realised per-model arrival rate.

  server.log.gpu_scheduler.log   "net_available: inf, queue_len: N"
    -> per-GPU admission-queue depth. net_available is *always* inf: the
       released code sets it to float("inf") in request_queue.py:137, so
       memory-based admission control never rejects. Rejections are therefore
       structurally zero, not merely unobserved -- reported as such.

  server.log.global_controller.log   ACTION lines + per-GPU/model state dumps
    -> model activation/deactivation (the time-sharing half of ballooning),
       migrations, per-GPU available memory.

  gpu_timeline.txt           nvidia-smi @2s -> device memory and utilisation.

  <exp>_slo.json             analyze_slo.py's recomputed per-request stats.

Writes, next to the results:
  <exp>_timeseries.csv   1 s bins: per-model tokens/running/queue/throughput,
                         per-GPU queue + device memory + utilisation
  <exp>_summary.csv      one row: the rate-vs-X table columns

  python collect_metrics.py --exp probe_glob_on_ts1 --tag probe
"""
import argparse
import csv
import json
import os
import re
from collections import defaultdict

TS = r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d+)"
BATCH = re.compile(
    TS + r" GPU=(\d+) Worker (\d+) \((model_\d+)\) TP=\d+\] (Prefill|Decode) batch\.(.*)"
)
QLEN = re.compile(TS + r" GPU_Scheduler_(\d+)\] net_available: (\S+), queue_len: (\d+)")
# Three distinct ACTION shapes, and a migration is ONE line naming both halves:
#   "ACTION: deactivate model_3:0 on GPU 0. Reason: idle instance eviction"
#   "ACTION: activate inactive model model_4 on GPU 1. Reason: inactive models ..."
#   "ACTION: deactivate model_4 on GPU 1 and activate model_4 on GPU 0. Reason: migrate model"
# Matching a leading verb alone books a migration as a bare deactivation and
# loses the migration count entirely, so classify on the Reason instead.
ACTION = re.compile(TS + r" GlobalController\] ACTION: (.*?)\. Reason: (.*)")
FIELDS = re.compile(r"#?([a-z\- ]+?)(?: \(token/s\))?: ([0-9.]+)")


def epoch(s):
    import datetime as dt
    return dt.datetime.strptime(s, "%Y-%m-%d %H:%M:%S.%f").timestamp()


def parse_batches(path):
    rows = []
    if not os.path.exists(path):
        return rows
    for line in open(path, errors="replace"):
        m = BATCH.match(line)
        if not m:
            continue
        t, gpu, worker, model, kind, rest = m.groups()
        f = {k.strip(): float(v) for k, v in FIELDS.findall(rest)}
        rows.append({
            "t": epoch(t), "gpu": int(gpu), "model": model, "kind": kind,
            "running": f.get("running-req", 0.0),
            "queue": f.get("queue-req", 0.0),
            # Only Decode lines carry "#token". Defaulting it to 0 on Prefill
            # lines and then binning last-value-wins punches zeros into the KV
            # occupancy series, so absent stays None and the binner skips it.
            "tokens": f.get("token"),                 # KV tokens resident
            "token_usage": f.get("token usage"),      # fraction of this model's pool
            "gen_tps": f.get("gen throughput", 0.0),
            "new_seq": f.get("new-seq", 0.0),
            "new_token": f.get("new-token", 0.0),
        })
    return rows


def parse_qlen(path):
    out = []
    if not os.path.exists(path):
        return out
    for line in open(path, errors="replace"):
        m = QLEN.match(line)
        if m:
            t, gpu, avail, q = m.groups()
            out.append({"t": epoch(t), "gpu": int(gpu), "net_available": avail, "qlen": int(q)})
    return out


def parse_actions(path):
    out = []
    if not os.path.exists(path):
        return out
    for line in open(path, errors="replace"):
        m = ACTION.match(line)
        if m:
            t, what, reason = m.groups()
            if "migrate" in reason:
                kind = "migrate"
            elif what.startswith("activate"):
                kind = "activate"
            else:
                kind = "deactivate"
            out.append({"t": epoch(t), "kind": kind, "what": what.strip(),
                        "reason": reason.strip()})
    return out


def parse_gpu_timeline(path):
    out = []
    if not os.path.exists(path):
        return out
    for line in open(path, errors="replace"):
        p = line.strip().split(" ", 1)
        if len(p) != 2:
            continue
        try:
            t = float(p[0])
        except ValueError:
            continue
        for e in p[1].split(";"):
            f = [x.strip() for x in e.split(",")]
            if len(f) >= 3:
                try:
                    out.append({"t": t, "gpu": int(f[0]), "mem_mib": float(f[1]), "util": float(f[2])})
                except ValueError:
                    pass
    return out


def window_report(a, resdir, models):
    """Bucket per-request results by the ARRIVAL window they belong to.

    benchmark.py writes the request dump in trace order and never reorders it
    (run_tp_mode appends results in the order requests were created), so record
    i in the dump is request i in the pkl -- which is where the arrival time
    lives. The dump itself carries no timestamp, so this join is the only way to
    say "TTFT during the 4 req/s phase". analyze_slo.py's committed slowdown
    analysis relies on the same index correspondence.

    A rate ramp then reads directly as offered-rate vs achieved-rate/latency,
    i.e. the capacity curve, from one run instead of one run per rate.
    """
    import glob
    import pickle
    import numpy as np
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from analyze_slo import SLO_BASE
    from build_sharegpt_trace import _Unpickler, SLOT_RANK

    with open(a.trace, "rb") as f:
        _, trace = _Unpickler(f).load()
    rank_slot = {v: k for k, v in SLOT_RANK.items()}

    cand = sorted(glob.glob(os.path.join(resdir, "requests", f"{a.exp}_*_output_requests.json")),
                  key=os.path.getmtime)
    if not cand:
        print(f"  (no request dump for {a.exp}; skipping windows)")
        return None
    reqs = json.load(open(cand[-1]))
    if len(reqs) != len(trace):
        print(f"  (dump {len(reqs)} != trace {len(trace)}; skipping windows -- "
              "index join would be wrong)")
        return None

    rows = defaultdict(list)
    for r, tr in zip(reqs, trace):
        slot = rank_slot[int(tr.adapter_dir.split("-")[-1])]
        rows[int(tr.req_time // a.window)].append((f"model_{slot}", tr.req_time, r))

    def pctl(xs, q):
        return float(np.percentile(xs, q)) if xs else float("nan")

    out = []
    for w in sorted(rows):
        items = rows[w]
        ok = [(m, r) for m, _, r in items if r["success"]]
        span = a.window
        ttft = [r["ttft"] for _, r in ok]
        tpot = [r["tpot"] for _, r in ok]
        e2e = [r["latency"] for _, r in ok]
        hit = [r["ttft"] <= SLO_BASE[m][0] * a.ttft_slo_scale
               and r["tpot"] <= SLO_BASE[m][1] / 1000.0 * a.tpot_slo_scale for m, r in ok]
        row = {
            "window": w,
            "t_start_s": round(w * span, 1),
            "offered_rate_rps": round(len(items) / span, 3),
            "completed": len(ok),
            "out_tok_per_s": round(sum(r["output_len"] for _, r in ok) / span, 1),
            "attain_both": round(sum(hit) / len(items), 4) if items else "",
            "ttft_mean_ms": round(float(np.mean(ttft)) * 1000, 1) if ttft else "",
            "ttft_p50_ms": round(pctl(ttft, 50) * 1000, 1),
            "ttft_p95_ms": round(pctl(ttft, 95) * 1000, 1),
            "ttft_p99_ms": round(pctl(ttft, 99) * 1000, 1),
            "tpot_p50_ms": round(pctl(tpot, 50) * 1000, 2),
            "tpot_p95_ms": round(pctl(tpot, 95) * 1000, 2),
            "e2e_p50_s": round(pctl(e2e, 50), 2),
            "e2e_p95_s": round(pctl(e2e, 95), 2),
        }
        for m in models:
            sub = [r for mm, r in ok if mm == m]
            row[f"{m}_rate_rps"] = round(sum(1 for mm, _, _ in items if mm == m) / span, 3)
            row[f"{m}_ttft_p95_ms"] = round(pctl([r["ttft"] for r in sub], 95) * 1000, 1)
        out.append(row)

    path = os.path.join(resdir, f"{a.exp}_windows.csv")
    with open(path, "w", newline="") as f:
        w_ = csv.DictWriter(f, fieldnames=list(out[0]))
        w_.writeheader()
        w_.writerows(out)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, help="experiment name, e.g. probe_glob_on_ts1")
    ap.add_argument("--tag", required=True, help="results/<tag>/ namespace")
    ap.add_argument("--root", default=os.environ.get("PRISM_EXP", "/workspace/prism-exp/exp"))
    ap.add_argument("--bin", type=float, default=1.0, help="time-series bin seconds")
    ap.add_argument("--trace", default=None,
                    help="the .pkl this run replayed. Enables <exp>_windows.csv: "
                         "per-request stats bucketed by ARRIVAL window, which is how a "
                         "rate ramp becomes a capacity curve.")
    ap.add_argument("--window", type=float, default=90.0, help="--trace window seconds")
    ap.add_argument("--ttft-slo-scale", type=float, default=5.0)
    ap.add_argument("--tpot-slo-scale", type=float, default=2.0)
    a = ap.parse_args()

    logdir = os.path.join(a.root, "server-logs", a.exp)
    resdir = os.path.join(a.root, "results", a.tag)

    batches = parse_batches(os.path.join(logdir, "server.log"))
    qlens = parse_qlen(os.path.join(logdir, "server.log.gpu_scheduler.log"))
    actions = parse_actions(os.path.join(logdir, "server.log.global_controller.log"))
    gputl = parse_gpu_timeline(os.path.join(logdir, "gpu_timeline.txt"))
    if not batches:
        raise SystemExit(f"no batch lines in {logdir}/server.log -- wrong --exp?")

    models = sorted({r["model"] for r in batches}, key=lambda s: int(s.split("_")[1]))
    gpus = sorted({r["gpu"] for r in batches})
    t0 = min(r["t"] for r in batches)

    # --- bin everything on a common clock -----------------------------------
    # Batch lines are event-driven, not periodic, so within a bin we take the
    # LAST observed value for level quantities (KV tokens, running, queue) and
    # the SUM for count quantities (new sequences admitted, prefill tokens).
    bins = defaultdict(dict)
    for r in batches:
        b = int((r["t"] - t0) // a.bin)
        d = bins[b]
        m = r["model"]
        for k, v in (("tokens", r["tokens"]), ("running", r["running"]),
                     ("queue", r["queue"]), ("token_usage", r["token_usage"])):
            if v is not None:
                d[f"{m}_{k}"] = v
        if r["kind"] == "Decode" and r["gen_tps"]:
            d[f"{m}_gen_tps"] = r["gen_tps"]
        if r["kind"] == "Prefill":
            d[f"{m}_arrivals"] = d.get(f"{m}_arrivals", 0.0) + r["new_seq"]
            d[f"{m}_prefill_tok"] = d.get(f"{m}_prefill_tok", 0.0) + r["new_token"]
    for r in qlens:
        bins[int((r["t"] - t0) // a.bin)][f"gpu{r['gpu']}_sched_qlen"] = r["qlen"]
    for r in gputl:
        b = int((r["t"] - t0) // a.bin)
        bins[b][f"gpu{r['gpu']}_mem_mib"] = r["mem_mib"]
        bins[b][f"gpu{r['gpu']}_util"] = r["util"]
    for r in actions:
        b = int((r["t"] - t0) // a.bin)
        bins[b]["ctrl_actions"] = bins[b].get("ctrl_actions", 0) + 1

    cols = ["t_rel"]
    for m in models:
        cols += [f"{m}_{k}" for k in
                 ("arrivals", "running", "queue", "tokens", "token_usage", "gen_tps", "prefill_tok")]
    for g in gpus:
        cols += [f"gpu{g}_sched_qlen", f"gpu{g}_mem_mib", f"gpu{g}_util"]
    cols += ["ctrl_actions"]

    os.makedirs(resdir, exist_ok=True)
    ts_path = os.path.join(resdir, f"{a.exp}_timeseries.csv")
    with open(ts_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for b in sorted(bins):
            row = {"t_rel": round(b * a.bin, 1)}
            row.update(bins[b])
            w.writerow(row)

    # --- one-row summary -----------------------------------------------------
    slo_path = os.path.join(resdir, f"{a.exp}_slo.json")
    slo = json.load(open(slo_path)) if os.path.exists(slo_path) else {}
    allm = slo.get("per_model", {}).get("ALL", {})
    dur = slo.get("duration_s") or (max(r["t"] for r in batches) - t0)

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else ""

    peak_tokens = {m: max((r["tokens"] for r in batches if r["model"] == m and r["tokens"] is not None), default=0)
                   for m in models}
    peak_usage = {m: max((r["token_usage"] for r in batches if r["model"] == m and r["token_usage"] is not None), default=0)
                  for m in models}
    n_reject = 0            # see module docstring: admission control is disabled
    net_avail = {r["net_available"] for r in qlens} or {"n/a"}

    summary = {
        "exp": a.exp,
        "duration_s": round(dur, 1),
        "requests": slo.get("total_requests", ""),
        "completed": slo.get("completed", ""),
        "offered_rate_rps": round(slo.get("total_requests", 0) / dur, 3) if dur else "",
        "throughput_rps": round(allm.get("req_throughput_rps", 0), 3) if allm else "",
        "out_tok_throughput": round(allm.get("output_tok_throughput", 0), 1) if allm else "",
        "goodput_rps": round(allm.get("goodput_rps", 0), 3) if allm else "",
        "attain_ttft": allm.get("attain_ttft", ""),
        "attain_tpot": allm.get("attain_tpot", ""),
        "attain_both": allm.get("attain_both", ""),
        "ttft_mean_ms": allm.get("ttft_mean_ms", ""),
        "ttft_p50_ms": allm.get("ttft_p50_ms", ""),
        "ttft_p95_ms": allm.get("ttft_p95_ms", ""),
        "ttft_p99_ms": allm.get("ttft_p99_ms", ""),
        "tpot_mean_ms": allm.get("tpot_mean_ms", ""),
        "tpot_p50_ms": allm.get("tpot_p50_ms", ""),
        "tpot_p95_ms": allm.get("tpot_p95_ms", ""),
        "tpot_p99_ms": allm.get("tpot_p99_ms", ""),
        "e2e_p50_s": allm.get("e2e_p50_s", ""),
        "e2e_p95_s": allm.get("e2e_p95_s", ""),
        "e2e_p99_s": allm.get("e2e_p99_s", ""),
        "max_model_queue": max((r["queue"] for r in batches if r["queue"] is not None), default=0),
        "max_sched_qlen": max((r["qlen"] for r in qlens), default=0),
        "mean_sched_qlen": round(mean([r["qlen"] for r in qlens]), 2) if qlens else "",
        "peak_kv_tokens": json.dumps(peak_tokens),
        "peak_kv_pool_frac": json.dumps({k: round(v, 3) for k, v in peak_usage.items()}),
        "activations": sum(1 for r in actions if r["kind"] == "activate"),
        "deactivations": sum(1 for r in actions if r["kind"] == "deactivate"),
        "migrations": sum(1 for r in actions if r["kind"] == "migrate"),
        "idle_evictions": sum(1 for r in actions if "idle" in r.get("reason", "")),
        "rejected": n_reject,
        "admission_net_available": "/".join(sorted(net_avail)),
    }
    for g in gpus:
        gm = [r["mem_mib"] for r in gputl if r["gpu"] == g]
        gu = [r["util"] for r in gputl if r["gpu"] == g]
        summary[f"gpu{g}_mem_mean_gib"] = round(mean(gm) / 1024, 2) if gm else ""
        summary[f"gpu{g}_mem_max_gib"] = round(max(gm) / 1024, 2) if gm else ""
        summary[f"gpu{g}_util_mean"] = round(mean(gu), 1) if gu else ""

    sum_path = os.path.join(resdir, f"{a.exp}_summary.csv")
    with open(sum_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary))
        w.writeheader()
        w.writerow(summary)

    # --- optional: per-arrival-window request stats (the capacity curve) -----
    if a.trace:
        win_path = window_report(a, resdir, models)
    else:
        win_path = None

    print(f"{ts_path}  ({len(bins)} bins x {len(cols)} cols)")
    print(f"{sum_path}")
    if win_path:
        print(f"{win_path}")
    for k in ("duration_s", "requests", "throughput_rps", "attain_ttft", "attain_tpot",
              "ttft_p95_ms", "tpot_p95_ms", "max_model_queue", "max_sched_qlen",
              "peak_kv_pool_frac", "activations", "deactivations", "migrations", "idle_evictions",
              "rejected", "admission_net_available"):
        print(f"  {k:26s} {summary.get(k)}")


if __name__ == "__main__":
    main()
