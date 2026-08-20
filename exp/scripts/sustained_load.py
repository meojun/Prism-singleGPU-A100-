#!/usr/bin/env python3
"""Sustained open-loop load against one model, with per-request records.

Open loop on purpose: arrivals follow the schedule regardless of whether the
server is keeping up, so a stall shows as growing latency and a growing
in-flight count instead of quietly throttling the offered rate.  A closed-loop
driver would hide exactly the failure this test exists to catch.

TTFT is taken from the first streamed chunk, so it is a real time-to-first-token
and not an end-to-end time relabelled.  If the server does not stream, TTFT is
recorded as None rather than filled in with e2e -- a missing number is better
than a wrong one.

Usage:
  sustained_load.py --port P --model NAME --seconds N --rate R --out DIR
"""

import argparse
import asyncio
import json
import random
import statistics as stats
import time
from pathlib import Path

import aiohttp

PROMPTS = [
    "Summarise the causes of the 1929 financial crash in three sentences.",
    "Explain the difference between a mutex and a semaphore.",
    "Write a short paragraph about the history of the printing press.",
    "What are the trade-offs between B-trees and LSM trees?",
    "Describe how a compiler performs register allocation.",
    "Give a plain-language explanation of Bayes' theorem.",
]


async def one(session, url, model, rid, prompt, out_len, rec):
    payload = {
        "text": prompt,
        "sampling_params": {"ignore_eos": True, "max_new_tokens": out_len},
        "rid": rid, "model": model,
        "slo": 60.0, "slo_ttft": 10.0, "slo_tpot": 1.0,
        "prompt_len": len(prompt.split()), "output_len": out_len,
    }
    t0 = time.perf_counter()
    ttft = None
    try:
        async with session.post(url, json=payload) as resp:
            if resp.status != 200:
                rec.append({"rid": rid, "ok": False, "err": f"HTTP {resp.status}",
                            "ttft": None, "e2e": None, "out_len": out_len})
                return
            async for chunk in resp.content:
                if chunk.strip() and ttft is None:
                    ttft = time.perf_counter() - t0
        e2e = time.perf_counter() - t0
        rec.append({"rid": rid, "ok": True, "err": None, "ttft": ttft, "e2e": e2e,
                    "out_len": out_len,
                    "tpot": ((e2e - ttft) / max(1, out_len - 1)) if ttft else None})
    except Exception as e:  # noqa: BLE001 -- the failure mode is the datum
        rec.append({"rid": rid, "ok": False, "err": f"{type(e).__name__}: {e}",
                    "ttft": None, "e2e": None, "out_len": out_len})


async def drive(port, model, seconds, rate, out_dir, seed):
    rng = random.Random(seed)
    url = f"http://127.0.0.1:{port}/generate"
    rec, tasks = [], []
    timeout = aiohttp.ClientTimeout(total=None, sock_read=600)
    start = time.perf_counter()
    i = 0
    async with aiohttp.ClientSession(timeout=timeout) as session:
        while True:
            now = time.perf_counter() - start
            if now >= seconds:
                break
            prompt = rng.choice(PROMPTS)
            out_len = rng.choice([32, 64, 128, 192])
            tasks.append(asyncio.create_task(
                one(session, url, model, f"b70_{i}", prompt, out_len, rec)))
            i += 1
            await asyncio.sleep(rng.expovariate(rate) if rate > 0 else 0.1)
            if i % 50 == 0:
                inflight = sum(1 for t in tasks if not t.done())
                print(f"  t={now:6.1f}s sent={i} done={len(rec)} inflight={inflight}",
                      flush=True)
        print(f"  arrivals finished at {time.perf_counter()-start:.1f}s; draining",
              flush=True)
        if tasks:
            await asyncio.wait(tasks, timeout=900)
    wall = time.perf_counter() - start

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "requests.jsonl").open("w") as fh:
        for r in rec:
            fh.write(json.dumps(r) + "\n")

    ok = [r for r in rec if r["ok"]]
    def pct(vals, q):
        vals = sorted(v for v in vals if v is not None)
        if not vals:
            return None
        k = min(len(vals) - 1, int(round(q * (len(vals) - 1))))
        return vals[k]

    def block(name, vals):
        vals = [v for v in vals if v is not None]
        if not vals:
            return {"n": 0}
        return {"n": len(vals), "mean": stats.mean(vals),
                "p50": pct(vals, .50), "p95": pct(vals, .95),
                "p99": pct(vals, .99), "max": max(vals)}

    summary = {
        "wall_seconds": wall,
        "target_rate": rate,
        "sent": i,
        "completed": len(ok),
        "failed": len(rec) - len(ok),
        "never_returned": i - len(rec),
        "achieved_throughput_rps": len(ok) / wall if wall else 0,
        "output_tokens": sum(r["out_len"] for r in ok),
        "output_token_throughput": sum(r["out_len"] for r in ok) / wall if wall else 0,
        "ttft": block("ttft", [r.get("ttft") for r in ok]),
        "tpot": block("tpot", [r.get("tpot") for r in ok]),
        "e2e": block("e2e", [r.get("e2e") for r in ok]),
        "errors": sorted({r["err"] for r in rec if r["err"]})[:20],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0 if summary["failed"] == 0 and summary["completed"] > 0 else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--seconds", type=float, default=1800)
    ap.add_argument("--rate", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True)
    ns = ap.parse_args()
    raise SystemExit(asyncio.run(
        drive(ns.port, ns.model, ns.seconds, ns.rate, ns.out, ns.seed)))


if __name__ == "__main__":
    main()
