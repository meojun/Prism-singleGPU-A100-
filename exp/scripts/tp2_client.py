#!/usr/bin/env python3
"""Drive a running multi-model server and report per-request latency.

Used by run_tp2_validation.sh: enough load to prove a TP=2 model actually
serves, with the same TTFT/TPOT definitions the sweep uses so the numbers are
comparable.
"""
import argparse
import asyncio
import json
import random
import statistics
import time
from pathlib import Path

import aiohttp


async def one(session, url, rid, model, prompt, output_len, results):
    payload = {
        "text": prompt,
        "sampling_params": {"ignore_eos": True, "max_new_tokens": output_len},
        "rid": rid, "model": model,
        "slo": 10.0, "slo_ttft": 10.0, "slo_tpot": 1.0,
        "prompt_len": len(prompt.split()), "output_len": output_len,
    }
    start = time.perf_counter()
    ttft, ntok, err = None, 0, None
    try:
        async with session.post(url + "/generate", json=payload) as resp:
            if resp.status != 200:
                err = f"HTTP {resp.status}: {(await resp.text())[:200]}"
            else:
                async for chunk in resp.content:
                    chunk = chunk.strip()
                    if not chunk:
                        continue
                    data = json.loads(chunk.decode())
                    if data.get("text"):
                        if ttft is None:
                            ttft = time.perf_counter() - start
                        ntok += 1
    except Exception as exc:                              # noqa: BLE001
        err = f"{type(exc).__name__}: {exc}"
    end = time.perf_counter()
    results.append({
        "rid": rid, "model": model, "error": err,
        "success": err is None and ttft is not None,
        "ttft_s": ttft, "e2e_s": end - start,
        "output_len": output_len,
        "tpot_s": ((end - start) - ttft) / max(1, output_len - 1) if ttft else None,
    })


def pct(values, q):
    if not values:
        return None
    values = sorted(values)
    idx = min(len(values) - 1, max(0, int(round(q / 100 * (len(values) - 1)))))
    return values[idx]


def summarize(rows, key):
    vals = [r[key] for r in rows if r["success"] and r[key] is not None]
    if not vals:
        return {}
    return {
        "n": len(vals), "mean": statistics.fmean(vals),
        "p50": pct(vals, 50), "p95": pct(vals, 95),
        "p99": pct(vals, 99), "max": max(vals),
    }


async def main_async(args):
    random.seed(args.seed)
    prompts = [" ".join(random.choice(
        ["memory", "cache", "tensor", "kernel", "serving", "latency", "cluster"])
        for _ in range(random.randint(32, 256))) for _ in range(args.requests)]
    results = []
    conn = aiohttp.TCPConnector(limit=args.concurrency)
    timeout = aiohttp.ClientTimeout(total=args.timeout)
    async with aiohttp.ClientSession(connector=conn, timeout=timeout) as session:
        sem = asyncio.Semaphore(args.concurrency)

        async def guarded(i):
            async with sem:
                await one(session, args.url, f"tp2_{i}", args.models[i % len(args.models)],
                          prompts[i], args.output_len, results)
        await asyncio.gather(*(guarded(i) for i in range(args.requests)))

    ok = [r for r in results if r["success"]]
    report = {
        "requests": len(results),
        "successful": len(ok),
        "failed": len(results) - len(ok),
        "errors": sorted({r["error"] for r in results if r["error"]})[:10],
        "ttft": summarize(results, "ttft_s"),
        "tpot": summarize(results, "tpot_s"),
        "e2e": summarize(results, "e2e_s"),
        "per_model": {
            m: {"requests": sum(1 for r in results if r["model"] == m),
                "successful": sum(1 for r in results if r["model"] == m and r["success"])}
            for m in args.models
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"summary": report, "requests": results}, indent=2))
    print(json.dumps(report, indent=2))
    return 0 if report["failed"] == 0 and report["successful"] > 0 else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--requests", type=int, default=120)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--output-len", type=int, default=64)
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    raise SystemExit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()
