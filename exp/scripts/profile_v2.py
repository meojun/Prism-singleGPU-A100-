#!/usr/bin/env python3
"""Per-model profiling for the paper-faithful-v2 study.

Produces, for ONE model served solo on a dedicated GPU (no contention):

  1. SLO baselines   TTFT p95 / TPOT p95, the paper's Sec. 7.1 method
                     (one request at a time, no queueing).
  2. c_i, three ways, because the estimators disagree by ~5x and the choice
     drives Algorithm 2's feasibility test:

       E1 ratio        sum(prompt_tokens) / sum(TTFT)
                       what v1 used.  TTFT bundles queueing, tokenisation,
                       scheduling and the fixed per-request overhead into the
                       denominator, so on short prompts it mostly measures
                       overhead, not prefill speed.
       E2 slope        least-squares fit of ttft = a + p/c, c = 1/slope.
                       The marginal cost of one more prompt token.
       E3 prefill      MEASURED prefill interval, straight from the engine:
                       meta_info.prefill_finish_timestamp - out_queue_timestamp.
                       Reported both per-request (E3-solo) and as aggregate
                       throughput under a saturating prefill burst
                       (E3-saturated), which is what the paper's
                       "chunked-prefill speed c_i determined by the model"
                       actually denotes -- an engine capacity, not the
                       reciprocal of one request's latency.

Prompts are drawn in three length buckets (short / medium / long) so the
estimators can be compared across the range instead of at one prompt size.

    python profile_v2.py --url http://127.0.0.1:31000 --model model_1 \
        --model-path meta-llama/Llama-3.2-1B -o out.json
"""
import argparse
import asyncio
import json
import random
import statistics
import sys
import time

import aiohttp

BUCKETS = {"short": (32, 128), "medium": (256, 768), "long": (1024, 2048)}


def load_prompts(sharegpt, tokenizer, n_per_bucket, seed):
    """Real ShareGPT text, tokenised with THIS model's tokenizer so prompt_len
    is the count the engine will actually see."""
    with open(sharegpt) as f:
        data = json.load(f)
    texts = [c["conversations"][0]["value"] for c in data
             if len(c.get("conversations", [])) >= 2 and c["conversations"][0]["value"]]
    random.Random(seed).shuffle(texts)

    out = {b: [] for b in BUCKETS}
    for t in texts:
        if all(len(v) >= n_per_bucket for v in out.values()):
            break
        ids = tokenizer(t, add_special_tokens=False)["input_ids"]
        for name, (lo, hi) in BUCKETS.items():
            if len(out[name]) >= n_per_bucket:
                continue
            if len(ids) >= lo:
                cut = min(len(ids), random.Random(seed + len(ids)).randint(lo, hi))
                out[name].append((tokenizer.decode(ids[:cut]), cut))
                break
    return out


async def one(session, url, model, text, plen, out_len, rid):
    payload = {
        "text": text,
        "sampling_params": {"ignore_eos": True, "max_new_tokens": int(out_len)},
        "rid": rid, "model": model, "slo": 1e9, "slo_ttft": 1e9, "slo_tpot": 1e9,
        "prompt_len": int(plen), "output_len": int(out_len),
    }
    st = time.perf_counter()
    meta = {}
    ntok = 0
    async with session.post(url + "/generate", json=payload) as resp:
        if resp.status != 200:
            return {"ok": False, "err": f"http {resp.status}"}
        async for chunk in resp.content:
            chunk = chunk.strip()
            if not chunk:
                continue
            d = json.loads(chunk.decode())
            if d.get("text"):
                ntok += 1
            if d.get("meta_info"):
                meta = d["meta_info"]
    e2e_client = time.perf_counter() - st
    arr = meta.get("arrival_timestamp")
    outq = meta.get("out_queue_timestamp")
    pf = meta.get("prefill_finish_timestamp")
    fin = meta.get("finish_timestamp")
    # TTFT and TPOT use the harness's own SERVER-SIDE definitions
    # (benchmark.py:140-142), not client stream timing: this endpoint is
    # non-streaming, so a client-side first-token clock would just measure the
    # whole generation.  Reusing the harness definition also keeps these
    # baselines directly comparable with what the sweep reports.
    # finish_timestamp can be absent for a 1-2 token generation; the prefill
    # trio is what the c_i measurement needs, so it alone gates ok.
    ok = bool(arr and outq and pf)
    return {
        "ok": ok, "prompt_len": plen, "output_len": out_len,
        "ttft": (pf - arr) if ok else None,
        "tpot": ((fin - pf) / out_len) if (ok and fin and out_len > 8) else None,
        "e2e_client": e2e_client, "n_chunks": ntok,
        "arrival_timestamp": arr, "out_queue_timestamp": outq,
        "prefill_finish_timestamp": pf, "finish_timestamp": fin,
        # true prefill interval: dequeued to engine -> prefill done
        "prefill_s": (pf - outq) if ok else None,
        "queue_s": (outq - arr) if ok else None,
    }


async def run_sequential(url, model, prompts, out_len, reps):
    """Concurrency 1 -- the paper's no-contention Sec. 7.1 condition."""
    res = []
    timeout = aiohttp.ClientTimeout(total=600)
    async with aiohttp.ClientSession(timeout=timeout) as s:
        n = 0
        for _ in range(reps):
            for bucket, items in prompts.items():
                for text, plen in items:
                    r = await one(s, url, model, text, plen, out_len, f"seq-{n}")
                    r["bucket"] = bucket
                    res.append(r)
                    n += 1
    return res


async def run_saturated(url, model, prompts, concurrency, rounds):
    """Prefill-dominated burst (output_len=2) at high concurrency.

    Aggregate prefill throughput over the burst = the engine capacity the
    paper's c_i denotes.  Wall-clock window is taken from the engine's own
    timestamps, not the client's, so client-side send cost is excluded.
    """
    flat = [(t, p) for items in prompts.values() for (t, p) in items]
    res = []
    timeout = aiohttp.ClientTimeout(total=1200)
    async with aiohttp.ClientSession(timeout=timeout) as s:
        for rd in range(rounds):
            batch = [flat[(rd * concurrency + i) % len(flat)] for i in range(concurrency)]
            tasks = [one(s, url, model, t, p, 2, f"sat-{rd}-{i}")
                     for i, (t, p) in enumerate(batch)]
            res.extend(await asyncio.gather(*tasks))
    return res


def pct(xs, q):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    k = max(0, min(len(xs) - 1, int(round(q / 100.0 * (len(xs) - 1)))))
    return xs[k]


def fit_slope(pairs):
    """least squares ttft = a + p/c  ->  returns (c, intercept, r2)"""
    if len(pairs) < 3:
        return None, None, None
    xs = [p for p, _ in pairs]
    ys = [t for _, t in pairs]
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return None, None, None
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    inter = my - slope * mx
    ss_res = sum((y - (inter + slope * x)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    r2 = 1 - ss_res / ss_tot if ss_tot else None
    return (1.0 / slope if slope > 0 else None), inter, r2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True, help="slot name, e.g. model_1")
    ap.add_argument("--model-path", required=True, help="HF path, for the tokenizer")
    ap.add_argument("--sharegpt", default="/workspace/datasets/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json")
    ap.add_argument("--per-bucket", type=int, default=40)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--decode-len", type=int, default=128)
    ap.add_argument("--sat-concurrency", type=int, default=48)
    ap.add_argument("--sat-rounds", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model_path)
    prompts = load_prompts(a.sharegpt, tok, a.per_bucket, a.seed)
    print(f"[{a.model}] prompts: " + ", ".join(f"{k}={len(v)}" for k, v in prompts.items()))

    seq = asyncio.run(run_sequential(a.url, a.model, prompts, a.decode_len, a.reps))
    ok = [r for r in seq if r.get("ok")]
    print(f"[{a.model}] sequential ok={len(ok)}/{len(seq)}")
    if not ok:
        sys.exit(f"FATAL: no successful sequential requests for {a.model}")

    sat = asyncio.run(run_saturated(a.url, a.model, prompts, a.sat_concurrency, a.sat_rounds))
    sok = [r for r in sat if r.get("ok") and r.get("prefill_finish_timestamp")]
    print(f"[{a.model}] saturated ok={len(sok)}/{len(sat)}")

    # ---- estimators
    tot_p = sum(r["prompt_len"] for r in ok)
    tot_ttft = sum(r["ttft"] for r in ok)
    e1 = tot_p / tot_ttft if tot_ttft else None

    e2, inter, r2 = fit_slope([(r["prompt_len"], r["ttft"]) for r in ok])

    pf = [r for r in ok if r.get("prefill_s")]
    e3_solo = (sum(r["prompt_len"] for r in pf) / sum(r["prefill_s"] for r in pf)) if pf else None

    e3_sat = None
    if sok:
        t0 = min(r["out_queue_timestamp"] for r in sok)
        t1 = max(r["prefill_finish_timestamp"] for r in sok)
        if t1 > t0:
            e3_sat = sum(r["prompt_len"] for r in sok) / (t1 - t0)

    per_bucket = {}
    for b in BUCKETS:
        sub = [r for r in ok if r.get("bucket") == b]
        if not sub:
            continue
        spf = [r for r in sub if r.get("prefill_s")]
        per_bucket[b] = {
            "n": len(sub),
            "prompt_len_mean": statistics.fmean(r["prompt_len"] for r in sub),
            "ttft_mean_ms": 1000 * statistics.fmean(r["ttft"] for r in sub),
            "prefill_mean_ms": 1000 * statistics.fmean(r["prefill_s"] for r in spf) if spf else None,
            "queue_mean_ms": 1000 * statistics.fmean(r["queue_s"] for r in sub if r.get("queue_s")) if any(r.get("queue_s") for r in sub) else None,
            "E1_ratio": sum(r["prompt_len"] for r in sub) / sum(r["ttft"] for r in sub),
            "E3_prefill_solo": (sum(r["prompt_len"] for r in spf) / sum(r["prefill_s"] for r in spf)) if spf else None,
        }

    res = {
        "model": a.model, "model_path": a.model_path,
        "n_sequential": len(ok), "n_saturated": len(sok),
        "slo_baseline": {
            "ttft_p95_s": pct([r["ttft"] for r in ok], 95),
            "ttft_p50_s": pct([r["ttft"] for r in ok], 50),
            "tpot_p95_s": pct([r["tpot"] for r in ok], 95),
            "tpot_p50_s": pct([r["tpot"] for r in ok], 50),
        },
        "c_i_estimators": {
            "E1_ratio_sum_p_over_sum_ttft": e1,
            "E2_regression_slope": e2,
            "E2_intercept_s": inter,
            "E2_r2": r2,
            "E3_prefill_solo": e3_solo,
            "E3_prefill_saturated": e3_sat,
            "sat_concurrency": a.sat_concurrency,
        },
        "per_bucket": per_bucket,
        "raw_sequential": ok,
    }
    with open(a.out, "w") as f:
        json.dump(res, f, indent=2)
    e = res["c_i_estimators"]
    print(f"[{a.model}] c_i  E1={e['E1_ratio_sum_p_over_sum_ttft']:.0f}  "
          f"E2={e['E2_regression_slope'] or float('nan'):.0f}  "
          f"E3solo={e['E3_prefill_solo'] or float('nan'):.0f}  "
          f"E3sat={e['E3_prefill_saturated'] or float('nan'):.0f} tok/s")
    print(f"[{a.model}] SLO baseline TTFT p95={1000*res['slo_baseline']['ttft_p95_s']:.1f}ms  "
          f"TPOT p95={1000*res['slo_baseline']['tpot_p95_s']:.2f}ms")


if __name__ == "__main__":
    main()
