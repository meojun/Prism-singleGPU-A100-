#!/usr/bin/env python3
"""Build the PAIRED (shifting-bursty, steady) workloads for paper-faithful-v2.

The whole point is a controlled contrast: the two traces must differ in
ARRIVAL TIMING AND NOTHING ELSE.  Identical across the pair, by construction:

    models, request_id, model assignment, prompt text, prompt_len,
    output_len, per-model request count, total request count,
    experiment duration, average offered load, seed

Different:  when each request arrives.

Design of the bursty trace
--------------------------
Phases of random length (default 30-90 s).  In each phase every model is HOT,
MEDIUM, LOW or IDLE, redrawn independently -- the scheduler cannot see the next
hot set coming.  Crucially the per-phase rates are RENORMALISED so the
*aggregate* arrival rate is the same constant R in every phase.  The cluster
therefore sees identical total load at all times in both traces; the only thing
that moves is WHICH model is hot.  That isolates the effect Prism claims to
exploit -- reclaiming memory from idle models for hot ones -- from the trivial
confound of the cluster simply being busier at some moments.

Design of the steady trace
--------------------------
Per-model counts are taken from the bursty trace and its N_m arrivals are drawn
uniformly at random over [0, D).  N uniform points on an interval is exactly a
homogeneous Poisson process conditioned on its count, so steady is the constant-
rate counterpart of bursty with the request count held exactly equal.

    python build_paired_workload.py --rate 12 --duration 600 --seed 1 \
        --outdir exp/workloads/paper-faithful-v2
"""
import argparse
import json
import os
import pickle
import random

MODELS = {
    "model_1": "meta-llama/Llama-3.2-1B",
    "model_2": "Qwen/Qwen2.5-1.5B-Instruct",
    "model_3": "meta-llama/Llama-3.2-3B",
    "model_4": "Qwen/Qwen2.5-3B-Instruct",
    "model_5": "meta-llama/Llama-3.1-8B",
    "model_6": "Qwen/Qwen2.5-7B-Instruct",
}

# Relative long-run share of traffic.  Deliberately skewed -- the paper's own
# traces are (top slots carry ~90% of requests) -- but every model gets enough
# requests for its own percentiles to mean something.
BASE_SHARE = {"model_1": 1.0, "model_2": 1.0, "model_3": 1.5,
              "model_4": 1.5, "model_5": 2.5, "model_6": 2.5}

HOT_RANGE = (3.0, 6.0)
MEDIUM_RANGE = (0.8, 1.5)
LOW_RANGE = (0.10, 0.40)


class Request:
    """Structural stand-in for trace.Request; trace.py's CustomUnpickler maps
    any class named `Request` onto its own dataclass, so this pickles fine."""


def build_phases(rng, duration, phase_lo, phase_hi, models):
    """Phases of random length; hot set redrawn each phase.

    The hot set is sampled with a STALENESS weight (phases since a model was
    last hot) rather than uniformly. Uniform iid draws leave some model cold for
    an entire short trace, which starves its percentiles and makes per-model
    numbers unreadable -- an artefact of the generator, not a property of the
    workload. Staleness weighting keeps every tenant in rotation while leaving
    the next hot set unpredictable: the scheduler still cannot see it coming,
    because which of the stale models get picked, how many (1-3), and how long
    the phase lasts are all still random.
    """
    phases, t = [], 0.0
    since_hot = {m: 1 for m in models}
    while t < duration:
        length = min(rng.uniform(phase_lo, phase_hi), duration - t)
        if length < 5.0:
            if phases:
                phases[-1]["end"] = duration
                phases[-1]["len"] = phases[-1]["end"] - phases[-1]["start"]
            break
        n_hot = rng.randint(1, 3)
        pool, hot = models[:], []
        for _ in range(n_hot):
            wts = [since_hot[m] ** 2 for m in pool]
            pick = rng.choices(pool, weights=wts, k=1)[0]
            hot.append(pick)
            pool.remove(pick)
        for m in models:
            since_hot[m] = 1 if m in hot else since_hot[m] + 1
        rest = pool
        rng.shuffle(rest)
        # the rest split into medium / low / idle, at least one idle where possible
        n_idle = rng.randint(1, max(1, len(rest) - 1)) if rest else 0
        idle = rest[:n_idle]
        remaining = rest[n_idle:]
        n_med = rng.randint(0, len(remaining))
        medium = remaining[:n_med]
        low = remaining[n_med:]

        w = {}
        for m in hot:
            w[m] = BASE_SHARE[m] * rng.uniform(*HOT_RANGE)
        for m in medium:
            w[m] = BASE_SHARE[m] * rng.uniform(*MEDIUM_RANGE)
        for m in low:
            w[m] = BASE_SHARE[m] * rng.uniform(*LOW_RANGE)
        for m in idle:
            w[m] = 0.0
        phases.append({
            "start": t, "end": t + length, "len": length,
            "hot": hot, "medium": medium, "low": low, "idle": idle,
            "weights": w,
        })
        t += length
    return phases


def bursty_arrivals(rng, phases, rate, models):
    """Non-homogeneous Poisson with piecewise-constant, renormalised rates."""
    arrivals = {m: [] for m in models}
    for ph in phases:
        tot = sum(ph["weights"].values())
        ph["rates"] = {m: (rate * ph["weights"][m] / tot if tot > 0 else 0.0)
                       for m in models}
        for m in models:
            lam = ph["rates"][m] * ph["len"]
            if lam <= 0:
                continue
            # Poisson count, then uniform positions inside the phase
            n = poisson(rng, lam)
            for _ in range(n):
                arrivals[m].append(ph["start"] + rng.uniform(0, ph["len"]))
    for m in models:
        arrivals[m].sort()
    return arrivals


def poisson(rng, lam):
    """Knuth for small lam, normal approximation above 500."""
    if lam > 500:
        return max(0, int(round(rng.gauss(lam, lam ** 0.5))))
    L, k, p = pow(2.718281828459045, -lam), 0, 1.0
    while True:
        k += 1
        p *= rng.random()
        if p <= L:
            return k - 1


def steady_arrivals(rng, counts, duration, models):
    return {m: sorted(rng.uniform(0, duration) for _ in range(counts[m]))
            for m in models}


def sample_payloads(sharegpt, counts, seed, out_lo, out_hi):
    """One ShareGPT (prompt, answer) pair per request, tokenised with the
    tokenizer of the model that will serve it -- prompt_len must be the count
    the engine actually sees, and the six models do not share a tokenizer."""
    from transformers import AutoTokenizer
    with open(sharegpt) as f:
        data = json.load(f)
    pairs = [(c["conversations"][0]["value"], c["conversations"][1]["value"])
             for c in data if len(c.get("conversations", [])) >= 2
             and c["conversations"][0]["value"] and c["conversations"][1]["value"]]
    random.Random(seed).shuffle(pairs)

    payloads, cursor = {}, 0
    for m in sorted(counts):
        tok = AutoTokenizer.from_pretrained(MODELS[m])
        got = []
        while len(got) < counts[m]:
            if cursor >= len(pairs):
                raise SystemExit("ShareGPT exhausted -- lower the rate or duration")
            q, a = pairs[cursor]
            cursor += 1
            qi = tok(q, add_special_tokens=False)["input_ids"]
            if not 8 <= len(qi) <= 3072:
                continue
            ai = tok(a, add_special_tokens=False)["input_ids"]
            olen = max(out_lo, min(out_hi, len(ai) or out_lo))
            got.append({"prompt": tok.decode(qi), "prompt_len": len(qi),
                        "output_len": int(olen)})
        payloads[m] = got
        print(f"  {m:9s} {counts[m]:5d} reqs  "
              f"prompt_len mean={sum(g['prompt_len'] for g in got)/len(got):7.1f}  "
              f"output_len mean={sum(g['output_len'] for g in got)/len(got):6.1f}")
    return payloads


def make_requests(payloads, arrivals, slo_base, models):
    reqs = []
    for m in models:
        for i, (pl, at) in enumerate(zip(payloads[m], arrivals[m])):
            r = Request()
            r.req_id = f"{m}#{i}"
            r.prompt = pl["prompt"]
            r.prompt_len = pl["prompt_len"]
            r.output_len = pl["output_len"]
            r.arrival_time = at
            r.model = m
            r.slo = slo_base[m]["ttft"]
            r.slo_ttft = slo_base[m]["ttft"]
            r.slo_tpot = slo_base[m]["tpot"]
            reqs.append(r)
    reqs.sort(key=lambda r: r.arrival_time)
    return reqs


def dump(path, reqs):
    """(adapter_dirs, requests) -- the shape RealWorldTrace._load_requests expects.
    The sentinel in slot 0 is what flips trace.py onto the direct passthrough."""
    with open(path, "wb") as f:
        pickle.dump((["__PRISM_DIRECT__"], reqs), f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rate", type=float, required=True, help="aggregate req/s")
    ap.add_argument("--duration", type=float, default=600.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--phase-lo", type=float, default=30.0)
    ap.add_argument("--phase-hi", type=float, default=90.0)
    ap.add_argument("--out-lo", type=int, default=16)
    ap.add_argument("--out-hi", type=int, default=384)
    ap.add_argument("--sharegpt", default="/workspace/datasets/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json")
    ap.add_argument("--slo-base", default="/workspace/prism-exp/exp/configs/v2/slo_base.json")
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()

    os.makedirs(a.outdir, exist_ok=True)
    models = sorted(MODELS)
    slo_base = json.load(open(a.slo_base))
    rng = random.Random(a.seed)

    phases = build_phases(rng, a.duration, a.phase_lo, a.phase_hi, models)
    burst = bursty_arrivals(rng, phases, a.rate, models)
    counts = {m: len(v) for m, v in burst.items()}
    total = sum(counts.values())
    print(f"rate={a.rate} req/s  duration={a.duration}s  phases={len(phases)}  total={total}")

    steady = steady_arrivals(random.Random(a.seed + 10_000), counts, a.duration, models)
    assert {m: len(v) for m, v in steady.items()} == counts

    payloads = sample_payloads(a.sharegpt, counts, a.seed, a.out_lo, a.out_hi)

    tag = f"r{a.rate:g}_s{a.seed}"
    b_reqs = make_requests(payloads, burst, slo_base, models)
    s_reqs = make_requests(payloads, steady, slo_base, models)
    dump(os.path.join(a.outdir, f"bursty_{tag}.pkl"), b_reqs)
    dump(os.path.join(a.outdir, f"steady_{tag}.pkl"), s_reqs)

    shared = {m: [{"request_id": f"{m}#{i}", "model_id": m,
                   "prompt_tokens": p["prompt_len"],
                   "requested_output_length": p["output_len"]}
                  for i, p in enumerate(payloads[m])] for m in models}
    json.dump({
        "rate": a.rate, "duration": a.duration, "seed": a.seed,
        "total_requests": total, "per_model_requests": counts,
        "average_offered_load_req_s": total / a.duration,
        "prompt_tokens_total": {m: sum(p["prompt_len"] for p in payloads[m]) for m in models},
        "output_tokens_total": {m: sum(p["output_len"] for p in payloads[m]) for m in models},
        "requests": shared,
    }, open(os.path.join(a.outdir, f"paired_requests_{tag}.json"), "w"), indent=1)
    json.dump({
        "seed": a.seed, "rate": a.rate, "duration": a.duration,
        "phase_len_range": [a.phase_lo, a.phase_hi],
        "hot_multiplier_range": HOT_RANGE, "medium_multiplier_range": MEDIUM_RANGE,
        "low_multiplier_range": LOW_RANGE, "base_share": BASE_SHARE,
        "phases": [{k: v for k, v in ph.items()} for ph in phases],
    }, open(os.path.join(a.outdir, f"phases_{tag}.json"), "w"), indent=1)

    print(f"wrote bursty_{tag}.pkl / steady_{tag}.pkl / paired_requests_{tag}.json "
          f"/ phases_{tag}.json  in {a.outdir}")
    for m in models:
        print(f"  {m:9s} n={counts[m]:5d}  bursty span "
              f"[{burst[m][0]:.0f},{burst[m][-1]:.0f}]  " if counts[m] else f"  {m:9s} n=0",
              end="")
        print(f"steady span [{steady[m][0]:.0f},{steady[m][-1]:.0f}]" if counts[m] else "")


if __name__ == "__main__":
    main()
