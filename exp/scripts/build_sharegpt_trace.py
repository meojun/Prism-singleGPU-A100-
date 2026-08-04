#!/usr/bin/env python3
"""Build ShareGPT-backed traces in the format Prism's multi-model harness expects.

The harness does NOT have a ShareGPT loader.  `benchmark.py --real-trace <pkl>`
reads `req.prompt` straight out of the pickle (trace.py::generate_e2e_benchmark_reqs),
so switching datasets means producing a new pickle with the same structure:

    [ adapter_dirs: list[str] , requests: list[Request] ]

    Request: req_id, prompt, prompt_len, output_len, req_time,
             adapter_dir, model_dir, slo, slo_ttft, slo_tpot

Two variants are produced, because "use ShareGPT" is ambiguous:

  content  Arrival times, routing, prompt_len and output_len are taken from the
           ORIGINAL trace; only the prompt TEXT becomes real ShareGPT content,
           truncated to the same true token count.  Load is bit-for-bit the same
           as the original run, so results stay comparable to
           exp/results/sanity/REPORT.md -- the only variable that moved is content.

  full     Arrival times and routing are kept, but prompt/prompt_len/output_len
           all come from ShareGPT (sglang's standard filter).  This is a real
           chat workload shape (longer prefill, shorter decode) and is NOT
           comparable to the original baseline.

Why "true token count" and not prompt_len: benchmark.py sends {"text": req.prompt}
and the server tokenizes it, so the actual prefill cost is len(tokenize(text)),
not the prompt_len field.  The original trace is itself slightly inconsistent
here ("Hello "*13 has prompt_len 13 but tokenizes to 14) -- we reproduce the
TOKENIZED length so the GPU sees identical work, and leave the prompt_len field
untouched so Prism's scheduler (request_queue.py uses prompt_len) sees identical
input too.

Output length is not affected by any of this: benchmark.py sends
ignore_eos=True with max_new_tokens=output_len, so decode length is forced
regardless of what the model would have generated.

Usage:
    python exp/scripts/build_sharegpt_trace.py                  # both variants
    python exp/scripts/build_sharegpt_trace.py --variant content
"""
import argparse
import json
import os
import pickle
import random
import sys

DEFAULT_SHAREGPT = "/workspace/datasets/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json"
DEFAULT_OUTDIR = "/workspace/datasets/sharegpt"
TOKENIZER = "meta-llama/Llama-3.1-8B"


class Request:
    """Structural stand-in for trace.Request.

    trace.py's CustomUnpickler maps ANY class named "Request" onto its own
    dataclass regardless of the recorded module, so pickling instances of this
    class is enough -- the loader never imports this module.
    """


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == "Request":
            return Request
        return super().find_class(module, name)


def load_original(path):
    with open(path, "rb") as f:
        adapter_dirs, requests = _Unpickler(f).load()
    return adapter_dirs, requests


def load_sharegpt_pairs(path, seed):
    with open(path) as f:
        data = json.load(f)
    pairs = [
        (c["conversations"][0]["value"], c["conversations"][1]["value"])
        for c in data
        if len(c.get("conversations", [])) >= 2
    ]
    random.Random(seed).shuffle(pairs)
    return pairs


def fit_to_length(tok, src_ids, target):
    """Return text that re-tokenizes to exactly `target` tokens, if possible.

    Truncating token ids then decoding is not round-trip safe with BPE: the
    decoded string can re-encode to a different length.  Correct by nudging the
    id count until the re-encoded length matches, then give up gracefully.
    """
    n = min(target, len(src_ids))
    for _ in range(24):
        text = tok.decode(src_ids[:n])
        got = len(tok.encode(text, add_special_tokens=False))
        if got == target:
            return text, target
        if got > target:
            n -= got - target
            if n < 1:
                return text, got
        else:
            if n >= len(src_ids):
                return text, got
            n += min(target - got, len(src_ids) - n)
    return text, got


def build_content(tok, originals, pairs, seed):
    """Original load, real text. Only `prompt` changes.

    Every request gets a DISTINCT source conversation.  This matters: drawing
    with replacement from a small pool makes many short prompts the opening
    words of the same document, which fabricates shared prefixes.  Measured at
    30% simulated radix-cache reuse with a 400-doc pool vs 0.6% for genuinely
    distinct text -- an artefact that would silently inflate prefix-cache hit
    rates for anyone who turns --disable-radix-cache off.

    Assignment is smallest-fit (targets descending against a length-sorted
    pool), so the scarce long documents are spent only where a long prompt
    actually needs them.
    """
    import bisect

    targets = [len(tok.encode(r.prompt, add_special_tokens=False)) for r in originals]
    need = max(targets)

    # Pool must be big enough to give every request its own document, with
    # enough long ones for the long targets.
    n_long_needed = sum(1 for t in targets if t > 512)
    pool, long_enough, i = [], 0, 0
    while i < len(pairs) and (len(pool) < len(originals) * 2 or long_enough < n_long_needed + 32):
        ids = tok.encode(pairs[i][0], add_special_tokens=False)
        i += 1
        if len(ids) < 4:
            continue
        pool.append(ids)
        if len(ids) >= need:
            long_enough += 1
        if i > 40000:  # hard scan bound
            break

    pool.sort(key=len)
    lens = [len(x) for x in pool]

    # Headroom: fit_to_length corrects BPE round-trip drift by adding tokens, so
    # a document that is only exactly `target` long leaves it nothing to add and
    # the prompt lands short. Ask for slack; fall back if the pool runs dry.
    SLACK = 48
    order = sorted(range(len(originals)), key=lambda k: -targets[k])
    assigned = [None] * len(originals)
    for k in order:
        t = targets[k]
        j = bisect.bisect_left(lens, t + SLACK)
        if j >= len(pool):
            j = bisect.bisect_left(lens, t)
        if j >= len(pool):          # nothing long enough left -> take the longest
            j = len(pool) - 1
        assigned[k] = pool.pop(j)
        lens.pop(j)

    out, exact, drift = [], 0, []
    for orig, target, src in zip(originals, targets, assigned):
        text, got = fit_to_length(tok, src, target)
        r = Request()
        r.__dict__.update(orig.__dict__)   # keep req_time, adapter_dir, model_dir, ids, slos
        r.prompt = text                    # <- the only field that changes
        out.append(r)
        if got == target:
            exact += 1
        else:
            drift.append(got - target)
    return out, exact, drift


def build_full(tok, originals, pairs, seed):
    """Original arrival times + routing, ShareGPT prompt/lengths.

    Filter matches sglang's sample_sharegpt_requests (bench_serving.py) so the
    workload is the same shape everyone else benchmarks with.
    """
    picked, i = [], 0
    while len(picked) < len(originals) and i < len(pairs):
        prompt, completion = pairs[i]
        i += 1
        plen = len(tok.encode(prompt, add_special_tokens=False))
        olen = len(tok.encode(completion, add_special_tokens=False))
        if plen < 4 or olen < 4:
            continue
        if plen > 1024 or plen + olen > 2048:
            continue
        picked.append((prompt, plen, olen))
    if len(picked) < len(originals):
        raise SystemExit(f"only {len(picked)} usable ShareGPT samples, need {len(originals)}")

    out = []
    for orig, (prompt, plen, olen) in zip(originals, picked):
        r = Request()
        r.__dict__.update(orig.__dict__)   # keep req_time, adapter_dir, model_dir, req_id
        r.prompt = prompt
        r.prompt_len = plen
        r.output_len = olen
        out.append(r)
    return out


def prefix_reuse(tok, requests):
    """Fraction of prefill tokens a radix cache would serve from an existing prefix.

    The synthetic trace scores ~97% here (every "Hello "*n is a prefix of every
    longer one), which is why it must not be used with prefix caching enabled.
    Real, distinct text lands near zero.  Reported so a regression in how source
    documents are assigned cannot slip through unnoticed.
    """
    root, total, hit = {}, 0, 0
    for r in sorted(requests, key=lambda r: r.req_time):
        ids = tok.encode(r.prompt, add_special_tokens=False)
        total += len(ids)
        node, matched = root, 0
        for i in ids:
            if i in node:
                node = node[i]
                matched += 1
            else:
                break
        hit += matched
        node = root
        for i in ids:
            node = node.setdefault(i, {})
    return hit / total if total else 0.0


def summarize(tok, requests, label):
    import numpy as np
    real = np.array([len(tok.encode(r.prompt, add_special_tokens=False)) for r in requests])
    field = np.array([r.prompt_len for r in requests])
    olen = np.array([r.output_len for r in requests])
    p = lambda x: f"p50 {int(np.percentile(x,50))} / p90 {int(np.percentile(x,90))} / p99 {int(np.percentile(x,99))} / max {int(x.max())}"
    print(f"  [{label}] n={len(requests)}  span={max(r.req_time for r in requests):.1f}s")
    print(f"     실제 prefill 토큰 : {p(real)}")
    print(f"     prompt_len 필드   : {p(field)}")
    print(f"     output_len        : {p(olen)}")
    print(f"     prefix 재사용률   : {prefix_reuse(tok, requests)*100:.1f}%"
          f"  (합성 트레이스는 ~97%, 실제 텍스트는 ~1%)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--original", default=None, help="source real_trace.pkl")
    ap.add_argument("--sharegpt", default=DEFAULT_SHAREGPT)
    ap.add_argument("--outdir", default=DEFAULT_OUTDIR)
    ap.add_argument("--variant", choices=["content", "full", "both"], default="both")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    original = args.original or os.path.join(
        os.environ.get("PRISM_REPO", "/workspace/prism-exp/prism-research"),
        "benchmark/multi-model/real_trace.pkl",
    )

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER)

    adapter_dirs, originals = load_original(original)
    print(f"원본 트레이스: {len(originals)}건, 어댑터 {len(adapter_dirs)}개  ({original})")
    summarize(tok, originals, "original")

    pairs = load_sharegpt_pairs(args.sharegpt, args.seed)
    print(f"\nShareGPT: 2턴 이상 대화 {len(pairs)}개")

    os.makedirs(args.outdir, exist_ok=True)
    made = []

    if args.variant in ("content", "both"):
        print("\n[content] 원본 부하 유지, 텍스트만 ShareGPT")
        reqs, exact, drift = build_content(tok, originals, pairs, args.seed)
        out = os.path.join(args.outdir, "sharegpt_content.pkl")
        with open(out, "wb") as f:
            pickle.dump([adapter_dirs, reqs], f)
        print(f"     토큰 길이 정확 일치: {exact}/{len(reqs)}"
              + (f"  (drift {min(drift)}..{max(drift)} 토큰)" if drift else ""))
        summarize(tok, reqs, "content")
        made.append(out)

    if args.variant in ("full", "both"):
        print("\n[full] 도착 시각·라우팅만 유지, 길이까지 ShareGPT")
        reqs = build_full(tok, originals, pairs, args.seed)
        out = os.path.join(args.outdir, "sharegpt_full.pkl")
        with open(out, "wb") as f:
            pickle.dump([adapter_dirs, reqs], f)
        summarize(tok, reqs, "full")
        made.append(out)

    print("\n생성됨:")
    for m in made:
        print(f"  {m}  ({os.path.getsize(m)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
