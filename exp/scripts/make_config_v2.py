#!/usr/bin/env python3
"""Emit the placement config for the heterogeneous 6-model set (paper-faithful-v2).

The set is chosen so that KV cell size is NOT monotone in parameter count:

  slot      model                       weights GiB   KV B/token
  model_1   meta-llama/Llama-3.2-1B            2.28        32768
  model_2   Qwen/Qwen2.5-1.5B-Instruct         3.01        28672
  model_3   meta-llama/Llama-3.2-3B            6.00       114688
  model_4   Qwen/Qwen2.5-3B-Instruct           5.84        36864
  model_5   meta-llama/Llama-3.1-8B           15.08       131072
  model_6   Qwen/Qwen2.5-7B-Instruct          14.28        57344

model_3 carries 3.1x model_4's KV per token at the same parameter count, and
model_5 2.3x model_6's.  A model set where cell size tracks size makes KVPR a
relabelling of "sum of weights", and Algorithm 1's objective goes flat --
which is exactly what happened in the v1 study's 3 x Llama-3.1-8B setup.

Default placement is WEIGHT-BALANCED (LPT on weights).  It is what an operator
who knows model sizes but not the workload would write, it gives both GPUs
near-identical shared_kv, and it therefore keeps the starting point neutral:
any KVPR difference that later appears comes from the token-rate numerator,
i.e. from the workload, not from a pre-baked imbalance.

  python make_config_v2.py --num-gpus 2 -o exp/configs/v2/6model_2gpu.json
"""
import argparse
import json

MODELS = [
    ("model_1", "meta-llama/Llama-3.2-1B", 2.279296875, 32768),
    ("model_2", "Qwen/Qwen2.5-1.5B-Instruct", 3.0078125, 28672),
    ("model_3", "meta-llama/Llama-3.2-3B", 6.00, 114688),
    ("model_4", "Qwen/Qwen2.5-3B-Instruct", 5.8359375, 36864),
    ("model_5", "meta-llama/Llama-3.1-8B", 15.080078125, 131072),
    ("model_6", "Qwen/Qwen2.5-7B-Instruct", 14.283203125, 57344),
]


def assign(models, n_gpus, mode):
    if mode == "blocks":
        per = -(-len(models) // n_gpus)
        return {m[0]: min(i // per, n_gpus - 1) for i, m in enumerate(models)}
    if mode == "roundrobin":
        return {m[0]: i % n_gpus for i, m in enumerate(models)}
    load = {g: 0.0 for g in range(n_gpus)}      # balanced: LPT on weights
    out = {}
    for name, _p, w, _c in sorted(models, key=lambda m: -m[2]):
        g = min(load, key=lambda g: (load[g], g))
        out[name] = g
        load[g] += w
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-gpus", type=int, default=2)
    ap.add_argument("--placement", choices=["balanced", "blocks", "roundrobin"],
                    default="balanced")
    ap.add_argument("--pool", type=float, default=20.0,
                    help="max_memory_pool_size GiB per model; a virtual ceiling "
                         "under kvcached, so over-subscribing is intended")
    ap.add_argument("--max-mem", type=float, default=67.28,
                    help="per-GPU budget the scheduler may hand out (for the report only)")
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args()

    if len(MODELS) < a.num_gpus:
        raise SystemExit("every GPU needs at least one on:true model")
    place = assign(MODELS, a.num_gpus, a.placement)
    cfg = [{
        "model_name": n, "model_path": p, "tp_size": 1,
        "init_placements": [{"gpu_ids": [place[n]], "on": True,
                             "max_memory_pool_size": a.pool}],
    } for n, p, _w, _c in MODELS]
    with open(a.out, "w") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")

    print(f"wrote {a.out}  ({len(MODELS)} models, {a.num_gpus} GPUs, {a.placement})")
    worst = 0
    for g in range(a.num_gpus):
        mine = [m for m in MODELS if place[m[0]] == g]
        w = sum(m[2] for m in mine)
        worst = max(worst, len(mine))
        print(f"  GPU {g}: {' '.join(m[0] for m in mine)}")
        print(f"          weights {w:6.2f} GiB | shared_kv {a.max_mem - w:6.2f} GiB")
    print(f"  -> launch with WORKERS >= {worst}")


if __name__ == "__main__":
    main()
