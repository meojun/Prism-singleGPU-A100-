#!/usr/bin/env python3
"""Emit a Prism model-placement config for N GPUs.

The slot -> model mapping is NOT free. benchmark.py routes every run that
passes --model-paths (all of ours) into
`trace.py::generate_e2e_benchmark_reqs`, which hard-codes a per-slot SLO
baseline measured for one specific model. Using a different model in a slot
silently compares against the wrong baseline. Those eight slots are the paper's
§7.2/§7.3 "eight models on two shared GPUs" mix, so they are reproduced here
verbatim.

  python make_config.py --num-gpus 2 -o exp/configs/llama_2gpu_8model.json
  python make_config.py --num-gpus 1 --models 4 --placement balanced -o /tmp/c.json

Three placement modes:
  blocks      contiguous slot blocks (GPU0 = model_1..4, GPU1 = model_5..8 at
              N=2). Deliberately naive -- it is what you would write knowing
              nothing about the trace, and on the shipped real_trace.pkl it
              lands 80% of requests on GPU 0. This is what results/fig7 used
              and the right starting point for a global-placement ablation.
  roundrobin  slot order dealt round-robin. Also naive, but less lopsided
              (60/40 at N=2) because it splits the three heavy 8B slots up.
  balanced    greedy longest-processing-time by trace request count (49/51 at
              N=2). Use when you want placement out of the way as a variable.
"""
import argparse
import json

# slot -> (hf path, weight GiB from model_info.json, requests in real_trace.pkl)
# Model names are the comments in trace.py::generate_e2e_benchmark_reqs.
# Request counts were measured by counting adapter ranks [2,14,3,10,5,19,23,24].
SLOTS = {
    1: ("meta-llama/Llama-3.1-8B", 15.08, 296),
    2: ("meta-llama/Llama-3.2-3B", 6.00, 22),
    3: ("meta-llama/Llama-3.2-1B", 2.28, 22),
    4: ("meta-llama/Llama-3.1-8B", 15.08, 262),
    5: ("meta-llama/Llama-3.1-8B", 15.08, 120),
    6: ("meta-llama/Llama-3.2-1B", 2.28, 19),
    7: ("meta-llama/Llama-3.2-1B", 2.28, 11),
    8: ("meta-llama/Llama-3.2-1B", 2.28, 2),
}


def assign(slots, n_gpus, mode):
    if mode == "blocks":
        # ceil-sized contiguous chunks, so the remainder lands on the low GPUs
        per = -(-len(slots) // n_gpus)
        return {s: min(i // per, n_gpus - 1) for i, s in enumerate(slots)}
    if mode == "roundrobin":
        return {s: (i % n_gpus) for i, s in enumerate(slots)}
    load = {g: 0 for g in range(n_gpus)}
    out = {}
    for s in sorted(slots, key=lambda s: -SLOTS[s][2]):
        g = min(load, key=lambda g: (load[g], g))
        out[s] = g
        load[g] += SLOTS[s][2]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-gpus", type=int, required=True)
    ap.add_argument("--models", type=int, default=8, help="use slots 1..K (max 8)")
    ap.add_argument("--placement", choices=["blocks", "roundrobin", "balanced"], default="blocks")
    ap.add_argument("--pool", type=float, default=20.0,
                    help="max_memory_pool_size GiB per model. In elastic/prism mode this is a "
                         "virtual ceiling, so over-subscribing the GPU is intended.")
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args()

    if not 1 <= a.models <= 8:
        raise SystemExit("--models must be 1..8: trace.py's e2e path defines exactly 8 slots")
    if a.models < a.num_gpus:
        # launch_multi_model_server only starts a GPU scheduler for gpu_ids that
        # appear in the initial placement, so a GPU with no on:true model is dead
        # for the whole run -- nothing can ever be activated on it.
        raise SystemExit(f"--models ({a.models}) must be >= --num-gpus ({a.num_gpus}): "
                         "every GPU needs at least one on:true model")

    slots = list(range(1, a.models + 1))
    placement = assign(slots, a.num_gpus, a.placement)

    cfg = [{
        "model_name": f"model_{s}",
        "model_path": SLOTS[s][0],
        "tp_size": 1,
        "init_placements": [
            {"gpu_ids": [placement[s]], "on": True, "max_memory_pool_size": a.pool}
        ],
    } for s in slots]

    with open(a.out, "w") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")

    total_reqs = sum(SLOTS[s][2] for s in slots)
    print(f"wrote {a.out}  ({a.models} models, {a.num_gpus} GPUs, {a.placement})")
    for g in range(a.num_gpus):
        mine = [s for s in slots if placement[s] == g]
        w = sum(SLOTS[s][1] for s in mine)
        r = sum(SLOTS[s][2] for s in mine)
        share = 100 * r / total_reqs if total_reqs else 0
        print(f"  GPU {g}: {' '.join('model_%d' % s for s in mine)}")
        print(f"          weights {w:5.1f} GiB | trace requests {r:4d} ({share:.0f}%)")
    print(f"  -> launch with WORKERS >= {max(sum(1 for s in slots if placement[s] == g) for g in range(a.num_gpus))}")


if __name__ == "__main__":
    main()
