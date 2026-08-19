#!/usr/bin/env python3
"""Measure the three weight-loading arms on one fixed GPU pair.

    sequential              one model at a time, every byte host->device to the
                            target GPU.  No helper GPU, no GPU-GPU hop.
    v3-parallel-activation  what exp/paper-faithful-v3 actually ran: several
                            models activated concurrently, each one internally
                            using the released prototype's 2-way broker split,
                            out of pageable shared host memory.
    v4-parallel-loading     the same broker split, but the shared host pages are
                            page-locked in place first, so the copies become
                            real DMA instead of driver bounce-buffer staging.
    v4-pipelined-helper     additionally overlaps the NVLink hop with the PCIe
                            leg across sub-chunks.  Reported as an ablation: on
                            this box the per-sub-chunk event cost exceeds what
                            the overlap returns.

The last two are ordered after the first two on purpose: page-locking is a
property of the mapping, so once it is done it cannot be undone for later arms.

The payload is the model's real weight tensors, read from its safetensors
shards -- not a synthetic buffer -- held in shared (not pinned) CPU memory,
which is what ``share_memory()`` gives the prototype's ModelService.

Writes one JSON record per (arm, repetition) plus per-model rows.
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "patches/paper_faithful_v4"))
from parallel_loading_v4 import (  # noqa: E402
    StreamPool, copy_model_to_gpu_v4, enable_peer_access, register_host_memory,
)


def load_cpu_state(model_id, hf_home):
    """Read a model's weight tensors into shared CPU memory."""
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    path = snapshot_download(model_id, allow_patterns=["*.safetensors", "*.json"],
                             cache_dir=os.path.join(hf_home, "hub"))
    shards = sorted(Path(path).glob("*.safetensors"))
    if not shards:
        raise RuntimeError(f"no safetensors under {path}")
    state = {}
    for shard in shards:
        for key, tensor in load_file(str(shard)).items():
            state[key] = tensor.share_memory_()
    return state


def empty_gpu_state(cpu_state, gpu_id):
    return {k: torch.empty_like(v, device=f"cuda:{gpu_id}") for k, v in cpu_state.items()}


def state_bytes(state):
    return sum(t.numel() * t.element_size() for t in state.values())


# name -> (policy, split, concurrent_models, page_locked)
# Every v4 arm keeps the prototype's byte split and broker assignment, so each
# step adds exactly one mechanism and the delta is attributable.
ARMS = {
    "sequential":             ("direct", 1, False, False),
    "v3-parallel-activation": ("paper",  0, True,  False),
    "v4-parallel-loading":    ("paper",  0, True,  True),
    "v4-pipelined-helper":    ("v4",     0, True,  True),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--gpu-ids", default="0,1")
    ap.add_argument("--target-gpu", type=int, default=0)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--pipeline-depth", type=int, default=4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hf-home", default=os.environ.get("HF_HOME", "/workspace/.hf_home"))
    args = ap.parse_args()

    gpu_ids = [int(x) for x in args.gpu_ids.split(",")]
    assert args.target_gpu in gpu_ids
    peer = enable_peer_access(gpu_ids)
    print(f"peer access matrix: {peer}", flush=True)

    print("reading weights into shared CPU memory ...", flush=True)
    cpu_states, sizes = {}, {}
    for model_id in args.models:
        cpu_states[model_id] = load_cpu_state(model_id, args.hf_home)
        sizes[model_id] = state_bytes(cpu_states[model_id])
        print(f"  {model_id}: {sizes[model_id]/2**30:.2f} GiB", flush=True)

    pool = StreamPool(gpu_ids, per_gpu=3)
    records = []

    registered = None
    for arm, (policy, split, concurrent, page_locked) in ARMS.items():
        split = split or min(4, len(gpu_ids))
        if page_locked and registered is None:
            registered = {}
            for model_id in args.models:
                registered[model_id] = register_host_memory(cpu_states[model_id])
            print(f"page-locked host mappings: {registered}", flush=True)
        for rep in range(1, args.reps + 1):
            gpu_states = {m: empty_gpu_state(cpu_states[m], args.target_gpu)
                          for m in args.models}
            torch.cuda.synchronize(args.target_gpu)
            per_model = {}

            def one(model_id):
                threads = ThreadPoolExecutor(max_workers=max(4, split * 2))
                try:
                    t0 = time.perf_counter()
                    rec = copy_model_to_gpu_v4(
                        cpu_states[model_id], gpu_states[model_id], args.target_gpu,
                        threads, gpu_ids, pool, policy=policy, split=split,
                        pipeline_depth=args.pipeline_depth,
                        host_registered=bool(page_locked),
                        tag=f"{arm}/rep{rep}/{model_id}",
                    )
                    rec["model"] = model_id
                    rec["wall_seconds"] = time.perf_counter() - t0
                    per_model[model_id] = rec
                finally:
                    threads.shutdown(wait=True)

            t_start = time.perf_counter()
            if concurrent:
                with ThreadPoolExecutor(max_workers=len(args.models)) as outer:
                    list(outer.map(one, args.models))
            else:
                for model_id in args.models:
                    one(model_id)
            total = time.perf_counter() - t_start

            payload = sum(r["payload_bytes"] for r in per_model.values())
            row = {
                "arm": arm, "rep": rep, "policy": policy, "split": split,
                "pipeline_depth": args.pipeline_depth if policy == "v4" else None,
                "concurrent_models": concurrent,
                "host_page_locked": page_locked,
                "target_gpu": args.target_gpu, "gpu_ids": gpu_ids,
                "peer_access": peer,
                "num_models": len(args.models),
                "total_loading_seconds": total,
                "payload_bytes": payload,
                "aggregate_gbps": payload / total / 1e9,
                "bytes_h2d_direct": sum(r["bytes_h2d_direct"] for r in per_model.values()),
                "bytes_h2d_helper": sum(r["bytes_h2d_helper"] for r in per_model.values()),
                "bytes_p2p": sum(r["bytes_p2p"] for r in per_model.values()),
                "transfer_path": next(iter(per_model.values()))["transfer_path"],
                "per_model": per_model,
            }
            records.append(row)
            print(f"{arm} rep{rep}: {total:.3f}s  {row['aggregate_gbps']:.2f} GB/s  "
                  f"path={row['transfer_path']}", flush=True)

            del gpu_states
            torch.cuda.empty_cache()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"records": records, "model_bytes": sizes, "peer_access": peer,
                   "host_registration": registered}, fh, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
