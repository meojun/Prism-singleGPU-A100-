#!/usr/bin/env python3
"""Measure model migration between one fixed GPU pair, three ways.

    prototype-source-first  the released prototype's ordering: free the source
                            copy, then load the target from host memory.  The
                            model is resident nowhere in between, so the whole
                            transfer is service downtime.
    v3-target-first         exp/paper-faithful-v3: load the target from host
                            memory first and only then retire the source, so
                            the source keeps serving throughout.  Downtime
                            collapses; the transfer still crosses PCIe twice.
    v4-p2p-target-first     target-first as well, but the target is filled from
                            the source GPU's *resident* weights straight over
                            the GPU-GPU link.  The host is not involved at all.

Target-first ordering is what makes the third one possible: the source copy is
still alive and holding the weights when the target needs them.

Reports per migration: latency, service downtime, bytes moved, path, bandwidth.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "patches/paper_faithful_v4"))
from parallel_loading_v4 import (  # noqa: E402
    StreamPool, copy_model_to_gpu_v4, enable_peer_access, register_host_memory,
)


def nvlink_counters(gpu):
    """Bytes carried by this GPU's NVLink lanes, from the driver's counters.

    A bandwidth above the PCIe ceiling already implies NVLink, but the report
    should not have to infer the path.  These counters say it directly, so a
    run that quietly fell back to a host bounce buffer cannot be written up as
    an NVLink migration.
    """
    try:
        out = subprocess.run(["nvidia-smi", "nvlink", "-gt", "d", "-i", str(gpu)],
                             capture_output=True, text=True, timeout=20).stdout
    except Exception:
        return None
    tx = sum(int(v) for v in re.findall(r"Data Tx:\s+(\d+) KiB", out))
    rx = sum(int(v) for v in re.findall(r"Data Rx:\s+(\d+) KiB", out))
    if not tx and not rx:
        return None
    return {"tx_bytes": tx * 1024, "rx_bytes": rx * 1024}


def counter_delta(before, after):
    if not before or not after:
        return None
    return {k: after[k] - before[k] for k in before}


def load_cpu_state(model_id, hf_home):
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    path = snapshot_download(model_id, allow_patterns=["*.safetensors", "*.json"],
                             cache_dir=os.path.join(hf_home, "hub"))
    state = {}
    for shard in sorted(Path(path).glob("*.safetensors")):
        for key, tensor in load_file(str(shard)).items():
            state[key] = tensor.share_memory_()
    return state


def empty_on(state, gpu):
    return {k: torch.empty_like(v, device=f"cuda:{gpu}") for k, v in state.items()}


def nbytes_of(state):
    return sum(t.numel() * t.element_size() for t in state.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--gpu-ids", default="0,1")
    ap.add_argument("--source-gpu", type=int, default=0)
    ap.add_argument("--target-gpu", type=int, default=1)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hf-home", default=os.environ.get("HF_HOME", "/workspace/.hf_home"))
    args = ap.parse_args()

    gpu_ids = [int(x) for x in args.gpu_ids.split(",")]
    peer = enable_peer_access(gpu_ids)
    print(f"peer access matrix: {peer}", flush=True)
    if not peer.get(f"{args.source_gpu}->{args.target_gpu}"):
        print("WARNING: no direct peer access; a cross-device copy will be "
              "staged through the host and must NOT be reported as NVLink.",
              flush=True)

    pool = StreamPool(gpu_ids, per_gpu=3)
    records = []

    # Every model is read and page-locked up front and then kept alive for the
    # whole run.  Freeing one model's shared mapping and reading the next lets
    # the OS hand back the same virtual addresses, while CUDA still holds them
    # registered -- the next allocation then fails with "part or all of the
    # requested memory range is already mapped".  The server never hits this:
    # its ModelService registers once at startup and never releases.
    cpu_states, registrations = {}, {}
    for model_id in args.models:
        cpu_states[model_id] = load_cpu_state(model_id, args.hf_home)
        # V3 and V4 both run against page-locked host pages, so the only
        # difference between them is the transfer path, not the host mapping.
        registrations[model_id] = register_host_memory(cpu_states[model_id])
        print(f"read + page-locked {model_id}: "
              f"{nbytes_of(cpu_states[model_id])/2**30:.2f} GiB "
              f"{registrations[model_id]}", flush=True)

    for model_id in args.models:
        cpu_state = cpu_states[model_id]
        payload = nbytes_of(cpu_state)
        reg = registrations[model_id]
        print(f"\n=== {model_id}: {payload/2**30:.2f} GiB", flush=True)

        for arm in ("prototype-source-first", "v3-target-first", "v4-p2p-target-first"):
            for rep in range(1, args.reps + 1):
                # Source copy resident and serving.
                source = empty_on(cpu_state, args.source_gpu)
                threads = ThreadPoolExecutor(max_workers=8)
                copy_model_to_gpu_v4(cpu_state, source, args.source_gpu, threads,
                                     gpu_ids, pool, policy="paper",
                                     host_registered=True, tag="prime")
                torch.cuda.synchronize(args.source_gpu)

                nvl_src_before = nvlink_counters(args.source_gpu)
                nvl_dst_before = nvlink_counters(args.target_gpu)
                t_start = time.perf_counter()
                if arm == "prototype-source-first":
                    # Source is retired first: nothing can serve until the
                    # target is up, so downtime is the whole transfer.
                    del source
                    torch.cuda.empty_cache()
                    t_source_gone = time.perf_counter()
                    target = empty_on(cpu_state, args.target_gpu)
                    rec = copy_model_to_gpu_v4(cpu_state, target, args.target_gpu,
                                               threads, gpu_ids, pool, policy="paper",
                                               host_registered=True, tag=arm)
                    t_ready = time.perf_counter()
                    downtime = t_ready - t_source_gone
                elif arm == "v3-target-first":
                    target = empty_on(cpu_state, args.target_gpu)
                    rec = copy_model_to_gpu_v4(cpu_state, target, args.target_gpu,
                                               threads, gpu_ids, pool, policy="paper",
                                               host_registered=True, tag=arm)
                    t_ready = time.perf_counter()
                    del source
                    torch.cuda.empty_cache()
                    downtime = 0.0
                else:
                    target = empty_on(cpu_state, args.target_gpu)
                    rec = copy_model_to_gpu_v4(None, target, args.target_gpu,
                                               threads, gpu_ids, pool, policy="paper",
                                               source_state_dict=source, tag=arm)
                    t_ready = time.perf_counter()
                    del source
                    torch.cuda.empty_cache()
                    downtime = 0.0
                t_end = time.perf_counter()
                threads.shutdown(wait=True)
                nvlink_src = counter_delta(nvl_src_before, nvlink_counters(args.source_gpu))
                nvlink_dst = counter_delta(nvl_dst_before, nvlink_counters(args.target_gpu))

                row = {
                    "arm": arm, "rep": rep, "model": model_id,
                    "source_gpu": args.source_gpu, "target_gpu": args.target_gpu,
                    "peer_access": peer.get(f"{args.source_gpu}->{args.target_gpu}"),
                    "weight_bytes": payload,
                    "kv_bytes": 0,
                    "total_bytes": rec["payload_bytes"],
                    "migration_latency_s": t_ready - t_start,
                    "service_downtime_s": downtime,
                    "migration_total_s": t_end - t_start,
                    "transfer_path": rec["transfer_path"],
                    "effective_gbps": rec["payload_bytes"] / (t_ready - t_start) / 1e9,
                    "host_registration": reg,
                    "nvlink_delta_source_gpu": nvlink_src,
                    "nvlink_delta_target_gpu": nvlink_dst,
                    "detail": rec,
                }
                records.append(row)
                print(f"  {arm:24s} rep{rep}: latency={row['migration_latency_s']:.3f}s "
                      f"downtime={downtime:.3f}s {row['effective_gbps']:.1f} GB/s "
                      f"path={row['transfer_path']} "
                      f"nvlink_rx={(nvlink_dst or {}).get('rx_bytes', 0)/2**30:.2f}GiB", flush=True)

                del target
                torch.cuda.empty_cache()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"records": records, "peer_access": peer}, fh, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
