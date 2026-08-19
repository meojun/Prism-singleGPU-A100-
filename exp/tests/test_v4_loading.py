#!/usr/bin/env python3
"""Unit tests for the v4 weight-transfer path.

Correctness first: a faster loader that corrupts a weight is worthless, so
every policy is checked to reproduce the source tensors bit-for-bit.
"""
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "patches/paper_faithful_v4"))
from parallel_loading_v4 import (  # noqa: E402
    StreamPool, _broker_for, _slice_bounds, copy_model_to_gpu_v4,
    enable_peer_access, register_host_memory,
)

PASS, FAIL = [], []


def check(name, cond):
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")


def test_slice_bounds():
    print("_slice_bounds")
    for size, parts in [(8, 2), (7, 2), (1, 4), (100, 7), (0, 3), (5, 5), (3, 8)]:
        bounds = _slice_bounds(size, parts)
        covered = sum(e - s for s, e in bounds)
        check(f"size={size} parts={parts} covers exactly", covered == size)
        check(f"size={size} parts={parts} contiguous", 
              all(bounds[i][1] == bounds[i + 1][0] for i in range(len(bounds) - 1)))
        check(f"size={size} parts={parts} no empty slice",
              size == 0 or all(e > s for s, e in bounds))
    # The prototype asserts divisibility; the point of this helper is that an
    # odd leading dimension is fine.
    check("odd rows do not raise", _slice_bounds(291, 2) == [(0, 146), (146, 291)])


def test_broker_assignment():
    print("_broker_for")
    gpus = [0, 1]
    check("direct policy always targets the target",
          all(_broker_for(i, t, gpus, "direct") == t for i in range(4) for t in gpus))
    # The released prototype's formula: (slice + target + 1) % num_gpus.
    check("paper policy matches the prototype formula",
          [_broker_for(i, 0, gpus, "paper") for i in range(4)] == [1, 0, 1, 0])
    check("paper policy is symmetric in the target",
          [_broker_for(i, 1, gpus, "paper") for i in range(4)] == [0, 1, 0, 1])
    check("single gpu degenerates to direct",
          _broker_for(0, 0, [0], "paper") == 0)


def test_transfers():
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        print("transfers: SKIPPED (needs 2 visible GPUs)")
        return
    print("transfers")
    gpus = [0, 1]
    peer = enable_peer_access(gpus)
    check("peer access is available both ways",
          peer.get("0->1") and peer.get("1->0"))

    torch.manual_seed(0)
    src = {f"w{i}": torch.randn(64 + i, 128).share_memory_() for i in range(5)}
    src["odd"] = torch.randn(291, 32).share_memory_()
    expected_bytes = sum(t.numel() * t.element_size() for t in src.values())
    pool = StreamPool(gpus, per_gpu=3)

    for policy, split in (("direct", 1), ("paper", 2), ("v4", 2)):
        dst = {k: torch.empty_like(v, device="cuda:0") for k, v in src.items()}
        with ThreadPoolExecutor(max_workers=8) as threads:
            rec = copy_model_to_gpu_v4(src, dst, 0, threads, gpus, pool,
                                       policy=policy, split=split, tag=f"test/{policy}")
        exact = all(torch.equal(dst[k].cpu(), src[k]) for k in src)
        check(f"{policy}: every tensor arrives bit-exact", exact)
        check(f"{policy}: payload accounting matches the tensors",
              rec["payload_bytes"] == expected_bytes)
        check(f"{policy}: reported path is consistent with the byte split",
              (rec["bytes_h2d_helper"] == 0) == (rec["transfer_path"] == "host-to-device-only"))

    # GPU -> GPU: the migration path.
    resident = {k: v.to("cuda:0") for k, v in src.items()}
    dst = {k: torch.empty_like(v, device="cuda:1") for k, v in src.items()}
    with ThreadPoolExecutor(max_workers=8) as threads:
        rec = copy_model_to_gpu_v4(None, dst, 1, threads, gpus, pool,
                                   policy="paper", source_state_dict=resident, tag="test/p2p")
    check("p2p: every tensor arrives bit-exact",
          all(torch.equal(dst[k].cpu(), src[k]) for k in src))
    check("p2p: reported as a gpu-to-gpu transfer", rec["transfer_path"] == "gpu-to-gpu-p2p")
    check("p2p: no host-to-device bytes are counted",
          rec["bytes_h2d_direct"] == 0 and rec["bytes_h2d_helper"] == 0)

    summary = register_host_memory(src)
    check("host registration locks every tensor", summary["failed"] == 0)
    dst = {k: torch.empty_like(v, device="cuda:0") for k, v in src.items()}
    with ThreadPoolExecutor(max_workers=8) as threads:
        copy_model_to_gpu_v4(src, dst, 0, threads, gpus, pool, policy="paper",
                             host_registered=True, tag="test/locked")
    check("page-locked source still arrives bit-exact",
          all(torch.equal(dst[k].cpu(), src[k]) for k in src))


def main():
    test_slice_bounds()
    test_broker_assignment()
    test_transfers()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        for name in FAIL:
            print(f"  FAILED: {name}")
        raise SystemExit(1)
    print("ALL V4 LOADING TESTS PASSED")


if __name__ == "__main__":
    main()
