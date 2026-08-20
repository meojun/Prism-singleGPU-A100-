#!/usr/bin/env python3
"""Tests for KV-cache migration (paper §5.3).

Runs against the real kvcached allocator on two GPUs, not a mock, because the
properties worth proving are properties of that allocator: that a request's
slots are scattered, that a gather off the source is legal, and that the
target's own numbering can differ without corrupting anything.

Skips rather than fails when fewer than two GPUs are visible -- the transfer
half has nothing to say on one GPU.
"""
import os
import sys
import traceback

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "patches", "paper_faithful_v6"))
import kv_migration_v6 as kvm  # noqa: E402

from kvcached import ops as kvops  # noqa: E402
from kvcached.slab_allocator import KVCacheManager  # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  -- {detail}" if detail and not cond else ""))


HEADS, DIM, LAYERS, NTOK = 8, 128, 4, 4096
DTYPE = torch.float16
CELL = HEADS * DIM * DTYPE.itemsize


def make_pool(gpu):
    torch.cuda.set_device(gpu)
    kvops.init_kvcached(gpu_id=gpu, virtual_mem_size_gb=2, reserve_virtual_mem=False)
    k, v = kvops.sgl_alloc_kv_cache(NTOK, HEADS, DIM, DTYPE, f"cuda:{gpu}", LAYERS)
    mgr = KVCacheManager(NTOK, 1, CELL, num_layers=LAYERS)
    return k, v, mgr


def write_pattern(k, v, slots, gpu, seed):
    """Give every (layer, token) a distinct value so a mis-ordered scatter shows."""
    idx = torch.as_tensor(slots, dtype=torch.long, device=f"cuda:{gpu}")
    for L in range(LAYERS):
        base = torch.arange(len(slots), device=f"cuda:{gpu}", dtype=DTYPE).view(-1, 1, 1)
        k[L][idx] = base + (L * 1000) + seed
        v[L][idx] = -(base + (L * 1000) + seed)
    torch.cuda.synchronize(gpu)


def main():
    n = torch.cuda.device_count()
    print(f"visible GPUs: {n}")
    if n < 2:
        print("SKIP: KV migration tests need 2 GPUs")
        return 0

    SRC, DST = 0, 1
    print(f"peer access {SRC}->{DST}: {torch.cuda.can_device_access_peer(SRC, DST)}")

    print("\n-- source pool")
    k_src, v_src, mgr_src = make_pool(SRC)

    # Two requests interleaved, so neither owns a contiguous slot range: this is
    # the layout the real scheduler produces and the reason a gather is needed.
    a_slots = mgr_src.alloc(48)
    b_slots = mgr_src.alloc(32)
    check("source slots are allocated", len(a_slots) == 48 and len(b_slots) == 32)
    check("the two requests do not overlap", not (set(a_slots) & set(b_slots)))

    write_pattern(k_src, v_src, a_slots, SRC, seed=7)
    write_pattern(k_src, v_src, b_slots, SRC, seed=99)

    print("\n-- gather off the source")
    ka, va = kvm.gather_request_kv(k_src, v_src, a_slots, f"cuda:{SRC}")
    check("gather returns one tensor per layer", len(ka) == LAYERS and len(va) == LAYERS)
    check("gathered shape is [tokens, heads, dim]",
          tuple(ka[0].shape) == (48, HEADS, DIM), str(tuple(ka[0].shape)))
    check("gathered tensor lives on the source GPU", ka[0].device.index == SRC)
    check("gather owns its storage (source may now be freed)",
          ka[0].data_ptr() != k_src[0].data_ptr())

    cap = kvm.RequestKVCapsule(
        rid="req-a", model_name="model_1", origin_input_ids=[1, 2, 3],
        output_ids=[41, 42], sampling_params={"temperature": 0.0},
        arrival_time=123.0, slo=0.6, k=ka, v=va, source_gpu=SRC)
    check("capsule counts tokens", cap.num_tokens == 48)
    check("capsule reports bytes", cap.nbytes == 48 * HEADS * DIM * 2 * LAYERS * 2,
          str(cap.nbytes))
    check("capsule carries output_ids (the existing serialiser drops them)",
          cap.output_ids == [41, 42])

    expect_k = [t.clone().cpu() for t in ka]
    expect_v = [t.clone().cpu() for t in va]

    # Everything that has to be read off the source must be read NOW.
    # kvcached keeps one global allocator per process (allocator.cpp:14
    # g_allocator_), so initialising a pool on the target re-initialises that
    # allocator and destroys the source's FTensors -- reading k_src afterwards
    # is an illegal access, which is exactly how this was found.
    #
    # It is a constraint on the design, not just on this test: a KV migration
    # can never happen inside one process, which is why the wiring routes
    # capsules through the model service rather than copying pool to pool.
    ka2, va2 = kvm.gather_request_kv(k_src, v_src, b_slots, f"cuda:{SRC}")
    ka3, va3 = kvm.gather_request_kv(k_src, v_src, a_slots, f"cuda:{SRC}")

    print("\n-- transfer to the target")
    cap, path = kvm.transfer_capsule(cap, DST)
    check("transfer reports a path", path in ("gpu-to-gpu-p2p", "via-host"), path)
    check("capsule now lives on the target", cap.k[0].device.index == DST)
    check("bytes survive the hop bit-exact",
          all(torch.equal(cap.k[i].cpu(), expect_k[i]) for i in range(LAYERS)) and
          all(torch.equal(cap.v[i].cpu(), expect_v[i]) for i in range(LAYERS)))

    print("\n-- scatter into freshly allocated slots")
    # The transfer is proven above and the scatter is proven here, separately,
    # because a single process cannot hold two kvcached pools: allocator.cpp:14
    # keeps one global FTensorAllocator, so bringing up a pool on a second GPU
    # re-initialises it and invalidates the first.  (Verified: after that
    # re-init the new pool reports capacity but alloc() returns None.)
    #
    # That is a property of the library, not of this test, and it is why the
    # wiring routes capsules through the model service instead of copying pool
    # to pool -- in production the source and target pools live in different
    # engine processes and this never arises.
    #
    # So: bring the capsule back and scatter it into slots the source allocator
    # hands out fresh.  What that proves is what matters -- the target's slot
    # numbering is unrelated to the one the KV was read from, and the bytes
    # still land correctly.
    cap, back = kvm.transfer_capsule(cap, SRC)
    check("capsule can be brought back for the scatter half", back != "empty")

    mgr_src.alloc(21)                      # skew the numbering
    d_slots = mgr_src.alloc(48)
    check("fresh slots were allocated", d_slots is not None)
    if d_slots is None:
        print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
        return 1
    check("fresh slot ids differ from the ones the KV was read from",
          list(d_slots) != list(a_slots), f"{d_slots[:3]} vs {a_slots[:3]}")

    kvm.scatter_request_kv(k_src, v_src, d_slots, cap, f"cuda:{SRC}")
    torch.cuda.synchronize(SRC)

    idx_d = torch.as_tensor(d_slots, dtype=torch.long, device=f"cuda:{SRC}")
    ok = all(torch.equal(k_src[L][idx_d].cpu(), expect_k[L]) for L in range(LAYERS)) and \
         all(torch.equal(v_src[L][idx_d].cpu(), expect_v[L]) for L in range(LAYERS))
    check("round-trip is bit-exact under a different slot numbering", ok)

    print("\n-- token order is what preserves correctness")
    got = k_src[0][idx_d][:, 0, 0].cpu()
    check("token order is preserved end to end",
          torch.equal(got, torch.arange(48, dtype=DTYPE) + 7), str(got[:4]))

    print("\n-- mismatched slot count is refused, not silently truncated")
    try:
        kvm.scatter_request_kv(k_src, v_src, d_slots[:10], cap, f"cuda:{SRC}")
        check("scatter rejects a short slot vector", False, "no exception")
    except ValueError:
        check("scatter rejects a short slot vector", True)

    print("\n-- accounting")
    cap2 = kvm.RequestKVCapsule("req-b", "model_1", [9], [], {}, 1.0, 0.6,
                                ka2, va2, SRC)
    moved, skipped, rec = kvm.migrate_request_kv([cap2], DST, tag="unit")
    check("record reports a non-zero kv_bytes", rec["kv_bytes"] > 0, str(rec))
    check("record counts requests moved", rec["requests_moved"] == 1)
    check("record names the transfer path", rec["transfer_path"] != "none")
    check("record reports the cap so truncation cannot pass as completeness",
          "max_tokens_cap" in rec and "requests_skipped_over_cap" in rec)

    print("\n-- the cap is enforced and reported")
    old = kvm.MAX_TOKENS
    kvm.MAX_TOKENS = 10
    try:
        big = kvm.RequestKVCapsule("req-c", "model_1", [1], [], {}, 1.0, 0.6,
                                   ka3, va3, SRC)
        moved3, skipped3, rec3 = kvm.migrate_request_kv([big], DST, tag="cap")
        check("over-cap request is skipped, not truncated",
              len(moved3) == 0 and len(skipped3) == 1)
        check("skipped tokens are reported", rec3["tokens_skipped"] == 48, str(rec3))
    finally:
        kvm.MAX_TOKENS = old

    print("\n-- empty request is a no-op, not a crash")
    empty = kvm.RequestKVCapsule("req-d", "model_1", [], [], {}, 1.0, 0.6, [], [], SRC)
    _, p = kvm.transfer_capsule(empty, DST)
    check("empty capsule transfers as a no-op", p == "empty" and empty.num_tokens == 0)

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("FAILED: " + ", ".join(FAIL))
        return 1
    print("ALL KV MIGRATION TESTS PASSED")
    return 0


if __name__ == "__main__":
    try:
        _code = main()
    except Exception:
        traceback.print_exc()
        _code = 1
    # kvcached tears its FTensors down at interpreter exit, after the CUDA
    # runtime has begun unloading, and cuMemRelease then fails with driver
    # error 4 (cudaErrorCudartUnloading) and aborts the process.  That happens
    # after every assertion has already been decided, but it replaces our exit
    # code with a crash, which would read as a test failure to anything
    # checking $?.  Flush what we printed and leave without running the
    # destructors.  This is a shutdown-order bug in the library, not a result.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(_code)
