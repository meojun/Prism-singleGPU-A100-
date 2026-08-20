"""Can a capsule cross a process boundary at all?

The hang appeared when the first non-empty capsule list was handed over.  Two
very different causes fit that: sharing the weight handshake (a channel
problem, which a dedicated queue would fix), or moving capsules across
processes at all (which no channel change would fix).

This separates them without a server.  A child process receives capsules over a
torch.multiprocessing.Queue and reads their bytes, while the parent does what
the source engine does immediately afterwards -- drops its references and frees
its pool.  If that alone hangs or corrupts, the channel was never the problem.
"""
import sys, time, traceback
import torch
import torch.multiprocessing as mp

sys.path.insert(0, "/workspace/prism-exp/patches/paper_faithful_v6")

HEADS, DIM, LAYERS, NTOK = 8, 128, 4, 4096
DTYPE = torch.float16


def child(q, result_q):
    try:
        import kv_migration_v6 as kvm  # noqa
        got = q.get(timeout=60)
        torch.cuda.set_device(1)
        # Read every byte, the way a scatter would.
        sums = []
        for cap in got:
            ks = sum(float(t.float().sum().item()) for t in cap.k)
            vs = sum(float(t.float().sum().item()) for t in cap.v)
            sums.append((cap.rid, cap.num_tokens, ks, vs))
        torch.cuda.synchronize(1)
        result_q.put(("ok", sums))
    except Exception:
        result_q.put(("err", traceback.format_exc()))


def main():
    import kv_migration_v6 as kvm
    from kvcached import ops as kvops
    from kvcached.slab_allocator import KVCacheManager

    torch.cuda.set_device(0)
    kvops.init_kvcached(gpu_id=0, virtual_mem_size_gb=2, reserve_virtual_mem=False)
    k, v = kvops.sgl_alloc_kv_cache(NTOK, HEADS, DIM, DTYPE, "cuda:0", LAYERS)
    mgr = KVCacheManager(NTOK, 1, HEADS * DIM * 2, num_layers=LAYERS)

    caps = []
    for i in range(4):
        slots = mgr.alloc(220)
        idx = torch.as_tensor(slots, dtype=torch.long, device="cuda:0")
        for L in range(LAYERS):
            k[L][idx] = float(i + 1) + L          # distinct per capsule AND layer
            v[L][idx] = 100.0 + float(i + 1) + L  # never cancels against k
        torch.cuda.synchronize(0)
        kk, vv = kvm.gather_request_kv(k, v, slots, "cuda:0")
        caps.append(kvm.RequestKVCapsule(f"r{i}", "m", [1], [2], {}, 0.0, 0.6,
                                         kk, vv, 0))
    expect = {c.rid: (sum(float(t.float().sum().item()) for t in c.k),
                      sum(float(t.float().sum().item()) for t in c.v))
              for c in caps}
    assert all(a != 0 and b != 0 and a != b for a, b in expect.values()), \
        "expected values must be non-zero and distinguishable or a match proves nothing"
    print("parent: expected k/v sums", {r: (round(a, 1), round(b, 1))
                                        for r, (a, b) in expect.items()})
    print(f"parent: {len(caps)} capsules, "
          f"{sum(c.nbytes for c in caps)/2**20:.1f} MiB")

    ctx = mp.get_context("spawn")
    q, rq = ctx.Queue(), ctx.Queue()
    p = ctx.Process(target=child, args=(q, rq))
    p.start()

    t0 = time.time()
    q.put(caps)
    print(f"parent: put returned in {time.time()-t0:.3f}s")

    # Exactly what the source engine does next: drop refs, tear the pool down.
    caps = []
    mgr.page_allocator._stop_prealloc_thread()
    del mgr, k, v
    import gc; gc.collect()
    kvops.free_kv_cached_tensors()
    torch.cuda.empty_cache()
    print("parent: references dropped and pool released")

    try:
        status, payload = rq.get(timeout=90)
    except Exception:
        print("RESULT: child never answered -- HANG REPRODUCED without any handshake")
        p.terminate(); return 2
    p.join(timeout=30)
    if status == "err":
        print("RESULT: child raised\n" + payload); return 3
    ok = all(abs(expect[rid][0] - ks) < 1.0 and abs(expect[rid][1] - vs) < 1.0
             for rid, _, ks, vs in payload)
    print(f"child read back: {[(r, n, round(ks,1), round(vs,1)) for r, n, ks, vs in payload]}")
    print(f"RESULT: crossing succeeded, bytes {'MATCH' if ok else 'DIFFER'}")
    return 0 if ok else 4


if __name__ == "__main__":
    raise SystemExit(main())
