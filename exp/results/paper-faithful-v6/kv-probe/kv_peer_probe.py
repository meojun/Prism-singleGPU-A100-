import torch, traceback
from kvcached import ops as kvops
from kvcached.slab_allocator import KVCacheManager

GPU_SRC, GPU_DST = 0, 1
print("peer:", torch.cuda.can_device_access_peer(GPU_SRC, GPU_DST))

torch.cuda.set_device(GPU_SRC)
kvops.init_kvcached(gpu_id=GPU_SRC, virtual_mem_size_gb=2, reserve_virtual_mem=False)
k, v = kvops.sgl_alloc_kv_cache(num_tokens=4096, head_num=8, head_dim=128,
                                dtype=torch.float16, device=f"cuda:{GPU_SRC}",
                                num_layers=2)
print("k[0]", k[0].shape, k[0].dtype, k[0].device)

mgr = KVCacheManager(4096, 1, 8 * 128 * 2, num_layers=2)
slots = mgr.alloc(64)
print("slots", slots[:4], "... n =", len(slots))
idx = torch.tensor(slots, device=f"cuda:{GPU_SRC}", dtype=torch.long)
k[0][idx] = 1.234
torch.cuda.synchronize()

print("--- same-GPU gather")
g = k[0][idx].clone()
torch.cuda.synchronize()
print("   ok", tuple(g.shape), float(g.flatten()[0]))

print("--- peer copy of the GATHERED tensor (normal allocator)")
try:
    dst = torch.empty(tuple(g.shape), dtype=g.dtype, device=f"cuda:{GPU_DST}")
    dst.copy_(g); torch.cuda.synchronize()
    print("   OK", float(dst.flatten()[0]))
except Exception:
    print("   FAILED"); traceback.print_exc()

print("--- DIRECT peer read of the VMM-backed FTensor view")
try:
    direct = torch.empty((len(slots), 8, 128), dtype=torch.float16, device=f"cuda:{GPU_DST}")
    direct.copy_(k[0][idx]); torch.cuda.synchronize()
    print("   OK", float(direct.flatten()[0]))
except Exception:
    print("   FAILED"); traceback.print_exc()
