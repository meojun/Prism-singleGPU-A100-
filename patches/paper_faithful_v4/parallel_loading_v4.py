"""Instrumented parallel model weight loading (paper §5.3).

The released prototype already contains the paper's mechanism, in
``model_sevice.multi_thread_copy_model_to_gpu``: every weight tensor is cut
into ``max_threads`` slices and slice ``b`` is driven by *broker* GPU
``(b + target + 1) % num_gpus``.  A slice whose broker is the target is a plain
host->device copy; a slice whose broker is another GPU is staged into that
helper GPU and then pulled across the GPU-GPU link.  Two host->device engines
feed the target instead of one.

Measuring it exposed where the prototype leaves throughput behind.  Both legs
of the helper path -- host->helper, then helper->target -- are issued on the
*same* CUDA stream, so the NVLink hop cannot start until the whole PCIe leg has
landed.  The helper's two links run one after the other, and the second
host->device engine buys 1.36x rather than the ~2x its bandwidth implies.

Measuring it a second way exposed the larger loss.  The prototype holds its
master weights in ``share_memory()`` host pages, which are *pageable*: CUDA
cannot DMA out of them, so every host->device copy is staged synchronously
through the driver's own bounce buffer and ``non_blocking=True`` is silently
ignored.  Page-locking that same shared mapping in place with
``cudaHostRegister`` -- no copy, no extra host RAM, paid once at startup --
lifts a single link from 11.7 to 26.1 GB/s on this box, and is what makes the
overlap above possible at all.

Policies:

``direct``  every slice is a host->device copy to the target.  One PCIe link,
            no helper GPU, no GPU-GPU hop: the sequential baseline.
``paper``   the released prototype's formula, unchanged, serialised legs and
            all.  This is what exp/paper-faithful-v3 actually ran.
``v4``      identical byte split and identical broker assignment, but each
            helper slice is cut into ``pipeline_depth`` sub-chunks issued
            across two helper streams joined by events, so the NVLink hop for
            sub-chunk k overlaps the PCIe leg of sub-chunk k+1.

Overlap only pays once the host pages are locked, so the two mechanisms are
reported both together and separately -- see ``register_host_memory``.

Every transfer appends one ``[PAPER-LOAD-V4]`` JSON record with per-path byte
counts, wall time and bandwidth.
"""

import json
import logging
import os
import threading
import time

import torch

logger = logging.getLogger(__name__)

BROKER_POLICY = os.environ.get("PRISM_V4_BROKER_POLICY", "paper")
LOADING_SPLIT = int(os.environ.get("PRISM_V4_LOADING_SPLIT", "0"))
PIPELINE_DEPTH = int(os.environ.get("PRISM_V4_PIPELINE_DEPTH", "4"))
TRACE_PATH = os.environ.get("PRISM_V4_LOAD_TRACE", "")

_trace_lock = threading.Lock()


def _emit(record):
    logger.info("[PAPER-LOAD-V4] " + json.dumps(record, default=str))
    if TRACE_PATH:
        try:
            with _trace_lock, open(TRACE_PATH, "a") as fh:
                fh.write(json.dumps(record, default=str) + "\n")
        except OSError:
            pass


def enable_peer_access(gpu_ids):
    """Report, per ordered pair, whether direct peer access exists.

    A cross-device ``copy_`` still succeeds without it, but the driver stages
    it through a host bounce buffer -- the very thing the design avoids.  The
    report states the measured path rather than assuming NVLink.
    """
    matrix = {}
    for src in gpu_ids:
        for dst in gpu_ids:
            if src == dst:
                continue
            try:
                matrix[f"{src}->{dst}"] = bool(torch.cuda.can_device_access_peer(src, dst))
            except Exception:
                matrix[f"{src}->{dst}"] = False
    return matrix


_registered = set()
_register_lock = threading.Lock()


def register_host_memory(state_dict):
    """Page-lock an existing shared host mapping in place.

    ``pin_memory()`` would allocate a second copy of every weight; the master
    copy is shared across engine processes and duplicating it is not an option.
    ``cudaHostRegister`` locks the pages that are already there, so the cost is
    one pass at startup and no additional host memory.

    Returns a summary rather than raising: a tensor that cannot be registered
    still transfers correctly, only slower, and the report should say how many
    were locked instead of the run dying.
    """
    cudart = torch.cuda.cudart()
    locked = failed = already = 0
    t0 = time.perf_counter()
    for tensor in state_dict.values():
        ptr = tensor.data_ptr()
        nbytes = tensor.numel() * tensor.element_size()
        if not nbytes:
            continue
        with _register_lock:
            if ptr in _registered:
                already += 1
                continue
            try:
                rc = int(cudart.cudaHostRegister(ptr, nbytes, 0))
            except Exception:
                rc = -1
            if rc == 0:
                _registered.add(ptr)
                locked += 1
            else:
                failed += 1
    return {
        "locked": locked,
        "already_locked": already,
        "failed": failed,
        "seconds": time.perf_counter() - t0,
    }


class StreamPool:
    """Per-GPU streams.  The helper path needs two to overlap its legs."""

    def __init__(self, gpu_ids, per_gpu=3):
        self.streams = {
            g: [torch.cuda.Stream(device=f"cuda:{g}") for _ in range(per_gpu)]
            for g in gpu_ids
        }

    def get(self, gpu, idx=0):
        pool = self.streams[gpu]
        return pool[idx % len(pool)]

    def synchronize(self):
        for pool in self.streams.values():
            for stream in pool:
                stream.synchronize()


def _slice_bounds(size, parts):
    """Split ``size`` rows into ``parts`` contiguous ranges.

    The prototype asserts ``local_size % max_threads == 0``; that assert is why
    the split cannot be raised past 2.  Spreading the remainder over the first
    slices keeps every slice non-empty and covers the tensor exactly.
    """
    if not size:
        return [(0, 0)]
    parts = max(1, min(parts, size))
    base, extra = divmod(size, parts)
    bounds, start = [], 0
    for i in range(parts):
        end = start + base + (1 if i < extra else 0)
        if end > start:
            bounds.append((start, end))
        start = end
    return bounds


def _broker_for(slice_idx, target_gpu_id, gpu_ids, policy):
    if policy == "direct" or len(gpu_ids) < 2:
        return target_gpu_id
    return gpu_ids[(slice_idx + gpu_ids.index(target_gpu_id) + 1) % len(gpu_ids)]


def _copy_direct(dst, src, stream, non_blocking=True):
    with torch.cuda.stream(stream):
        dst.copy_(src, non_blocking=non_blocking)


def _copy_via_helper_serial(dst, src, helper, stream):
    """The prototype's helper path: both legs on one stream, in order."""
    with torch.cuda.stream(stream):
        staged = src.to(f"cuda:{helper}", non_blocking=True)
        dst.copy_(staged, non_blocking=True)


def _copy_via_helper_pipelined(dst, src, helper, pool, depth):
    """Overlap the PCIe leg and the NVLink leg across sub-chunks."""
    s_h2d = pool.get(helper, 0)
    s_p2p = pool.get(helper, 1)
    rows = src.shape[0] if src.dim() else 0
    chunks = _slice_bounds(rows, depth) if rows else [(0, 0)]
    for start, end in chunks:
        sub_src = src[start:end] if rows else src
        sub_dst = dst[start:end] if rows else dst
        with torch.cuda.stream(s_h2d):
            staged = sub_src.to(f"cuda:{helper}", non_blocking=True)
            event = torch.cuda.Event()
            event.record(s_h2d)
        with torch.cuda.stream(s_p2p):
            s_p2p.wait_event(event)
            sub_dst.copy_(staged, non_blocking=True)
            # The staging buffer is freed by the allocator only after the
            # consuming stream is done with it.
            staged.record_stream(s_p2p)


def copy_model_to_gpu_v4(
    cpu_state_dict,
    gpu_state_dict,
    target_gpu_id,
    executor,
    gpu_ids,
    pool,
    policy=None,
    split=None,
    pipeline_depth=None,
    source_state_dict=None,
    host_registered=None,
    tag="",
):
    """Fill ``gpu_state_dict`` on ``target_gpu_id`` and record how it got there.

    ``source_state_dict`` -- a live copy of the same model resident on another
    GPU -- makes this a pure device-to-device transfer with the host out of the
    loop entirely.  That is the migration path; ``cpu_state_dict`` is the
    cold-start path.
    """
    policy = policy or BROKER_POLICY
    split = split or LOADING_SPLIT or min(4, len(gpu_ids))
    depth = pipeline_depth or PIPELINE_DEPTH
    t0 = time.perf_counter()

    counts = {"h2d_direct": 0, "h2d_helper": 0, "p2p": 0}
    slices = 0
    futures = []

    def _plan(dst_t, src_t, broker, device_src):
        if device_src:
            # Source already on a GPU: one hop over the GPU-GPU link.
            _copy_direct(dst_t, src_t, pool.get(target_gpu_id, 0))
        elif broker == target_gpu_id:
            _copy_direct(dst_t, src_t, pool.get(target_gpu_id, 0))
        elif policy == "v4":
            _copy_via_helper_pipelined(dst_t, src_t, broker, pool, depth)
        else:
            _copy_via_helper_serial(dst_t, src_t, broker, pool.get(broker, 0))

    for key, gpu_tensor in gpu_state_dict.items():
        device_src = source_state_dict is not None
        src_full = source_state_dict[key] if device_src else cpu_state_dict[key]
        rows = gpu_tensor.shape[0] if gpu_tensor.dim() else 0
        itemsize = gpu_tensor.element_size()
        per_row = (gpu_tensor.numel() // rows * itemsize) if rows else gpu_tensor.numel() * itemsize

        if not rows:
            futures.append(executor.submit(_plan, gpu_tensor, src_full, target_gpu_id, device_src))
            counts["p2p" if device_src else "h2d_direct"] += gpu_tensor.numel() * itemsize
            slices += 1
            continue

        for idx, (start, end) in enumerate(_slice_bounds(rows, split)):
            broker = _broker_for(idx, target_gpu_id, gpu_ids, policy)
            nbytes = (end - start) * per_row
            futures.append(
                executor.submit(_plan, gpu_tensor[start:end], src_full[start:end], broker, device_src)
            )
            if device_src:
                counts["p2p"] += nbytes
            elif broker == target_gpu_id:
                counts["h2d_direct"] += nbytes
            else:
                # Crosses PCIe once and the GPU-GPU link once.
                counts["h2d_helper"] += nbytes
                counts["p2p"] += nbytes
            slices += 1

    for future in futures:
        future.result()
    pool.synchronize()
    torch.cuda.synchronize(target_gpu_id)
    elapsed = time.perf_counter() - t0

    payload = counts["p2p"] if source_state_dict is not None else (
        counts["h2d_direct"] + counts["h2d_helper"])
    record = {
        "tag": tag,
        "policy": policy,
        "split": split,
        "pipeline_depth": depth if policy == "v4" else None,
        "target_gpu": target_gpu_id,
        "gpu_ids": list(gpu_ids),
        "source": "gpu" if source_state_dict is not None else "cpu",
        "host_registered": host_registered,
        "slices": slices,
        "payload_bytes": payload,
        "bytes_h2d_direct": counts["h2d_direct"],
        "bytes_h2d_helper": counts["h2d_helper"],
        "bytes_p2p": counts["p2p"],
        "seconds": elapsed,
        "payload_gbps": (payload / elapsed / 1e9) if elapsed > 0 else 0.0,
        "transfer_path": (
            "gpu-to-gpu-p2p" if source_state_dict is not None
            else "host-to-device-only" if counts["h2d_helper"] == 0
            else "host-to-device + helper-gpu-p2p"
        ),
    }
    _emit(record)
    return record
