"""KV-cache migration (paper §5.3) -- transport and request capsule.

The paper migrates a model's KV cache with the model.  The released prototype
does not, and the reason is not a missing transfer: it is that there is nothing
to transfer at the moment it tears a model down.

Every migration path emits ``DeactivateAction(preempt=False, ...)``
(``simple_global.py:822,918``; the paper-faithful policies subclass that class
and override only ``_find_optimal_migrations``, so they inherit it).  With
``preempt=False`` the scheduler runs ``_run_to_completion_normal`` and drains
the in-flight batch before teardown, so the KV is consumed rather than dropped.
That is a coherent design -- it avoids needing KV migration by paying a longer
teardown -- but it is a *different* design from the paper's, which retracts
in-flight requests and moves their KV so the target can resume decoding.

Turning that on means switching the migration path to
``preempt=True, preempt_mode=RECOMPUTE`` so ``_retract_running_batch_normal``
runs, and then moving what it would otherwise free.  Both live behind
``--enable-kv-migration``; with the flag off this module is never entered and
the default arm keeps reproducing the released prototype.

Layout, and why a gather is not an optimisation but a requirement
----------------------------------------------------------------
kvcached hands SGLang one virtually-contiguous ``FTensor`` per layer;
``k_buffer[l]`` and ``v_buffer[l]`` are its two halves, each shaped
``[num_tokens, head_num, head_dim]`` (``kvcached/ops.py:81-107``).  A request's
tokens occupy *scattered* slot indices -- ``req_to_token[req_pool_idx, :n]`` --
and slot ``i`` sits on page ``i * cell_size // 2MiB``
(``slab_allocator.py:483``), a page it generally shares with other requests.
So a request's KV is never a contiguous range and must be gathered whichever
way it is then moved.

That gather is also what makes the transfer legal.  ``csrc/page.cpp`` maps each
2 MiB page with ``cuMemSetAccess`` for the owning device only, so a peer read of
a source ``FTensor`` should fault.  It does not, because ``k_buffer[l][slots]``
is advanced indexing with the index tensor on the source device: the gather
executes *there* and materialises a temporary in the ordinary caching
allocator, and only that temporary crosses the link.  Measured on this box in
``exp/results/paper-faithful-v6/kv-probe/``.  No kvcached change is needed.

What the target cannot inherit
------------------------------
``req_pool_idx``, ``prefix_indices`` and ``last_node`` are indices into the
*source* engine's pools and are meaningless on the target; the target
re-allocates its own and rebuilds ``req_to_token`` from them.  The KV bytes are
positional, so they survive the renumbering as long as token order is kept --
which is why the capsule stores the gathered tensors in token order rather than
storing slot ids.

``output_ids`` is the only record of decode progress, and the prototype's
existing serialiser drops it: ``_convert_req_to_frontend_reqs``
(``scheduler.py:1772-1824``) rebuilds a ``GenerateReqInput`` from
``origin_input_ids`` alone and hardcodes ``output_len=512``.  Reusing it would
silently restart every migrated request from its prompt, which is exactly the
recomputation this mechanism exists to avoid, so the capsule carries its own
fields.

Every migration appends one ``[PAPER-KV-V6]`` JSON record, so ``kv_bytes`` in
the migration CSVs comes from measurement rather than staying a hardcoded 0.
"""

import json
import logging
import os
import threading
import time

import torch

logger = logging.getLogger(__name__)

TRACE_PATH = os.environ.get("PRISM_V6_KV_TRACE", "")
ENABLED = os.environ.get("PRISM_V6_KV_MIGRATION", "0") == "1"
# Cap what one migration may carry.  A retracted batch can be large, and the
# point of the mechanism is to save recomputation, not to move an unbounded
# amount of memory while both GPUs wait.  Requests past the cap fall back to
# the prototype's behaviour (recompute), and the record says how many did.
MAX_TOKENS = int(os.environ.get("PRISM_V6_KV_MAX_TOKENS", "65536"))

_trace_lock = threading.Lock()


def _emit(record):
    logger.info("[PAPER-KV-V6] " + json.dumps(record, default=str))
    if TRACE_PATH:
        try:
            with _trace_lock, open(TRACE_PATH, "a") as fh:
                fh.write(json.dumps(record, default=str) + "\n")
        except OSError:
            pass


class RequestKVCapsule:
    """One retracted request: its scheduler state and its KV, in token order.

    Deliberately not a ``Req``.  A ``Req`` carries tokenizer handles, radix-tree
    nodes and source-pool indices that cannot cross a process boundary or would
    be wrong on the target; this carries only what the target needs to rebuild
    one.
    """

    __slots__ = ("rid", "model_name", "origin_input_ids", "output_ids",
                 "sampling_params", "arrival_time", "slo", "k", "v",
                 "num_tokens", "source_gpu")

    def __init__(self, rid, model_name, origin_input_ids, output_ids,
                 sampling_params, arrival_time, slo, k, v, source_gpu):
        self.rid = rid
        self.model_name = model_name
        self.origin_input_ids = list(origin_input_ids)
        self.output_ids = list(output_ids)
        self.sampling_params = sampling_params
        self.arrival_time = arrival_time
        self.slo = slo
        self.k = k                      # list[layer] -> [n, head_num, head_dim]
        self.v = v
        self.num_tokens = k[0].shape[0] if k else 0
        self.source_gpu = source_gpu

    @property
    def nbytes(self):
        return sum(t.numel() * t.element_size() for t in self.k) + \
               sum(t.numel() * t.element_size() for t in self.v)

    def __repr__(self):
        return (f"RequestKVCapsule(rid={self.rid}, tokens={self.num_tokens}, "
                f"layers={len(self.k)}, bytes={self.nbytes})")


def gather_request_kv(k_buffer, v_buffer, slots, device):
    """Pull one request's KV off the source GPU, in token order.

    ``slots`` is ``req_to_token[req_pool_idx, :n]`` -- the ordered slot index per
    token.  Runs on the source device so the VMM pages are only ever read
    locally (see the module docstring).
    """
    if len(slots) == 0:
        return [], []
    idx = torch.as_tensor(slots, dtype=torch.long, device=device)
    # .clone() so the result owns its storage: the caller is about to free the
    # slots these were read from.
    k = [k_buffer[i][idx].clone() for i in range(len(k_buffer))]
    v = [v_buffer[i][idx].clone() for i in range(len(v_buffer))]
    return k, v


def transfer_capsule(capsule, target_gpu, stream=None):
    """Move a capsule's tensors to ``target_gpu``.

    Peer-to-peer when the pair allows it, otherwise through the host -- the
    same fallback shape as v4's weight path, so a PCIe-only box degrades
    instead of failing.
    """
    if capsule.num_tokens == 0:
        return capsule, "empty"
    src = capsule.source_gpu
    peer = bool(torch.cuda.can_device_access_peer(src, target_gpu)) \
        if src != target_gpu else True
    path = "gpu-to-gpu-p2p" if peer else "via-host"

    def _move(t):
        if peer:
            return t.to(f"cuda:{target_gpu}", non_blocking=True)
        return t.cpu().to(f"cuda:{target_gpu}", non_blocking=True)

    ctx = torch.cuda.stream(stream) if stream is not None else _NullCtx()
    with ctx:
        capsule.k = [_move(t) for t in capsule.k]
        capsule.v = [_move(t) for t in capsule.v]
    if stream is not None:
        stream.synchronize()
    else:
        torch.cuda.synchronize(target_gpu)
    capsule.source_gpu = target_gpu
    return capsule, path


def scatter_request_kv(k_buffer, v_buffer, slots, capsule, device):
    """Write a capsule's KV into freshly allocated slots on the target.

    ``slots`` come from the *target's* allocator and bear no relation to the
    source's numbering; token order is what preserves correctness.
    """
    if capsule.num_tokens == 0:
        return
    if len(slots) != capsule.num_tokens:
        raise ValueError(
            f"slot count {len(slots)} != capsule tokens {capsule.num_tokens}")
    idx = torch.as_tensor(slots, dtype=torch.long, device=device)
    for i in range(len(k_buffer)):
        k_buffer[i][idx] = capsule.k[i]
        v_buffer[i][idx] = capsule.v[i]


def migrate_request_kv(capsules, target_gpu, tag=""):
    """Transfer a batch of capsules and account for it.

    Returns ``(moved, skipped, record)``.  The cap is enforced here rather than
    at the call site so the record always reports what was left behind -- a
    silent truncation would read as "everything migrated".
    """
    t0 = time.time()
    moved, skipped, budget = [], [], MAX_TOKENS
    for c in capsules:
        if c.num_tokens <= budget:
            moved.append(c)
            budget -= c.num_tokens
        else:
            skipped.append(c)

    paths, nbytes = set(), 0
    for c in moved:
        nbytes += c.nbytes
        _, p = transfer_capsule(c, target_gpu)
        paths.add(p)
    seconds = time.time() - t0

    record = {
        "tag": tag,
        "target_gpu": target_gpu,
        "requests_moved": len(moved),
        "requests_skipped_over_cap": len(skipped),
        "tokens_moved": sum(c.num_tokens for c in moved),
        "tokens_skipped": sum(c.num_tokens for c in skipped),
        "kv_bytes": nbytes,
        "seconds": round(seconds, 6),
        "kv_gbps": round(nbytes / seconds / 1e9, 3) if seconds > 0 else None,
        "transfer_path": "+".join(sorted(paths)) if paths else "none",
        "max_tokens_cap": MAX_TOKENS,
    }
    _emit(record)
    return moved, skipped, record


class _NullCtx:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False
