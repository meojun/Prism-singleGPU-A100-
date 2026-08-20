"""TP worker-slot groups for the worker-pool path.

Why this module exists
----------------------
In the released prototype ``launch_worker_pool_engines`` builds one engine per
``(GPU, worker slot)`` and hands each one ``[gpu_id]`` -- a list of length 1.
The obvious reading is that the fix is to hand it the model's TP group instead.
It is not, because of an ordering problem:

* a worker-pool engine's ``tp_size`` is fixed when the **server starts**;
* which model occupies a slot is decided at **runtime**, by the scheduler.

So "pass the model's tp_size" is not expressible -- at launch time nobody knows
which model will land in the slot.  The slots themselves have to carry the
type.

What a TP slot group is
-----------------------
A TP=k group is a *static* tuple of k distinct GPUs plus one worker-slot id.
One engine spans it with k scheduler processes, rank r running on ``gpu_ids[r]``.
Because membership is fixed at launch, the runtime side stays simple:

* the slot is **owned** by rank0's GPU -- that pool assigns it to a model;
* on every other GPU of the group the same slot id is **reserved** forever and
  is never handed to anything else.

That is worth stating plainly because the handoff expected otherwise.  The
handoff assumed the k pools would have to acquire k slots together and roll
back on partial failure.  With static membership there is no partial state to
roll back: activation touches exactly one pool (rank0's), so it is already
atomic.  The cost of that simplicity is stranded capacity -- a reserved slot on
a non-rank0 GPU is unusable by TP=1 models even while the group is idle.  That
is a real trade and is recorded rather than hidden.

Why the slot id must be shared across the group
-----------------------------------------------
``worker_pool_model_runner.py:121`` derives each rank's accounting segment as
``ipc_{gpu_id}_{worker_id}_{user}`` from *that rank's own* gpu_id.  So a group
must carry a single ``worker_id``, and that id must be free on every GPU in the
group.  Ids are therefore allocated globally, not per GPU, which means a GPU's
slot ids are a sparse subset of the integers -- ``range(num_workers)`` is no
longer a valid enumeration of them.  Callers must use :meth:`SlotPlan.worker_ids_on`.

Determinism
-----------
``build_slot_plan`` is a pure function of ``(num_gpus, workers_per_gpu,
tp_sizes, max_groups_per_tp_size)``.  The server launcher and each per-GPU
scheduler process call it independently and must agree, so nothing here may
depend on iteration order of a set, wall-clock, or process identity.
"""

from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class TPSlot:
    """One worker slot.  ``tp_size == 1`` is the prototype's per-GPU slot."""

    worker_id: int
    tp_size: int
    gpu_ids: Tuple[int, ...]

    @property
    def owner_gpu(self) -> int:
        """The GPU whose scheduler drives this slot (rank 0)."""
        return self.gpu_ids[0]

    def role_on(self, gpu_id: int) -> str:
        """``owner`` where the pool may assign it, ``shadow`` where it is only reserved."""
        if gpu_id == self.owner_gpu:
            return "owner"
        if gpu_id in self.gpu_ids:
            return "shadow"
        return "absent"

    def rank_of(self, gpu_id: int) -> int:
        return self.gpu_ids.index(gpu_id)

    def as_dict(self) -> Dict:
        return {
            "worker_id": self.worker_id,
            "tp_size": self.tp_size,
            "gpu_ids": list(self.gpu_ids),
            "owner_gpu": self.owner_gpu,
        }


@dataclass
class SlotPlan:
    """Every slot on every GPU, and which of them are TP groups."""

    num_gpus: int
    workers_per_gpu: int
    slots: List[TPSlot] = field(default_factory=list)

    # -- lookups ---------------------------------------------------------
    def slots_on(self, gpu_id: int) -> List[TPSlot]:
        return [s for s in self.slots if gpu_id in s.gpu_ids]

    def worker_ids_on(self, gpu_id: int) -> List[int]:
        """The slot ids that actually exist on this GPU.

        Sparse in general.  This is what replaces ``range(num_workers)`` for
        anything that walks a GPU's slots (the worker pool, and the resource
        manager's per-slot memory readers).
        """
        return sorted(s.worker_id for s in self.slots_on(gpu_id))

    def owned_slots_on(self, gpu_id: int) -> List[TPSlot]:
        return [s for s in self.slots_on(gpu_id) if s.role_on(gpu_id) == "owner"]

    def groups(self, tp_size: Optional[int] = None) -> List[TPSlot]:
        """The TP>1 groups, optionally filtered to one tp_size."""
        out = [s for s in self.slots if s.tp_size > 1]
        if tp_size is not None:
            out = [s for s in out if s.tp_size == tp_size]
        return out

    def group_for_gpus(self, gpu_ids: Sequence[int]) -> Optional[TPSlot]:
        """The group occupying exactly this ordered GPU tuple, if one exists."""
        want = tuple(gpu_ids)
        for s in self.slots:
            if s.gpu_ids == want:
                return s
        return None

    def tp_sizes_available(self) -> List[int]:
        return sorted({s.tp_size for s in self.slots})

    def as_dict(self) -> Dict:
        return {
            "num_gpus": self.num_gpus,
            "workers_per_gpu": self.workers_per_gpu,
            "slots": [s.as_dict() for s in self.slots],
            "slots_per_gpu": {
                g: self.worker_ids_on(g) for g in range(self.num_gpus)
            },
        }


def build_slot_plan(
    num_gpus: int,
    workers_per_gpu: int,
    tp_sizes: Sequence[int] = (),
    max_groups_per_tp_size: Optional[int] = None,
) -> SlotPlan:
    """Lay out every worker slot in the cluster.

    Slots ``0 .. workers_per_gpu-1`` on each GPU stay exactly what the released
    prototype builds: one TP=1 engine each.  Anything with ``tp_size > 1`` is
    appended afterwards with globally-unique ids, so with no TP models in the
    config the plan is byte-for-byte the prototype's layout.

    ``tp_sizes`` are the distinct TP sizes the model configs ask for.  For each
    k > 1 every k-subset of GPUs is enumerated, which is what gives Algorithm 1
    a real choice of placement -- with only one candidate group per model the
    anti-affinity constraint would be satisfied by construction and there would
    be nothing to measure.

    ``max_groups_per_tp_size`` caps that enumeration.  Each group costs k
    scheduler processes and k CUDA contexts, so the full C(n,k) is not always
    affordable.  When it bites, the *dropped* groups are recorded on the plan
    rather than silently omitted.
    """
    if num_gpus < 1:
        raise ValueError(f"num_gpus must be >= 1, got {num_gpus}")
    if workers_per_gpu < 0:
        raise ValueError(f"workers_per_gpu must be >= 0, got {workers_per_gpu}")

    slots: List[TPSlot] = []

    # 1. the prototype's per-GPU slots, unchanged.
    for worker_id in range(workers_per_gpu):
        for gpu_id in range(num_gpus):
            slots.append(TPSlot(worker_id=worker_id, tp_size=1, gpu_ids=(gpu_id,)))

    # 2. TP groups.  Ids continue past the per-GPU block and are global, since
    #    a group's id has to be free on every GPU it spans.
    next_id = workers_per_gpu
    dropped: List[Tuple[int, Tuple[int, ...]]] = []
    for k in sorted({int(t) for t in tp_sizes}):
        if k <= 1:
            continue
        if k > num_gpus:
            # Not satisfiable here: k distinct GPUs are required and do not exist.
            dropped.append((k, ()))
            continue
        combos = list(combinations(range(num_gpus), k))
        keep = combos if max_groups_per_tp_size is None else combos[:max_groups_per_tp_size]
        for combo in keep:
            slots.append(TPSlot(worker_id=next_id, tp_size=k, gpu_ids=tuple(combo)))
            next_id += 1
        for combo in combos[len(keep):]:
            dropped.append((k, combo))

    plan = SlotPlan(num_gpus=num_gpus, workers_per_gpu=workers_per_gpu, slots=slots)
    plan.dropped_groups = dropped  # type: ignore[attr-defined]
    return plan


def tp_sizes_from_model_configs(model_configs) -> List[int]:
    """Distinct ``tp_size`` values the configured models ask for."""
    sizes = set()
    for mc in model_configs or []:
        sizes.add(int(getattr(mc, "tp_size", 1) or 1))
    return sorted(sizes)
