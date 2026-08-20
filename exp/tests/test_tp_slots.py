#!/usr/bin/env python3
"""Unit tests for the TP worker-slot plan.

The plan is computed independently in the server launcher and in every per-GPU
scheduler process, and they must agree without talking to each other.  So the
properties worth testing are determinism, the no-TP fallback being *exactly*
the prototype's layout, and the two invariants the runtime depends on:

  * a group's ``worker_id`` is free on every GPU it spans (otherwise two engines
    collide on one accounting segment, ``ipc_{gpu}_{worker}_{user}``);
  * a group's GPUs are distinct (the paper's anti-affinity constraint, which
    here is also a hard requirement -- NCCL will not put two ranks of one
    communicator on the same device).

Run: python3 exp/tests/test_tp_slots.py
"""

import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "patches/paper_faithful_tp"))

from tp_slots import build_slot_plan, tp_sizes_from_model_configs  # noqa: E402

FAILED = []


def check(cond, label):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}")
    if not cond:
        FAILED.append(label)


def case_prototype_layout_unchanged():
    print("case 1: no TP models -> the prototype's layout, byte for byte")
    plan = build_slot_plan(num_gpus=4, workers_per_gpu=2, tp_sizes=[1])
    check(len(plan.slots) == 8, "4 GPUs x 2 workers = 8 slots")
    check(all(s.tp_size == 1 for s in plan.slots), "every slot is TP=1")
    check(all(len(s.gpu_ids) == 1 for s in plan.slots), "every slot spans one GPU")
    check(
        all(plan.worker_ids_on(g) == [0, 1] for g in range(4)),
        "slot ids are 0..workers_per_gpu-1 on every GPU",
    )
    check(plan.groups() == [], "no TP groups exist")
    # An empty tp_sizes must behave the same as [1].
    check(
        build_slot_plan(4, 2, tp_sizes=[]).as_dict() == plan.as_dict(),
        "tp_sizes=[] is identical to tp_sizes=[1]",
    )


def case_group_enumeration():
    print("case 2: TP=2 on 4 GPUs enumerates all six pairs")
    plan = build_slot_plan(num_gpus=4, workers_per_gpu=2, tp_sizes=[1, 2])
    groups = plan.groups(2)
    check(len(groups) == 6, "C(4,2) = 6 groups")
    check(
        {g.gpu_ids for g in groups} == set(combinations(range(4), 2)),
        "the six pairs are exactly the 2-subsets of {0,1,2,3}",
    )
    # Placement freedom is the whole point: with one candidate the constraint
    # would be satisfied by construction and there would be nothing to measure.
    check(len(groups) >= 2, "more than one placement candidate exists")


def case_worker_id_free_on_every_gpu_it_spans():
    print("case 3: a group's slot id is free on every GPU it spans")
    plan = build_slot_plan(num_gpus=4, workers_per_gpu=2, tp_sizes=[1, 2])
    ok = True
    for gpu in range(plan.num_gpus):
        ids = plan.worker_ids_on(gpu)
        if len(ids) != len(set(ids)):
            ok = False
    check(ok, "no GPU has a duplicated slot id")

    # The real invariant behind it: two different slots never share an id on a
    # GPU, because that id names a single shared-memory segment there.
    collisions = []
    for gpu in range(plan.num_gpus):
        seen = {}
        for s in plan.slots_on(gpu):
            if s.worker_id in seen:
                collisions.append((gpu, s.worker_id))
            seen[s.worker_id] = s
    check(not collisions, f"no (gpu, slot id) collisions {collisions or ''}")


def case_distinct_gpus_per_group():
    print("case 4: every group's GPUs are distinct")
    for n, k in ((4, 2), (4, 4), (8, 2), (8, 4)):
        plan = build_slot_plan(num_gpus=n, workers_per_gpu=2, tp_sizes=[k])
        bad = [g.gpu_ids for g in plan.groups() if len(set(g.gpu_ids)) != len(g.gpu_ids)]
        check(not bad, f"num_gpus={n} tp={k}: no group repeats a GPU {bad or ''}")


def case_owner_and_shadow_roles():
    print("case 5: exactly one owner per group, the rest are shadows")
    plan = build_slot_plan(num_gpus=4, workers_per_gpu=2, tp_sizes=[1, 2])
    ok_owner, ok_shadow = True, True
    for s in plan.groups():
        owners = [g for g in s.gpu_ids if s.role_on(g) == "owner"]
        shadows = [g for g in s.gpu_ids if s.role_on(g) == "shadow"]
        if owners != [s.gpu_ids[0]]:
            ok_owner = False
        if len(shadows) != s.tp_size - 1:
            ok_shadow = False
    check(ok_owner, "the owner is rank0's GPU and nothing else")
    check(ok_shadow, "the other k-1 GPUs are shadows")
    check(
        plan.groups()[0].role_on(99) == "absent",
        "a GPU outside the group has no role in it",
    )
    # A shadow slot must never be handed out by its GPU's pool.
    gpu1_owned = {s.worker_id for s in plan.owned_slots_on(1)}
    check(2 not in gpu1_owned, "GPU1 does not own slot 2, which GPU0 drives")


def case_determinism():
    print("case 6: same inputs -> same plan, in any process")
    a = build_slot_plan(4, 2, tp_sizes=[1, 2, 4])
    b = build_slot_plan(4, 2, tp_sizes=[4, 2, 1])          # order must not matter
    c = build_slot_plan(4, 2, tp_sizes=[1, 1, 2, 2, 4])    # duplicates must not matter
    check(a.as_dict() == b.as_dict(), "tp_sizes order does not change the plan")
    check(a.as_dict() == c.as_dict(), "duplicate tp_sizes do not change the plan")
    check(
        [s.worker_id for s in a.slots] == [s.worker_id for s in build_slot_plan(4, 2, [1, 2, 4]).slots],
        "repeated calls agree",
    )


def case_tp_larger_than_cluster():
    print("case 7: tp_size larger than the cluster is dropped, not faked")
    plan = build_slot_plan(num_gpus=2, workers_per_gpu=2, tp_sizes=[1, 4])
    check(plan.groups(4) == [], "TP=4 yields no group on a 2-GPU box")
    check((4, ()) in getattr(plan, "dropped_groups", []), "the drop is recorded")
    # TP=4 on exactly 4 GPUs is the single whole-cluster group.
    plan4 = build_slot_plan(num_gpus=4, workers_per_gpu=2, tp_sizes=[4])
    check(len(plan4.groups(4)) == 1, "TP=4 on 4 GPUs is one group")
    check(plan4.groups(4)[0].gpu_ids == (0, 1, 2, 3), "and it spans every GPU")


def case_cap_records_what_it_drops():
    print("case 8: a cap on group count reports what it dropped")
    plan = build_slot_plan(4, 2, tp_sizes=[2], max_groups_per_tp_size=2)
    check(len(plan.groups(2)) == 2, "cap honoured")
    check(len(getattr(plan, "dropped_groups", [])) == 4, "the other four are recorded")
    # Silent truncation would read as "all pairs covered" in the report.
    check(
        all(k == 2 for k, _ in plan.dropped_groups),
        "dropped entries carry their tp_size",
    )


def case_lookup_helpers():
    print("case 9: lookups used by placement")
    plan = build_slot_plan(4, 2, tp_sizes=[1, 2])
    check(plan.group_for_gpus([1, 2]) is not None, "an existing pair is found")
    check(plan.group_for_gpus((1, 2)).tp_size == 2, "and it is the TP=2 group")
    check(plan.group_for_gpus([2, 1]) is None, "order matters: (2,1) is not (1,2)")
    check(plan.group_for_gpus([0, 0]) is None, "a colliding pair has no group")
    check(plan.tp_sizes_available() == [1, 2], "available tp sizes are reported")


def case_sizes_from_configs():
    print("case 10: tp_sizes read off the model configs")

    class MC:
        def __init__(self, tp):
            self.tp_size = tp

    check(tp_sizes_from_model_configs([MC(1), MC(2), MC(2)]) == [1, 2], "distinct and sorted")
    check(tp_sizes_from_model_configs([]) == [], "no configs -> nothing")

    class Bare:
        pass

    check(tp_sizes_from_model_configs([Bare()]) == [1], "a missing tp_size means 1")
    check(tp_sizes_from_model_configs([MC(None)]) == [1], "tp_size=None means 1")


def main():
    for fn in (
        case_prototype_layout_unchanged,
        case_group_enumeration,
        case_worker_id_free_on_every_gpu_it_spans,
        case_distinct_gpus_per_group,
        case_owner_and_shadow_roles,
        case_determinism,
        case_tp_larger_than_cluster,
        case_cap_records_what_it_drops,
        case_lookup_helpers,
        case_sizes_from_configs,
    ):
        fn()
    print()
    if FAILED:
        print(f"{len(FAILED)} TP SLOT TEST(S) FAILED:")
        for f in FAILED:
            print(f"  - {f}")
        return 1
    print("ALL TP SLOT TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
