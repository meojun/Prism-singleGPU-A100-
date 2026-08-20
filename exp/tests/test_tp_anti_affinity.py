#!/usr/bin/env python3
"""Unit tests for the TP anti-affinity filter in Algorithm 1.

The point of these tests is not that the constraint is *satisfied* -- that is
easy to arrange and proves nothing.  It is that the constraint **binds**: there
are placements the unconstrained argmin genuinely wants to make, and the filter
changes them.  Case 2 is the one that matters; if it ever stops failing with the
filter off, the experiment in step 4 is measuring nothing.

Run: python3 exp/tests/test_tp_anti_affinity.py   (needs the venv on sys.path)
"""

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "prism-research/python"))

from sglang.multi_model.scheduling.policy.kvpr_global_tp import (  # noqa: E402
    KVPRGlobalPolicyTP,
)

FAILED = []


def check(cond, label):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}")
    if not cond:
        FAILED.append(label)


def make_policy(num_gpus, tp_sizes, anti_affinity, tau=0.0, gpu_mem=80.0):
    """A policy instance with the collaborators Algorithm 1 actually reads."""
    model_configs = [
        SimpleNamespace(model_name=name, tp_size=k) for name, k in tp_sizes.items()
    ]
    server_args = SimpleNamespace(
        enable_tp_anti_affinity=anti_affinity,
        model_configs=model_configs,
    )
    weights_info = {name: {"model_size": 8.0} for name in tp_sizes}
    return KVPRGlobalPolicyTP(
        num_gpus=num_gpus,
        gpu_mem=gpu_mem,
        model_weights_info=weights_info,
        workers_per_gpu=2,
        tau=tau,
        rate_window=30.0,
        migration_cooldown=30.0,
        tpot_slo_s={name: 0.05 for name in tp_sizes},
        server_args=server_args,
    )


def place(policy, current_groups, rates, num_gpus):
    weights = {m: {"weighted_token_rate": r} for m, r in rates.items()}
    free = {i: 80.0 for i in range(num_gpus)}
    return policy._greedy_placement_tp(weights, current_groups, free)


def case_tp1_unchanged():
    print("case 1: tp_size=1 models are untouched by the filter")
    for aa in (False, True):
        pol = make_policy(4, {"m1": 1, "m2": 1}, anti_affinity=aa)
        plan = place(pol, {"m1": (0,), "m2": (1,)}, {"m1": 10.0, "m2": 5.0}, 4)
        check(all(len(g) == 1 for g in plan.values()),
              f"aa={aa}: every TP=1 model still occupies exactly one GPU")
        check(pol._tp_audit["aa_violations"] == 0,
              f"aa={aa}: no violation is counted for TP=1 models")


def _imbalanced():
    """A cluster where the constraint genuinely binds, and why.

    Per-shard argmin charges each GPU as it places, so it *usually* spreads a
    group by itself and anti-affinity is satisfied for free.  It only binds when
    one GPU is so much less loaded than every alternative that it is still the
    argmin after taking a shard.  That is not exotic -- it is what a
    freshly-drained GPU looks like.

    Three heavy TP=1 models take GPUs 0..2 in the planning order (they sort
    first, by rate), leaving GPU 3 empty.  Then a light TP=2 model arrives: its
    first shard goes to GPU 3, and charging it barely moves GPU 3's ratio, so
    the unconstrained argmin sends the second shard to GPU 3 as well.
    """
    sizes = {"heavy_a": 1, "heavy_b": 1, "heavy_c": 1, "tp": 2}
    current = {"heavy_a": (0,), "heavy_b": (1,), "heavy_c": (2,), "tp": (0, 1)}
    rates = {"heavy_a": 1000.0, "heavy_b": 1000.0, "heavy_c": 1000.0, "tp": 1.0}
    return sizes, current, rates


def case_constraint_actually_binds():
    print("case 2: the filter changes a placement the argmin wanted to make")
    sizes, current, rates = _imbalanced()
    off = make_policy(4, sizes, anti_affinity=False)
    plan_off = place(off, current, rates, 4)
    on = make_policy(4, sizes, anti_affinity=True)
    plan_on = place(on, current, rates, 4)

    print(f"    OFF -> tp on {plan_off['tp']}   ON -> tp on {plan_on['tp']}")
    check(off._tp_audit["aa_violations"] > 0,
          "OFF: the unconstrained argmin is recorded as wanting a collision")
    check(len(set(plan_on["tp"])) == 2,
          "ON: the two shards land on two distinct GPUs")
    check(on._tp_audit["aa_diverted"] > 0,
          "ON: the filter is recorded as having diverted a shard")
    check(plan_on["tp"] != plan_off["tp"],
          "the ON and OFF plans differ -- there is something to measure")


def case_off_can_produce_illegal_plan():
    print("case 3: with the filter off, an illegal placement is actually emitted")
    sizes, current, rates = _imbalanced()
    off = make_policy(4, sizes, anti_affinity=False)
    plan = place(off, current, rates, 4)
    collided = len(set(plan["tp"])) < len(plan["tp"])
    print(f"    OFF plan = tp on {plan['tp']}  collided={collided}")
    check(collided, "OFF emits two shards on one GPU -- the constraint has "
                    "something to constrain")


def case_balanced_cluster_does_not_bind():
    print("case 4a: on a balanced cluster the constraint does NOT bind (recorded, "
          "not hidden)")
    # Reported because it is the common case, and because a study that only
    # showed the binding scenario would overstate how often this matters.
    off = make_policy(4, {"tp": 2}, anti_affinity=False)
    plan_off = place(off, {"tp": (0, 1)}, {"tp": 100.0}, 4)
    on = make_policy(4, {"tp": 2}, anti_affinity=True)
    plan_on = place(on, {"tp": (0, 1)}, {"tp": 100.0}, 4)
    print(f"    OFF -> {plan_off['tp']}   ON -> {plan_on['tp']}")
    check(len(set(plan_off["tp"])) == 2,
          "the unconstrained argmin already spreads the shards")
    check(plan_off["tp"] == plan_on["tp"],
          "ON changes nothing here")
    check(off._tp_audit["aa_violations"] == 0,
          "and no violation is counted, so the counters do not inflate")


def case_on_never_emits_illegal_plan():
    print("case 4b: with the filter on, no plan repeats a GPU within a model")
    scenarios = [
        (4, {"tp": 2}, {"tp": (0, 1)}, {"tp": 100.0}),
        (4, {"tp": 4}, {"tp": (0, 1, 2, 3)}, {"tp": 40.0}),
        (4, {"a": 2, "b": 2}, {"a": (0, 1), "b": (2, 3)}, {"a": 90.0, "b": 10.0}),
        (8, {"tp": 4}, {"tp": (0, 1, 2, 3)}, {"tp": 50.0}),
    ]
    for n, sizes, cur, rates in scenarios:
        pol = make_policy(n, sizes, anti_affinity=True)
        plan = place(pol, cur, rates, n)
        bad = {m: g for m, g in plan.items() if len(set(g)) != len(g)}
        check(not bad, f"num_gpus={n} {sizes}: no model repeats a GPU {bad or ''}")


def case_group_size_preserved():
    print("case 5: a TP=k model is planned onto exactly k GPUs")
    for k in (2, 4):
        pol = make_policy(4, {"tp": k}, anti_affinity=True)
        plan = place(pol, {"tp": tuple(range(k))}, {"tp": 50.0}, 4)
        check(len(plan["tp"]) == k, f"tp_size={k}: plan has {k} entries")
        check(pol._tp_audit["shards_placed"] == k, f"tp_size={k}: k shards placed")


def case_counterfactual_is_recorded_when_off():
    print("case 6: the OFF arm still records what the constraint would have done")
    # This is what makes the ON/OFF comparison measurable rather than anecdotal:
    # the OFF run must count the violations it commits, not stay silent.
    sizes, current, rates = _imbalanced()
    off = make_policy(4, sizes, anti_affinity=False)
    place(off, current, rates, 4)
    check(off._tp_audit["aa_violations"] > 0, "OFF counts its own violations")
    check(off._tp_audit["aa_diverted"] == 0, "OFF diverts nothing, by definition")
    check(off.anti_affinity is False, "OFF reports itself as off")


def case_line8_records_the_alternative():
    print("case 7: every shard decision carries its unconstrained alternative")
    sizes, current, rates = _imbalanced()
    on = make_policy(4, sizes, anti_affinity=True)
    place(on, current, rates, 4)
    rows = [r for r in on._last_line8 if r["tp_size"] == 2]
    check(len(rows) == 2, "one row per shard")
    check(all("best_gpu_unconstrained" in r for r in rows),
          "each row records what the argmin would have chosen unfiltered")
    check(all("anti_affinity_filtered" in r for r in rows),
          "each row records which candidates the filter removed")
    check(any(r["anti_affinity_filtered"] for r in rows),
          "at least one shard had a candidate removed")


def case_tp_size_lookup():
    print("case 8: tp_size comes from the model configs")
    pol = make_policy(4, {"a": 2, "b": 1}, anti_affinity=True)
    check(pol.tp_size_of("a") == 2, "configured tp_size is used")
    check(pol.tp_size_of("b") == 1, "tp_size=1 is used")
    check(pol.tp_size_of("unknown") == 1, "an unknown model defaults to 1")


def main():
    for fn in (
        case_tp1_unchanged,
        case_constraint_actually_binds,
        case_off_can_produce_illegal_plan,
        case_balanced_cluster_does_not_bind,
        case_on_never_emits_illegal_plan,
        case_group_size_preserved,
        case_counterfactual_is_recorded_when_off,
        case_line8_records_the_alternative,
        case_tp_size_lookup,
    ):
        fn()
    print()
    if FAILED:
        print(f"{len(FAILED)} TP ANTI-AFFINITY TEST(S) FAILED:")
        for f in FAILED:
            print(f"  - {f}")
        return 1
    print("ALL TP ANTI-AFFINITY TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
