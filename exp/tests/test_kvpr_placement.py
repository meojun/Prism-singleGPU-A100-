#!/usr/bin/env python3
"""Test A — KVPR placement (paper Algorithm 1) on synthetic GPUs and models.

Run:  python exp/tests/test_kvpr_placement.py
"""
import sys
import time

from sglang.multi_model.scheduling.model_queue_tracker import Req
from sglang.multi_model.scheduling.policy.kvpr_global import KVPRGlobalPolicy
from sglang.multi_model.scheduling.state import ModelInstanceState, ModelState
from sglang.srt.managers.io_struct import MemoryUsage

FAILURES = []


def check(name, cond, detail=""):
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        FAILURES.append(name)


class StubQueue:
    """Minimal stand-in for ModelQueueTracker.

    Algorithm 1 reads exactly two things from it: arrivals (for the input token
    rate) and the engine-reported decode throughput.
    """

    def __init__(self, name, arrivals_prompt_lens, decode_tput, now):
        self.model_name = name
        self.decode_token_tput = decode_tput
        self.received_reqs = {}
        for i, plen in enumerate(arrivals_prompt_lens):
            r = Req(rid=f"{name}-{i}")
            r.arrival_time = now - 1.0     # inside any sane window
            r.prompt_len = plen
            r.is_warmup = False
            self.received_reqs[r.rid] = r


def make_policy(num_gpus=2, gpu_mem=67.28, tau=0.10, sizes=None, cells=None, tpot_ms=None):
    models = sizes or {"model_1": 15.08, "model_4": 15.08, "model_5": 15.08}
    cells = cells or {m: 131072 for m in models}
    info = {m: {"model_size": models[m], "cell_size": cells[m]} for m in models}
    pol = KVPRGlobalPolicy(
        num_gpus=num_gpus,
        gpu_mem=gpu_mem,
        model_weights_info=info,
        workers_per_gpu=4,
        tau=tau,
        rate_window=30.0,
    )
    if tpot_ms:
        pol.tpot_slo_s = {m: v / 1000.0 for m, v in tpot_ms.items()}
    return pol


def instances(mapping, num_gpus=2):
    """model -> ACTIVE instance on its GPU, INACTIVE elsewhere."""
    d = {}
    for gpu, models in mapping.items():
        for m in models:
            d.setdefault(m, [])
            for g in range(num_gpus):
                d[m].append(
                    ModelInstanceState(
                        model_name=m,
                        model_path=m,
                        instance_idx=g,
                        gpu_ids=[g],
                        memory_usage=MemoryUsage(
                            total_used_memory=15.08,
                            model_weights_memory=15.08,
                            memory_pool_memory=0,
                            req_to_token_pool_memory=0,
                            token_to_kv_pool_memory=0,
                        ),
                        init_memory_pool_size=0,
                        state=ModelState.ACTIVE if g == gpu else ModelState.INACTIVE,
                    )
                )
    return d


now = time.time()

print("== 1. weighted_token_rate = token_rate * token_size / SLO ==")
pol = make_policy(tpot_ms={"model_1": 10.0, "model_4": 10.0, "model_5": 10.0})
# 100 arrivals x 10 tokens over a 30 s window = 1000/30 tok/s input; decode 50 tok/s
q = StubQueue("model_1", [10] * 100, 50.0, now)
w = pol._weighted_token_rates({"model_1": q}, ["model_1"])["model_1"]
exp_input = 1000 / 30.0
check("input token rate", abs(w["input_token_rate"] - exp_input) < 1e-6, w)
check("decode token rate", abs(w["decode_token_rate"] - 50.0) < 1e-9, w)
check("token_rate = input + decode", abs(w["token_rate"] - (exp_input + 50.0)) < 1e-9, w)
expected = (exp_input + 50.0) * 131072 / 0.010
check("weighted = rate*size/slo", abs(w["weighted_token_rate"] - expected) < 1e-3, w)

print("== 2. tighter TPOT SLO raises the weight proportionally ==")
pol2 = make_policy(tpot_ms={"model_1": 5.0, "model_4": 10.0, "model_5": 10.0})
qs = {"model_1": StubQueue("model_1", [10] * 100, 50.0, now),
      "model_4": StubQueue("model_4", [10] * 100, 50.0, now)}
ws = pol2._weighted_token_rates(qs, ["model_1", "model_4"])
ratio = ws["model_1"]["weighted_token_rate"] / ws["model_4"]["weighted_token_rate"]
check("half the SLO -> double the weight", abs(ratio - 2.0) < 1e-9, ratio)

print("== 3. shared_kv = budget - sum(weights on that GPU) ==")
pol = make_policy()
mapping = {0: ["model_1"], 1: ["model_4", "model_5"]}
check("GPU0 shared_kv", abs(pol._shared_kv(mapping, 0) - (67.28 - 15.08)) < 1e-9)
check("GPU1 shared_kv", abs(pol._shared_kv(mapping, 1) - (67.28 - 30.16)) < 1e-9)

print("== 4. KVPR = aggregate weighted rate / shared_kv ==")
weights = {m: {"weighted_token_rate": r} for m, r in
           [("model_1", 100.0), ("model_4", 100.0), ("model_5", 100.0)]}
kvpr = pol._kvpr(mapping, weights)
check("GPU0 KVPR", abs(kvpr[0] - 100.0 / (67.28 - 15.08)) < 1e-9, kvpr)
check("GPU1 KVPR", abs(kvpr[1] - 200.0 / (67.28 - 30.16)) < 1e-9, kvpr)
check("the 2-model GPU is the more pressured one", kvpr[1] > kvpr[0], kvpr)

print("== 5. models are considered in DESCENDING weighted token rate ==")
pol = make_policy(tpot_ms={"model_1": 10.0, "model_4": 10.0, "model_5": 10.0})
qs = {
    "model_1": StubQueue("model_1", [10] * 10, 0.0, now),    # smallest
    "model_4": StubQueue("model_4", [10] * 100, 0.0, now),   # largest
    "model_5": StubQueue("model_5", [10] * 50, 0.0, now),
}
ws = pol._weighted_token_rates(qs, list(qs))
order = [m for m, _ in sorted(ws.items(),
         key=lambda kv: kv[1]["weighted_token_rate"], reverse=True)]
check("descending order", order == ["model_4", "model_5", "model_1"], order)

print("== 6. migration fires and picks the LOWEST resulting KVPR ==")
# GPU0: model_1 (idle).  GPU1: model_4 + model_5, both hot.
# Moving either hot model to GPU0 cuts peak KVPR roughly in half -> well over tau.
pol = make_policy(tau=0.10, tpot_ms={"model_1": 10.0, "model_4": 10.0, "model_5": 10.0})
qs = {
    "model_1": StubQueue("model_1", [], 0.0, now),
    "model_4": StubQueue("model_4", [100] * 100, 200.0, now),
    "model_5": StubQueue("model_5", [100] * 100, 200.0, now),
}
mapping = {0: ["model_1"], 1: ["model_4", "model_5"]}
free = {0: 40.0, 1: 40.0}
mig = pol._find_optimal_migrations(instances(mapping), qs, free, None, mapping)
check("exactly one migration", len(mig) == 1, mig)
if mig:
    name, src, dst = mig[0]
    check("moves a hot model off GPU1", name in ("model_4", "model_5") and src == 1, mig)
    check("target is GPU0", dst == 0, mig)

print("== 7. tau gates the migration ==")
# Same imbalance, but tau=0.99 demands a 99% improvement that does not exist.
pol = make_policy(tau=0.99, tpot_ms={"model_1": 10.0, "model_4": 10.0, "model_5": 10.0})
mig = pol._find_optimal_migrations(instances(mapping), qs, free, None, mapping)
check("no migration when tau is unreachable", mig == [], mig)

print("== 8. balanced load -> no migration ==")
pol = make_policy(tau=0.10, tpot_ms={"model_1": 10.0, "model_4": 10.0, "model_5": 10.0})
qs_bal = {
    "model_1": StubQueue("model_1", [100] * 100, 200.0, now),
    "model_4": StubQueue("model_4", [100] * 100, 200.0, now),
}
bal_map = {0: ["model_1"], 1: ["model_4"]}
mig = pol._find_optimal_migrations(instances(bal_map), qs_bal, free, None, bal_map)
check("symmetric placement is left alone", mig == [], mig)

print("== 9. a GPU is never emptied ==")
# GPU1 holds a single model; moving it would leave GPU1 with no scheduler.
pol = make_policy(tau=0.01, tpot_ms={"model_1": 10.0, "model_4": 10.0, "model_5": 10.0})
qs_one = {
    "model_1": StubQueue("model_1", [], 0.0, now),
    "model_4": StubQueue("model_4", [100] * 200, 400.0, now),
}
one_map = {0: ["model_1"], 1: ["model_4"]}
mig = pol._find_optimal_migrations(instances(one_map), qs_one, free, None, one_map)
check("last model on a GPU stays put", mig == [], mig)

print("== 10. insufficient target memory blocks the move ==")
pol = make_policy(tau=0.10, tpot_ms={"model_1": 10.0, "model_4": 10.0, "model_5": 10.0})
tight = {0: 1.0, 1: 1.0}   # 1 GiB free, model needs 15.08
mig = pol._find_optimal_migrations(instances(mapping), qs, tight, None, mapping)
check("no migration without room for the weights", mig == [], mig)

print("== 11. zero measured load -> no migration ==")
pol = make_policy(tau=0.10, tpot_ms={"model_1": 10.0, "model_4": 10.0, "model_5": 10.0})
qs_idle = {m: StubQueue(m, [], 0.0, now) for m in ("model_1", "model_4", "model_5")}
mig = pol._find_optimal_migrations(instances(mapping), qs_idle, free, None, mapping)
check("idle system is left alone", mig == [], mig)

print()
if FAILURES:
    print(f"FAILED: {len(FAILURES)} check(s): {FAILURES}")
    sys.exit(1)
print("ALL KVPR PLACEMENT TESTS PASSED")
