#!/usr/bin/env python3
"""Integration regression test for Algorithm-2 global runtime ordering.

Forces the cross-model sequence A(model1), B(model2), C(model1), then verifies
that the selector, Redis backend dispatch, engine admission acknowledgement,
and prefill-completion gate all observe A,B,C in that order.
"""

import json
import logging
import sys
from pathlib import Path
from threading import Event, Lock
from types import SimpleNamespace
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[2] / "prism-research"
sys.path.insert(0, str(REPO / "python"))

from sglang.multi_model.scheduling.gpu.gpu_scheduler import GPUScheduler
from sglang.multi_model.scheduling.gpu.request_queue import RequestQueue
from sglang.srt.managers.io_struct import (
    BackendAdmitReq,
    BatchRunReq,
    GenerateReqInput,
    PrefillCompleteReq,
)
from sglang.srt.managers.scheduler import Scheduler


class RecordingRedis:
    def __init__(self):
        self.sent = []

    def send_pyobj(self, key, obj):
        self.sent.append((key, obj))


class Capacity:
    def available_size(self):
        return 100_000

    def evictable_size(self):
        return 0


class SharedBarrierRedis:
    def __init__(self, requests):
        self.queues = requests
        self.next_seq = 1

    def recv_pyobj_non_block(self, key, count=1):
        queue = self.queues.setdefault(key, [])
        out, self.queues[key] = queue[:count], queue[count:]
        return out

    def get_int(self, key):
        return self.next_seq

    def compare_and_advance_int(self, key, expected):
        if self.next_seq != expected:
            return False
        self.next_seq += 1
        return True

    def close(self):
        pass


def make_backend(model, redis):
    engine = Scheduler.__new__(Scheduler)
    engine.tp_rank = 0
    engine.tp_size = 1
    engine.token_to_kv_pool = Capacity()
    engine.tree_cache = Capacity()
    engine.max_prefill_tokens = 8192
    engine.chunked_prefill_size = 8192
    engine.num_received_requests = 0
    engine.total_input_len = 0
    engine.waiting_queue = []
    engine._alg2_staged_generation_reqs = []
    engine._alg2_runtime_gate = True
    engine._alg2_admission_seq_key = "alg2-next"
    engine.model_name = model
    engine.gpu_id = 0
    engine.server_args = SimpleNamespace(
        backend_generate_request_key_prefix="backend"
    )
    engine.redis_client = redis
    return engine


def make_req(rid, model, deadline_offset):
    return GenerateReqInput(
        input_ids=[1] * 100,
        sampling_params={"max_new_tokens": 4},
        rid=rid,
        model=model,
        arrival_time=1000.0,
        slo=deadline_offset,
        prompt_len=100,
        output_len=4,
    )


def make_gpu_scheduler():
    scheduler = GPUScheduler.__new__(GPUScheduler)
    scheduler.gpu_id = 0
    scheduler._mh_runtime_gate = True
    scheduler._mh_gate_lock = Lock()
    scheduler._mh_outstanding_prefills = {}
    scheduler._mh_dispatch_seq = 0
    scheduler._mh_next_backend_admit_seq = 1
    scheduler._mh_next_prefill_start_seq = 1
    scheduler._shutdown_event = Event()
    scheduler.server_args = SimpleNamespace(
        backend_generate_request_key_prefix="backend"
    )
    scheduler.redis_client = RecordingRedis()
    return scheduler


def reserve(scheduler, req):
    scheduler._mh_dispatch_seq += 1
    req.alg2_seq = scheduler._mh_dispatch_seq
    record = {
        "seq": scheduler._mh_dispatch_seq,
        "rid": req.rid,
        "model": req.model,
        "dispatch_time": 1000.0,
        "predicted_exec_s": 0.01,
        "backend_admit_time": None,
        "start_time": None,
        "backend_admitted": False,
        "started": False,
    }
    scheduler._mh_outstanding_prefills[req.rid] = record


def main():
    queue = RequestQueue({"model1": 1, "model2": 1})
    queue.configure_moore_hodgson(
        enabled=True,
        prefill_speed={"model1": 10_000.0, "model2": 10_000.0},
    )
    expected = [
        make_req("A", "model1", 10.0),
        make_req("B", "model2", 20.0),
        make_req("C", "model1", 30.0),
    ]
    queue.add_requests(expected)
    gpu = make_gpu_scheduler()

    runtime_log_lines = []

    def capture_log(message, *args, **kwargs):
        if isinstance(message, str) and message.startswith("[PAPER-ALG2-RUNTIME] "):
            runtime_log_lines.append(message.split("] ", 1)[1])

    actual = []
    with patch(
        "sglang.multi_model.scheduling.gpu.request_queue_mh.time.time",
        return_value=1000.0,
    ), patch(
        "sglang.multi_model.scheduling.gpu.gpu_scheduler.logger.info",
        side_effect=capture_log,
    ):
        admitted = queue.admission_control(
            available_resources=float("inf"),
            model_backend_queue_lens={"model1": 0, "model2": 0},
            model_states={"model1": "activated", "model2": "activated"},
            allow_sending_when_activating=True,
            mh_outstanding_work_s=0.0,
            mh_dispatch_budget=3,
        )
        assert [req.rid for req in admitted] == ["A", "B", "C"]
        for req in admitted:
            reserve(gpu, req)
        # The whole bounded window is dispatched before any start ACK.
        gpu._send_to_backend_queue(admitted)

        for req in admitted:
            gpu._handle_mh_backend_admit(
                BackendAdmitReq(
                    rids=[req.rid], model=req.model,
                    alg2_seqs=[req.alg2_seq], admit_time=1000.05,
                    gpu_id=0,
                )
            )
            gpu._handle_mh_prefill_start(
                BatchRunReq(
                    rids=[req.rid], model=req.model, run_time=1000.1,
                    gpu_id=0, alg2_seqs=[req.alg2_seq],
                )
            )
            assert req.rid in gpu._mh_outstanding_prefills
            actual.append(req.rid)

        # Independent engines may finish in a different order.  Completion
        # removes accounting; it does not retroactively change admission order.
        for req in (expected[1], expected[0], expected[2]):
            gpu._handle_mh_prefill_complete(
                PrefillCompleteReq(
                    rids=[req.rid], model=req.model, complete_time=1000.2,
                    gpu_id=0,
                )
            )

        assert not gpu._mh_outstanding_prefills

    assert actual == ["A", "B", "C"], actual
    assert [key for key, _ in gpu.redis_client.sent] == [
        "backend:model1", "backend:model2", "backend:model1"
    ]

    events = [json.loads(line) for line in runtime_log_lines]
    assert [e["event"] for e in events] == [
        "dispatch", "dispatch", "dispatch",
        "backend_admit", "prefill_start",
        "backend_admit", "prefill_start",
        "backend_admit", "prefill_start",
        "prefill_complete", "prefill_complete", "prefill_complete",
    ]
    assert [e["actual_rids"][0] for e in events] == [
        "A", "B", "C", "A", "A", "B", "B", "C", "C", "B", "A", "C"
    ]
    assert all(e["order_ok"] for e in events)

    # Exercise the production backend fetch/staging method. Both model queues
    # are prefetched, but only the shared next sequence is released.
    shared = SharedBarrierRedis({
        "backend:model1": [expected[0], expected[2]],
        "backend:model2": [expected[1]],
    })
    model1 = make_backend("model1", shared)
    model2 = make_backend("model2", shared)
    assert [r.rid for r in model1.recv_generation_requests()] == ["A"]
    assert model2.recv_generation_requests() == []
    assert [r.rid for r in model1._alg2_staged_generation_reqs] == ["C"]
    assert [r.rid for r in model2._alg2_staged_generation_reqs] == ["B"]
    assert shared.compare_and_advance_int("alg2-next", 1)
    assert [r.rid for r in model2.recv_generation_requests()] == ["B"]
    assert shared.compare_and_advance_int("alg2-next", 2)
    assert [r.rid for r in model1.recv_generation_requests()] == ["C"]

    # The backend barrier must fail closed if an independent model attempts to
    # admit B before A, even though both were already dispatched.
    violation = make_gpu_scheduler()
    reserve(violation, expected[0])
    reserve(violation, expected[1])
    try:
        violation._handle_mh_backend_admit(
            BackendAdmitReq(
                rids=["B"], model="model2", alg2_seqs=[2],
                admit_time=1000.05, gpu_id=0,
            )
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("cross-model backend reorder was not rejected")
    assert violation._shutdown_event.is_set()

    logging.getLogger(__name__).info("runtime events: %s", events)
    print("PASS: selector and runtime admission order = A(model1), B(model2), C(model1)")
    print("PASS: independent-backend reorder is detected and fails closed")


if __name__ == "__main__":
    main()
