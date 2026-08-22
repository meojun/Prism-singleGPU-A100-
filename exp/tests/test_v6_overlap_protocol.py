#!/usr/bin/env python3
"""CPU-only protocol checks for V6 prepare/commit activation."""

import unittest
import inspect
import queue
from types import SimpleNamespace
from unittest.mock import patch

from sglang.multi_model.scheduling.action import ActivateAction
from sglang.multi_model.scheduling.gpu.gpu_scheduler import GPUScheduler
from sglang.multi_model.scheduling.gpu.worker_pool import WorkerPool
from sglang.multi_model.scheduling.state import ModelState
from sglang.srt.managers.scheduler import Scheduler


class FakeResponse:
    status_code = 200

    def __init__(self, success=True):
        self.success = success

    def json(self):
        return {"success": self.success, "memory_usage": None}

    def raise_for_status(self):
        raise AssertionError("unexpected HTTP error")


class FakeInstance:
    def __init__(self):
        self.state = ModelState.INACTIVE
        self.memory_usage = None
        self.gpu_ids = []
        self.activate_calls = 0

    def on_activate(self, memory_usage, gpu_id=None):
        self.activate_calls += 1
        self.state = ModelState.ACTIVE
        self.memory_usage = memory_usage
        self.gpu_ids = [gpu_id]


class FakeSocket:
    def __init__(self):
        self.messages = []

    def send_pyobj(self, value):
        self.messages.append(value)


class FakeRelay:
    def __init__(self):
        self.sent = []
        self.kv_replies = {"engine-1": queue.Queue()}
        self.kv_replies["engine-1"].put(("__kv_clear_ack__", 3))

    def put(self, value):
        self.sent.append(value)


class FakeStashRelay:
    def __init__(self, valid_ack=True):
        self.sent = []
        self.valid_ack = valid_ack
        self.kv_replies = {"engine-1": queue.Queue()}

    def put(self, value):
        self.sent.append(value)
        if value[0] == "__kv_stash__":
            capsules, _source_engine = value[3]
            ack = (
                "__kv_stash_ack__" if self.valid_ack else "__bad_ack__",
                value[1],
                len(capsules),
            )
            self.kv_replies["engine-1"].put(ack)


class V6OverlapProtocolTests(unittest.TestCase):
    def execute(self, phase, success=True):
        instance = FakeInstance()
        action = ActivateAction(
            model_name="model_1", instance_idx=0, gpu_id=1, phase=phase
        )
        with patch(
            "sglang.multi_model.scheduling.action.requests.post",
            return_value=FakeResponse(success),
        ) as post:
            result = action.execute("http://controller", {"model_1": [instance]})
        return result, instance, post.call_args.kwargs["json"]

    def test_prepare_reserves_target_without_exposing_active_state(self):
        result, instance, payload = self.execute("prepare")
        self.assertTrue(result)
        self.assertEqual(instance.state, ModelState.ACTIVATING)
        self.assertEqual(instance.activate_calls, 0)
        self.assertEqual(payload["phase"], "prepare")

    def test_commit_is_the_only_phase_that_exposes_active_state(self):
        result, instance, payload = self.execute("commit")
        self.assertTrue(result)
        self.assertEqual(instance.state, ModelState.ACTIVE)
        self.assertEqual(instance.activate_calls, 1)
        self.assertEqual(payload["phase"], "commit")

    def test_failed_prepare_does_not_change_controller_state(self):
        result, instance, _ = self.execute("prepare", success=False)
        self.assertFalse(result)
        self.assertEqual(instance.state, ModelState.INACTIVE)

    def test_commit_reuses_the_worker_assigned_by_prepare(self):
        pool = WorkerPool.__new__(WorkerPool)
        pool.gpu_id = 1
        pool._model_to_worker = {"model_1": 7}
        socket = FakeSocket()
        pool._worker_to_ipc_name = {7: socket}
        req = SimpleNamespace(model_name="model_1", gpu_id=1, phase="commit")

        self.assertTrue(pool.handle_activate_model(req))
        self.assertEqual(socket.messages, [req])
        self.assertEqual(pool._model_to_worker, {"model_1": 7})

    def test_commit_without_prepare_is_rejected(self):
        pool = WorkerPool.__new__(WorkerPool)
        pool.gpu_id = 1
        pool._model_to_worker = {}
        pool._worker_to_ipc_name = {}
        req = SimpleNamespace(model_name="model_1", gpu_id=1, phase="commit")
        self.assertFalse(pool.handle_activate_model(req))

    def test_prepared_target_is_not_eligible_to_consume_requests(self):
        scheduler = SimpleNamespace(_model_states={
            "source_model": "activated",
            "prepared_target": "prepared",
            "committing_target": "committing",
        })
        self.assertEqual(
            GPUScheduler._get_active_or_activating_model_names(scheduler),
            ["source_model"],
        )

    def test_prepare_acknowledges_stale_stash_clear(self):
        relay = FakeRelay()
        scheduler = SimpleNamespace(
            _v6_kv_enabled=lambda: True,
            tp_worker=SimpleNamespace(model_runner=SimpleNamespace(
                input_queue=relay, engine_id="engine-1"
            )),
        )
        self.assertTrue(
            Scheduler._v6_clear_stale_kv(scheduler, "model/path")
        )
        self.assertEqual(
            relay.sent,
            [("__kv_clear__", "model/path", None, "engine-1")],
        )

    def test_stale_clear_precedes_target_model_load(self):
        source = inspect.getsource(Scheduler.handle_activate_request)
        self.assertLess(
            source.index("self._v6_clear_stale_kv("),
            source.index("self.tp_worker.activate_model_runner("),
        )

    def test_source_stash_waits_for_relay_ack_before_removing_requests(self):
        relay = FakeStashRelay()
        capsule = SimpleNamespace(rid="req-1", num_tokens=7, nbytes=56)
        waiting = [SimpleNamespace(rid="req-1")]
        scheduler = SimpleNamespace(
            _v6_kv_enabled=lambda: True,
            tp_worker=SimpleNamespace(model_runner=SimpleNamespace(
                input_queue=relay,
                engine_id="engine-1",
                model_path="model/path",
            )),
            _v6_captured=[capsule],
            _v6_capture_failures=0,
            waiting_queue=waiting,
            gpu_id=0,
        )
        self.assertTrue(Scheduler._v6_stash_captured(scheduler))
        self.assertEqual(waiting, [])
        self.assertEqual(
            relay.sent,
            [("__kv_stash__", "model/path", 0, ([capsule], "engine-1"))],
        )

    def test_bad_stash_ack_keeps_requests_for_source_recovery(self):
        relay = FakeStashRelay(valid_ack=False)
        capsule = SimpleNamespace(rid="req-1", num_tokens=7, nbytes=56)
        waiting = [SimpleNamespace(rid="req-1")]
        scheduler = SimpleNamespace(
            _v6_kv_enabled=lambda: True,
            tp_worker=SimpleNamespace(model_runner=SimpleNamespace(
                input_queue=relay,
                engine_id="engine-1",
                model_path="model/path",
            )),
            _v6_captured=[capsule],
            _v6_capture_failures=0,
            waiting_queue=waiting,
            gpu_id=0,
        )
        self.assertFalse(Scheduler._v6_stash_captured(scheduler))
        self.assertEqual([req.rid for req in waiting], ["req-1"])


if __name__ == "__main__":
    unittest.main()
