#!/usr/bin/env python3
"""V5_2 instrumentation.  Measurement only -- no behaviour is changed.

Two costs are unexplained after the v4/v5 sweeps, and both are measured here
rather than guessed at.

(a) Turning Algorithm 2 on costs 39% of goodput at steady 8 req/s while it
    defers nothing at all (1161 of 1161 eligible selected, 0 deferred).  So the
    cost is not its admission decisions.  The GPU scheduler's loop is the only
    other thing that differs, so this times the loop: how long an iteration
    takes, and how that time splits between reading available memory, the Redis
    queue-length round trips, and admission_control itself.  If the loop period
    stretches, newly arrived requests wait longer before dispatch, which would
    show up as exactly the uniform latency penalty observed.

(b) A deactivation takes 5.4 s as the server sees it while the engine's own
    teardown is 0.96 s -- 82% of it is somewhere in between.  This stamps every
    hop the request passes through so the gap can be attributed instead of
    speculated about.

Both accumulate counters and emit a JSON line every few seconds, so the
instrument itself stays far below what it measures.
"""
import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def replace(path, old, new, probe=None):
    text = path.read_text()
    if probe and probe in text:
        return
    if old not in text:
        if new in text:
            return
        raise RuntimeError(f"anchor not found in {path}: {old[:140]!r}")
    path.write_text(text.replace(old, new, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=str(ROOT / "prism-research"))
    ns = ap.parse_args()
    repo = Path(ns.repo).resolve()
    mm = repo / "python/sglang/multi_model"
    gpu = mm / "scheduling/gpu/gpu_scheduler.py"
    handler = mm / "request_handler_worker_pool.py"

    # ---------------------------------------------------------------- (a) loop
    replace(gpu,
        "        try:\n"
        "            while not self._shutdown_event.is_set():\n"
        "                available_kv_cache_memory = (\n"
        "                    self.resource_manager.get_available_kv_cache_memory()\n"
        "                )\n",
        "        # V5_2 (a): time the loop, since Algorithm 2 costs goodput while\n"
        "        # deferring nothing -- the loop period is the other thing that\n"
        "        # differs between the arms.\n"
        "        import json as _json\n"
        "        _v5 = {\"iters\": 0, \"t_mem\": 0.0, \"t_redis\": 0.0, \"t_admit\": 0.0,\n"
        "               \"t_send\": 0.0, \"t_iter\": 0.0, \"admitted\": 0, \"iter_max\": 0.0}\n"
        "        _v5_last = time.time()\n"
        "        try:\n"
        "            while not self._shutdown_event.is_set():\n"
        "                _t0 = time.perf_counter()\n"
        "                available_kv_cache_memory = (\n"
        "                    self.resource_manager.get_available_kv_cache_memory()\n"
        "                )\n"
        "                _t1 = time.perf_counter()\n",
        probe='"iters": 0, "t_mem"')

    replace(gpu,
        "                    for model_name in active_models\n"
        "                }\n\n"
        "                # Pop requests from priority queue with admission control\n",
        "                    for model_name in active_models\n"
        "                }\n"
        "                _t2 = time.perf_counter()\n\n"
        "                # Pop requests from priority queue with admission control\n",
        probe="                _t2 = time.perf_counter()")

    replace(gpu,
        "                for model_name, reqs in reqs_can_be_admitted.items():\n"
        "                    logger.info(f\"Admitted {len(reqs)} requests for {model_name}\")\n"
        "                self._send_to_backend_queue(reqs_can_be_admitted)\n\n"
        "                time.sleep(0.01)\n",
        "                _t3 = time.perf_counter()\n"
        "                for model_name, reqs in reqs_can_be_admitted.items():\n"
        "                    logger.info(f\"Admitted {len(reqs)} requests for {model_name}\")\n"
        "                self._send_to_backend_queue(reqs_can_be_admitted)\n"
        "                _t4 = time.perf_counter()\n"
        "                _v5[\"iters\"] += 1\n"
        "                _v5[\"t_mem\"] += _t1 - _t0\n"
        "                _v5[\"t_redis\"] += _t2 - _t1\n"
        "                _v5[\"t_admit\"] += _t3 - _t2\n"
        "                _v5[\"t_send\"] += _t4 - _t3\n"
        "                _v5[\"t_iter\"] += _t4 - _t0\n"
        "                _v5[\"iter_max\"] = max(_v5[\"iter_max\"], _t4 - _t0)\n"
        "                _v5[\"admitted\"] += sum(len(r) for r in reqs_can_be_admitted.values())\n"
        "                if time.time() - _v5_last >= 5.0:\n"
        "                    logger.info(\"[V5-LOOP] \" + _json.dumps(\n"
        "                        dict(_v5, gpu_id=self.gpu_id, window_s=time.time() - _v5_last)))\n"
        "                    _v5 = {k: (0.0 if isinstance(v, float) else 0) for k, v in _v5.items()}\n"
        "                    _v5_last = time.time()\n\n"
        "                time.sleep(0.01)\n",
        probe="[V5-LOOP]")

    # ------------------------------------------------------------- (b) hops
    replace(handler,
        "    async def _send_req_and_wait_for_response(\n"
        "        self, req: Union[ActivateReqInput, DeactivateReqInput]\n"
        "    ):\n",
        "    async def _send_req_and_wait_for_response(\n"
        "        self, req: Union[ActivateReqInput, DeactivateReqInput]\n"
        "    ):\n"
        "        # V5_2 (b): 82% of a deactivation sits between this send and the\n"
        "        # engine's own teardown.  Stamp the hops so it can be attributed.\n"
        "        import json as _json, time as _time\n"
        "        _v5_t0 = _time.time()\n",
        probe="_v5_t0 = _time.time()")

    replace(handler,
        "        self.rid_to_state[rid] = state\n"
        "        self._send_req_to_gpu_scheduler(req)\n\n"
        "        # wait for the response\n"
        "        try:\n"
        "            await state.event.wait()\n",
        "        self.rid_to_state[rid] = state\n"
        "        self._send_req_to_gpu_scheduler(req)\n"
        "        _v5_sent = _time.time()\n\n"
        "        # wait for the response\n"
        "        try:\n"
        "            await state.event.wait()\n"
        "            _v5_done = _time.time()\n"
        "            logger.info(\"[V5-HOP] \" + _json.dumps({\n"
        "                \"action\": type(req).__name__,\n"
        "                \"model\": getattr(req, \"model_name\", None),\n"
        "                \"gpu_id\": getattr(req, \"gpu_id\", None),\n"
        "                \"register_s\": _v5_sent - _v5_t0,\n"
        "                \"wait_for_engine_s\": _v5_done - _v5_sent,\n"
        "                \"total_s\": _v5_done - _v5_t0,\n"
        "            }))\n",
        probe="[V5-HOP]")

    checks = {gpu: ["[V5-LOOP]", "_v5[\"t_admit\"]"],
              handler: ["[V5-HOP]", "wait_for_engine_s"]}
    missing = [f"{p}: {n}" for p, ns_ in checks.items() for n in ns_ if n not in p.read_text()]
    if missing:
        raise RuntimeError("v5_2 instrumentation verification failed:\n" + "\n".join(missing))
    print("paper-faithful-v5_2 instrumentation applied (measurement only)")


if __name__ == "__main__":
    main()
