#!/usr/bin/env python3
"""Apply the opt-in paper-faithful-v4 path on top of paper-faithful-v3.

v4 changes three things and leaves everything else in v3 exactly as it was:

1.  The ModelService page-locks its shared host weights once at startup, so
    host->device transfers become real DMA instead of driver bounce-buffer
    staging.  Measured on this box: 11.7 -> 26.1 GB/s on a single link.
2.  When a model is migrated and a live copy is still resident on the source
    GPU -- which target-first ordering guarantees -- the target is filled from
    that copy straight over NVLink instead of from host memory.
3.  Algorithm 1 logs the whole placement plan it computed, not only the one
    migration it emitted, so planner intent and runtime effect can be compared.

Everything is behind environment switches read by the ModelService process, so
a v3 run through the same binary is bit-for-bit the v3 path.
"""
import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def replace(path, old, new, count=1, probe=None):
    text = path.read_text()
    if probe and probe in text:
        return
    if old not in text:
        if new in text:
            return
        raise RuntimeError(f"patch anchor not found in {path}: {old[:160]!r}")
    path.write_text(text.replace(old, new, count))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=str(ROOT / "prism-research"))
    ns = ap.parse_args()
    repo = Path(ns.repo).resolve()
    mm = repo / "python/sglang/multi_model"
    if not mm.is_dir():
        raise RuntimeError(f"not a prism-research checkout: {repo}")

    # bootstrap.sh installs the profiled model_info.json (cell_size and
    # model_size per model path) over the upstream copy, which lists only 24
    # models and none of the Qwen2.5 Instruct ones this study uses.  A
    # `git checkout` inside prism-research silently reverts it, and the run
    # then dies at GPU-scheduler start with "not found in the profiled model
    # info file".  Re-assert it here so the patch chain alone can rebuild a
    # clean checkout into a working tree.
    profiled_info = ROOT / "setup/model_info.json"
    if profiled_info.exists():
        shutil.copyfile(profiled_info, mm / "utils/model_info.json")

    shutil.copyfile(HERE / "parallel_loading_v4.py", mm / "parallel_loading_v4.py")
    shutil.copyfile(HERE / "kvpr_global_v4.py", mm / "scheduling/policy/kvpr_global_v4.py")

    # ---------------------------------------------------------------- service
    service = mm / "model_sevice.py"
    replace(service,
        "class ModelService:\n",
        '''_V4 = None


def _v4():
    """Lazy import so a v3 run never touches v4 code."""
    global _V4
    if _V4 is None:
        import os
        from sglang.multi_model import parallel_loading_v4 as mod
        _V4 = (mod, os.environ.get("PRISM_V4_PAGELOCK") == "1",
               os.environ.get("PRISM_V4_P2P_MIGRATION") == "1")
    return _V4


class ModelService:
''',
        probe="def _v4():")

    # Page-lock the shared host weights once, and prepare the resident registry
    # that makes a GPU-to-GPU migration possible.
    replace(service,
        "        else:\n"
        "            self.executor = ThreadPoolExecutor(max_workers=max_threads)\n"
        "            self.streams = {i: torch.cuda.Stream(device=f\"cuda:{i}\") for i in gpu_ids}\n",
        "        else:\n"
        "            self.executor = ThreadPoolExecutor(max_workers=max_threads)\n"
        "            self.streams = {i: torch.cuda.Stream(device=f\"cuda:{i}\") for i in gpu_ids}\n"
        "        mod, pagelock, p2p = _v4()\n"
        "        self.v4_pool = mod.StreamPool(gpu_ids, per_gpu=3)\n"
        "        self.v4_p2p = p2p\n"
        "        # model_key -> (gpu_id, state_dict) of the copy that is live on a\n"
        "        # GPU right now.  A migration reads its weights from here.\n"
        "        self.v4_resident = {}\n"
        "        self.v4_peer = mod.enable_peer_access(gpu_ids)\n"
        "        if pagelock:\n"
        "            # Only this worker's own models: the router sends a given\n"
        "            # model to a fixed worker, so locking all of them in every\n"
        "            # worker would pin the same pages N times for nothing.\n"
        "            import os as _os, zlib as _zlib\n"
        "            _n = int(_os.environ.get(\"PRISM_V4_NUM_SERVICE_WORKERS\", \"1\"))\n"
        "            _own = {\n"
        "                k: i % _n for i, k in enumerate(sorted(self.model_dict))\n"
        "            } if _n > 1 else {}\n"
        "            summary = {\n"
        "                k: mod.register_host_memory(m.state_dict())\n"
        "                for k, m in self.model_dict.items()\n"
        "                if _n <= 1 or _own.get(k) == self.service_id\n"
        "            }\n"
        "            logging.info(\n"
        "                f\"[PAPER-LOAD-V4] worker {self.service_id}/{_n} \"\n"
        "                f\"host page-lock: {summary}\")\n"
        "        logging.info(f\"[PAPER-LOAD-V4] peer access: {self.v4_peer} p2p={p2p}\")\n",
        probe="self.v4_resident = {}")

    # Release control message: the engine tells us when its copy is gone, so we
    # never pin another process's freed GPU memory.
    replace(service,
        "    model_service = ModelService(\n"
        "        cpu_model_dict, input_queue, output_queues, max_threads, gpu_ids, num_shards\n"
        "    )\n",
        "    model_service = ModelService(\n"
        "        cpu_model_dict, input_queue, output_queues, max_threads, gpu_ids,\n"
        "        num_shards, instance,\n"
        "    )\n",
        probe="num_shards, instance,")

    # The model service runs several worker processes sharing ONE queue, so a
    # model can be loaded by worker A and later reloaded by worker B -- and the
    # registry of what is resident where lives inside a worker.  Give each
    # worker its own queue and route by model, so every load of a given model
    # lands on the same worker and its registry stays coherent.  Different
    # models still go to different workers, so cross-model concurrency is
    # unchanged.  Routing is applied to every arm, not just v4, so it is not a
    # difference between them.
    replace(service,
        "class ModelService:\n",
        '''class _ModelServiceRouter:
    """Queue facade that sends a model to a fixed worker.

    Fixed, because the record of which GPU currently holds a model lives
    inside a worker: if two loads of the same model reach different
    workers, the migration finds no source and silently falls back to
    host memory.

    The assignment is built once from the sorted model list and travels
    with this object when it is pickled into the spawned processes, so
    every process agrees on it -- ``hash()`` cannot be used here, being
    salted per process.  Sorted round-robin also spreads models evenly,
    which a checksum modulo does not: on this six-model set crc32
    collides onto four workers and leaves two idle.
    """

    def __init__(self, queues, model_keys=()):
        self.queues = queues
        self.mapping = {
            key: i % len(queues) for i, key in enumerate(sorted(model_keys))
        }

    def _index(self, model_key):
        key = str(model_key)
        if key in self.mapping:
            return self.mapping[key]
        # Anything absent at construction still has to land somewhere
        # deterministic.
        return zlib.crc32(key.encode()) % len(self.queues)

    def put(self, item, *args, **kwargs):
        key = item[1] if item and item[0] == "__release__" else item[0]
        self.queues[self._index(key)].put(item, *args, **kwargs)

    def __len__(self):
        return len(self.queues)


class ModelService:
''',
        probe="class _ModelServiceRouter:")

    replace(service, "import gc\n", "import gc\nimport zlib\n", probe="import zlib")

    server = mm / "multi_model_server.py"
    replace(server,
        """    input_queue = torch.multiprocessing.Queue()
    output_queues = {
        engine_id: torch.multiprocessing.Queue() for engine_id in engine_ids
    }
    max_loading_threads = min(4, num_devices)
    num_shards = 1
    num_model_service_workers = multi_model_server_args.num_model_service_workers
    from sglang.multi_model.model_sevice import run_model_service
    for service_worker_id in range(num_model_service_workers):
        p = torch.multiprocessing.Process(
            target=run_model_service,
            args=(
                multi_model_server_args,
                cpu_model_dict,
                input_queue,
                output_queues,""",
        """    output_queues = {
        engine_id: torch.multiprocessing.Queue() for engine_id in engine_ids
    }
    max_loading_threads = min(4, num_devices)
    num_shards = 1
    num_model_service_workers = multi_model_server_args.num_model_service_workers
    from sglang.multi_model.model_sevice import _ModelServiceRouter, run_model_service
    # One queue per worker; the router picks a worker per model.
    worker_queues = [
        torch.multiprocessing.Queue() for _ in range(num_model_service_workers)
    ]
    input_queue = _ModelServiceRouter(worker_queues, cpu_model_dict.keys())
    os.environ["PRISM_V4_NUM_SERVICE_WORKERS"] = str(num_model_service_workers)
    for service_worker_id in range(num_model_service_workers):
        p = torch.multiprocessing.Process(
            target=run_model_service,
            args=(
                multi_model_server_args,
                cpu_model_dict,
                worker_queues[service_worker_id],
                output_queues,""",
        probe="_ModelServiceRouter(worker_queues, cpu_model_dict.keys())")

    # A release arrives as ("__release__", model_key, gpu_id, None): the engine
    # has dropped its GPU copy, so ours must go too.  Handled after the Empty
    # guard so the queue-drain path is untouched.
    replace(service,
        '            logging.info(f"Model key: {model_key}, engine id: {engine_id}, target gpu id: {target_gpu_id}")\n',
        '            if model_key == "__release__":\n'
        '                released_key, released_gpu = engine_id, target_gpu_id\n'
        '                held = self.v4_resident.get(released_key)\n'
        '                if held is not None and held[0] == released_gpu:\n'
        '                    del self.v4_resident[released_key]\n'
        '                    gc.collect()\n'
        '                    torch.cuda.empty_cache()\n'
        '                    logging.info(\n'
        '                        f"[PAPER-LOAD-V4] released resident {released_key} "\n'
        '                        f"on gpu {released_gpu}")\n'
        '                continue\n'
        '            logging.info(f"Model key: {model_key}, engine id: {engine_id}, target gpu id: {target_gpu_id}")\n',
        probe='if model_key == "__release__":')

    replace(service,
        '''                else:
                    futures = multi_thread_copy_model_to_gpu(
                        self.model_dict[model_key].state_dict(),
                        gpu_model.state_dict(),
                        target_gpu_id,
                        self.executor,
                        len(self.gpu_ids),
                        self.max_threads,
                        self.streams,
                        1,
                        0,
                    )
                    self.output_queue[engine_id].put("success")
                    for future in futures:
                        future.result()
                    t1 = time.perf_counter()''',
        '''                else:
                    mod, _pagelock, _p2p = _v4()
                    resident = self.v4_resident.get(model_key)
                    source_sd = None
                    if (self.v4_p2p and resident is not None
                            and resident[0] != target_gpu_id
                            and self.v4_peer.get(f"{resident[0]}->{target_gpu_id}")):
                        # Target-first ordering keeps the source copy alive, so
                        # its weights can be pulled straight across NVLink.
                        source_sd = resident[1]
                    self.output_queue[engine_id].put("success")
                    mod.copy_model_to_gpu_v4(
                        self.model_dict[model_key].state_dict(),
                        gpu_model.state_dict(),
                        target_gpu_id,
                        self.executor,
                        list(self.gpu_ids),
                        self.v4_pool,
                        source_state_dict=source_sd,
                        host_registered=_pagelock,
                        tag=f"{model_key}|engine={engine_id}|src={None if source_sd is None else resident[0]}",
                    )
                    if self.v4_p2p:
                        self.v4_resident[model_key] = (target_gpu_id, gpu_model.state_dict())
                    t1 = time.perf_counter()''',
        probe="mod.copy_model_to_gpu_v4(")

    # The engine keeps its own reference; ours must not outlive it.
    replace(service,
        "                del gpu_model\n",
        "                if not self.v4_p2p:\n"
        "                    del gpu_model\n",
        probe="if not self.v4_p2p:")

    # ----------------------------------------------------------------- engine
    runner = repo / "python/sglang/srt/model_executor/model_runner.py"
    replace(runner,
        '''    def delete_gpu_model(self):
        if hasattr(self, "model") and self.model is not None:
            del self.model
            self.model = None''',
        '''    def delete_gpu_model(self):
        if hasattr(self, "model") and self.model is not None:
            # Tell the model service to drop its reference first. It may hold
            # this model's weights as a migration source, and a reference held
            # across process boundaries would keep the memory alive after the
            # engine has released it.
            import os as _os
            if _os.environ.get("PRISM_V4_P2P_MIGRATION") == "1" and self.input_queue is not None:
                try:
                    self.input_queue.put(
                        ("__release__", self.model_path, self.gpu_id, None))
                except Exception:
                    pass
            del self.model
            self.model = None''',
        probe="__release__")

    # flashinfer sizes its prefill scratch buffer from FLASHINFER_WORKSPACE_SIZE,
    # but reads it straight out of the environment as a string -- so setting the
    # variable at all makes torch.empty() fail.  The default 384 MiB is too small
    # for Qwen2.5-7B's GQA ratio here (it asks for ~420 MiB and the run dies with
    # "Failed to allocate memory for batch_prefill_tmp_v"), so the variable has to
    # work.  Applies to every arm.
    gcfg = repo / "python/sglang/global_config.py"
    replace(gcfg,
        "        self.flashinfer_workspace_size = os.environ.get(\n"
        "            \"FLASHINFER_WORKSPACE_SIZE\", 384 * 1024 * 1024\n"
        "        )\n",
        "        self.flashinfer_workspace_size = int(\n"
        "            os.environ.get(\"FLASHINFER_WORKSPACE_SIZE\", 384 * 1024 * 1024)\n"
        "        )\n",
        probe="self.flashinfer_workspace_size = int(")

    # -------------------------------------------------------------- policy v4
    controller = mm / "scheduling/controller_global.py"
    replace(controller,
        "from sglang.multi_model.scheduling.policy.kvpr_global_v3 import KVPRGlobalPolicyV3\n",
        "from sglang.multi_model.scheduling.policy.kvpr_global_v3 import KVPRGlobalPolicyV3\n"
        "from sglang.multi_model.scheduling.policy.kvpr_global_v4 import KVPRGlobalPolicyV4\n",
        probe="import KVPRGlobalPolicyV4")
    replace(controller,
        '        elif self.server_args.policy == "kvpr-global-v3":\n',
        '        elif self.server_args.policy in ("kvpr-global-v3", "kvpr-global-v4"):\n',
        probe='in ("kvpr-global-v3", "kvpr-global-v4")')
    replace(controller,
        '            self.policy = KVPRGlobalPolicyV3(\n',
        '            _cls = (KVPRGlobalPolicyV4\n'
        '                    if self.server_args.policy == "kvpr-global-v4"\n'
        '                    else KVPRGlobalPolicyV3)\n'
        '            self.policy = _cls(\n',
        probe="_cls = (KVPRGlobalPolicyV4")

    # Time every control action.  Migration latency, target-ready time and
    # service downtime are all derived from these records; without them a
    # migration is only a count.  Applied to the shared code path, so every arm
    # is instrumented identically.
    replace(controller,
        """                future_to_action = {
                    threads.submit(
                        action.execute,
                        self.server_args.url(),
                        self.model_instance_state_dict,
                        600.0 if overlap_migration else None,
                    ): action
                    for action in batch
                }""",
        """                def _timed_execute(action):
                    _t0 = time.time()
                    _ok = False
                    try:
                        _result = action.execute(
                            self.server_args.url(),
                            self.model_instance_state_dict,
                            600.0 if overlap_migration else None,
                        )
                        _ok = _result is not False
                        return _result
                    finally:
                        _t1 = time.time()
                        logger.info("[PAPER-ACTION-V4] " + json.dumps({
                            "action": type(action).__name__,
                            "model": getattr(action, "model_name", None),
                            "gpu_id": getattr(action, "gpu_id", None),
                            "instance_idx": getattr(action, "instance_idx", None),
                            "start": _t0, "end": _t1, "duration_s": _t1 - _t0,
                            "success": _ok,
                        }, default=str))

                future_to_action = {
                    threads.submit(_timed_execute, action): action
                    for action in batch
                }""",
        probe="[PAPER-ACTION-V4]")

    args = mm / "multi_model_server_args.py"
    replace(args, '                "kvpr-global-v3",\n',
        '                "kvpr-global-v3",\n                "kvpr-global-v4",\n',
        probe='                "kvpr-global-v4",')

    checks = {
        service: ["def _v4():", "self.v4_resident = {}", "mod.copy_model_to_gpu_v4(",
                  '__release__', "if not self.v4_p2p:", "class _ModelServiceRouter:"],
        mm / "multi_model_server.py": ["_ModelServiceRouter(worker_queues, cpu_model_dict.keys())"],
        runner: ["__release__"],
        controller: ["KVPRGlobalPolicyV4", 'in ("kvpr-global-v3", "kvpr-global-v4")',
                     "[PAPER-ACTION-V4]"],
        args: ["kvpr-global-v4"],
        repo / "python/sglang/global_config.py": ["self.flashinfer_workspace_size = int("],
        mm / "parallel_loading_v4.py": ["register_host_memory"],
        mm / "utils/model_info.json": ["Qwen/Qwen2.5-1.5B-Instruct"],
        mm / "scheduling/policy/kvpr_global_v4.py": ["KVPRGlobalPolicyV4"],
    }
    missing = [f"{p}: {n}" for p, needles in checks.items()
               for n in needles if n not in p.read_text()]
    if missing:
        raise RuntimeError("v4 patch verification failed:\n" + "\n".join(missing))
    print(f"paper-faithful-v4 applied ({len(checks)} files, all landing points verified)")


if __name__ == "__main__":
    main()
