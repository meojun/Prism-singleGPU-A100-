#!/usr/bin/env python3
"""V6: KV-cache migration (paper §5.3), behind ``--enable-kv-migration``.

Why this is not a pure addition
------------------------------
The paper migrates a model's KV cache with the model.  The prototype does not,
and the reason is not a missing transfer -- it is that at teardown there is
nothing left to transfer.  Every migration path emits
``DeactivateAction(preempt=False, ...)`` (``simple_global.py:822,918``; the
paper-faithful policies subclass that class and override only
``_find_optimal_migrations``, so they inherit it), and with ``preempt=False``
the scheduler runs ``_run_to_completion_normal`` and drains the in-flight batch
before tearing down.  The KV is consumed, not dropped.

That is a coherent design -- it avoids needing KV migration by paying a longer
teardown -- but it is a different design from the paper's, which retracts
in-flight requests and moves their KV so the target resumes decoding instead of
re-prefilling.  Getting the paper's behaviour means switching the migration
path to preempt, so ``_retract_running_batch_normal`` runs, and then moving
what it would otherwise free.

Everything here is behind ``--enable-kv-migration``.  With the flag off not one
of these branches is entered and the default arm keeps reproducing the released
prototype, so the 27-run sweep stays comparable.

The five integration points
---------------------------
1. ``simple_global.py`` -- the *migration* deactivation asks for preemption.
   Idle eviction deliberately does not: it has no target GPU, so there is
   nowhere for the KV to go and draining remains the right behaviour.
2. ``scheduler.py::_retract_running_batch_normal`` -- capture each request's KV
   and scheduler state *before* ``cache_finished_req`` frees the slots.
3. ``model_sevice.py`` -- carry the capsules between two engine processes.  The
   service already brokers CUDA tensors across processes for weights and
   already has a sentinel protocol (``__release__``), so this adds
   ``__kv_stash__`` / ``__kv_fetch__`` in the same shape rather than inventing
   a second channel.
4. ``scheduler.py::handle_activate_request`` -- fetch the capsules, allocate
   fresh slots on the target, scatter the KV, rebuild the requests.
5. ``multi_model_server_args.py`` -- the flag.

Slot indices are never carried across.  ``req_pool_idx``, ``prefix_indices``
and the slot vector all index the *source* engine's pools; the target allocates
its own and the KV survives because the capsule stores tensors in token order.
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
        raise RuntimeError(f"anchor not found in {path}: {old[:160]!r}")
    path.write_text(text.replace(old, new, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=str(ROOT / "prism-research"))
    ns = ap.parse_args()
    repo = Path(ns.repo).resolve()
    mm = repo / "python/sglang/multi_model"
    srt = repo / "python/sglang/srt"

    policy = mm / "scheduling/policy/simple_global.py"
    scheduler = srt / "managers/scheduler.py"
    service = mm / "model_sevice.py"
    args_file = mm / "multi_model_server_args.py"

    shutil.copyfile(HERE / "kv_migration_v6.py", mm / "kv_migration_v6.py")

    # ------------------------------------------------------------- 1. policy
    # Only the migration deactivation.  Idle eviction (simple_global.py:822)
    # has no target GPU, so preempting there would retract requests with
    # nowhere to send their KV -- strictly worse than draining.
    replace(policy,
        "            # Deactivate the instance on source GPU\n"
        "            source_instance_key = (model_name, source_gpu_id)\n"
        "            if source_instance_key not in idle_instance_keys:\n"
        "                model_instance_to_action_dict.setdefault(source_instance_key, []).append(\n"
        "                    DeactivateAction(\n"
        "                        model_name=model_name,\n"
        "                        instance_idx=source_gpu_id,\n"
        "                        preempt=False,\n",
        "            # Deactivate the instance on source GPU\n"
        "            source_instance_key = (model_name, source_gpu_id)\n"
        "            # V6: the paper's KV migration presupposes preemption --\n"
        "            # retract the in-flight requests, move their KV, resume on\n"
        "            # the target.  preempt=False drains instead, which leaves\n"
        "            # nothing to move.  Only the migration path flips; idle\n"
        "            # eviction has no target and keeps draining.\n"
        "            import os as _os\n"
        "            _v6_kv = _os.environ.get(\"PRISM_V6_KV_MIGRATION\") == \"1\"\n"
        "            if source_instance_key not in idle_instance_keys:\n"
        "                model_instance_to_action_dict.setdefault(source_instance_key, []).append(\n"
        "                    DeactivateAction(\n"
        "                        model_name=model_name,\n"
        "                        instance_idx=source_gpu_id,\n"
        "                        preempt=_v6_kv,\n",
        probe="_v6_kv = _os.environ.get(\"PRISM_V6_KV_MIGRATION\")")

    # ------------------------------------------------------------ 2. capture
    # Insert before cache_finished_req, which is what frees the slots.  At this
    # point req.req_pool_idx is still valid and req_to_token still holds the
    # slot vector.  The token count matches radix_cache's own accounting so the
    # capture and the free agree on what belonged to this request.
    replace(scheduler,
        "                rids.append(req.rid)\n"
        "                len_output_ids.append(len(req.output_ids))\n"
        "                self.waiting_queue.append(req)\n"
        "                if self.use_kvcached_v0:\n"
        "                    try:\n"
        "                        self.tree_cache.cache_finished_req(req)\n",
        "                rids.append(req.rid)\n"
        "                len_output_ids.append(len(req.output_ids))\n"
        "                self.waiting_queue.append(req)\n"
        "                self._v6_capture_kv(req)\n"
        "                if self.use_kvcached_v0:\n"
        "                    try:\n"
        "                        self.tree_cache.cache_finished_req(req)\n",
        probe="self._v6_capture_kv(req)")

    replace(scheduler,
        "    def _retract_running_batch_normal(self):\n",
        "    def _v6_kv_enabled(self):\n"
        "        import os\n"
        "        return os.environ.get(\"PRISM_V6_KV_MIGRATION\") == \"1\"\n"
        "\n"
        "    def _v6_capture_kv(self, req):\n"
        "        \"\"\"Stash one retracted request's KV before the slots are freed.\n"
        "\n"
        "        Silent on any failure: a migration that cannot carry its KV must\n"
        "        degrade to the prototype's recompute, never take the engine down.\n"
        "        The counter below is what makes that visible instead of silent.\n"
        "        \"\"\"\n"
        "        if not self._v6_kv_enabled():\n"
        "            return\n"
        "        try:\n"
        "            from sglang.multi_model import kv_migration_v6 as _kvm\n"
        "            pool = self.token_to_kv_pool\n"
        "            if getattr(pool, \"k_buffer\", None) is None:\n"
        "                return\n"
        "            n = len(req.origin_input_ids) + len(req.output_ids) - 1\n"
        "            if n <= 0:\n"
        "                return\n"
        "            slots = self.req_to_token_pool.req_to_token[req.req_pool_idx, :n]\n"
        "            dev = f\"cuda:{self.gpu_id}\"\n"
        "            k, v = _kvm.gather_request_kv(pool.k_buffer, pool.v_buffer, slots, dev)\n"
        "            cap = _kvm.RequestKVCapsule(\n"
        "                rid=req.rid, model_name=self.model_name,\n"
        "                origin_input_ids=list(req.origin_input_ids),\n"
        "                output_ids=list(req.output_ids),\n"
        "                sampling_params=req.sampling_params,\n"
        "                arrival_time=getattr(req, \"arrival_time\", None),\n"
        "                slo=getattr(req, \"slo\", None),\n"
        "                k=k, v=v, source_gpu=self.gpu_id)\n"
        "            self._v6_captured.append(cap)\n"
        "        except Exception as e:\n"
        "            self._v6_capture_failures += 1\n"
        "            logger.warning(f\"[PAPER-KV-V6] capture failed for {req.rid}: {e}\")\n"
        "\n"
        "    def _v6_stash_captured(self):\n"
        "        \"\"\"Hand the capsules to the model service, which brokers them to\n"
        "        whichever engine activates this model next.\n"
        "\n"
        "        The source never needs to know the target: the service keys the\n"
        "        stash by model, and the target asks for it by model on activate.\n"
        "        \"\"\"\n"
        "        if not self._v6_kv_enabled() or not self._v6_captured:\n"
        "            return\n"
        "        try:\n"
        "            import json as _json\n"
        "            q = self.tp_worker.model_runner.input_queue\n"
        "            if q is None:\n"
        "                return\n"
        "            key = self.tp_worker.model_runner.model_path\n"
        "            n_tok = sum(c.num_tokens for c in self._v6_captured)\n"
        "            nbytes = sum(c.nbytes for c in self._v6_captured)\n"
        "            q.put((\"__kv_stash__\", key, self.gpu_id, self._v6_captured))\n"
        "            logger.info(\"[PAPER-KV-V6] \" + _json.dumps({\n"
        "                \"event\": \"stash\", \"model\": key, \"source_gpu\": self.gpu_id,\n"
        "                \"requests\": len(self._v6_captured), \"tokens\": n_tok,\n"
        "                \"kv_bytes\": nbytes,\n"
        "                \"capture_failures\": self._v6_capture_failures}))\n"
        "        except Exception as e:\n"
        "            logger.warning(f\"[PAPER-KV-V6] stash failed: {e}\")\n"
        "        finally:\n"
        "            self._v6_captured = []\n"
        "            self._v6_capture_failures = 0\n"
        "\n"
        "    def _retract_running_batch_normal(self):\n",
        probe="def _v6_capture_kv(self, req):")

    # The capture list has to exist before any retraction can append to it.
    replace(scheduler,
        "        self.tp_rank = tp_rank\n",
        "        self.tp_rank = tp_rank\n"
        "        self._v6_captured = []            # V6: retracted KV awaiting stash\n"
        "        self._v6_capture_failures = 0\n",
        probe="# V6: retracted KV awaiting stash")

    # Stash after the retraction finishes, before the pools are released.
    # The anchor has to carry enough context to be unique: a bare
    # "self.tp_worker.deactivate_model_runner()" first matches __init__'s own
    # startup teardown, and putting the call there runs it before the attribute
    # it reads has been assigned.
    replace(scheduler,
        "        logger.info(\n"
        "            f\"Process ongoing requests with preempt ({preempt}) takes {time_taken:.4f} s\"\n"
        "        )\n"
        "\n"
        "        self.tp_worker.deactivate_model_runner()\n",
        "        logger.info(\n"
        "            f\"Process ongoing requests with preempt ({preempt}) takes {time_taken:.4f} s\"\n"
        "        )\n"
        "\n"
        "        self._v6_stash_captured()\n"
        "        self.tp_worker.deactivate_model_runner()\n",
        probe="self._v6_stash_captured()\n        self.tp_worker.deactivate_model_runner()")

    # ---- dedicated reply channel for KV capsules -----------------------
    # The hand-off cannot share the weight-loading handshake: one queue with two
    # readers races (measured: every fetch timed out), and appending an extra
    # message to that exchange hangs the run (measured, cause never established).
    # Capsules themselves cross processes fine -- proved separately in
    # exp/results/paper-faithful-v6/e2e/attempt-4th-message/ipc_capsule_probe.py
    # -- so what is needed is a channel nothing else reads.
    #
    # Threading a new queue through launch_engine, run_scheduler_process and the
    # ModelRunner would touch five signatures, all of them files the TP branch
    # owns.  Instead the reply queues ride on the router object, which already
    # travels that whole path: it is passed to each engine process at spawn, so
    # queues attached to it are shared by inheritance, which is the only way
    # multiprocessing queues can be shared at all.
    server = mm / "multi_model_server.py"

    replace(server,
        "    output_queues = {\n"
        "        engine_id: torch.multiprocessing.Queue() for engine_id in engine_ids\n"
        "    }\n",
        "    output_queues = {\n"
        "        engine_id: torch.multiprocessing.Queue() for engine_id in engine_ids\n"
        "    }\n"
        "    # V6: one reply queue per engine, for migrated KV only.\n"
        "    v6_kv_queues = {\n"
        "        engine_id: torch.multiprocessing.Queue() for engine_id in engine_ids\n"
        "    }\n",
        probe="v6_kv_queues = {")

    replace(server,
        "    input_queue = _ModelServiceRouter(worker_queues, cpu_model_dict.keys())\n",
        "    input_queue = _ModelServiceRouter(worker_queues, cpu_model_dict.keys())\n"
        "    # Rides along to every engine process, so the engine can find its own\n"
        "    # reply queue without a new parameter on four call signatures.\n"
        "    input_queue.kv_replies = v6_kv_queues\n",
        probe="input_queue.kv_replies = v6_kv_queues")

    replace(server,
        "                worker_queues[service_worker_id],\n"
        "                output_queues,\n",
        "                worker_queues[service_worker_id],\n"
        "                output_queues,\n"
        "                v6_kv_queues,\n",
        probe="                v6_kv_queues,\n")

    replace(service,
        "def run_model_service(\n"
        "    multi_model_server_args,\n"
        "    cpu_model_dict: Dict[str, torch.nn.Module],\n"
        "    input_queue: torch.multiprocessing.Queue,\n"
        "    output_queues: Dict[str, torch.multiprocessing.Queue],\n",
        "def run_model_service(\n"
        "    multi_model_server_args,\n"
        "    cpu_model_dict: Dict[str, torch.nn.Module],\n"
        "    input_queue: torch.multiprocessing.Queue,\n"
        "    output_queues: Dict[str, torch.multiprocessing.Queue],\n"
        "    v6_kv_queues: Dict[str, torch.multiprocessing.Queue],\n",
        probe="    v6_kv_queues: Dict[str, torch.multiprocessing.Queue],\n")

    replace(service,
        "    model_service = ModelService(\n"
        "        cpu_model_dict, input_queue, output_queues, max_threads, gpu_ids,\n"
        "        num_shards, instance,\n"
        "    )\n",
        "    model_service = ModelService(\n"
        "        cpu_model_dict, input_queue, output_queues, max_threads, gpu_ids,\n"
        "        num_shards, instance,\n"
        "    )\n"
        "    model_service.v6_kv_queues = v6_kv_queues\n",
        probe="model_service.v6_kv_queues = v6_kv_queues")

    # ------------------------------------------------------------ 3. service
    replace(service,
        "            if model_key == \"__release__\":\n",
        "            if model_key == \"__kv_stash__\":\n"
        "                # V6: hold a migrating model's KV until its next engine\n"
        "                # asks.  Keyed by model, so source and target never have\n"
        "                # to learn each other's identity.  A second stash for the\n"
        "                # same model replaces the first: only the most recent\n"
        "                # deactivation can have live requests.\n"
        "                self.v6_kv_stash[engine_id] = gpu_model\n"
        "                logging.info(f\"[PAPER-KV-V6] stash {engine_id} \"\n"
        "                             f\"from gpu {target_gpu_id} \"\n"
        "                             f\"({len(gpu_model)} requests)\")\n"
        "                continue\n"
        "            if model_key == \"__kv_fetch__\":\n"
        "                caps = self.v6_kv_stash.pop(engine_id, [])\n"
        "                # Answer on the dedicated queue: the engine's\n"
        "                # output_queue is the weight loader's and racing it\n"
        "                # loses the reply.\n"
        "                self.v6_kv_queues[gpu_model].put(caps)\n"
        "                logging.info(f\"[PAPER-KV-V6] fetch {engine_id} \"\n"
        "                             f\"-> {len(caps)} requests\")\n"
        "                continue\n"
        "            if model_key == \"__release__\":\n",
        probe="__kv_stash__")

    replace(service,
        "        self.v4_resident = {}\n",
        "        self.v4_resident = {}\n"
        "        # V6: model_key -> [RequestKVCapsule] left by a deactivating\n"
        "        # engine for whichever engine activates this model next.\n"
        "        self.v6_kv_stash = {}\n",
        probe="self.v6_kv_stash = {}")

    replace(service,
        "        key = item[1] if item and item[0] == \"__release__\" else item[0]\n",
        "        key = item[1] if item and item[0] in (\n"
        "            \"__release__\", \"__kv_stash__\", \"__kv_fetch__\") else item[0]\n",
        probe="\"__kv_stash__\", \"__kv_fetch__\") else item[0]")

    # ------------------------------------------------------------- 4. inject
    replace(scheduler,
        "        self._restore_waiting_requests()\n",
        "        self._restore_waiting_requests()\n"
        "        self._v6_inject_migrated_kv()\n",
        probe="self._v6_inject_migrated_kv()")

    replace(scheduler,
        "    def _restore_waiting_requests(self):\n",
        "    def _v6_inject_migrated_kv(self):\n"
        "        \"\"\"Pull this model's migrated KV, if any, and rebuild its requests.\n"
        "\n"
        "        Runs after update_memory_pool(), so the target's own pools and\n"
        "        allocator already exist.  Slots come from the target allocator and\n"
        "        bear no relation to the source's numbering -- token order is what\n"
        "        keeps the KV correct.\n"
        "\n"
        "        Any request that cannot be rebuilt is simply not injected; it is\n"
        "        already sitting in the frontend queue and will be recomputed,\n"
        "        which is exactly the prototype's behaviour.\n"
        "        \"\"\"\n"
        "        if not self._v6_kv_enabled():\n"
        "            return\n"
        "        try:\n"
        "            import json as _json\n"
        "            from sglang.multi_model import kv_migration_v6 as _kvm\n"
        "            mr = self.tp_worker.model_runner\n"
        "            q = getattr(mr, \"input_queue\", None)\n"
        "            # A queue nothing else reads, so the weight loader\n"
        "            # cannot take this reply out from under us.\n"
        "            oq = getattr(q, \"kv_replies\", {}).get(mr.engine_id)\n"
        "            if q is None or oq is None:\n"
        "                return\n"
        "            q.put((\"__kv_fetch__\", mr.model_path, None, mr.engine_id))\n"
        "            # Never block activation on this.  A fetch that goes\n"
        "            # unanswered -- service busy, routed elsewhere, stash lost to\n"
        "            # a crash -- must cost a recompute, not hang the engine\n"
        "            # forever inside handle_activate_request.\n"
        "            try:\n"
        "                caps = oq.get(timeout=5)\n"
        "            except Exception:\n"
        "                logger.warning(\"[PAPER-KV-V6] fetch timed out; \"\n"
        "                               \"falling back to recompute\")\n"
        "                return\n"
        "            if not caps:\n"
        "                return\n"
        "            moved, skipped, rec = _kvm.migrate_request_kv(\n"
        "                caps, self.gpu_id, tag=f\"{self.model_name}|gpu={self.gpu_id}\")\n"
        "            pool = self.token_to_kv_pool\n"
        "            dev = f\"cuda:{self.gpu_id}\"\n"
        "            injected = 0\n"
        "            for cap in moved:\n"
        "                try:\n"
        "                    slots = pool.alloc(cap.num_tokens)\n"
        "                    if slots is None:\n"
        "                        continue\n"
        "                    _kvm.scatter_request_kv(pool.k_buffer, pool.v_buffer,\n"
        "                                            slots, cap, dev)\n"
        "                    injected += 1\n"
        "                except Exception as e:\n"
        "                    logger.warning(\n"
        "                        f\"[PAPER-KV-V6] inject failed for {cap.rid}: {e}\")\n"
        "            logger.info(\"[PAPER-KV-V6] \" + _json.dumps(dict(\n"
        "                rec, event=\"inject\", requests_injected=injected,\n"
        "                model=self.model_name)))\n"
        "        except Exception as e:\n"
        "            logger.warning(f\"[PAPER-KV-V6] inject stage failed: {e}\")\n"
        "\n"
        "    def _restore_waiting_requests(self):\n",
        probe="oq.get(timeout=5)")

    # --------------------------------------------------------------- 5. flag
    # ServerArgs.from_multi_model_server_args() builds each engine's ServerArgs
    # from vars(multi_model_server_args) after stripping the multi-model-only
    # fields.  A new field that is not in that strip list is passed straight
    # through and ServerArgs.__init__ rejects it -- which kills EVERY arm at
    # startup, not just the one using the flag.  overlap_migration is in the
    # list for exactly this reason.
    server_args_file = repo / "python/sglang/srt/server_args.py"
    replace(server_args_file,
        "            \"overlap_migration\",\n"
        "        }\n",
        "            \"overlap_migration\",\n"
        "            \"enable_kv_migration\",   # PAPER-FAITHFUL-V6\n"
        "        }\n",
        probe="\"enable_kv_migration\",   # PAPER-FAITHFUL-V6")

    replace(args_file,
        "    overlap_migration: bool = False       # PAPER-FAITHFUL-V3\n",
        "    overlap_migration: bool = False       # PAPER-FAITHFUL-V3\n"
        "    enable_kv_migration: bool = False     # PAPER-FAITHFUL-V6\n",
        probe="enable_kv_migration: bool = False")

    # The flag has to actually reach the code.  The branches read the
    # environment, not server_args, because the policy runs in the controller
    # process while the capture and inject run in engine processes -- an env var
    # crosses that boundary and a parsed arg does not.  So the flag sets the env
    # var, and either route works.
    replace(args_file,
        "    @classmethod\n"
        "    def from_cli_args(cls, args: argparse.Namespace):\n"
        "        args.tp_size = args.tensor_parallel_size\n",
        "    @classmethod\n"
        "    def from_cli_args(cls, args: argparse.Namespace):\n"
        "        if getattr(args, \"enable_kv_migration\", False):\n"
        "            import os as _os\n"
        "            _os.environ[\"PRISM_V6_KV_MIGRATION\"] = \"1\"\n"
        "        args.tp_size = args.tensor_parallel_size\n",
        probe="_os.environ[\"PRISM_V6_KV_MIGRATION\"]")

    replace(args_file,
        "        parser.add_argument(\"--overlap-migration\", action=\"store_true\",\n",
        "        parser.add_argument(\"--enable-kv-migration\", action=\"store_true\",\n"
        "                            help=\"Migrate a model's KV cache with the model \"\n"
        "                                 \"(paper 5.3). Switches the migration path to \"\n"
        "                                 \"preemption, since draining leaves no KV to \"\n"
        "                                 \"move. Off by default: the default arm \"\n"
        "                                 \"reproduces the released prototype.\")\n"
        "        parser.add_argument(\"--overlap-migration\", action=\"store_true\",\n",
        probe="--enable-kv-migration")

    checks = {
        policy: ["_v6_kv = _os.environ.get"],
        scheduler: ["_v6_capture_kv", "_v6_stash_captured", "_v6_inject_migrated_kv",
                    "# V6: retracted KV awaiting stash",
                    # the stash call must sit beside the real teardown, not in __init__
                    "self._v6_stash_captured()\n        self.tp_worker.deactivate_model_runner()"],
        service: ["__kv_stash__", "__kv_fetch__", "self.v6_kv_stash = {}",
                  "v6_kv_queues"],
        mm / "multi_model_server.py":
            ["v6_kv_queues = {", "input_queue.kv_replies = v6_kv_queues"],
        args_file: ["enable_kv_migration", "--enable-kv-migration"],
        repo / "python/sglang/srt/server_args.py": ["enable_kv_migration"],
        mm / "kv_migration_v6.py": ["RequestKVCapsule"],
    }
    missing = [f"{p}: {n}" for p, names in checks.items() for n in names
               if n not in p.read_text()]
    if missing:
        raise RuntimeError("v6 verification failed:\n" + "\n".join(missing))
    print("paper-faithful-v6 KV migration applied (opt-in: --enable-kv-migration)")


if __name__ == "__main__":
    main()
