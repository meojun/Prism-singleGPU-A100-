"""Apply the V6 two-phase overlap protocol to an already patched tree.

The protocol keeps the target private while its weights and pools are prepared.
Only after the source has quiesced and enqueued its latest KV does a commit make
the target eligible to consume backend requests.
"""

from pathlib import Path


def apply_overlap(repo: Path, replace):
    mm = repo / "python/sglang/multi_model"
    srt = repo / "python/sglang/srt"
    action = mm / "scheduling/action.py"
    controller = mm / "scheduling/controller_global.py"
    worker_pool = mm / "scheduling/gpu/worker_pool.py"
    gpu_scheduler = mm / "scheduling/gpu/gpu_scheduler.py"
    io_struct = srt / "managers/io_struct.py"
    scheduler = srt / "managers/scheduler.py"

    # The HTTP request and its completion carry the phase end to end.  ``full``
    # is the exact legacy behaviour used for non-migration activations.
    replace(
        io_struct,
        "    memory_pool_size: Optional[float] = None\n\n"
        "    def __post_init__(self):\n"
        "        if self.rid is None:\n"
        "            self.rid = uuid.uuid4().hex\n\n\n"
        "@dataclass\n"
        "class ActivateReqOutput:\n",
        "    memory_pool_size: Optional[float] = None\n"
        "    phase: str = \"full\"  # PAPER-FAITHFUL-V6: full|prepare|commit\n\n"
        "    def __post_init__(self):\n"
        "        if self.rid is None:\n"
        "            self.rid = uuid.uuid4().hex\n"
        "        if self.phase not in (\"full\", \"prepare\", \"commit\"):\n"
        "            raise ValueError(f\"Invalid activation phase: {self.phase}\")\n\n\n"
        "@dataclass\n"
        "class ActivateReqOutput:\n",
        probe="phase: str = \"full\"  # PAPER-FAITHFUL-V6",
    )
    replace(
        io_struct,
        "    gpu_id: Optional[int] = None\n\n\n"
        "@dataclass\n"
        "class FinishReq:\n",
        "    gpu_id: Optional[int] = None\n"
        "    phase: str = \"full\"  # PAPER-FAITHFUL-V6\n\n\n"
        "@dataclass\n"
        "class FinishReq:\n",
        probe="class ActivateReqOutput:\n",
    )
    # The broad probe above only tells us the class exists; ensure the output
    # field was actually added when upgrading an older generated tree.
    text = io_struct.read_text()
    output_block = text[text.index("class ActivateReqOutput:"):text.index("class FinishReq:")]
    if "phase: str" not in output_block:
        replace(
            io_struct,
            "    gpu_id: Optional[int] = None\n\n\n"
            "@dataclass\n"
            "class FinishReq:\n",
            "    gpu_id: Optional[int] = None\n"
            "    phase: str = \"full\"  # PAPER-FAITHFUL-V6\n\n\n"
            "@dataclass\n"
            "class FinishReq:\n",
        )

    replace(
        action,
        "class ActivateAction(BaseAction):\n"
        "    memory_pool_size: Optional[int] = None\n"
        "    gpu_id: Optional[int] = None\n",
        "class ActivateAction(BaseAction):\n"
        "    memory_pool_size: Optional[int] = None\n"
        "    gpu_id: Optional[int] = None\n"
        "    phase: str = \"full\"  # PAPER-FAITHFUL-V6\n",
        probe="phase: str = \"full\"  # PAPER-FAITHFUL-V6",
    )
    replace(
        action,
        "            if response.status_code == 200:\n"
        "                if response.json()[\"memory_usage\"] is None:\n",
        "            if response.status_code == 200:\n"
        "                payload = response.json()\n"
        "                # Preserve the legacy full-activation semantics, but a\n"
        "                # failed prepare/commit must fence the following phase.\n"
        "                if self.phase != \"full\" and not payload.get(\"success\", True):\n"
        "                    return False\n"
        "                if payload[\"memory_usage\"] is None:\n",
        probe='self.phase != "full" and not payload.get("success", True)',
    )
    replace(
        action,
        "                        response.json()[\"memory_usage\"]\n",
        "                        payload[\"memory_usage\"]\n",
        probe='payload["memory_usage"]',
    )
    replace(
        action,
        "            if response.status_code == 200:\n"
        "                if response.json()[\"memory_usage\"] is None:\n",
        "            if response.status_code == 200:\n"
        "                payload = response.json()\n"
        "                if self.preempt and not payload.get(\"success\", True):\n"
        "                    return False\n"
        "                if payload[\"memory_usage\"] is None:\n",
        probe='self.preempt and not payload.get("success", True)',
    )
    replace(
        action,
        "                model_instance_state_dict[self.model_name][\n"
        "                    self.instance_idx\n"
        "                ].on_activate(memory_usage, gpu_id=self.gpu_id)\n"
        "                success = True\n",
        "                instance = model_instance_state_dict[self.model_name][\n"
        "                    self.instance_idx\n"
        "                ]\n"
        "                if self.phase == \"prepare\":\n"
        "                    # Prepared targets own memory but remain invisible to\n"
        "                    # placement/routing until the commit response.\n"
        "                    instance.state = ModelState.ACTIVATING\n"
        "                    instance.memory_usage = memory_usage\n"
        "                    if self.gpu_id is not None:\n"
        "                        instance.gpu_ids = [self.gpu_id]\n"
        "                else:\n"
        "                    instance.on_activate(memory_usage, gpu_id=self.gpu_id)\n"
        "                success = True\n",
        probe="if self.phase == \"prepare\":",
    )

    # A commit is sent to the worker already assigned during prepare; it must
    # not allocate a second slot or reject the model as already served.
    replace(
        worker_pool,
        "        # Check if the model is already served by a worker\n"
        "        if model_name in self._model_to_worker:\n"
        "            logger.info(f\"Model {model_name} is already served by a worker\")\n"
        "            return False\n",
        "        phase = getattr(req, \"phase\", \"full\")\n"
        "        if phase == \"commit\":\n"
        "            worker_id = self._model_to_worker.get(model_name)\n"
        "            if worker_id is None:\n"
        "                logger.error(f\"Cannot commit unprepared model {model_name}\")\n"
        "                return False\n"
        "            self._worker_to_ipc_name[worker_id].send_pyobj(req)\n"
        "            logger.info(f\"[PAPER-OVERLAP-V6] commit dispatched \"\n"
        "                        f\"model={model_name} gpu={self.gpu_id} \"\n"
        "                        f\"worker={worker_id}\")\n"
        "            return True\n\n"
        "        # Check if the model is already served by a worker\n"
        "        if model_name in self._model_to_worker:\n"
        "            logger.info(f\"Model {model_name} is already served by a worker\")\n"
        "            return False\n",
        probe="Cannot commit unprepared model",
    )

    # Keep prepared/committing targets out of frontend and backend admission.
    # They still reserve resources, and only the activation completion moves a
    # committed target into the existing ``activated`` serving state.
    replace(
        gpu_scheduler,
        "                if success:\n"
        "                    self._set_model_state(model_name, \"activated\")\n",
        "                if success:\n"
        "                    phase = getattr(message, \"phase\", \"full\")\n"
        "                    self._set_model_state(\n"
        "                        model_name,\n"
        "                        \"prepared\" if phase == \"prepare\" else \"activated\",\n"
        "                    )\n",
        probe='"prepared" if phase == "prepare" else "activated"',
    )
    replace(
        gpu_scheduler,
        "                    # No special handling needed here, only take action when tokenizer sends activation result\n"
        "                    self._set_model_state(message.model_name, \"activating\")\n",
        "                    phase = getattr(message, \"phase\", \"full\")\n"
        "                    self._set_model_state(\n"
        "                        message.model_name,\n"
        "                        \"preparing\" if phase == \"prepare\"\n"
        "                        else \"committing\" if phase == \"commit\"\n"
        "                        else \"activating\",\n"
        "                    )\n",
        probe='"preparing" if phase == "prepare"',
    )
    replace(
        gpu_scheduler,
        "        states_need_resource = (\"activating\", \"activated\", \"deactivating\")\n",
        "        states_need_resource = (\n"
        "            \"preparing\", \"prepared\", \"committing\",\n"
        "            \"activating\", \"activated\", \"deactivating\",\n"
        "        )\n",
        probe='"preparing", "prepared", "committing"',
    )

    # Engine-side state: prepare loads weights/pools but deliberately leaves
    # _activated false, so the event loop cannot consume the shared backend
    # queue.  Commit performs the KV join and only then flips _activated.
    if "self._v6_prepared_model" not in scheduler.read_text():
        replace(
            scheduler,
            "        self._v6_capture_failures = 0\n",
            "        self._v6_capture_failures = 0\n"
            "        self._v6_prepared_model = None   # V6: private until commit\n",
        )
    replace(
        scheduler,
        "        if self._activated:\n"
        "            if self.tp_rank == 0:\n"
        "                logger.warning(\"Scheduler is already activated\")\n",
        "        phase = getattr(recv_req, \"phase\", \"full\")\n"
        "        if phase == \"commit\":\n"
        "            success = self._v6_prepared_model == recv_req.model_name\n"
        "            if success:\n"
        "                commit_start = time.time()\n"
        "                self._restore_waiting_requests()\n"
        "                self._v6_inject_migrated_kv()\n"
        "                self._activated = True\n"
        "                self._v6_prepared_model = None\n"
        "                logger.info(\"[PAPER-OVERLAP-V6] \" + json.dumps({\n"
        "                    \"event\": \"target_commit\", \"model\": self.model_name,\n"
        "                    \"gpu\": self.gpu_id, \"rid\": recv_req.rid,\n"
        "                    \"start\": commit_start, \"ready\": time.time()}))\n"
        "            else:\n"
        "                logger.error(\"[PAPER-OVERLAP-V6] commit without matching prepare \"\n"
        "                             f\"model={recv_req.model_name}\")\n"
        "            if self.tp_rank == 0:\n"
        "                activate_req_output = ActivateReqOutput(\n"
        "                    rid=recv_req.rid, gpu_id=gpu_id,\n"
        "                    model_name=recv_req.model_name,\n"
        "                    instance_idx=recv_req.instance_idx, success=success,\n"
        "                    memory_usage=self.get_memory_usage(), phase=phase)\n"
        "                self.send_to_detokenizer.send_pyobj(activate_req_output)\n"
        "                self.redis_client.send_pyobj(\n"
        "                    key=f\"{self.server_args.engine_to_gpu_scheduler_key_prefix}:{self.gpu_id}\",\n"
        "                    obj=activate_req_output)\n"
        "            return\n\n"
        "        if self._activated or self._v6_prepared_model is not None:\n"
        "            if self.tp_rank == 0:\n"
        "                logger.warning(\"Scheduler is already activated or prepared\")\n",
        probe='if phase == "commit":\n            success = self._v6_prepared_model',
    )
    replace(
        scheduler,
        "        self._restore_waiting_requests()\n"
        "        self._v6_inject_migrated_kv()\n"
        "        logger.info(\"Waiting requests restored\")\n\n"
        "        if not self._activated:\n"
        "            self._activated = True\n"
        "            logger.info(\n"
        "                f\"Scheduler activated. Activation takes {time.perf_counter() - start_time:.2f} s\"\n"
        "            )\n"
        "        else:\n"
        "            logger.warning(\"Scheduler is already activated\")\n",
        "        if phase == \"prepare\":\n"
        "            self._v6_prepared_model = recv_req.model_name\n"
        "            logger.info(\"[PAPER-OVERLAP-V6] \" + json.dumps({\n"
        "                \"event\": \"target_ready\", \"model\": self.model_name,\n"
        "                \"gpu\": self.gpu_id, \"rid\": recv_req.rid,\n"
        "                \"ready\": time.time(),\n"
        "                \"prepare_duration_s\": time.perf_counter() - start_time}))\n"
        "        else:\n"
        "            self._restore_waiting_requests()\n"
        "            self._v6_inject_migrated_kv()\n"
        "            self._activated = True\n"
        "            logger.info(\"Waiting requests restored\")\n"
        "            logger.info(\n"
        "                f\"Scheduler activated. Activation takes {time.perf_counter() - start_time:.2f} s\"\n"
        "            )\n",
        probe='"event": "target_ready"',
    )
    # Both the already-live rejection and the normal completion must echo the
    # phase so the GPU scheduler can distinguish prepared from routable.
    text = scheduler.read_text()
    method_start = text.index("    def handle_activate_request")
    method_end = text.index("    def handle_deactivate_request", method_start)
    block = text[method_start:method_end]
    block = block.replace(
        "                    memory_usage=self.get_memory_usage(),\n"
        "                )",
        "                    memory_usage=self.get_memory_usage(),\n"
        "                    phase=phase,\n"
        "                )",
    ).replace(
        "                memory_usage=memory_usage,\n"
        "            )",
        "                memory_usage=memory_usage,\n"
        "                phase=phase,\n"
        "            )",
    )
    scheduler.write_text(text[:method_start] + block + text[method_end:])

    replace(
        scheduler,
        "        if self._activated:\n"
        "            self._activated = False\n"
        "            logger.info(f\"Scheduler receives the deactivate requests with rid: {recv_req.rid}\")\n",
        "        if self._activated or self._v6_prepared_model is not None:\n"
        "            quiesce = time.time()\n"
        "            was_serving = self._activated\n"
        "            self._activated = False\n"
        "            self._v6_prepared_model = None\n"
        "            logger.info(\"[PAPER-OVERLAP-V6] \" + json.dumps({\n"
        "                \"event\": \"source_quiesce\", \"model\": self.model_name,\n"
        "                \"gpu\": self.gpu_id, \"rid\": recv_req.rid,\n"
        "                \"time\": quiesce, \"was_serving\": was_serving}))\n"
        "            logger.info(f\"Scheduler receives the deactivate requests with rid: {recv_req.rid}\")\n",
        probe='"event": "source_quiesce"',
    )

    # Controller-side observability and failure fencing.  A failed prepare
    # prevents source teardown and its matching commit.
    text = controller.read_text()
    text = text.replace(
        "            if (overlap_migration and not kv_migration\n"
        "                    and isinstance(batch[0], DeactivateAction)):\n",
        "            if overlap_migration and isinstance(batch[0], DeactivateAction):\n",
    )
    controller.write_text(text)
    replace(
        controller,
        "        failed_activation_models = set()\n",
        "        failed_activation_models = set()\n"
        "        failed_deactivation_models = set()\n"
        "        deactivating_models = {\n"
        "            a.model_name for a in actions if isinstance(a, DeactivateAction)\n"
        "        }\n"
        "        migration_targets = {\n"
        "            a.model_name: a for a in actions\n"
        "            if isinstance(a, ActivateAction)\n"
        "            and a.model_name in deactivating_models\n"
        "        }\n",
        probe="failed_deactivation_models = set()",
    )
    replace(
        controller,
        "            with ThreadPoolExecutor(max_workers=max_workers) as threads:\n",
        "            if (overlap_migration and isinstance(batch[0], ActivateAction)\n"
        "                    and getattr(batch[0], \"phase\", \"full\") == \"commit\"):\n"
        "                batch = [a for a in batch\n"
        "                         if a.model_name not in failed_activation_models]\n"
        "                if not batch:\n"
        "                    continue\n\n"
        "            with ThreadPoolExecutor(max_workers=max_workers) as threads:\n",
        probe='getattr(batch[0], "phase", "full") == "commit"',
    )
    replace(
        controller,
        "                def _timed_execute(action):\n"
        "                    _t0 = time.time()\n"
        "                    _ok = False\n",
        "                def _timed_execute(action):\n"
        "                    _t0 = time.time()\n"
        "                    _tracker = self.model_queues.get(action.model_name)\n"
        "                    _finish_before = (\n"
        "                        _tracker.get_last_finished_time()\n"
        "                        if _tracker is not None else float(\"-inf\")\n"
        "                    )\n"
        "                    _ok = False\n",
        probe="_finish_before = (",
    )
    replace(
        controller,
        "                        _t1 = time.time()\n"
        "                        logger.info(\"[PAPER-ACTION-V4] \" + json.dumps({\n",
        "                        _t1 = time.time()\n"
        "                        _finish_after = (\n"
        "                            _tracker.get_last_finished_time()\n"
        "                            if _tracker is not None else float(\"-inf\")\n"
        "                        )\n"
        "                        _phase = getattr(action, \"phase\", \"full\")\n"
        "                        logger.info(\"[PAPER-ACTION-V4] \" + json.dumps({\n",
        probe="_finish_after = (",
    )
    if ('"phase": getattr(action, "phase", "full")' not in controller.read_text()
            and '"phase": _phase' not in controller.read_text()):
        replace(
            controller,
            "                            \"instance_idx\": getattr(action, \"instance_idx\", None),\n"
            "                            \"start\": _t0, \"end\": _t1, \"duration_s\": _t1 - _t0,\n",
            "                            \"instance_idx\": getattr(action, \"instance_idx\", None),\n"
            "                            \"phase\": getattr(action, \"phase\", \"full\"),\n"
            "                            \"start\": _t0, \"end\": _t1, \"duration_s\": _t1 - _t0,\n",
        )
    replace(
        controller,
        "                            \"phase\": getattr(action, \"phase\", \"full\"),\n"
        "                            \"start\": _t0, \"end\": _t1, \"duration_s\": _t1 - _t0,\n",
        "                            \"phase\": _phase,\n"
        "                            \"start\": _t0, \"end\": _t1, \"duration_s\": _t1 - _t0,\n"
        "                            \"last_finish_before\": _finish_before,\n"
        "                            \"last_finish_after\": _finish_after,\n"
        "                            \"source_completed_during_prepare\": (\n"
        "                                _phase == \"prepare\" and _finish_after >= _t0),\n",
        probe='"source_completed_during_prepare"',
    )
    replace(
        controller,
        "                        future.result()\n"
        "                    except Exception:\n",
        "                        result = future.result()\n"
        "                        if result is False:\n"
        "                            if isinstance(action, ActivateAction):\n"
        "                                failed_activation_models.add(action.model_name)\n"
        "                            elif isinstance(action, DeactivateAction):\n"
        "                                failed_deactivation_models.add(action.model_name)\n"
        "                    except Exception:\n",
        probe="failed_deactivation_models.add(action.model_name)",
    )
    replace(
        controller,
        "                        if isinstance(action, ActivateAction):\n"
        "                            failed_activation_models.add(action.model_name)\n"
        "                        logger.error(f\"Error executing action: {get_exception_traceback()}\")\n"
        "        toc = time.time()\n",
        "                        if isinstance(action, ActivateAction):\n"
        "                            failed_activation_models.add(action.model_name)\n"
        "                        elif isinstance(action, DeactivateAction):\n"
        "                            failed_deactivation_models.add(action.model_name)\n"
        "                        logger.error(f\"Error executing action: {get_exception_traceback()}\")\n"
        "\n"
        "            if (kv_migration and isinstance(batch[0], DeactivateAction)\n"
        "                    and failed_deactivation_models):\n"
        "                # A prepared target must not leak if source quiesce fails.\n"
        "                # Roll it back before fencing the matching commit.\n"
        "                for model_name in sorted(failed_deactivation_models):\n"
        "                    target = migration_targets.get(model_name)\n"
        "                    if target is None:\n"
        "                        continue\n"
        "                    failed_activation_models.add(model_name)\n"
        "                    rollback = DeactivateAction(\n"
        "                        model_name=model_name,\n"
        "                        instance_idx=target.instance_idx,\n"
        "                        gpu_id=target.gpu_id,\n"
        "                        preempt=False, evict_waiting_requests=False,\n"
        "                    )\n"
        "                    try:\n"
        "                        rollback.execute(\n"
        "                            self.server_args.url(),\n"
        "                            self.model_instance_state_dict, 600.0)\n"
        "                        logger.warning(\n"
        "                            \"[PAPER-OVERLAP-V6] rolled back prepared target \"\n"
        "                            f\"for {model_name} after source deactivation failure\")\n"
        "                    except Exception:\n"
        "                        logger.error(\n"
        "                            \"[PAPER-OVERLAP-V6] target rollback failed: \"\n"
        "                            + get_exception_traceback())\n"
        "        toc = time.time()\n",
        probe="rolled back prepared target",
    )

    required = {
        action: ["phase: str", 'self.phase == "prepare"',
                 'not payload.get("success", True)',
                 'self.preempt and not payload.get("success", True)'],
        io_struct: ["Invalid activation phase", "phase: str"],
        worker_pool: ["Cannot commit unprepared model"],
        gpu_scheduler: ["preparing", "prepared", "committing"],
        scheduler: ["_v6_prepared_model", '"event": "target_ready"',
                    '"event": "source_quiesce"', '"event": "target_commit"'],
        controller: ['getattr(batch[0], "phase", "full") == "commit"',
                     '"phase": _phase', "rolled back prepared target",
                     '"source_completed_during_prepare"'],
    }
    missing = [f"{path}: {needle}" for path, needles in required.items()
               for needle in needles if needle not in path.read_text()]
    if missing:
        raise RuntimeError("V6 overlap verification failed:\n" + "\n".join(missing))
