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


def patch_response_delivery(path):
    """Make streamed response delivery lossless under migration bursts.

    The upstream request handler uses one level-triggered Event for a stream.
    A response arriving between the consumer's wakeup and event.clear() loses
    its edge; if it is the final response the HTTP request waits forever.  It
    also publishes a request before registering its response state.  Both
    races are observable once overlap makes resumed requests complete in tight
    bursts, so register first and carry each response through an asyncio.Queue.
    """
    if "PAPER-OVERLAP-V6-RESPONSE-QUEUE" in path.read_text():
        return

    replace(path,
        "class ReqState:\n"
        "    \"\"\"Store the state a request.\"\"\"\n"
        "\n"
        "    out_list: List\n"
        "    finished: bool\n"
        "    event: asyncio.Event\n",
        "class ReqState:\n"
        "    \"\"\"Store the state a request.\"\"\"\n"
        "\n"
        "    # PAPER-OVERLAP-V6-RESPONSE-QUEUE\n"
        "    out_list: List\n"
        "    finished: bool\n"
        "    event: asyncio.Event\n"
        "    # Events are level-triggered and lose back-to-back stream edges.\n"
        "    response_queue: Optional[asyncio.Queue] = None\n")

    replace(path,
        "        # Send request to the generation queue with model name in the key\n"
        "        model = obj.model\n",
        "        # Register before publishing: a resumed/short request can reply\n"
        "        # before the caller gets another scheduling turn.\n"
        "        event = asyncio.Event()\n"
        "        state = ReqState([], False, event, asyncio.Queue())\n"
        "        self.rid_to_state[single_request_obj.rid] = state\n"
        "\n"
        "        # Send request to the generation queue with model name in the key\n"
        "        model = obj.model\n")
    replace(path,
        "        except asyncio.exceptions.CancelledError:\n"
        "            raise\n"
        "        except Exception as e:\n"
        "            logger.error(f\"Error when sending request to {model}: {e}\")\n",
        "        except asyncio.exceptions.CancelledError:\n"
        "            self.rid_to_state.pop(single_request_obj.rid, None)\n"
        "            raise\n"
        "        except Exception as e:\n"
        "            self.rid_to_state.pop(single_request_obj.rid, None)\n"
        "            logger.error(f\"Error when sending request to {model}: {e}\")\n")
    replace(path,
        "        return single_request_obj.rid, single_request_obj.input_ids\n",
        "        return single_request_obj.rid, single_request_obj.input_ids, state\n")
    replace(path,
        "            rid, input_ids = await self._send_single_request(\n",
        "            rid, input_ids, state = await self._send_single_request(\n")
    replace(path,
        "        # Recv results\n"
        "        event = asyncio.Event()\n"
        "        state = ReqState([], False, event)\n"
        "        self.rid_to_state[rid] = state\n"
        "\n",
        "        # Recv results (state was registered before Redis publish).\n")
    replace(path,
        "                rid, _ = await self._send_single_request(\n"
        "                    obj, index, input_id_index=i, is_cache_for_prefill=False\n"
        "                )\n"
        "\n"
        "                event = asyncio.Event()\n"
        "                state = ReqState([], False, event)\n"
        "                self.rid_to_state[rid] = state\n",
        "                rid, _, state = await self._send_single_request(\n"
        "                    obj, index, input_id_index=i, is_cache_for_prefill=False\n"
        "                )\n")
    replace(path,
        "                await asyncio.wait_for(state.event.wait(), timeout=4)\n",
        "                out, finished = await asyncio.wait_for(\n"
        "                    state.response_queue.get(), timeout=4\n"
        "                )\n")
    replace(path,
        "                    state.out_list[-1],\n",
        "                    out,\n")
    replace(path,
        "            else:  # isinstance(obj, (EmbeddingReqInput, RewardReqInput))\n"
        "                out = state.out_list[-1]\n"
        "\n",
        "")
    replace(path,
        "            if self.server_args.log_requests and state.finished:\n",
        "            if self.server_args.log_requests and finished:\n")
    replace(path,
        "            state.out_list = []\n"
        "            if state.finished:\n",
        "            if finished:\n")
    replace(path,
        "            state.event.clear()\n"
        "            yield out\n",
        "            yield out\n")
    replace(path,
        "                await asyncio.wait_for(state.event.wait(), timeout=4)\n"
        "                break\n",
        "                _, finished = await asyncio.wait_for(\n"
        "                    state.response_queue.get(), timeout=4\n"
        "                )\n"
        "                if finished:\n"
        "                    break\n")
    replace(path,
        "        assert state.finished\n"
        "        del self.rid_to_state[rid]\n",
        "        del self.rid_to_state[rid]\n")
    replace(path,
        "                state.out_list.append(out_dict)\n"
        "                state.finished = recv_obj.finished_reason[i] is not None\n"
        "                state.event.set()\n",
        "                finished = recv_obj.finished_reason[i] is not None\n"
        "                if state.response_queue is not None:\n"
        "                    state.response_queue.put_nowait((out_dict, finished))\n"
        "                else:\n"
        "                    state.out_list.append(out_dict)\n"
        "                    state.finished = finished\n"
        "                    state.event.set()\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=str(ROOT / "prism-research"))
    ns = ap.parse_args()
    repo = Path(ns.repo).resolve()
    mm = repo / "python/sglang/multi_model"
    srt = repo / "python/sglang/srt"

    policy = mm / "scheduling/policy/simple_global.py"
    controller = mm / "scheduling/controller_global.py"
    scheduler = srt / "managers/scheduler.py"
    service = mm / "model_sevice.py"
    args_file = mm / "multi_model_server_args.py"
    request_handler = mm / "request_handler_worker_pool.py"

    shutil.copyfile(HERE / "kv_migration_v6.py", mm / "kv_migration_v6.py")
    shutil.copyfile(HERE / "action_order_v6.py", mm / "scheduling/action_order_v6.py")

    # Lossless request/response lifecycle is a prerequisite for testing KV
    # overlap: otherwise a completed engine request can leave its HTTP caller
    # hung forever and make a healthy GPU look idle.
    patch_response_delivery(request_handler)

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

    # V3 overlap migration deliberately activates the target before retiring
    # the source. V6 splits that activation into prepare and commit: prepare
    # loads the target while the source still serves, source deactivation then
    # captures/stashes the latest KV, and commit injects it before routing.
    replace(controller,
        "        overlap_migration = getattr(self.server_args, \"overlap_migration\", False)\n"
        "        if not overlap_migration:\n"
        "            batches = [(actions, len(actions))]\n"
        "        else:\n"
        "            # The HTTP control path cannot safely process two concurrent model\n"
        "            # activations for the same GPU. Serialize activations, then apply\n"
        "            # unrelated actions, and only then retire source copies.\n"
        "            activations = [a for a in actions if isinstance(a, ActivateAction)]\n"
        "            middle = [\n"
        "                a for a in actions\n"
        "                if not isinstance(a, (ActivateAction, DeactivateAction))\n"
        "            ]\n"
        "            deactivations = [a for a in actions if isinstance(a, DeactivateAction)]\n"
        "            batches = [\n"
        "                (batch, workers)\n"
        "                for batch, workers in (\n"
        "                    (activations, 1),\n"
        "                    (middle, len(middle)),\n"
        "                    (deactivations, len(deactivations)),\n"
        "                )\n"
        "                if batch\n"
        "            ]\n",
        "        overlap_migration = getattr(self.server_args, \"overlap_migration\", False)\n"
        "        kv_migration = getattr(self.server_args, \"enable_kv_migration\", False)\n"
        "        from sglang.multi_model.scheduling.action_order_v6 import (\n"
        "            build_action_batches,\n"
        "        )\n"
        "        batches = build_action_batches(\n"
        "            actions, overlap_migration, kv_migration,\n"
        "            ActivateAction, DeactivateAction,\n"
        "        )\n",
        probe="build_action_batches(")

    # ------------------------------------------------------------ 2. capture
    # Insert before cache_finished_req, which is what frees the slots.  At this
    # point req.req_pool_idx is still valid and req_to_token still holds the
    # slot vector.  The token count matches radix_cache's own accounting so the
    # capture and the free agree on what belonged to this request.
    # Capture before _retract_running_batch_normal clears scheduler state such
    # as logprob_start_len and prefix_indices.
    replace(scheduler,
        "            for req in self.running_batch.reqs:\n"
        "                req.prefix_indices = []\n",
        "            for req in self.running_batch.reqs:\n"
        "                self._v6_capture_kv(req)\n"
        "                req.prefix_indices = []\n",
        probe="for req in self.running_batch.reqs:\n                self._v6_capture_kv(req)")
    # Upgrade trees produced by the earlier generator, which captured after
    # mutating the request.
    old_late_capture = (
        "                self.waiting_queue.append(req)\n"
        "                self._v6_capture_kv(req)\n"
        "                if self.use_kvcached_v0:\n"
    )
    if old_late_capture in scheduler.read_text():
        scheduler.write_text(scheduler.read_text().replace(
            old_late_capture,
            "                self.waiting_queue.append(req)\n"
            "                if self.use_kvcached_v0:\n",
            1,
        ))

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
        "                k=k, v=v, source_gpu=self.gpu_id,\n"
        "                request_state={name: getattr(req, name, None) for name in (\n"
        "                    \"origin_input_text\", \"lora_path\", \"return_logprob\",\n"
        "                    \"top_logprobs_num\", \"stream\", \"logprob_start_len\",\n"
        "                    \"vid\", \"decoded_text\", \"surr_offset\", \"read_offset\",\n"
        "                    \"completion_tokens_wo_jump_forward\",\n"
        "                    \"normalized_prompt_logprob\", \"input_token_logprobs\",\n"
        "                    \"input_top_logprobs\", \"output_token_logprobs\",\n"
        "                    \"output_top_logprobs\", \"out_queue_timestamp\",\n"
        "                    \"prefill_finish_timestamp\", \"decode_timestamps\",\n"
        "                    \"regex_fsm_state\")})\n"
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
        "        if not self._v6_kv_enabled():\n"
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
        "            captured_rids = {c.rid for c in self._v6_captured}\n"
        "            self.waiting_queue[:] = [\n"
        "                req for req in self.waiting_queue if req.rid not in captured_rids\n"
        "            ]\n"
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

    # Upgrade capsule/request cleanup pieces in trees produced by an older V6
    # generator, whose method-level probe skips the insertion above.
    replace(scheduler,
        "                k=k, v=v, source_gpu=self.gpu_id)\n",
        "                k=k, v=v, source_gpu=self.gpu_id,\n"
        "                request_state={name: getattr(req, name, None) for name in (\n"
        "                    \"origin_input_text\", \"lora_path\", \"return_logprob\",\n"
        "                    \"top_logprobs_num\", \"stream\", \"logprob_start_len\",\n"
        "                    \"vid\", \"decoded_text\", \"surr_offset\", \"read_offset\",\n"
        "                    \"completion_tokens_wo_jump_forward\",\n"
        "                    \"normalized_prompt_logprob\", \"input_token_logprobs\",\n"
        "                    \"input_top_logprobs\", \"output_token_logprobs\",\n"
        "                    \"output_top_logprobs\", \"out_queue_timestamp\",\n"
        "                    \"prefill_finish_timestamp\", \"decode_timestamps\",\n"
        "                    \"regex_fsm_state\")})\n",
        probe="request_state={name: getattr(req, name, None)")

    replace(scheduler,
        "            q.put((\"__kv_stash__\", key, self.gpu_id, self._v6_captured))\n"
        "            logger.info(\"[PAPER-KV-V6] \" + _json.dumps({\n",
        "            q.put((\"__kv_stash__\", key, self.gpu_id, self._v6_captured))\n"
        "            captured_rids = {c.rid for c in self._v6_captured}\n"
        "            self.waiting_queue[:] = [\n"
        "                req for req in self.waiting_queue if req.rid not in captured_rids\n"
        "            ]\n"
        "            logger.info(\"[PAPER-KV-V6] \" + _json.dumps({\n",
        probe="captured_rids = {c.rid for c in self._v6_captured}")

    # Capture state has to exist before any retraction can append to it.
    if "# V6: retracted KV awaiting stash" not in scheduler.read_text():
        replace(scheduler,
            "        self.tp_rank = tp_rank\n",
            "        self.tp_rank = tp_rank\n"
            "        self._v6_captured = []            # V6: retracted KV awaiting stash\n"
            "        self._v6_capture_failures = 0\n")
    # Remove the obsolete rid->slot join map from older generated trees.
    obsolete_pending = (
        "        self._v6_pending_requests = {}    # V6: rid -> (capsule, target slots)\n"
    )
    if obsolete_pending in scheduler.read_text():
        scheduler.write_text(scheduler.read_text().replace(obsolete_pending, "", 1))

    # Stash after the retraction finishes, before the pools are released.
    # The anchor has to carry enough context to be unique: a bare
    # "self.tp_worker.deactivate_model_runner()" first matches __init__'s own
    # startup teardown, and putting the call there runs it before the attribute
    # it reads has been assigned.
    if "self._v6_stash_captured()" in scheduler.read_text():
        replace(scheduler,
            "        self._v6_stash_captured()\n"
            "        self.tp_worker.deactivate_model_runner()\n",
            "        self._v6_stash_captured()\n"
            "        if self._v6_kv_enabled():\n"
            "            # Captured requests were removed after a successful stash;\n"
            "            # any leftovers failed capture/stash and must recompute.\n"
            "            self._evict_all_waiting_requests()\n"
            "        self.tp_worker.deactivate_model_runner()\n",
            probe="any leftovers failed capture/stash and must recompute")
    else:
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
            "        if self._v6_kv_enabled():\n"
            "            # Captured requests were removed after a successful stash;\n"
            "            # any leftovers failed capture/stash and must recompute.\n"
            "            self._evict_all_waiting_requests()\n"
            "        self.tp_worker.deactivate_model_runner()\n",
            probe="any leftovers failed capture/stash and must recompute")

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
        "                from sglang.multi_model import kv_migration_v6 as _kvm\n"
        "                gpu_model = _kvm.clone_capsules_for_relay(gpu_model)\n"
        "                self.v6_kv_stash[engine_id] = gpu_model\n"
        "                logging.info(f\"[PAPER-KV-V6] relay-clone {engine_id} \"\n"
        "                             f\"({len(gpu_model)} requests)\")\n"
        "                logging.info(f\"[PAPER-KV-V6] stash {engine_id} \"\n"
        "                             f\"from gpu {target_gpu_id} \"\n"
        "                             f\"({len(gpu_model)} requests)\")\n"
        "                continue\n"
        "            if model_key == \"__kv_fetch__\":\n"
        "                # None means the source stash has not reached this relay\n"
        "                # yet; [] means it arrived with no in-flight requests.\n"
        "                caps = self.v6_kv_stash.pop(engine_id, None)\n"
        "                # Answer on the dedicated queue: the engine's\n"
        "                # output_queue is the weight loader's and racing it\n"
        "                # loses the reply.\n"
        "                self.v6_kv_queues[gpu_model].put(caps)\n"
        "                logging.info(f\"[PAPER-KV-V6] fetch {engine_id} \"\n"
        "                             f\"-> {None if caps is None else len(caps)} requests\")\n"
        "                continue\n"
        "            if model_key == \"__release__\":\n",
        probe="[PAPER-KV-V6] relay-clone")

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
        probe="\"__kv_stash__\", \"__kv_fetch__\"")

    # ------------------------------------------------------------- 4. inject
    if "self._v6_inject_migrated_kv()" not in scheduler.read_text():
        replace(scheduler,
            "        self._restore_waiting_requests()\n",
            "        self._restore_waiting_requests()\n"
            "        self._v6_inject_migrated_kv()\n")
    else:
        obsolete_clear = (
            "        # update_memory_pool() rebuilt the allocator.  Any unmatched\n"
            "        # entries from a prior activation point at invalid slots.\n"
            "        self._v6_pending_requests = {}\n"
        )
        if obsolete_clear in scheduler.read_text():
            scheduler.write_text(scheduler.read_text().replace(obsolete_clear, "", 1))

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
        "        The capsule is also the request body.  If KV placement fails or\n"
        "        exceeds the transfer cap, rebuild it with an empty prefix so it\n"
        "        recomputes instead of disappearing.\n"
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
        "            # Queue puts from different engine processes have no global\n"
        "            # ordering guarantee. Retry a not-yet-present (None) stash; an\n"
        "            # empty list is an acknowledged source quiesce with no KV.\n"
        "            deadline = time.monotonic() + 5\n"
        "            caps = None\n"
        "            while caps is None:\n"
        "                q.put((\"__kv_fetch__\", mr.model_path, None, mr.engine_id))\n"
        "                try:\n"
        "                    caps = oq.get(timeout=max(0.01, deadline - time.monotonic()))\n"
        "                except Exception:\n"
        "                    logger.warning(\"[PAPER-KV-V6] fetch timed out; \"\n"
        "                                   \"falling back to recompute\")\n"
        "                    return\n"
        "                if caps is None:\n"
        "                    if time.monotonic() >= deadline:\n"
        "                        logger.warning(\"[PAPER-KV-V6] stash readiness timed out\")\n"
        "                        return\n"
        "                    time.sleep(0.01)\n"
        "            if not caps:\n"
        "                return\n"
        "            moved, skipped, rec = _kvm.migrate_request_kv(\n"
        "                caps, self.gpu_id, tag=f\"{self.model_name}|gpu={self.gpu_id}\")\n"
        "            pool = self.token_to_kv_pool\n"
        "            dev = f\"cuda:{self.gpu_id}\"\n"
        "            injected = 0\n"
        "            for cap in moved:\n"
        "                slots = None\n"
        "                try:\n"
        "                    slots = pool.alloc(cap.num_tokens)\n"
        "                    if slots is None:\n"
        "                        continue\n"
        "                    _kvm.scatter_request_kv(pool.k_buffer, pool.v_buffer,\n"
        "                                            slots, cap, dev)\n"
        "                    old = self._v6_pending_requests.pop(cap.rid, None)\n"
        "                    if old is not None:\n"
        "                        pool.free(old[1])\n"
        "                    self._v6_pending_requests[cap.rid] = (cap, slots)\n"
        "                    injected += 1\n"
        "                except Exception as e:\n"
        "                    if slots is not None:\n"
        "                        pool.free(slots)\n"
        "                    logger.warning(\n"
        "                        f\"[PAPER-KV-V6] inject failed for {cap.rid}: {e}\")\n"
        "            logger.info(\"[PAPER-KV-V6] \" + _json.dumps(dict(\n"
        "                rec, event=\"inject\", requests_injected=injected,\n"
        "                model=self.model_name)))\n"
        "        except Exception as e:\n"
        "            logger.warning(f\"[PAPER-KV-V6] inject stage failed: {e}\")\n"
        "\n"
        "    def _restore_waiting_requests(self):\n",
        probe="def _v6_inject_migrated_kv(self):")

    # The capsule is the request body: BatchRetractDecodeReq only updates queue
    # accounting and never emits another GenerateReqInput.  Rebuild and enqueue
    # requests here, falling back to full recompute if KV placement fails.
    replace(scheduler,
        "            moved, skipped, rec = _kvm.migrate_request_kv(\n"
        "                caps, self.gpu_id, tag=f\"{self.model_name}|gpu={self.gpu_id}\")\n"
        "            pool = self.token_to_kv_pool\n"
        "            dev = f\"cuda:{self.gpu_id}\"\n"
        "            injected = 0\n"
        "            for cap in moved:\n"
        "                slots = None\n"
        "                try:\n"
        "                    slots = pool.alloc(cap.num_tokens)\n"
        "                    if slots is None:\n"
        "                        continue\n"
        "                    _kvm.scatter_request_kv(pool.k_buffer, pool.v_buffer,\n"
        "                                            slots, cap, dev)\n"
        "                    old = self._v6_pending_requests.pop(cap.rid, None)\n"
        "                    if old is not None:\n"
        "                        pool.free(old[1])\n"
        "                    self._v6_pending_requests[cap.rid] = (cap, slots)\n"
        "                    injected += 1\n"
        "                except Exception as e:\n"
        "                    if slots is not None:\n"
        "                        pool.free(slots)\n"
        "                    logger.warning(\n"
        "                        f\"[PAPER-KV-V6] inject failed for {cap.rid}: {e}\")\n"
        "            logger.info(\"[PAPER-KV-V6] \" + _json.dumps(dict(\n"
        "                rec, event=\"inject\", requests_injected=injected,\n"
        "                model=self.model_name)))\n",
        "            moved, skipped, rec = _kvm.migrate_request_kv(\n"
        "                caps, self.gpu_id, tag=f\"{self.model_name}|gpu={self.gpu_id}\")\n"
        "            pool = self.token_to_kv_pool\n"
        "            dev = f\"cuda:{self.gpu_id}\"\n"
        "            injected = recomputed = rebuild_failures = 0\n"
        "\n"
        "            def _build(cap, slots=None):\n"
        "                if slots is None:\n"
        "                    req = _kvm.build_recomputed_request(\n"
        "                        Req, cap, self.tokenizer)\n"
        "                else:\n"
        "                    req = _kvm.build_resumed_request(\n"
        "                        Req, cap, slots, self.tokenizer)\n"
        "                if (req.sampling_params.json_schema is not None or\n"
        "                        req.sampling_params.regex is not None):\n"
        "                    if req.sampling_params.json_schema is not None:\n"
        "                        req.regex_fsm, regex = self.regex_fsm_cache.query(\n"
        "                            (\"json\", req.sampling_params.json_schema))\n"
        "                    else:\n"
        "                        req.regex_fsm, regex = self.regex_fsm_cache.query(\n"
        "                            (\"regex\", req.sampling_params.regex))\n"
        "                    if not self.disable_regex_jump_forward:\n"
        "                        req.jump_forward_map = self.jump_forward_cache.query(regex)\n"
        "                self.waiting_queue.append(req)\n"
        "                return req\n"
        "\n"
        "            for cap in moved:\n"
        "                slots = None\n"
        "                try:\n"
        "                    slots = pool.alloc(cap.num_tokens)\n"
        "                    if slots is None:\n"
        "                        raise RuntimeError(\"target KV slots unavailable\")\n"
        "                    _kvm.scatter_request_kv(pool.k_buffer, pool.v_buffer,\n"
        "                                            slots, cap, dev)\n"
        "                    req = _build(cap, slots)\n"
        "                    injected += 1\n"
        "                    logger.info(\"[PAPER-KV-V6] \" + _json.dumps({\n"
        "                        \"event\": \"resume\", \"rid\": req.rid,\n"
        "                        \"tokens\": len(slots), \"gpu\": self.gpu_id,\n"
        "                        \"prompt_tokens\": len(req.origin_input_ids),\n"
        "                        \"output_tokens\": len(req.output_ids),\n"
        "                        \"resume_extend_len\": req.extend_input_len}))\n"
        "                except Exception as e:\n"
        "                    if slots is not None:\n"
        "                        pool.free(slots)\n"
        "                    try:\n"
        "                        req = _build(cap)\n"
        "                        recomputed += 1\n"
        "                        logger.warning(\"[PAPER-KV-V6] \" + _json.dumps({\n"
        "                            \"event\": \"recompute\", \"rid\": req.rid,\n"
        "                            \"reason\": str(e), \"gpu\": self.gpu_id}))\n"
        "                    except Exception as rebuild_error:\n"
        "                        rebuild_failures += 1\n"
        "                        logger.warning(\n"
        "                            f\"[PAPER-KV-V6] rebuild failed for {cap.rid}: \"\n"
        "                            f\"{rebuild_error}\")\n"
        "            for cap in skipped:\n"
        "                try:\n"
        "                    req = _build(cap)\n"
        "                    recomputed += 1\n"
        "                    logger.info(\"[PAPER-KV-V6] \" + _json.dumps({\n"
        "                        \"event\": \"recompute\", \"rid\": req.rid,\n"
        "                        \"reason\": \"over-cap\", \"gpu\": self.gpu_id}))\n"
        "                except Exception as e:\n"
        "                    rebuild_failures += 1\n"
        "                    logger.warning(\n"
        "                        f\"[PAPER-KV-V6] rebuild failed for {cap.rid}: {e}\")\n"
        "            logger.info(\"[PAPER-KV-V6] \" + _json.dumps(dict(\n"
        "                rec, event=\"inject\", requests_injected=injected,\n"
        "                requests_recomputed=recomputed,\n"
        "                request_rebuild_failures=rebuild_failures,\n"
        "                model=self.model_name)))\n",
        probe="req = _kvm.build_resumed_request(")

    # Remove obsolete request-arrival joins from upgraded generated trees.
    obsolete_join = (
        "        _v6_pending = self._v6_pending_requests.pop(req.rid, None)\n"
        "        if _v6_pending is not None:\n"
        "            import json as _json\n"
        "            from sglang.multi_model import kv_migration_v6 as _kvm\n"
        "            _cap, _slots = _v6_pending\n"
        "            try:\n"
        "                _kvm.prepare_resumed_request(req, _cap, _slots)\n"
        "                logger.info(\"[PAPER-KV-V6] \" + _json.dumps({\n"
        "                    \"event\": \"resume\", \"rid\": req.rid,\n"
        "                    \"tokens\": len(_slots), \"gpu\": self.gpu_id}))\n"
        "            except Exception as e:\n"
        "                self.token_to_kv_pool.free(_slots)\n"
        "                logger.warning(\n"
        "                    f\"[PAPER-KV-V6] resume failed for {req.rid}: {e}; \"\n"
        "                    \"falling back to recompute\")\n"
    )
    if obsolete_join in scheduler.read_text():
        scheduler.write_text(scheduler.read_text().replace(obsolete_join, "", 1))

    obsolete_abort = (
        "        _v6_pending = self._v6_pending_requests.pop(recv_req.rid, None)\n"
        "        if _v6_pending is not None:\n"
        "            self.token_to_kv_pool.free(_v6_pending[1])\n"
        "            logger.info(f\"[PAPER-KV-V6] released pending abort {recv_req.rid}\")\n"
    )
    if obsolete_abort in scheduler.read_text():
        scheduler.write_text(scheduler.read_text().replace(obsolete_abort, "", 1))

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

    # The phase protocol is kept in a separate generator so the transport and
    # request-capsule patch above remains independently reviewable.
    from apply_overlap_v6 import apply_overlap
    apply_overlap(repo, replace)

    # Fence migration generations in the relay.  A model can leave an empty
    # stash behind; without an acknowledged clear, the next commit may consume
    # that stale [] before the current source's non-empty stash arrives.
    replace(service,
        "            \"__release__\", \"__kv_stash__\", \"__kv_fetch__\") else item[0]\n",
        "            \"__release__\", \"__kv_stash__\", \"__kv_fetch__\",\n"
        "            \"__kv_clear__\") else item[0]\n",
        probe='"__kv_clear__") else item[0]')
    replace(service,
        "            if model_key == \"__kv_fetch__\":\n",
        "            if model_key == \"__kv_clear__\":\n"
        "                stale = self.v6_kv_stash.pop(engine_id, None)\n"
        "                self.v6_kv_queues[gpu_model].put(\n"
        "                    (\"__kv_clear_ack__\", 0 if stale is None else len(stale))\n"
        "                )\n"
        "                logging.info(\n"
        "                    f\"[PAPER-KV-V6] clear {engine_id} \"\n"
        "                    f\"dropped={0 if stale is None else len(stale)}\"\n"
        "                )\n"
        "                continue\n"
        "            if model_key == \"__kv_fetch__\":\n",
        probe='if model_key == "__kv_clear__"')
    replace(scheduler,
        "        assert self.running_batch is None, \"Running batch should be None\"\n"
        "\n"
        "        start_time = time.perf_counter()\n",
        "        assert self.running_batch is None, \"Running batch should be None\"\n"
        "\n"
        "        if phase == \"prepare\" and not self._v6_clear_stale_kv(\n"
        "            self.model_names_to_model_paths[recv_req.model_name]\n"
        "        ):\n"
        "            logger.error(\n"
        "                \"[PAPER-OVERLAP-V6] stale KV clear failed; refusing prepare \"\n"
        "                f\"model={recv_req.model_name}\"\n"
        "            )\n"
        "            if self.tp_rank == 0:\n"
        "                activate_req_output = ActivateReqOutput(\n"
        "                    rid=recv_req.rid, gpu_id=gpu_id,\n"
        "                    model_name=recv_req.model_name,\n"
        "                    instance_idx=recv_req.instance_idx, success=False,\n"
        "                    memory_usage=self.get_memory_usage(), phase=phase)\n"
        "                self.send_to_detokenizer.send_pyobj(activate_req_output)\n"
        "                self.redis_client.send_pyobj(\n"
        "                    key=f\"{self.server_args.engine_to_gpu_scheduler_key_prefix}:{self.gpu_id}\",\n"
        "                    obj=activate_req_output)\n"
        "            return\n"
        "\n"
        "        start_time = time.perf_counter()\n",
        probe="stale KV clear failed; refusing prepare")
    replace(scheduler,
        "    def _restore_waiting_requests(self):\n",
        "    def _v6_clear_stale_kv(self, model_key):\n"
        "        \"\"\"ACK a stale-stash clear before source quiesce begins.\"\"\"\n"
        "        if not self._v6_kv_enabled():\n"
        "            return True\n"
        "        try:\n"
        "            q = self.tp_worker.model_runner.input_queue\n"
        "            engine_id = self.tp_worker.model_runner.engine_id\n"
        "            oq = getattr(q, \"kv_replies\", {}).get(engine_id)\n"
        "            if q is None or oq is None:\n"
        "                return False\n"
        "            q.put((\"__kv_clear__\", model_key, None, engine_id))\n"
        "            ack = oq.get(timeout=5)\n"
        "            ok = isinstance(ack, tuple) and ack[0] == \"__kv_clear_ack__\"\n"
        "            if ok:\n"
        "                logger.info(\n"
        "                    \"[PAPER-KV-V6] stale-clear acknowledged \"\n"
        "                    f\"model={model_key} dropped={ack[1]}\"\n"
        "                )\n"
        "            return ok\n"
        "        except Exception as e:\n"
        "            logger.warning(f\"[PAPER-KV-V6] stale-clear failed: {e}\")\n"
        "            return False\n"
        "\n"
        "    def _restore_waiting_requests(self):\n",
        probe="def _v6_clear_stale_kv(self, model_key):")

    # A multiprocessing.Queue is FIFO per producer, not across the source and
    # target engine processes.  Merely putting the stash before returning from
    # deactivation therefore does not fence commit: a target fetch can overtake
    # the source put.  ACK relay ownership after cloning, and make source
    # deactivation fail closed until that ACK arrives.
    service_text = service.read_text()
    if "__kv_stash_ack__" not in service_text:
        old = (
            "                gpu_model = _kvm.clone_capsules_for_relay(gpu_model)\n"
            "                self.v6_kv_stash[engine_id] = gpu_model\n"
        )
        new = (
            "                gpu_model, source_engine_id = gpu_model\n"
            "                gpu_model = _kvm.clone_capsules_for_relay(gpu_model)\n"
            "                self.v6_kv_stash[engine_id] = gpu_model\n"
            "                self.v6_kv_queues[source_engine_id].put(\n"
            "                    (\"__kv_stash_ack__\", engine_id, len(gpu_model))\n"
            "                )\n"
        )
        if old not in service_text:
            raise RuntimeError("cannot add V6 stash acknowledgement")
        service.write_text(service_text.replace(old, new, 1))

    scheduler_text = scheduler.read_text()
    stash_start = scheduler_text.find("\n    def _v6_stash_captured(self):\n")
    stash_end = scheduler_text.find("\n    def _retract_running_batch_normal(self):\n", stash_start)
    if stash_start < 0 or stash_end < 0:
        raise RuntimeError("cannot delimit V6 stash method")
    stash_method = '''
    def _v6_stash_captured(self):
        """Publish the latest KV and wait until the relay owns its clone."""
        if not self._v6_kv_enabled():
            return True
        try:
            import json as _json
            mr = self.tp_worker.model_runner
            q = mr.input_queue
            engine_id = mr.engine_id
            oq = getattr(q, "kv_replies", {}).get(engine_id)
            if q is None or oq is None:
                return False
            key = mr.model_path
            n_tok = sum(c.num_tokens for c in self._v6_captured)
            nbytes = sum(c.nbytes for c in self._v6_captured)
            q.put(("__kv_stash__", key, self.gpu_id,
                   (self._v6_captured, engine_id)))
            ack = oq.get(timeout=30)
            if not (isinstance(ack, tuple) and len(ack) == 3
                    and ack[0] == "__kv_stash_ack__"
                    and ack[1] == key
                    and ack[2] == len(self._v6_captured)):
                raise RuntimeError(f"invalid stash acknowledgement: {ack!r}")
            captured_rids = {c.rid for c in self._v6_captured}
            self.waiting_queue[:] = [
                req for req in self.waiting_queue if req.rid not in captured_rids
            ]
            logger.info("[PAPER-KV-V6] " + _json.dumps({
                "event": "stash", "model": key, "source_gpu": self.gpu_id,
                "requests": len(self._v6_captured), "tokens": n_tok,
                "kv_bytes": nbytes,
                "capture_failures": self._v6_capture_failures}))
            return True
        except Exception as e:
            logger.warning(f"[PAPER-KV-V6] stash failed: {e}")
            return False
        finally:
            self._v6_captured = []
            self._v6_capture_failures = 0
'''
    scheduler_text = scheduler_text[:stash_start] + stash_method + scheduler_text[stash_end:]
    if "stash_ok = self._v6_stash_captured()" not in scheduler_text:
        old = "        self._v6_stash_captured()\n        if self._v6_kv_enabled():\n"
        new = (
            "        stash_ok = self._v6_stash_captured()\n"
            "        if not stash_ok:\n"
            "            self._activated = True\n"
            "            logger.error(\n"
            "                \"[PAPER-KV-V6] relay did not acknowledge stash; \"\n"
            "                f\"keeping source active model={self.model_name}\"\n"
            "            )\n"
            "            if self.tp_rank == 0:\n"
            "                deactivate_req_output = DeactivateReqOutput(\n"
            "                    rid=recv_req.rid, gpu_id=gpu_id,\n"
            "                    model_name=recv_req.model_name,\n"
            "                    instance_idx=recv_req.instance_idx, success=False,\n"
            "                    memory_usage=self.get_memory_usage())\n"
            "                self.send_to_detokenizer.send_pyobj(deactivate_req_output)\n"
            "                self.redis_client.send_pyobj(\n"
            "                    key=f\"{self.server_args.engine_to_gpu_scheduler_key_prefix}:{self.gpu_id}\",\n"
            "                    obj=deactivate_req_output)\n"
            "            return\n"
            "        if self._v6_kv_enabled():\n"
        )
        if old not in scheduler_text:
            raise RuntimeError("cannot fence deactivation on V6 stash acknowledgement")
        scheduler_text = scheduler_text.replace(old, new, 1)
    scheduler.write_text(scheduler_text)

    # Upgrade older generated trees deterministically.  Early V6 generators
    # could leave two versions of this method: a request-arrival join and a
    # later direct-resume implementation.  A broad probe then saw the newer
    # token in one copy and skipped upgrading the other.  Keep the first method
    # only, then normalize its fetch barrier and continuation evidence.
    text = scheduler.read_text()
    method_marker = "\n    def _v6_inject_migrated_kv(self):\n"
    while text.count(method_marker) > 1:
        second = text.find(method_marker, text.find(method_marker) + 1)
        following = text.find("\n    def _restore_waiting_requests(self):\n", second)
        if following < 0:
            raise RuntimeError("cannot delimit duplicate V6 inject method")
        text = text[:second] + text[following:]
    probe_block = (
        "            _kvp = getattr(q, \"kv_replies\", {})\n"
        "            logger.info(\n"
        "                \"[KV-PROBE engine] key=%r have=%r id=%r\"\n"
        "                % (mr.engine_id, sorted(_kvp.keys()),\n"
        "                   id(_kvp.get(mr.engine_id)))\n"
        "            )\n"
    )
    text = text.replace(probe_block, "")
    old_fetch = (
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
    )
    new_fetch = (
        "            # Queue puts from different engine processes have no global\n"
        "            # ordering guarantee. Retry a not-yet-present (None) stash; an\n"
        "            # empty list is an acknowledged source quiesce with no KV.\n"
        "            deadline = time.monotonic() + 5\n"
        "            caps = None\n"
        "            while caps is None:\n"
        "                q.put((\"__kv_fetch__\", mr.model_path, None, mr.engine_id))\n"
        "                try:\n"
        "                    caps = oq.get(timeout=max(0.01, deadline - time.monotonic()))\n"
        "                except Exception:\n"
        "                    logger.warning(\"[PAPER-KV-V6] fetch timed out; \"\n"
        "                                   \"falling back to recompute\")\n"
        "                    return\n"
        "                if caps is None:\n"
        "                    if time.monotonic() >= deadline:\n"
        "                        logger.warning(\"[PAPER-KV-V6] stash readiness timed out\")\n"
        "                        return\n"
        "                    time.sleep(0.01)\n"
    )
    text = text.replace(old_fetch, new_fetch)
    old_resume = (
        "                        \"event\": \"resume\", \"rid\": req.rid,\n"
        "                        \"tokens\": len(slots), \"gpu\": self.gpu_id}))\n"
    )
    new_resume = (
        "                        \"event\": \"resume\", \"rid\": req.rid,\n"
        "                        \"tokens\": len(slots), \"gpu\": self.gpu_id,\n"
        "                        \"prompt_tokens\": len(req.origin_input_ids),\n"
        "                        \"output_tokens\": len(req.output_ids),\n"
        "                        \"resume_extend_len\": req.extend_input_len}))\n"
    )
    text = text.replace(old_resume, new_resume)
    text = text.replace(
        "        if not self._v6_kv_enabled() or not self._v6_captured:\n",
        "        if not self._v6_kv_enabled():\n",
    )
    scheduler.write_text(text)

    checks = {
        policy: ["_v6_kv = _os.environ.get"],
        controller: ["build_action_batches", "failed_activation_models",
                     '"phase": _phase', '"source_completed_during_prepare"'],
        scheduler: ["_v6_capture_kv", "_v6_stash_captured", "_v6_inject_migrated_kv",
                    "_v6_clear_stale_kv",
                    "stash_ok = self._v6_stash_captured()",
                    'ack[0] == "__kv_stash_ack__"',
                    "# V6: retracted KV awaiting stash",
                    "request_state={name: getattr(req, name, None)",
                    "req = _kvm.build_resumed_request(",
                    "any leftovers failed capture/stash and must recompute"],
        service: ["__kv_stash__", "__kv_fetch__", "__kv_clear__",
                  "__kv_stash_ack__",
                  "self.v6_kv_stash = {}",
                  "v6_kv_queues"],
        mm / "multi_model_server.py":
            ["v6_kv_queues = {", "input_queue.kv_replies = v6_kv_queues"],
        args_file: ["enable_kv_migration", "--enable-kv-migration"],
        repo / "python/sglang/srt/server_args.py": ["enable_kv_migration"],
        mm / "kv_migration_v6.py": ["RequestKVCapsule", "clone_capsules_for_relay",
                                    "build_resumed_request", "build_recomputed_request"],
        mm / "scheduling/action_order_v6.py": ["prepares", "commits"],
    }
    missing = [f"{p}: {n}" for p, names in checks.items() for n in names
               if n not in p.read_text()]
    if missing:
        raise RuntimeError("v6 verification failed:\n" + "\n".join(missing))
    obsolete = [name for name in (
        "_v6_pending_requests", "prepare_resumed_request(req, _cap, _slots)",
        "released pending abort",
    ) if name in scheduler.read_text()]
    if obsolete:
        raise RuntimeError(f"obsolete V6 request joins remain: {obsolete}")
    if scheduler.read_text().count("def _v6_inject_migrated_kv(self):") != 1:
        raise RuntimeError("V6 inject method is not unique")
    print("paper-faithful-v6 KV migration applied (opt-in: --enable-kv-migration)")


if __name__ == "__main__":
    main()
