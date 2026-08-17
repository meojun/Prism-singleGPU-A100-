#!/usr/bin/env python3
"""Apply the opt-in paper-faithful-v3 path on top of paper-faithful-v2."""
import argparse
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

def replace(path, old, new, count=1, probe=None):
    text = path.read_text()
    if probe and probe in text:
        return
    if old not in text:
        if new in text:
            return
        raise RuntimeError(f"patch anchor not found in {path}: {old[:120]!r}")
    path.write_text(text.replace(old, new, count))


def replace_block(path, start, end, new):
    text = path.read_text()
    start_idx = text.find(start)
    end_idx = text.find(end, start_idx)
    if start_idx < 0 or end_idx < 0:
        raise RuntimeError(f"block anchors not found in {path}")
    path.write_text(text[:start_idx] + new + text[end_idx:])


def replace_last_block(path, start, end, new):
    text = path.read_text()
    start_idx = text.rfind(start)
    end_idx = text.find(end, start_idx)
    if start_idx < 0 or end_idx < 0:
        raise RuntimeError(f"block anchors not found in {path}")
    path.write_text(text[:start_idx] + new + text[end_idx + len(end):])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=str(ROOT / "prism-research"))
    args_ns = parser.parse_args()
    repo = Path(args_ns.repo).resolve()
    mm = repo / "python/sglang/multi_model"
    if not mm.is_dir():
        raise RuntimeError(f"not a prism-research checkout: {repo}")

    shutil.copyfile(HERE / "kvpr_global_v3.py", mm / "scheduling/policy/kvpr_global_v3.py")
    shutil.copyfile(HERE / "overlap_migration.py", mm / "scheduling/overlap_migration.py")

    controller = mm / "scheduling/controller_global.py"
    replace(controller,
        "from sglang.multi_model.scheduling.policy.simple_global import SimpleGlobalPolicy\n",
        "from sglang.multi_model.scheduling.policy.simple_global import SimpleGlobalPolicy\n"
        "from sglang.multi_model.scheduling.policy.kvpr_global_v3 import KVPRGlobalPolicyV3\n",
        probe="from sglang.multi_model.scheduling.policy.kvpr_global_v3 import")
    replace(controller,
        "from sglang.multi_model.scheduling.action import BaseAction\n",
        "from sglang.multi_model.scheduling.action import (\n"
        "    ActivateAction,\n    BaseAction,\n    DeactivateAction,\n)\n",
        probe="    ActivateAction,\n")
    branch = (
        '        elif self.server_args.policy == "kvpr-global-v3":\n'
        '            # PAPER-FAITHFUL-V3: literal Algorithm 1 line-8 absolute tau.\n'
        '            import json as _json, os as _os\n'
        '            _tpot = {}\n'
        '            _f = getattr(self.server_args, "slo_base_file", None)\n'
        '            if _f and _os.path.exists(_f):\n'
        '                _raw = _json.load(open(_f))\n'
        '                _scale = float(getattr(self.server_args, "kvpr_tpot_slo_scale", 1.0) or 1.0)\n'
        '                _p2n = {v: k for k, v in self.model_names_to_model_paths.items()}\n'
        '                for _k, _v in _raw.items():\n'
        '                    if isinstance(_v, dict):\n'
        '                        _v = _v.get("tpot") or _v.get("tpot_slo") or _v.get("tpot_p95_ms")\n'
        '                    if _v is None:\n'
        '                        continue\n'
        '                    _v = float(_v)\n'
        '                    _v = _v / 1000.0 if _v > 1.0 else _v\n'
        '                    _name = _k if _k in self.model_names_to_model_paths else _p2n.get(_k)\n'
        '                    if _name:\n'
        '                        _tpot[_name] = _v * _scale\n'
        '            self.policy = KVPRGlobalPolicyV3(\n'
        '                num_gpus=len(self.gpu_ids), gpu_mem=gpu_mem,\n'
        '                model_weights_info=self.model_weights_info_after_renamed,\n'
        '                workers_per_gpu=self.server_args.workers_per_gpu,\n'
        '                tau=float(getattr(self.server_args, "kvpr_tau", 0.35)),\n'
        '                rate_window=float(getattr(self.server_args, "kvpr_rate_window", 30.0)),\n'
        '                migration_cooldown=float(getattr(self.server_args, "kvpr_migration_cooldown", 30.0)),\n'
        '                tpot_slo_s=_tpot,\n'
        '            )\n'
    )
    replace(controller,
        '        else:\n            raise ValueError(f"Unknown policy: {self.server_args.policy}")',
        branch + '        else:\n            raise ValueError(f"Unknown policy: {self.server_args.policy}")',
        probe='self.server_args.policy == "kvpr-global-v3"')
    replace(controller,
        '        logger.info(f"Using policy: {self.policy.__class__.__name__}")\n',
        '        self.policy._target_first_actions = bool(\n'
        '            getattr(self.server_args, "overlap_migration", False)\n'
        '        )\n'
        '        logger.info(f"Using policy: {self.policy.__class__.__name__}")\n',
        probe='self.policy._target_first_actions =')

    execute_actions = '''    def execute_actions(self, actions: List[BaseAction]):
        """Execute actions, with a readiness barrier for overlap migration."""
        tic = time.time()
        overlap_migration = getattr(self.server_args, "overlap_migration", False)
        if not overlap_migration:
            batches = [(actions, len(actions))]
        else:
            # The HTTP control path cannot safely process two concurrent model
            # activations for the same GPU. Serialize activations, then apply
            # unrelated actions, and only then retire source copies.
            activations = [a for a in actions if isinstance(a, ActivateAction)]
            middle = [
                a for a in actions
                if not isinstance(a, (ActivateAction, DeactivateAction))
            ]
            deactivations = [a for a in actions if isinstance(a, DeactivateAction)]
            batches = [
                (batch, workers)
                for batch, workers in (
                    (activations, 1),
                    (middle, len(middle)),
                    (deactivations, len(deactivations)),
                )
                if batch
            ]

        failed_activation_models = set()
        for batch, max_workers in batches:
            if overlap_migration and isinstance(batch[0], DeactivateAction):
                safe_batch = []
                for action in batch:
                    if action.model_name not in failed_activation_models:
                        safe_batch.append(action)
                        continue
                    active_copies = sum(
                        instance.state == ModelState.ACTIVE
                        for instance in self.model_instance_state_dict[action.model_name]
                    )
                    if active_copies > 1:
                        safe_batch.append(action)
                    else:
                        logger.error(
                            "Skipping source deactivation for %s after target "
                            "activation failed",
                            action.model_name,
                        )
                batch = safe_batch
                if not batch:
                    continue

            with ThreadPoolExecutor(max_workers=max_workers) as threads:
                future_to_action = {
                    threads.submit(
                        action.execute,
                        self.server_args.url(),
                        self.model_instance_state_dict,
                        600.0 if overlap_migration else None,
                    ): action
                    for action in batch
                }
                for future, action in future_to_action.items():
                    try:
                        future.result()
                    except Exception:
                        if isinstance(action, ActivateAction):
                            failed_activation_models.add(action.model_name)
                        logger.error(f"Error executing action: {get_exception_traceback()}")
        toc = time.time()
        logger.info(
            f"Executed {len(actions)} actions in {toc - tic:.2f} seconds. Actions: {actions}"
        )
        self.print_model_instance_state_dict()

'''
    replace_block(
        controller,
        "    def execute_actions(self, actions: List[BaseAction]):\n",
        "    def _init_model_instance_state_dict(\n",
        execute_actions,
    )

    args = mm / "multi_model_server_args.py"
    replace(args, '    kvpr_tpot_slo_scale: float = 1.0\n',
        '    kvpr_tpot_slo_scale: float = 1.0\n'
        '    parallel_model_loading: bool = False  # PAPER-FAITHFUL-V3\n'
        '    overlap_migration: bool = False       # PAPER-FAITHFUL-V3\n',
        probe='parallel_model_loading: bool')
    replace(args, '                "kvpr-global",   # PAPER-FAITHFUL\n',
        '                "kvpr-global",   # PAPER-FAITHFUL\n                "kvpr-global-v3",\n',
        probe='                "kvpr-global-v3",')
    replace(args, '        parser.add_argument("--kvpr-tpot-slo-scale", type=float,\n',
        '        parser.add_argument("--parallel-model-loading", action="store_true",\n'
        '            help="Load initial model placements concurrently (v3).")\n'
        '        parser.add_argument("--overlap-migration", action="store_true",\n'
        '            help="Activate target before draining source during migration (v3).")\n'
        '        parser.add_argument("--kvpr-tpot-slo-scale",\n',
        probe='parser.add_argument("--parallel-model-loading"')

    server_args = repo / "python/sglang/srt/server_args.py"
    replace(server_args, '            "kvpr_tpot_slo_scale",\n',
        '            "kvpr_tpot_slo_scale",\n            "parallel_model_loading",\n            "overlap_migration",\n',
        probe='            "parallel_model_loading",')
    server = mm / "multi_model_server.py"
    replace(server, '    num_model_service_workers = multi_model_server_args.num_model_service_workers\n    num_model_service_workers = 1\n',
        '    num_model_service_workers = multi_model_server_args.num_model_service_workers\n')

    gpu = mm / "scheduling/gpu/gpu_scheduler.py"
    replace(gpu, 'import atexit\n', 'import atexit\nfrom concurrent.futures import ThreadPoolExecutor\n',
        probe='from concurrent.futures import ThreadPoolExecutor')
    old = ('        for model_name in init_model_names:\n'
           '            self._set_model_state(model_name, "activating")\n'
           '            activate_req = ActivateReqInput(\n'
           '                model_name=model_name,\n'
           '                instance_idx=self.gpu_id,\n'
           '                gpu_id=self.gpu_id,\n'
           '            )\n'
           '            self.worker_pool.handle_activate_model(activate_req)\n')
    new = ('        def _activate(model_name):\n'
           '            self._set_model_state(model_name, "activating")\n'
           '            return self.worker_pool.handle_activate_model(ActivateReqInput(\n'
           '                model_name=model_name, instance_idx=self.gpu_id, gpu_id=self.gpu_id))\n'
           '        if getattr(self.server_args, "parallel_model_loading", False) and len(init_model_names) > 1:\n'
           '            with ThreadPoolExecutor(max_workers=len(init_model_names)) as pool:\n'
           '                list(pool.map(_activate, init_model_names))\n'
           '        else:\n'
           '            for model_name in init_model_names:\n'
           '                _activate(model_name)\n')
    replace(gpu, old, new)

    handler = mm / "request_handler_worker_pool.py"
    old = ('    async def deactivate(self, req: DeactivateReqInput):\n'
           '        logger.info(f"Sending deactivate request to GPU scheduler {req.gpu_id}")\n'
           '        self._send_req_to_gpu_scheduler(req)\n'
           '        return True, None\n\n'
           '    async def activate(self, req: ActivateReqInput):\n'
           '        logger.info(f"Sending activate request to GPU scheduler {req.gpu_id}")\n'
           '        self._send_req_to_gpu_scheduler(req)\n'
           '        return True, None\n')
    new = ('    async def deactivate(self, req: DeactivateReqInput):\n'
           '        logger.info(f"Sending deactivate request to GPU scheduler {req.gpu_id}")\n'
           '        if getattr(self.server_args, "overlap_migration", False):\n'
           '            return await self._send_req_and_wait_for_response(req)\n'
           '        self._send_req_to_gpu_scheduler(req)\n'
           '        return True, None\n\n'
           '    async def activate(self, req: ActivateReqInput):\n'
           '        logger.info(f"Sending activate request to GPU scheduler {req.gpu_id}")\n'
           '        if getattr(self.server_args, "overlap_migration", False):\n'
           '            return await self._send_req_and_wait_for_response(req)\n'
           '        self._send_req_to_gpu_scheduler(req)\n'
           '        return True, None\n')
    replace(handler, old, new)
    replace(handler,
        '        # send request to the corresponding gpu scheduler\n'
        '        self._send_req_to_gpu_scheduler(req)\n\n'
        '        # await for the response\n'
        '        rid = req.rid\n'
        '        event = asyncio.Event()\n'
        '        state = ReqState([], False, event)\n'
        '        self.rid_to_state[rid] = state\n',
        '        # Register the waiter before sending so a fast response cannot be lost.\n'
        '        rid = req.rid\n'
        '        event = asyncio.Event()\n'
        '        state = ReqState([], False, event)\n'
        '        self.rid_to_state[rid] = state\n'
        '        self._send_req_to_gpu_scheduler(req)\n',
        probe='Register the waiter before sending')

    action = mm / "scheduling/action.py"
    replace(action,
        '        self, url: str, model_instance_state_dict: Dict[str, List[ModelInstanceState]]\n'
        '    ):',
        '        self, url: str, model_instance_state_dict: Dict[str, List[ModelInstanceState]],\n'
        '        request_timeout: Optional[float] = None,\n'
        '    ):',
        count=4,
        probe='request_timeout: Optional[float] = None')
    replace(action, 'timeout=REQUEST_TIMEOUT',
        'timeout=request_timeout or REQUEST_TIMEOUT', count=3,
        probe='timeout=request_timeout or REQUEST_TIMEOUT')

    simple = mm / "scheduling/policy/simple_global.py"
    replace(simple,
        '        model_queues: Dict[str, ModelQueueTracker],\n'
        '        model_instance_state_dict: Dict[str, List[ModelInstanceState]]\n'
        '    ) -> Optional[int]:',
        '        model_queues: Dict[str, ModelQueueTracker],\n'
        '        model_instance_state_dict: Dict[str, List[ModelInstanceState]],\n'
        '        gpu_planned_active_count: Optional[Dict[int, int]] = None,\n'
        '    ) -> Optional[int]:',
        probe='gpu_planned_active_count: Optional[Dict[int, int]] = None')
    replace(simple,
        '            for gpu_id in cluster:\n'
        '                # Check if the GPU has enough memory\n',
        '            for gpu_id in cluster:\n'
        '                # Do not reserve more models than the worker pool can serve.\n'
        '                if (gpu_planned_active_count is not None and\n'
        '                        gpu_planned_active_count[gpu_id] >= self.workers_per_gpu):\n'
        '                    continue\n\n'
        '                # Check if the GPU has enough memory\n',
        probe='Do not reserve more models than the worker pool can serve.')
    replace(simple,
        '        activation_plan = {}\n'
        '        for model_name in sorted_inactive_models:\n',
        '        activation_plan = {}\n'
        '        # Include target slots reserved earlier in this cycle. Source slots\n'
        '        # remain occupied until target-first activation has completed.\n'
        '        gpu_planned_active_count = {\n'
        '            gpu_id: len(gpu_to_active_instances[gpu_id])\n'
        '            for gpu_id in range(self.num_gpus)\n'
        '        }\n'
        '        for _, _, target_gpu_id in migrations:\n'
        '            gpu_planned_active_count[target_gpu_id] += 1\n'
        '        for model_name in sorted_inactive_models:\n',
        probe='Include target slots reserved earlier in this cycle.')
    replace(simple,
        '                gpu_available_memory, sorted_clusters, model_queues, model_instance_state_dict\n'
        '            )',
        '                gpu_available_memory, sorted_clusters, model_queues,\n'
        '                model_instance_state_dict, gpu_planned_active_count,\n'
        '            )',
        probe='model_instance_state_dict, gpu_planned_active_count,')
    replace(simple,
        '            if target_gpu is not None:\n'
        '                activation_plan[(model_name, target_gpu)] = target_gpu\n',
        '            if target_gpu is not None:\n'
        '                activation_plan[(model_name, target_gpu)] = target_gpu\n'
        '                gpu_planned_active_count[target_gpu] += 1\n',
        probe='gpu_planned_active_count[target_gpu] += 1')
    action_order = '''        all_actions = []
        _target_first = bool(getattr(self, "_target_first_actions", False))

        for actions in model_instance_to_action_dict.values():
            sorted_actions = sorted(actions, key=lambda action: (
                1 if isinstance(action, DeactivateAction) else 0) if _target_first else (
                0 if isinstance(action, DeactivateAction) else 1))
            all_actions.extend(sorted_actions)

        # Migration actions live under separate (model, instance) keys. A
        # per-key sort alone cannot guarantee target-first ordering globally.
        if _target_first:
            all_actions.sort(key=lambda action: 1 if isinstance(action, DeactivateAction) else 0)

        return all_actions'''
    replace_last_block(
        simple,
        '        all_actions = []\n',
        '        return all_actions',
        action_order,
    )

    checks = {
        controller: ["KVPRGlobalPolicyV3", "failed_activation_models", "ActivateAction,", "_target_first_actions"],
        handler: ["Register the waiter before sending", "overlap_migration"],
        action: ["request_timeout: Optional[float] = None", "request_timeout or REQUEST_TIMEOUT"],
        simple: ["gpu_planned_active_count", "worker pool can serve"],
        args: ["kvpr-global-v3", "parallel_model_loading", "overlap_migration"],
        gpu: ["parallel_model_loading", "ThreadPoolExecutor"],
    }
    missing = [f"{path}: {needle}" for path, needles in checks.items()
               for needle in needles if needle not in path.read_text()]
    if missing:
        raise RuntimeError("v3 patch verification failed:\n" + "\n".join(missing))
    print("paper-faithful-v3 applied")

if __name__ == "__main__":
    main()
