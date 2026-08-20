#!/usr/bin/env python3
"""Tensor parallelism in the worker-pool path, plus the paper's TP anti-affinity.

Everything here is behind ``--enable-tp-worker-pool`` and, for the placement
constraint, ``--enable-tp-anti-affinity``.  With both off the released
prototype's behaviour is reproduced line for line: the slot plan degenerates to
the prototype's ``(gpu, worker)`` grid and no other edited path is reached.

Why TP=2 died in the released prototype
---------------------------------------
The recorded failure is ``Model model_5 not found in shared cpu models`` at
activation, and the handoff read it as a third, independent defect in the
weight lookup.  It is not independent -- it is a symptom, and the cause is one
line further up than the handoff guessed.

``ServerArgs.from_multi_model_server_args`` lists ``"tp_size"`` in
``keys_to_remove`` (``server_args.py:265``) and pops it (``:276-277``).  The
worker-pool call site passes no ``instance_config``, so the branch that would
restore it (``:286``) never runs and ``tp_size`` falls back to the class default
of 1 (``:74``).  So ``--tensor-parallel-size 2`` was **silently discarded** on
its way to a worker-pool engine.

The decisive evidence is that the *uniform*-TP attempt failed the same way.  Had
tp_size really arrived as 2, ``launch_engine``'s
``assert len(tp_rank_range) == len(gpu_ids)`` would have fired first on 2 != 1.
It did not; the run got as far as the weight lookup, and the recorded error
lines are labelled ``TP0`` for both ``Worker=0`` and ``Worker=1``.  Rank 1 was
never created.

The CPU side was never broken: ``load_shared_cpu_models`` already keys by
``(model_path, tp_size)`` and already loads one shard per rank.  Restoring
tp_size makes the key match, so this patch does not touch it.

What actually needed building
-----------------------------
An engine's ``tp_size`` is fixed when the server starts, but which model lands
in a worker slot is decided at runtime.  So the slot has to carry the type --
see ``tp_slots.py`` for the layout and for why a group's slot id must be global.

Two things that were expected to be hard turned out not to be, and both are
recorded because the handoff budgeted for them:

* **No distributed slot acquisition.**  Group membership is static, so
  activation touches exactly one worker pool (rank0's) and there is no
  partially-acquired group to roll back.
* **No control-plane work.**  Upstream already broadcasts activate/deactivate
  across a TP group: only rank0 binds ``recv_from_gpu_scheduler``
  (``scheduler.py:157-168``) and ``recv_gpu_scheduler_requests`` fans it out
  with ``broadcast_pyobj`` when ``tp_size != 1`` (``scheduler.py:628-646``).

A limitation that is kept rather than worked around: ``model_runner.py:133-136``
disables the model-service weight path when ``tp_size > 1``, so a TP model uses
neither v4's parallel weight loading nor its P2P migration.  TP-arm timings are
therefore not comparable with non-TP-arm timings, and the report says so.
"""

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def replace(path, old, new, probe=None):
    """Apply one edit, idempotently, and refuse to guess which site was meant.

    Two failure modes this guards against, both of which have actually bitten
    this project:

    * a non-unique anchor.  ``text.replace(old, new, 1)`` silently takes the
      FIRST match, so an anchor that also appears in, say, ``__init__`` inserts
      the edit somewhere harmless-looking and the server dies at startup.  An
      ambiguous anchor is a bug in the patch, not something to resolve by
      position, so it raises.
    * a non-unique probe.  If the probe string is one another edit already
      wrote, this edit is skipped without a word.  So the probe is required to
      be absent-or-ours, and it must not match more than once.
    """
    text = path.read_text()
    if probe is not None:
        hits = text.count(probe)
        if hits > 1:
            raise RuntimeError(
                f"probe is not unique in {path} ({hits} matches): {probe[:100]!r}"
            )
        if hits == 1:
            return
    n = text.count(old)
    if n == 0:
        if new in text:
            return
        raise RuntimeError(f"anchor not found in {path}: {old[:140]!r}")
    if n > 1:
        raise RuntimeError(
            f"anchor is not unique in {path} ({n} matches): {old[:140]!r}"
        )
    path.write_text(text.replace(old, new, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=str(ROOT / "prism-research"))
    ns = ap.parse_args()
    repo = Path(ns.repo).resolve()
    mm = repo / "python/sglang/multi_model"
    args_py = mm / "multi_model_server_args.py"
    server = mm / "multi_model_server.py"
    pool = mm / "scheduling/gpu/worker_pool.py"
    resource = mm / "scheduling/gpu/resource_manager.py"
    gpu_sched = mm / "scheduling/gpu/gpu_scheduler.py"
    runner = repo / "python/sglang/srt/model_executor/model_runner.py"

    # The slot plan is a pure function of the server args, so the launcher and
    # every per-GPU scheduler process compute it independently and agree.
    shutil.copyfile(HERE / "tp_slots.py", mm / "tp_slots.py")

    # A model the GPU scheduler has never been told about kills it at startup:
    #   ValueError: Model path ... not found in the profiled model info file
    # and the traceback is then masked by GPUScheduler.__del__ raising
    # AttributeError on a half-built object.  The 70B TP target is added here
    # rather than in v4's copy of model_info.json, so the two patches do not
    # fight over the same file's contents.
    #
    # cell_size is 2 (K and V) x layers x kv_heads x head_dim x dtype_bytes,
    # which reproduces the committed 131072 for Llama-3.1-8B exactly.
    # model_size is the measured on-disk safetensors total in GiB.
    # NOTE: the upstream file's own large-model rows are not trustworthy --
    # meta-llama/Llama-3.3-70B-Instruct is listed at model_size 17.5 GiB, which
    # is off by ~7.5x.  Only the row added here was checked against the weights.
    import json as _json
    _info_path = mm / "utils/model_info.json"
    _info = _json.loads(_info_path.read_text())
    _extra = _json.loads((HERE / "tp_model_info.json").read_text())
    if any(_info.get(k) != v for k, v in _extra.items()):
        _info.update(_extra)
        _info_path.write_text(_json.dumps(_info, indent=4))

    # ------------------------------------------------------------------ args
    replace(args_py,
        "    parallel_model_loading: bool = False  # PAPER-FAITHFUL-V3\n"
        "    overlap_migration: bool = False       # PAPER-FAITHFUL-V3\n",
        "    parallel_model_loading: bool = False  # PAPER-FAITHFUL-V3\n"
        "    overlap_migration: bool = False       # PAPER-FAITHFUL-V3\n"
        "    # PAPER-FAITHFUL-TP.  Both default off: the released prototype is\n"
        "    # reproduced exactly unless they are asked for.\n"
        "    enable_tp_worker_pool: bool = False\n"
        "    tp_max_groups: int = 0            # 0 = every k-subset of the GPUs\n"
        "    enable_tp_anti_affinity: bool = False\n"
        "    enable_tp_anti_affinity_strict: bool = False\n",
        probe="enable_tp_worker_pool: bool")

    replace(args_py,
        "        parser.add_argument(\"--overlap-migration\", action=\"store_true\",\n"
        "            help=\"Activate target before draining source during migration (v3).\")\n",
        "        parser.add_argument(\"--overlap-migration\", action=\"store_true\",\n"
        "            help=\"Activate target before draining source during migration (v3).\")\n"
        "        # PAPER-FAITHFUL-TP\n"
        "        parser.add_argument(\"--enable-tp-worker-pool\", action=\"store_true\",\n"
        "            help=\"Let worker-pool engines span several GPUs so TP>1 models can run \"\n"
        "                 \"under the GPU scheduler and the migration machinery.\")\n"
        "        parser.add_argument(\"--tp-max-groups\", type=int,\n"
        "            default=MultiModelServerArgs.tp_max_groups,\n"
        "            help=\"Cap on TP groups enumerated per tp_size (0 = all). Dropped \"\n"
        "                 \"groups are logged, never silently omitted.\")\n"
        "        parser.add_argument(\"--enable-tp-anti-affinity\", action=\"store_true\",\n"
        "            help=\"Paper Appendix A.2.2: a TP part whose minimum-KVPR GPU already \"\n"
        "                 \"holds another part of the same model goes to the second-lowest \"\n"
        "                 \"KVPR GPU instead.\")\n"
        "        parser.add_argument(\"--enable-tp-anti-affinity-strict\", action=\"store_true\",\n"
        "            help=\"Stronger than the paper: pick the lowest-KVPR GPU that holds no \"\n"
        "                 \"part of this model. Differs from A.2.2 only for tp_size>=3, where \"\n"
        "                 \"the paper's second-lowest GPU may itself collide.\")\n",
        probe="--enable-tp-worker-pool")

    # The multi-model args are splatted into ServerArgs after keys_to_remove is
    # applied, so any field that exists only on MultiModelServerArgs must be
    # listed there or ServerArgs.__init__ rejects it.  Same treatment the v2
    # flags already get two lines above.
    # Anchored on parallel_model_loading alone, NOT on the set's closing brace.
    # The v6 KV-migration patch appends its own key just before that brace, so
    # an anchor spanning it matches only when this patch runs first -- and the
    # two patches are developed on separate branches with no fixed order.  This
    # line is unique in the file and neither patch consumes it.
    replace(repo / "python/sglang/srt/server_args.py",
        "            \"parallel_model_loading\",\n",
        "            \"parallel_model_loading\",\n"
        "            \"enable_tp_worker_pool\",   # PAPER-FAITHFUL-TP\n"
        "            \"tp_max_groups\",\n"
        "            \"enable_tp_anti_affinity\",\n"
        "            \"enable_tp_anti_affinity_strict\",\n",
        probe="\"enable_tp_worker_pool\",   # PAPER-FAITHFUL-TP")

    # ---------------------------------------------------------------- server
    replace(server,
        "from sglang.multi_model.multi_model_server_args import MultiModelServerArgs\n",
        "from sglang.multi_model.multi_model_server_args import MultiModelServerArgs\n"
        "from sglang.multi_model.tp_slots import (  # PAPER-FAITHFUL-TP\n"
        "    plan_for_server_args as tp_slot_plan_for,\n"
        ")\n",
        probe="from sglang.multi_model.tp_slots import")

    # The model service needs an id for every engine that exists, including the
    # TP groups -- otherwise output_queues[engine_id] raises at launch.
    replace(server,
        "            engine_ids = []\n"
        "            workers_per_gpu = multi_model_server_args.workers_per_gpu\n"
        "            num_gpus = multi_model_server_args.num_gpus\n"
        "            for gpu_id in range(num_gpus):\n"
        "                for worker_id in range(workers_per_gpu):\n"
        "                    engine_ids.append(f\"{gpu_id}_{worker_id}\")\n",
        "            # PAPER-FAITHFUL-TP: one id per engine in the slot plan.  A TP\n"
        "            # group is named after rank0's GPU, which is where its control\n"
        "            # socket is bound.\n"
        "            engine_ids = [\n"
        "                f\"{_s.owner_gpu}_{_s.worker_id}\"\n"
        "                for _s in tp_slot_plan_for(multi_model_server_args).slots\n"
        "            ]\n",
        probe="one id per engine in the slot plan")

    # The wiring itself.
    replace(server,
        "    engine_launch_args = []\n"
        "    start_port = multi_model_server_args.port\n"
        "    workers_per_gpu = multi_model_server_args.workers_per_gpu\n"
        "    num_gpus = multi_model_server_args.num_gpus\n"
        "    for gpu_id in range(num_gpus):\n"
        "        for worker_id in range(workers_per_gpu):\n"
        "            engine_id = f\"{gpu_id}_{worker_id}\"\n"
        "            server_args = ServerArgs.from_multi_model_server_args(\n"
        "                multi_model_server_args=multi_model_server_args,\n"
        "                worker_id=worker_id,\n"
        "            )\n"
        "            # Allocate ports for inter-process communications\n"
        "            port_args = PortArgs.init_with_request_handler_ipc_name(\n"
        "                start_port, request_handler_ipc_name, schedulers_to_controller_ipc_name\n"
        "            )\n"
        "            start_port = port_args.nccl_port\n"
        "            engine_launch_args.append(\n"
        "                (\n"
        "                    server_args,\n"
        "                    port_args,\n"
        "                    [gpu_id],\n"
        "                    worker_id,\n",
        "    engine_launch_args = []\n"
        "    start_port = multi_model_server_args.port\n"
        "    # PAPER-FAITHFUL-TP: iterate the slot plan rather than a (gpu, worker)\n"
        "    # grid.  A TP=k slot is one engine spanning k GPUs; with TP off the plan\n"
        "    # is exactly the grid and this loop is the prototype's.\n"
        "    _tp_plan = tp_slot_plan_for(multi_model_server_args)\n"
        "    logger.info(\"[PAPER-TP] slot plan: \" + json.dumps(_tp_plan.as_dict()))\n"
        "    for _dropped_k, _dropped_combo in getattr(_tp_plan, \"dropped_groups\", []):\n"
        "        logger.warning(\n"
        "            \"[PAPER-TP] group dropped (not silently omitted): \"\n"
        "            f\"tp_size={_dropped_k} gpus={_dropped_combo or 'none satisfiable'}\"\n"
        "        )\n"
        "    for _slot in _tp_plan.slots:\n"
        "        if True:\n"
        "            gpu_id = _slot.owner_gpu\n"
        "            worker_id = _slot.worker_id\n"
        "            engine_id = f\"{gpu_id}_{worker_id}\"\n"
        "            server_args = ServerArgs.from_multi_model_server_args(\n"
        "                multi_model_server_args=multi_model_server_args,\n"
        "                worker_id=worker_id,\n"
        "            )\n"
        "            # from_multi_model_server_args drops tp_size (it is in\n"
        "            # keys_to_remove) and the worker-pool call passes no\n"
        "            # instance_config, so tp_size silently became 1 no matter what\n"
        "            # --tensor-parallel-size said.  That is why TP=2 died in the\n"
        "            # shared-CPU lookup: the engine asked for (path, 1) while the\n"
        "            # loader had stored (path, 2).  Restore it from the slot.\n"
        "            server_args.tp_size = _slot.tp_size\n"
        "            # Allocate ports for inter-process communications\n"
        "            port_args = PortArgs.init_with_request_handler_ipc_name(\n"
        "                start_port, request_handler_ipc_name, schedulers_to_controller_ipc_name\n"
        "            )\n"
        "            start_port = port_args.nccl_port\n"
        "            engine_launch_args.append(\n"
        "                (\n"
        "                    server_args,\n"
        "                    port_args,\n"
        "                    list(_slot.gpu_ids),\n"
        "                    worker_id,\n",
        probe="server_args.tp_size = _slot.tp_size")

    # ----------------------------------------------------------- worker pool
    replace(pool,
        "from sglang.utils import cleanup_zmq_ipc\n",
        "from sglang.multi_model.tp_slots import TPSlot  # PAPER-FAITHFUL-TP\n"
        "from sglang.utils import cleanup_zmq_ipc\n",
        probe="from sglang.multi_model.tp_slots import TPSlot")

    replace(pool,
        "    def __init__(self, num_workers: int, gpu_id: int, zmq_context: zmq.Context):\n",
        "    def __init__(\n"
        "        self,\n"
        "        num_workers: int,\n"
        "        gpu_id: int,\n"
        "        zmq_context: zmq.Context,\n"
        "        slot_plan=None,            # PAPER-FAITHFUL-TP\n"
        "        model_tp_sizes=None,       # PAPER-FAITHFUL-TP\n"
        "    ):\n",
        probe="slot_plan=None,            # PAPER-FAITHFUL-TP")

    replace(pool,
        "        self.num_workers = num_workers\n"
        "        self.gpu_id = gpu_id\n"
        "        self._model_to_worker: Dict[str, int] = {}\n"
        "        self._free_workers: Deque[int] = deque()\n"
        "        self._worker_to_ipc_name: Dict[int, zmq.Socket] = {}\n"
        "        self.zmq_context = zmq_context\n"
        "        self._init_workers()\n",
        "        self.num_workers = num_workers\n"
        "        self.gpu_id = gpu_id\n"
        "        self._model_to_worker: Dict[str, int] = {}\n"
        "        self._free_workers: Deque[int] = deque()\n"
        "        self._worker_to_ipc_name: Dict[int, zmq.Socket] = {}\n"
        "        self.zmq_context = zmq_context\n"
        "        # PAPER-FAITHFUL-TP.  Slots are typed by tp_size, because an\n"
        "        # engine's tp_size is fixed at server start while the model that\n"
        "        # occupies the slot is chosen here at runtime.\n"
        "        #\n"
        "        # A group's non-rank0 GPUs hold the same slot id as a *shadow*:\n"
        "        # reserved permanently, never assignable.  That reservation is what\n"
        "        # keeps a second engine off the group's accounting segment\n"
        "        # (ipc_{gpu}_{worker}_{user}), and because it is static there is no\n"
        "        # partially-acquired group to roll back -- activation touches only\n"
        "        # rank0's pool.\n"
        "        if slot_plan is None:\n"
        "            self._slots = {\n"
        "                i: TPSlot(worker_id=i, tp_size=1, gpu_ids=(gpu_id,))\n"
        "                for i in range(num_workers)\n"
        "            }\n"
        "        else:\n"
        "            self._slots = {s.worker_id: s for s in slot_plan.slots_on(gpu_id)}\n"
        "        self._owned_ids = sorted(\n"
        "            wid for wid, s in self._slots.items()\n"
        "            if s.role_on(gpu_id) == \"owner\"\n"
        "        )\n"
        "        self._shadow_ids = sorted(\n"
        "            wid for wid, s in self._slots.items()\n"
        "            if s.role_on(gpu_id) == \"shadow\"\n"
        "        )\n"
        "        self._model_tp_sizes = dict(model_tp_sizes or {})\n"
        "        self._init_workers()\n"
        "        logger.info(\n"
        "            f\"[PAPER-TP] WorkerPool gpu={gpu_id} owned={self._owned_ids} \"\n"
        "            f\"shadow={self._shadow_ids} \"\n"
        "            + str({wid: s.tp_size for wid, s in sorted(self._slots.items())})\n"
        "        )\n",
        probe="PAPER-FAITHFUL-TP.  Slots are typed by tp_size")

    replace(pool,
        "    def _init_workers(self):\n"
        "        \"\"\"Initialize all workers.\"\"\"\n"
        "        for i in range(self.num_workers):\n"
        "            worker = Worker(i, self.gpu_id)\n"
        "            self._free_workers.append(worker.worker_id)\n"
        "            self._worker_to_ipc_name[i] = self.zmq_context.socket(zmq.PUSH)\n"
        "            self._worker_to_ipc_name[i].connect(f\"ipc://{worker.ipc_name}\")\n",
        "    def _init_workers(self):\n"
        "        \"\"\"Initialize the slots this GPU owns.\n"
        "\n"
        "        Shadow slots get neither a free-list entry nor a socket: nothing\n"
        "        binds ``gpu_scheduler_{this gpu}_to_worker_{id}`` for them, because\n"
        "        only rank0 binds a group's control socket (scheduler.py:157-168).\n"
        "        Connecting anyway would queue activations into a socket no process\n"
        "        ever reads.\n"
        "        \"\"\"\n"
        "        for i in self._owned_ids:\n"
        "            worker = Worker(i, self.gpu_id)\n"
        "            self._free_workers.append(worker.worker_id)\n"
        "            self._worker_to_ipc_name[i] = self.zmq_context.socket(zmq.PUSH)\n"
        "            self._worker_to_ipc_name[i].connect(f\"ipc://{worker.ipc_name}\")\n",
        probe="Shadow slots get neither a free-list entry nor a socket")

    replace(pool,
        "    def get_idle_worker(self):\n"
        "        \"\"\"\n"
        "        Get an idle worker.\n"
        "        \n"
        "        Returns:\n"
        "            worker_id: Idle worker ID, or None if no idle worker available\n"
        "        \"\"\"\n"
        "        if len(self._free_workers) == 0:\n"
        "            return None\n"
        "        return self._free_workers[0]\n",
        "    def get_idle_worker(self, tp_size: int = 1):\n"
        "        \"\"\"An idle slot whose tp_size matches, or None.\n"
        "\n"
        "        A TP=1 model must not take a TP=k engine and vice versa: the engine\n"
        "        already holds its rank layout and its shard of the CPU weights, so\n"
        "        the two cannot meet.\n"
        "        \"\"\"\n"
        "        for worker_id in self._free_workers:\n"
        "            slot = self._slots.get(worker_id)\n"
        "            if slot is not None and slot.tp_size == tp_size:\n"
        "                return worker_id\n"
        "        return None\n",
        probe="An idle slot whose tp_size matches")

    replace(pool,
        "        worker_id = self.get_idle_worker()\n"
        "        if worker_id is None:\n"
        "            logger.error(f\"No idle worker found for GPU {self.gpu_id}\")\n"
        "            return False\n",
        "        # PAPER-FAITHFUL-TP: match the model's TP size to a slot of that type.\n"
        "        tp_size = self._model_tp_sizes.get(model_name, 1)\n"
        "        worker_id = self.get_idle_worker(tp_size)\n"
        "        if worker_id is None:\n"
        "            logger.error(\n"
        "                f\"No idle tp_size={tp_size} worker for GPU {self.gpu_id}; \"\n"
        "                f\"free={list(self._free_workers)} \"\n"
        "                + str({w: self._slots[w].tp_size for w in self._free_workers\n"
        "                       if w in self._slots})\n"
        "            )\n"
        "            return False\n"
        "        _slot = self._slots[worker_id]\n"
        "        if _slot.tp_size > 1:\n"
        "            logger.info(\n"
        "                \"[PAPER-TP] activating \"\n"
        "                f\"{model_name} tp_size={_slot.tp_size} on gpus={list(_slot.gpu_ids)} \"\n"
        "                f\"slot={worker_id} (owner gpu={_slot.owner_gpu})\"\n"
        "            )\n",
        probe="match the model's TP size to a slot of that type")

    # ------------------------------------------------------- resource manager
    replace(resource,
        "        num_workers: Optional[int] = None,\n"
        "    ):\n",
        "        num_workers: Optional[int] = None,\n"
        "        slot_plan=None,          # PAPER-FAITHFUL-TP\n"
        "        model_tp_sizes=None,     # PAPER-FAITHFUL-TP\n"
        "    ):\n",
        probe="slot_plan=None,          # PAPER-FAITHFUL-TP")

    replace(resource,
        "        self.num_workers = num_workers\n"
        "        \n"
        "        self._init_model_names_mappings(engine_info_dict)\n",
        "        self.num_workers = num_workers\n"
        "        # PAPER-FAITHFUL-TP: a GPU's slot ids are a sparse subset once TP\n"
        "        # groups exist, because a group's id has to be free on every GPU it\n"
        "        # spans and is therefore allocated globally.  range(num_workers)\n"
        "        # would name segments that do not exist on this GPU.\n"
        "        if slot_plan is None:\n"
        "            self._worker_ids = list(range(num_workers or 0))\n"
        "        else:\n"
        "            self._worker_ids = slot_plan.worker_ids_on(gpu_id)\n"
        "        self._model_tp_sizes = dict(model_tp_sizes or {})\n"
        "        \n"
        "        self._init_model_names_mappings(engine_info_dict)\n",
        probe="a GPU's slot ids are a sparse subset")

    replace(resource,
        "            self._worker_to_mem_reader = {}\n"
        "            for worker_id in range(self.num_workers):\n"
        "                self._worker_to_mem_reader[worker_id] = MemoryUsageReader(\n"
        "                    f\"ipc_{self.gpu_id}_{worker_id}_{self.username}\"\n"
        "                )\n",
        "            self._worker_to_mem_reader = {}\n"
        "            for worker_id in self._worker_ids:  # PAPER-FAITHFUL-TP sparse (readers)\n"
        "                self._worker_to_mem_reader[worker_id] = MemoryUsageReader(\n"
        "                    f\"ipc_{self.gpu_id}_{worker_id}_{self.username}\"\n"
        "                )\n",
        probe="PAPER-FAITHFUL-TP sparse (readers)")

    replace(resource,
        "            used_sum = 0\n"
        "            for worker_id in range(self.num_workers):\n"
        "                used_sum += self._worker_to_mem_reader[\n"
        "                    worker_id\n"
        "                ].get_memory_usage_in_bytes()\n"
        "            return used_sum\n",
        "            used_sum = 0\n"
        "            for worker_id in self._worker_ids:  # PAPER-FAITHFUL-TP sparse (usage)\n"
        "                used_sum += self._worker_to_mem_reader[\n"
        "                    worker_id\n"
        "                ].get_memory_usage_in_bytes()\n"
        "            return used_sum\n",
        probe="PAPER-FAITHFUL-TP sparse (usage)")

    # A TP=k model puts 1/k of its weights on each GPU it spans.  Charging the
    # full size on every GPU would understate free KV memory k-fold and starve
    # Algorithm 2's feasibility test.
    replace(resource,
        "            self._model_names_to_weights_memory = {\n"
        "                model_name: model_path_to_model_size[\n"
        "                    self.model_names_to_model_paths[model_name]\n"
        "                ]\n"
        "                for model_name in self.model_names_to_model_paths.keys()\n"
        "            }",          # 이 파일은 개행 없이 끝난다
        "            # PAPER-FAITHFUL-TP: weights are sharded, so this GPU holds\n"
        "            # model_size / tp_size, not model_size.\n"
        "            self._model_names_to_weights_memory = {\n"
        "                model_name: model_path_to_model_size[\n"
        "                    self.model_names_to_model_paths[model_name]\n"
        "                ] / max(1, self._model_tp_sizes.get(model_name, 1))\n"
        "                for model_name in self.model_names_to_model_paths.keys()\n"
        "            }",
        probe="weights are sharded, so this GPU holds")

    # -------------------------------------------------------- gpu scheduler
    # tp_slots is a leaf module: importing it here creates no cycle, whereas
    # importing multi_model_server would (it already imports gpu_scheduler at
    # :66) and would pull FastAPI/uvicorn into every scheduler process.
    replace(gpu_sched,
        "from sglang.multi_model.scheduling.gpu.worker_pool import WorkerPool\n",
        "from sglang.multi_model.scheduling.gpu.worker_pool import WorkerPool\n"
        "from sglang.multi_model.tp_slots import (  # PAPER-FAITHFUL-TP\n"
        "    plan_for_server_args as tp_slot_plan_for,\n"
        ")\n",
        probe="plan_for_server_args as tp_slot_plan_for")

    # Two construction sites only; the plan is recomputed here rather than
    # shipped, so nothing new crosses a process boundary.
    replace(gpu_sched,
        "            self.worker_pool = WorkerPool(\n"
        "                self.server_args.workers_per_gpu, self.gpu_id, self.context\n"
        "            )\n",
        "            self._tp_slot_plan = tp_slot_plan_for(self.server_args)\n"
        "            self._tp_model_sizes = {\n"
        "                mc.model_name: int(getattr(mc, \"tp_size\", 1) or 1)\n"
        "                for mc in (getattr(self.server_args, \"model_configs\", None) or [])\n"
        "            }\n"
        "            self.worker_pool = WorkerPool(\n"
        "                self.server_args.workers_per_gpu,\n"
        "                self.gpu_id,\n"
        "                self.context,\n"
        "                slot_plan=self._tp_slot_plan,\n"
        "                model_tp_sizes=self._tp_model_sizes,\n"
        "            )\n",
        probe="self._tp_slot_plan = tp_slot_plan_for(self.server_args)")

    replace(gpu_sched,
        "            num_workers=multi_model_server_args.workers_per_gpu,\n"
        "        )\n",
        "            num_workers=multi_model_server_args.workers_per_gpu,\n"
        "            slot_plan=getattr(self, \"_tp_slot_plan\", None),        # PAPER-FAITHFUL-TP\n"
        "            model_tp_sizes=getattr(self, \"_tp_model_sizes\", None),  # PAPER-FAITHFUL-TP\n"
        "        )\n",
        probe="slot_plan=getattr(self, \"_tp_slot_plan\", None)")

    # ------------------------------------------------- controller (step 2/3)
    controller = mm / "scheduling/controller_global.py"
    shutil.copyfile(HERE / "kvpr_global_tp.py", mm / "scheduling/policy/kvpr_global_tp.py")

    # A TP group stops being collapsed to its rank-0 GPU.  With every model at
    # tp_size=1 this is a no-op (gpu_ids is a 1-element list either way), so it
    # needs no flag; it only starts to matter once a group spans GPUs.
    replace(controller,
        "            # NOTE(ke): For TP case, only consider rank0 state\n"
        "            gpu_ids = set([mod.gpu_ids[0] for mod in models])\n",
        "            # PAPER-FAITHFUL-TP: a TP group is a multi-GPU object.  The\n"
        "            # upstream note kept only rank0, which makes every GPU past\n"
        "            # rank0 invisible to placement -- and an anti-affinity\n"
        "            # constraint over GPUs nobody can see cannot be expressed.\n"
        "            # Identical to the old line when every model is tp_size=1.\n"
        "            gpu_ids = set()\n"
        "            for mod in models:\n"
        "                gpu_ids.update(mod.gpu_ids)\n",
        probe="PAPER-FAITHFUL-TP: a TP group is a multi-GPU object")

    replace(controller,
        "from sglang.multi_model.scheduling.policy.kvpr_global_v4 import KVPRGlobalPolicyV4\n",
        "from sglang.multi_model.scheduling.policy.kvpr_global_v4 import KVPRGlobalPolicyV4\n"
        "from sglang.multi_model.scheduling.policy.kvpr_global_tp import KVPRGlobalPolicyTP\n"
        "from sglang.multi_model.tp_slots import (  # PAPER-FAITHFUL-TP\n"
        "    plan_for_server_args as tp_slot_plan_for,\n"
        ")\n",
        probe="import KVPRGlobalPolicyTP")

    replace(controller,
        '        elif self.server_args.policy in ("kvpr-global-v3", "kvpr-global-v4"):\n',
        '        elif self.server_args.policy in ("kvpr-global-v3", "kvpr-global-v4",\n'
        '                                        "kvpr-global-tp"):\n',
        probe='"kvpr-global-tp"):')

    replace(controller,
        "            _cls = (KVPRGlobalPolicyV4\n"
        "                    if self.server_args.policy == \"kvpr-global-v4\"\n"
        "                    else KVPRGlobalPolicyV3)\n"
        "            self.policy = _cls(\n",
        "            _cls = (KVPRGlobalPolicyTP\n"
        "                    if self.server_args.policy == \"kvpr-global-tp\"\n"
        "                    else KVPRGlobalPolicyV4\n"
        "                    if self.server_args.policy == \"kvpr-global-v4\"\n"
        "                    else KVPRGlobalPolicyV3)\n"
        "            # Only the TP policy needs the server args (for tp_size per\n"
        "            # model and the anti-affinity flag); the others keep the\n"
        "            # keyword-only signature they already had.\n"
        "            _extra = ({\"server_args\": self.server_args}\n"
        "                      if self.server_args.policy == \"kvpr-global-tp\" else {})\n"
        "            self.policy = _cls(\n",
        probe="_cls = (KVPRGlobalPolicyTP")

    replace(controller,
        "                tpot_slo_s=_tpot,\n"
        "            )\n"
        "        else:\n"
        "            raise ValueError(f\"Unknown policy: {self.server_args.policy}\")\n",
        "                tpot_slo_s=_tpot,\n"
        "                **_extra,  # PAPER-FAITHFUL-TP\n"
        "            )\n"
        "        else:\n"
        "            raise ValueError(f\"Unknown policy: {self.server_args.policy}\")\n",
        probe="**_extra,  # PAPER-FAITHFUL-TP")

    replace(args_py, '                "kvpr-global-v4",\n',
        '                "kvpr-global-v4",\n                "kvpr-global-tp",\n',
        probe='                "kvpr-global-tp",')

    # --------------------------------------------------------- model runner
    # The worker-pool log prefix stamps rank0's gpu_id on every rank, so a TP
    # run reads as though every rank sat on one GPU -- which is why the v4
    # attempt recovered no rank->GPU map at all.  Emit the mapping from the one
    # place that knows both, in the exact shape collect_tp2_evidence.py greps
    # for, instead of editing the shared log prefix.
    replace(runner,
        "        if tp_size > 1:\n"
        "            logger.warning(\n"
        "                \"Tensor parallelism is enabled, model service will not be used.\"\n"
        "            )\n"
        "            self.use_model_service = False\n",
        "        if tp_size > 1:\n"
        "            logger.warning(\n"
        "                \"Tensor parallelism is enabled, model service will not be used.\"\n"
        "            )\n"
        "            self.use_model_service = False\n"
        "        # PAPER-FAITHFUL-TP evidence line.  Every rank prints where it\n"
        "        # actually is; the shared log prefix cannot, since it carries\n"
        "        # rank0's gpu_id for the whole group.\n"
        "        logger.info(\n"
        "            f\"[PAPER-TP] engine rank: tp_rank={tp_rank} gpu_id={gpu_id} \"\n"
        "            f\"tp_size={tp_size} model_service={self.use_model_service}\"\n"
        "        )\n",
        probe="[PAPER-TP] engine rank:")

    # ------------------------------------------------------------- verify
    checks = {
        mm / "tp_slots.py": ["def build_slot_plan(", "def plan_for_server_args("],
        mm / "utils/model_info.json": ["meta-llama/Llama-3.1-70B"],
        args_py: ["enable_tp_worker_pool: bool", "--enable-tp-anti-affinity"],
        server: ["plan_for_server_args as tp_slot_plan_for", "server_args.tp_size = _slot.tp_size",
                 "list(_slot.gpu_ids)"],
        pool: ["def get_idle_worker(self, tp_size: int = 1)", "self._owned_ids",
               "self._shadow_ids"],
        resource: ["self._worker_ids", "self._model_tp_sizes.get(model_name, 1)"],
        gpu_sched: ["slot_plan=self._tp_slot_plan", "plan_for_server_args as tp_slot_plan_for"],
        runner: ["[PAPER-TP] engine rank:"],
        mm / "scheduling/controller_global.py": [
            "import KVPRGlobalPolicyTP", "gpu_ids.update(mod.gpu_ids)",
            "**_extra,  # PAPER-FAITHFUL-TP"],
        mm / "scheduling/policy/kvpr_global_tp.py": ["class KVPRGlobalPolicyTP"],
        repo / "python/sglang/srt/server_args.py": ["\"enable_tp_worker_pool\","],
    }
    missing = [f"{p}: {n}" for p, names in checks.items() for n in names
               if n not in p.read_text()]
    if missing:
        raise RuntimeError("paper-faithful-tp verification failed:\n" + "\n".join(missing))
    print("paper-faithful-tp applied (10 files, all landing points verified)")


if __name__ == "__main__":
    main()
