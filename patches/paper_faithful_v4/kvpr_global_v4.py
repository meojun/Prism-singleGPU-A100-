"""Algorithm 1 with a runtime placement audit.

v3 already implements the paper's line-8 rule literally.  What it does not
record is the gap between what the planner decided and what the cluster ended
up looking like: ``_greedy_placement`` computes a target GPU for *every* model,
but a cycle emits at most one migration, so the plan and the runtime placement
can disagree for a long time -- or forever -- and nothing said so.

v4 keeps v3's decision rule and its one-migration-per-cycle emission unchanged.
Emitting the whole plan at once is not obviously the paper's intent and is
demonstrably worse here: each migration is a real transfer, the rate estimate
underneath is a 30 s sliding window, and moving several models against one
window's estimate is what produced the thrashing documented in
``docs/paper_faithful/design_analysis.md`` §5a.  So the behaviour is left
alone and the *evidence* is added instead:

* the full placement plan for every cycle, model by model;
* for every model the plan wants to move, why it was not moved this cycle --
  cooldown, memory infeasibility, last model on its GPU, or simply not the
  cycle's best candidate;
* a convergence gap: how many models sit somewhere other than where the
  planner wants them.

Whether one-at-a-time converges is then a measured question, not an assumption.
"""

import json
import logging
import time
from typing import Dict, List, Tuple

from sglang.multi_model.scheduling.model_queue_tracker import ModelQueueTracker
from sglang.multi_model.scheduling.policy.kvpr_global_v3 import KVPRGlobalPolicyV3
from sglang.multi_model.scheduling.state import ModelInstanceState

logger = logging.getLogger(__name__)


class KVPRGlobalPolicyV4(KVPRGlobalPolicyV3):
    """v3's rule, plus the planner-versus-runtime record v3 never kept."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._audit = {
            "cycles": 0,
            "placement_decisions": 0,
            "migrations_emitted": 0,
            "suppressed_by_tau": 0,
            "rejected_by_memory": 0,
            "rejected_last_model_on_gpu": 0,
            "deferred_by_cooldown": 0,
            "not_best_candidate": 0,
        }

    def _find_optimal_migrations(
        self,
        model_instance_state_dict: Dict[str, List[ModelInstanceState]],
        model_queues: Dict[str, ModelQueueTracker],
        gpu_available_memory: Dict[int, float],
        model_violation_stats,
        gpu_to_model_mapping: Dict[int, List[str]],
    ) -> List[Tuple[str, int, int]]:
        self._cycle += 1
        self._audit["cycles"] += 1
        now = time.time()
        current_gpu = {
            name: gpu_id
            for gpu_id, names in gpu_to_model_mapping.items()
            for name in names
            if name in self.model_weights_info
        }
        if not current_gpu:
            return []

        weights = self._weighted_token_rates(model_queues, list(current_gpu))
        kvpr_now = self._kvpr(gpu_to_model_mapping, weights)
        peak_now = self._peak(kvpr_now)

        cooldown_active = (now - self._last_migration_time) < self.migration_cooldown
        plan = self._greedy_placement(weights, current_gpu, gpu_available_memory)
        line8 = list(getattr(self, "_last_line8", []))
        self._audit["placement_decisions"] += len(line8)
        for row in line8:
            if row["chosen_gpu"] == row["current_gpu"] and row["best_gpu"] != row["current_gpu"]:
                self._audit["suppressed_by_tau"] += 1

        # Everything the plan wants moved, and what stopped each one.
        wanted, blocked, candidates = [], [], []
        for name, dst in plan.items():
            src = current_gpu[name]
            if dst == src:
                continue
            wanted.append({"model": name, "from": src, "to": dst})
            if len(gpu_to_model_mapping.get(src, [])) <= 1:
                blocked.append({"model": name, "reason": "last active model on source gpu"})
                self._audit["rejected_last_model_on_gpu"] += 1
                continue
            need = self.model_weights_info[name]["model_size"]
            if gpu_available_memory.get(dst, 0.0) < need:
                blocked.append({
                    "model": name, "reason": "target memory infeasible",
                    "need_gib": need, "free_gib": gpu_available_memory.get(dst, 0.0),
                })
                self._audit["rejected_by_memory"] += 1
                continue
            sim = {gpu: list(models) for gpu, models in gpu_to_model_mapping.items()}
            sim[src].remove(name)
            sim.setdefault(dst, []).append(name)
            candidates.append((self._peak(self._kvpr(sim, weights)), name, src, dst))

        # How far the cluster is from the plan right now.
        misplaced = [m for m, dst in plan.items() if dst != current_gpu[m]]

        chosen = None
        decision, reason = None, ""
        if peak_now <= 0:
            reason = "no measured load"
        elif cooldown_active:
            reason = "migration cooldown active"
            self._audit["deferred_by_cooldown"] += len(candidates)
        elif not candidates:
            reason = "line-8 placement unchanged or every candidate blocked"
        else:
            peak_after, name, src, dst = min(candidates)
            chosen = {"model": name, "from": src, "to": dst, "peak_kvpr_after": peak_after}
            decision, reason = "MIGRATE", "literal line-8 absolute delta exceeds tau"
            self._last_migration_time = now
            self._audit["migrations_emitted"] += 1
            self._audit["not_best_candidate"] += len(candidates) - 1

        logger.info("[PAPER-ALG1-V4] " + json.dumps({
            "cycle": self._cycle,
            "timestamp": now,
            "tau_mode": "absolute-line8",
            "tau": self.tau,
            "kvpr": kvpr_now,
            "peak_kvpr": peak_now,
            "line8": line8,
            "placement_plan": plan,
            "current_placement": current_gpu,
            "plan_wants_moved": wanted,
            "blocked": blocked,
            "convergence_gap": len(misplaced),
            "misplaced_models": misplaced,
            "cooldown_active": cooldown_active,
            "candidate": chosen,
            "migration_decision": decision,
            "migration_reason": reason,
            "audit_totals": dict(self._audit),
        }, default=str))

        if chosen is None:
            return []
        return [(chosen["model"], chosen["from"], chosen["to"])]
