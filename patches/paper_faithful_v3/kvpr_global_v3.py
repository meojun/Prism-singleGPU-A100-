"""Prism Algorithm 1 with the paper's literal absolute-tau decision rule."""

import json
import logging
import time
from typing import Dict, List, Tuple

from sglang.multi_model.scheduling.model_queue_tracker import ModelQueueTracker
from sglang.multi_model.scheduling.policy.kvpr_global import KVPRGlobalPolicy
from sglang.multi_model.scheduling.state import ModelInstanceState

logger = logging.getLogger(__name__)


class KVPRGlobalPolicyV3(KVPRGlobalPolicy):
    """Use ``current_r - best_r > tau`` exactly as printed in Algorithm 1.

    V2 used a relative reduction in cluster peak KVPR because the paper omits
    tau's value and units.  V3 deliberately restores the literal, absolute
    threshold.  Tau is calibrated once on this machine and then held fixed for
    every workload/rate/system comparison.
    """

    def _weighted_token_rates(self, model_queues, model_names):
        """Return Algorithm-1 rates in GiB/s/SLO so they match shared_kv GiB.

        The base v2 policy can leave ``cell_size`` in bytes because its final
        migration gate is relative and the common 2**30 scale cancels.  V3's
        literal absolute tau cannot: both numerator and denominator must use
        the same memory unit.
        """
        out = super()._weighted_token_rates(model_queues, model_names)
        for row in out.values():
            row["token_size_bytes"] = row["token_size"]
            row["token_size"] /= 2**30
            row["weighted_token_rate"] /= 2**30
        return out

    def _greedy_placement(
        self,
        weights: Dict[str, Dict[str, float]],
        current_gpu: Dict[str, int],
        gpu_available_memory: Dict[int, float],
    ) -> Dict[str, int]:
        order = sorted(
            current_gpu, key=lambda m: (-weights[m]["weighted_token_rate"], m)
        )
        shared_kv = {i: self.gpu_mem for i in range(self.num_gpus)}
        w_rate = {i: 0.0 for i in range(self.num_gpus)}
        target: Dict[str, int] = {}
        self._last_line8 = []

        for name in order:
            current = current_gpu[name]
            model_size = self.model_weights_info[name]["model_size"]
            candidates = [
                i
                for i in range(self.num_gpus)
                if shared_kv[i] > model_size
                and (i == current or gpu_available_memory.get(i, 0.0) >= model_size)
            ] or [current]
            ratios = {
                i: w_rate[i] / shared_kv[i] if shared_kv[i] > 0 else float("inf")
                for i in candidates
            }
            best = min(ratios, key=lambda i: (ratios[i], i))
            best_r = ratios[best]
            current_r = (
                w_rate[current] / shared_kv[current]
                if shared_kv.get(current, 0) > 0
                else float("inf")
            )
            delta = current_r - best_r
            chosen = best if delta > self.tau else current
            self._last_line8.append(
                {
                    "model": name,
                    "current_gpu": current,
                    "best_gpu": best,
                    "current_r": current_r,
                    "best_r": best_r,
                    "absolute_delta": delta,
                    "tau": self.tau,
                    "chosen_gpu": chosen,
                }
            )
            target[name] = chosen
            w_rate[chosen] += weights[name]["weighted_token_rate"]
            shared_kv[chosen] -= model_size
        return target

    def _find_optimal_migrations(
        self,
        model_instance_state_dict: Dict[str, List[ModelInstanceState]],
        model_queues: Dict[str, ModelQueueTracker],
        gpu_available_memory: Dict[int, float],
        model_violation_stats,
        gpu_to_model_mapping: Dict[int, List[str]],
    ) -> List[Tuple[str, int, int]]:
        self._cycle += 1
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

        def emit(decision, reason, candidate=None, peak_after=None):
            logger.info(
                "[PAPER-ALG1-V3] "
                + json.dumps(
                    {
                        "cycle": self._cycle,
                        "tau_mode": "absolute-line8",
                        "tau": self.tau,
                        "kvpr": kvpr_now,
                        "peak_kvpr": peak_now,
                        "line8": getattr(self, "_last_line8", []),
                        "candidate": candidate,
                        "peak_kvpr_after": peak_after,
                        "migration_decision": decision,
                        "migration_reason": reason,
                    },
                    default=str,
                )
            )

        if peak_now <= 0:
            emit(None, "no measured load")
            return []
        if now - self._last_migration_time < self.migration_cooldown:
            emit(None, "migration cooldown active")
            return []

        target = self._greedy_placement(weights, current_gpu, gpu_available_memory)
        candidates = []
        for name, dst in target.items():
            src = current_gpu[name]
            if dst == src or len(gpu_to_model_mapping.get(src, [])) <= 1:
                continue
            if gpu_available_memory.get(dst, 0.0) < self.model_weights_info[name]["model_size"]:
                continue
            sim = {gpu: list(models) for gpu, models in gpu_to_model_mapping.items()}
            sim[src].remove(name)
            sim.setdefault(dst, []).append(name)
            peak_after = self._peak(self._kvpr(sim, weights))
            candidates.append((peak_after, name, src, dst))

        if not candidates:
            emit(None, "line-8 placement unchanged or candidate blocked")
            return []
        peak_after, name, src, dst = min(candidates)
        self._last_migration_time = now
        candidate = {"model": name, "from": src, "to": dst}
        emit("MIGRATE", "literal line-8 absolute delta exceeds tau", candidate, peak_after)
        return [(name, src, dst)]
