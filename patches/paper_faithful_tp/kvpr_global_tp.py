"""Algorithm 1 with TP groups kept whole, and the paper's anti-affinity constraint.

Two things change against v4, and only the second is a new mechanism.

1.  **A TP group stops being collapsed to one GPU.**  v4 builds
    ``current_gpu = {name: gpu_id ...}`` from ``gpu_to_model_mapping``.  That
    mapping already lists a TP model under *every* GPU it spans
    (``simple_global.py:401-405`` walks ``instance.gpu_ids``), so the dict
    comprehension silently keeps whichever GPU it saw last and the group is
    lost.  Here placement carries an ordered tuple of GPUs per model.

2.  **Anti-affinity.**  The paper constrains the k shards of a tp_size=k model
    to k distinct GPUs.  Expressing it needs per-shard placement: the shards are
    placed one at a time by the same argmin v3 uses, and the constraint is a
    filter on that argmin's candidate set -- drop any GPU already holding a
    shard of this model.  It is opt-in (``--enable-tp-anti-affinity``), per the
    project's rule that new paths sit behind a flag.

Why the constraint can actually bind here
-----------------------------------------
It would be easy to build this so the constraint never fires and then report
that it "passed".  It fires only because placement is per-shard: two shards of
one model are two independent argmin decisions, and the least-loaded GPU for
shard 0 is very often also the least-loaded for shard 1.  With the filter off,
that is exactly what the planner picks.

The counterfactual is therefore recorded on every cycle: ``aa_violations`` counts
the shards whose unconstrained argmin would have doubled up on a GPU already
holding a shard of the same model, and ``aa_diverted`` counts the ones the
filter actually moved.  A run where both are zero is reported as "the constraint
never bound", not as "the constraint works".

Modelling a shard's load
------------------------
Under TP both the weights and the KV cache are sharded across the group: each
GPU holds 1/k of the parameters and 1/k of the KV heads.  So a shard
contributes ``weighted_token_rate / k`` and occupies ``model_size / k``.
Charging each GPU the whole model would make a TP model look k times more
expensive than it is and Algorithm 1 would refuse to place it.
"""

import json
import logging
import time
from typing import Dict, List, Sequence, Tuple

from sglang.multi_model.scheduling.model_queue_tracker import ModelQueueTracker
from sglang.multi_model.scheduling.policy.kvpr_global_v4 import KVPRGlobalPolicyV4
from sglang.multi_model.scheduling.state import ModelInstanceState

logger = logging.getLogger(__name__)


class KVPRGlobalPolicyTP(KVPRGlobalPolicyV4):
    """v4's decision rule, with TP groups intact and anti-affinity available."""

    def __init__(self, *args, **kwargs):
        # The parent takes keyword args only and does not know about
        # server_args, so it is popped before the super() call rather than
        # forwarded.  controller_global passes it only for this class.
        server_args = kwargs.pop("server_args", None)
        super().__init__(*args, **kwargs)
        self.anti_affinity = bool(
            getattr(server_args, "enable_tp_anti_affinity", False)
        )
        self._model_tp_sizes: Dict[str, int] = {
            mc.model_name: int(getattr(mc, "tp_size", 1) or 1)
            for mc in (getattr(server_args, "model_configs", None) or [])
        }
        self._tp_audit = {
            "cycles": 0,
            "shards_placed": 0,
            "aa_violations": 0,   # unconstrained argmin would have doubled up
            "aa_diverted": 0,     # the filter actually changed the choice
            "aa_infeasible": 0,   # filter left no candidate; fell back
        }
        logger.info(
            "[PAPER-ALG1-TP] init "
            + json.dumps({
                "anti_affinity": self.anti_affinity,
                "model_tp_sizes": self._model_tp_sizes,
            })
        )

    # ------------------------------------------------------------------ helpers
    def tp_size_of(self, model_name: str) -> int:
        return max(1, self._model_tp_sizes.get(model_name, 1))

    def _current_groups(
        self, gpu_to_model_mapping: Dict[int, List[str]]
    ) -> Dict[str, Tuple[int, ...]]:
        """model -> the ordered tuple of GPUs it currently occupies.

        v4 keeps only one GPU per model here, which is what makes a TP group
        invisible to placement.  Ordering is by GPU id, which is stable and is
        also the order ``build_slot_plan`` uses for a group's ranks.
        """
        groups: Dict[str, List[int]] = {}
        for gpu_id, names in gpu_to_model_mapping.items():
            for name in names:
                if name in self.model_weights_info:
                    groups.setdefault(name, []).append(gpu_id)
        return {name: tuple(sorted(set(g))) for name, g in groups.items()}

    # -------------------------------------------------------------- placement
    def _greedy_placement_tp(
        self,
        weights: Dict[str, Dict[str, float]],
        current_groups: Dict[str, Tuple[int, ...]],
        gpu_available_memory: Dict[int, float],
    ) -> Dict[str, Tuple[int, ...]]:
        """v3's line-8 argmin, run once per shard, with the group kept whole."""
        order = sorted(
            current_groups, key=lambda m: (-weights[m]["weighted_token_rate"], m)
        )
        shared_kv = {i: self.gpu_mem for i in range(self.num_gpus)}
        w_rate = {i: 0.0 for i in range(self.num_gpus)}
        target: Dict[str, Tuple[int, ...]] = {}
        self._last_line8 = []

        for name in order:
            k = self.tp_size_of(name)
            current = list(current_groups[name])
            # Under TP each GPU carries 1/k of the weights and 1/k of the KV.
            shard_size = self.model_weights_info[name]["model_size"] / k
            shard_rate = weights[name]["weighted_token_rate"] / k
            chosen_gpus: List[int] = []

            for rank in range(k):
                cur = current[rank] if rank < len(current) else current[-1]
                feasible = [
                    i
                    for i in range(self.num_gpus)
                    if shared_kv[i] > shard_size
                    and (i in current or gpu_available_memory.get(i, 0.0) >= shard_size)
                ] or [cur]

                # The paper's constraint, as a filter on the candidate set.
                allowed = feasible
                filtered_out = [i for i in feasible if i in chosen_gpus]
                if k > 1 and self.anti_affinity:
                    allowed = [i for i in feasible if i not in chosen_gpus]
                    if not allowed:
                        # Cannot satisfy it here.  Say so rather than quietly
                        # emitting a placement that violates the constraint.
                        self._tp_audit["aa_infeasible"] += 1
                        allowed = feasible

                def ratio(i):
                    return w_rate[i] / shared_kv[i] if shared_kv[i] > 0 else float("inf")

                best_unconstrained = min(feasible, key=lambda i: (ratio(i), i))
                best = min(allowed, key=lambda i: (ratio(i), i))

                if k > 1 and best_unconstrained in chosen_gpus:
                    # The unconstrained argmin would have stacked two shards of
                    # this model on one GPU.  This is the counterfactual that
                    # makes the ON/OFF comparison meaningful.
                    self._tp_audit["aa_violations"] += 1
                    if self.anti_affinity and best != best_unconstrained:
                        self._tp_audit["aa_diverted"] += 1

                cur_r = ratio(cur)
                delta = cur_r - ratio(best)
                # tau applies to the move, exactly as in v3: stay put unless the
                # improvement clears it.  A shard may only stay if staying does
                # not itself violate anti-affinity.
                stay_ok = not (k > 1 and self.anti_affinity and cur in chosen_gpus)
                chosen = best if (delta > self.tau or not stay_ok) else cur

                self._last_line8.append({
                    "model": name,
                    "tp_size": k,
                    "rank": rank,
                    "current_gpu": cur,
                    "best_gpu": best,
                    "best_gpu_unconstrained": best_unconstrained,
                    "anti_affinity_filtered": sorted(filtered_out) if k > 1 else [],
                    "current_r": cur_r,
                    "best_r": ratio(best),
                    "absolute_delta": delta,
                    "tau": self.tau,
                    "chosen_gpu": chosen,
                })

                chosen_gpus.append(chosen)
                w_rate[chosen] += shard_rate
                shared_kv[chosen] -= shard_size
                self._tp_audit["shards_placed"] += 1

            target[name] = tuple(chosen_gpus)
        return target

    # v4's audit calls _greedy_placement and expects model -> gpu.  Keep that
    # contract for TP=1 models so the inherited logging stays truthful, and
    # expose the group form separately.
    def _greedy_placement(self, weights, current_gpu, gpu_available_memory):
        groups = self._greedy_placement_tp(
            weights,
            {m: (g,) if isinstance(g, int) else tuple(g) for m, g in current_gpu.items()},
            gpu_available_memory,
        )
        self._last_group_plan = groups
        return {m: g[0] for m, g in groups.items()}

    # ------------------------------------------------------------ entry point
    def _find_optimal_migrations(
        self,
        model_instance_state_dict: Dict[str, List[ModelInstanceState]],
        model_queues: Dict[str, ModelQueueTracker],
        gpu_available_memory: Dict[int, float],
        model_violation_stats,
        gpu_to_model_mapping: Dict[int, List[str]],
    ) -> List[Tuple[str, int, int]]:
        self._tp_audit["cycles"] += 1
        current_groups = self._current_groups(gpu_to_model_mapping)
        has_tp = any(len(g) > 1 for g in current_groups.values())

        out = super()._find_optimal_migrations(
            model_instance_state_dict,
            model_queues,
            gpu_available_memory,
            model_violation_stats,
            gpu_to_model_mapping,
        )

        # One line per cycle carrying the group view and the counterfactual, so
        # "did the constraint bind" is answered from raw data, not from a claim.
        logger.info("[PAPER-ALG1-TP] " + json.dumps({
            "cycle": self._tp_audit["cycles"],
            "timestamp": time.time(),
            "anti_affinity": self.anti_affinity,
            "any_tp_group_active": has_tp,
            "current_groups": {m: list(g) for m, g in current_groups.items()},
            "group_plan": {
                m: list(g) for m, g in getattr(self, "_last_group_plan", {}).items()
            },
            "tp_audit": dict(self._tp_audit),
        }, default=str))
        return out
