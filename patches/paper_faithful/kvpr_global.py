"""Paper Algorithm 1 -- Model Placement to Maximize Memory Headroom (KVPR).

Transcribed from *Prism: Cost-Efficient Multi-LLM Serving via GPU Memory
Ballooning*, Algorithm 1:

    1: sort models by t_j * tz_j / s_j descending
    2: for i = 1..N:  shared_kv_i <- C ; w_token_rate_i <- 0
    4: for k = 1..M:
    6:     best_r, best_idx <- (min, argmin)_i  w_token_rate_i / shared_kv_i
    7:     current_r <- w_token_rate_{g_k} / shared_kv_{g_k}
    8:     best_gpu  <- best_idx if current_r - best_r > tau else g_k
    9:     assign m_k to best_gpu
   10:     w_token_rate_{best_gpu} += ...
   11:     shared_kv_{best_gpu}    -= w_k

Terms (paper Require clause):
    t_j   token rate         tz_j  token size (KV bytes/token)
    w_j   model weight size  s_j   latency SLO         g_j  current GPU

Only `_find_optimal_migrations` is overridden; idle eviction, model
activation, action emission and every threshold outside Algorithm 1 are
inherited unchanged from SimpleGlobalPolicy so that the two arms differ in
placement policy alone.

--------------------------------------------------------------------------
ASSUMPTION / OUR IMPLEMENTATION DECISION  (the paper does not fix these)
--------------------------------------------------------------------------
A1. tau is dimensionless here.  KVPR carries units of
    (tokens/s * bytes/token / s) / GiB, so the paper's literal absolute test
    `current_r - best_r > tau` has no scale-free meaning and the paper gives
    neither a unit nor a value.  We keep Algorithm 1's greedy pass verbatim
    and gate the resulting placement change on the RELATIVE reduction of the
    cluster's PEAK KVPR:
        (peak_now - peak_after) / peak_now > tau
    "Bounding the maximum KVPR across the cluster" is the paper's own stated
    objective for this greedy approximation (Analysis paragraph / App. A.2),
    so this preserves the intent while making tau comparable across setups.
A2. Line 10 of the printed algorithm reads `+= r_k / s_k`, which contradicts
    the line-1 sort key `t_j * tz_j / s_j` (it is a leftover from arXiv v1,
    where the numerator was a request rate).  We use `t_k * tz_k / s_k`
    consistently for both the sort key and the accumulator.
A3. s_j is read as the TPOT SLO: KVPR models memory pressure, and it is TPOT
    that memory headroom governs (paper Sec. 6.2 Analysis).
A4. t_j = admitted input tokens/s (sliding window) + decode tokens/s reported
    by the engine -- "the rate at which the KV cache actually grows".
    Window length is not specified by the paper; we reuse the prototype's own
    30 s ModelRequestTracker window so window length is not a confound.
A5. At most one migration is emitted per scheduling cycle, matching the
    released prototype, because a migration is stop-the-world here.
A6. A GPU is never emptied: the upstream launcher only starts a GPU scheduler
    for GPUs present in the initial placement, so an emptied GPU stays dead
    for the rest of the run.
A7. A candidate GPU must have free memory >= the model's weights.
"""

import json
import logging
import time
from typing import Dict, List, Optional, Tuple

from sglang.multi_model.scheduling.model_queue_tracker import ModelQueueTracker
from sglang.multi_model.scheduling.policy.simple_global import SimpleGlobalPolicy
from sglang.multi_model.scheduling.state import ModelInstanceState, ModelState

logger = logging.getLogger(__name__)


class KVPRGlobalPolicy(SimpleGlobalPolicy):
    def __init__(
        self,
        num_gpus: int,
        gpu_mem: float,
        model_weights_info: Dict[str, Dict[str, float]],
        workers_per_gpu: int,
        tau: float = 0.35,
        rate_window: float = 30.0,
        migration_cooldown: float = 30.0,
        tpot_slo_s: Optional[Dict[str, float]] = None,
    ):
        super().__init__(num_gpus, gpu_mem, model_weights_info, workers_per_gpu)
        self.tau = tau
        self.rate_window = rate_window
        self.migration_cooldown = migration_cooldown
        # Default TPOT SLO if the slo-base-file carries no entry for a model.
        self.tpot_slo_s: Dict[str, float] = dict(tpot_slo_s or {})
        self._default_tpot_slo_s = 0.05
        self._last_migration_time = float("-inf")
        self._cycle = 0

    # ------------------------------------------------------------------ terms

    def _tpot_slo(self, model_name: str) -> float:
        return self.tpot_slo_s.get(model_name, self._default_tpot_slo_s)

    def _weighted_token_rates(
        self, model_queues: Dict[str, ModelQueueTracker], model_names: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """t_j, tz_j, s_j and w_token_rate_j = t_j * tz_j / s_j per model."""
        now = time.time()
        out: Dict[str, Dict[str, float]] = {}
        for name in model_names:
            queue = model_queues.get(name)
            input_tokens = 0.0
            if queue is not None:
                for req in queue.received_reqs.values():
                    if getattr(req, "is_warmup", False):
                        continue
                    at = getattr(req, "arrival_time", None)
                    if at is None or now - at > self.rate_window:
                        continue
                    input_tokens += req.prompt_len or 0
            input_rate = input_tokens / self.rate_window
            decode_rate = float(getattr(queue, "decode_token_tput", 0.0) or 0.0)
            token_rate = input_rate + decode_rate
            token_size = float(self.model_weights_info[name]["cell_size"])
            slo = self._tpot_slo(name)
            out[name] = {
                "input_token_rate": input_rate,
                "decode_token_rate": decode_rate,
                "token_rate": token_rate,
                "token_size": token_size,
                "tpot_slo_s": slo,
                "weighted_token_rate": token_rate * token_size / slo,
            }
        return out

    def _shared_kv(self, gpu_to_model_mapping: Dict[int, List[str]], gpu_id: int) -> float:
        """shared_kv_i = C - sum of resident model weights on GPU i."""
        used = sum(
            self.model_weights_info[m]["model_size"]
            for m in gpu_to_model_mapping.get(gpu_id, [])
        )
        return self.gpu_mem - used

    def _kvpr(
        self,
        gpu_to_model_mapping: Dict[int, List[str]],
        weights: Dict[str, Dict[str, float]],
    ) -> Dict[int, float]:
        """KVPR_i = sum_j w_token_rate_j / shared_kv_i."""
        out = {}
        for gpu_id in range(self.num_gpus):
            agg = sum(
                weights.get(m, {}).get("weighted_token_rate", 0.0)
                for m in gpu_to_model_mapping.get(gpu_id, [])
            )
            kv = self._shared_kv(gpu_to_model_mapping, gpu_id)
            out[gpu_id] = agg / kv if kv > 0 else float("inf")
        return out

    @staticmethod
    def _peak(kvpr: Dict[int, float]) -> float:
        return max(kvpr.values()) if kvpr else 0.0

    # ------------------------------------------------------- Algorithm 1 pass

    def _greedy_placement(
        self,
        weights: Dict[str, Dict[str, float]],
        current_gpu: Dict[str, int],
        gpu_available_memory: Dict[int, float],
    ) -> Dict[str, int]:
        """Algorithm 1 lines 1-12 verbatim; returns model -> target GPU."""
        # line 1: descending w_token_rate (name breaks ties deterministically)
        order = sorted(
            current_gpu,
            key=lambda m: (-weights[m]["weighted_token_rate"], m),
        )
        # line 2-3
        shared_kv = {i: self.gpu_mem for i in range(self.num_gpus)}
        w_rate = {i: 0.0 for i in range(self.num_gpus)}

        target: Dict[str, int] = {}
        for name in order:
            g_k = current_gpu[name]
            w_k = self.model_weights_info[name]["model_size"]

            # A7: only GPUs that can physically hold the weights are candidates.
            # The current GPU always qualifies -- the weights are already there.
            cand = [
                i
                for i in range(self.num_gpus)
                if shared_kv[i] > w_k
                and (i == g_k or gpu_available_memory.get(i, 0.0) >= w_k)
            ]
            if not cand:
                cand = [g_k]

            # line 6
            ratios = {i: (w_rate[i] / shared_kv[i] if shared_kv[i] > 0 else float("inf"))
                      for i in cand}
            best_idx = min(ratios, key=lambda i: (ratios[i], i))
            best_r = ratios[best_idx]
            # line 7
            current_r = (
                w_rate[g_k] / shared_kv[g_k] if shared_kv.get(g_k, 0) > 0 else float("inf")
            )
            # line 8 -- gated later on peak KVPR (assumption A1); the greedy pass
            # itself takes the argmin, which is the tau -> 0 reading of line 8.
            best_gpu = best_idx if best_r < current_r else g_k

            # lines 9-11
            target[name] = best_gpu
            w_rate[best_gpu] += weights[name]["weighted_token_rate"]
            shared_kv[best_gpu] -= w_k
        return target

    # --------------------------------------------------------------- override

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

        current_gpu: Dict[str, int] = {}
        for gpu_id, names in gpu_to_model_mapping.items():
            for name in names:
                if name in self.model_weights_info:
                    current_gpu[name] = gpu_id
        if not current_gpu:
            return []

        weights = self._weighted_token_rates(model_queues, list(current_gpu))
        kvpr_now = self._kvpr(gpu_to_model_mapping, weights)
        peak_now = self._peak(kvpr_now)

        # ---- per-cycle observability (Section 7 of the task brief)
        rows = []
        for name, gpu_id in sorted(current_gpu.items()):
            w = weights[name]
            rows.append(
                {
                    "model": name,
                    "token_rate": round(w["token_rate"], 3),
                    "input_token_rate": round(w["input_token_rate"], 3),
                    "decode_token_rate": round(w["decode_token_rate"], 3),
                    "token_size": w["token_size"],
                    "tpot_slo_s": w["tpot_slo_s"],
                    "w_token_rate": w["weighted_token_rate"],
                    "current_gpu": gpu_id,
                }
            )
        shared_kv_now = {
            i: self._shared_kv(gpu_to_model_mapping, i) for i in range(self.num_gpus)
        }

        def emit(decision, reason, cand=None, peak_after=None, improvement=None):
            logger.info(
                "[PAPER-ALG1] "
                + json.dumps(
                    {
                        "cycle": self._cycle,
                        "t": round(now, 3),
                        "models": rows,
                        "shared_kv": {str(k): round(v, 3) for k, v in shared_kv_now.items()},
                        "kvpr": {str(k): v for k, v in kvpr_now.items()},
                        "peak_kvpr": peak_now,
                        "candidate": cand,
                        "peak_kvpr_after": peak_after,
                        "improvement": improvement,
                        "tau": self.tau,
                        "migration_decision": decision,
                        "migration_reason": reason,
                    },
                    default=str,
                )
            )

        if peak_now <= 0:
            emit(None, "no measured load (peak KVPR == 0)")
            return []
        if now - self._last_migration_time < self.migration_cooldown:
            emit(None, "migration cooldown active")
            return []

        target = self._greedy_placement(weights, current_gpu, gpu_available_memory)
        diffs = [(m, current_gpu[m], g) for m, g in target.items() if g != current_gpu[m]]
        if not diffs:
            emit(None, "greedy placement equals current placement")
            return []

        # A5: evaluate each single move on its own and keep the best one.
        best = None
        for name, src, dst in diffs:
            if len(gpu_to_model_mapping.get(src, [])) <= 1:
                continue                                        # A6
            if gpu_available_memory.get(dst, 0.0) < self.model_weights_info[name]["model_size"]:
                continue                                        # A7
            sim = {g: list(ms) for g, ms in gpu_to_model_mapping.items()}
            sim[src].remove(name)
            sim.setdefault(dst, []).append(name)
            peak_after = self._peak(self._kvpr(sim, weights))
            improvement = (peak_now - peak_after) / peak_now
            if best is None or improvement > best[3]:
                best = ((name, src, dst), peak_after, sim, improvement)

        if best is None:
            emit(None, "every candidate move blocked (would empty a GPU / no room)")
            return []

        (name, src, dst), peak_after, _sim, improvement = best
        cand = {"model": name, "from": src, "to": dst}
        if improvement > self.tau:
            self._last_migration_time = now
            emit("MIGRATE", "peak KVPR improvement exceeds tau", cand, peak_after, improvement)
            logger.info(f"[PAPER-ALG1] MIGRATE {name} gpu{src} -> gpu{dst} "
                        f"improvement={improvement:.4f} tau={self.tau}")
            return [(name, src, dst)]

        emit(None, "peak KVPR improvement below tau", cand, peak_after, improvement)
        return []
