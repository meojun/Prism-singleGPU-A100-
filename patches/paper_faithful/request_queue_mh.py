"""Moore-Hodgson (paper Algorithm 2) branch for RequestQueue.

Injected into
``sglang/multi_model/scheduling/gpu/request_queue.py`` by apply_patches.py.
The prototype's own ``admission_control`` is untouched and still runs whenever
``--enable-moore-hodgson`` is absent, so the released-prototype arm is bit-for-bit
the upstream path.

--------------------------------------------------------------------------
ASSUMPTION / OUR IMPLEMENTATION DECISION
--------------------------------------------------------------------------
B1. The paper does not say what happens to the requests Algorithm 2 sheds.
    Moore-Hodgson minimises the NUMBER of late jobs on the premise that every
    job still runs -- late ones simply run after the on-time ones.  So:
      * deferred with d_i >  now  -> requeued (real backpressure, retried next
        round with its original a_i / d_i)
      * deferred with d_i <= now  -> dispatched after S at lowest priority.
    Holding the second class back is a livelock: the verdict is stable across
    rounds, so they are shed forever (observed in the v1 sweep: 328k rounds
    with the client blocked).  Regression-tested by test_moore_hodgson.py #9.
B2. c_i is the model's aggregate chunked-prefill token throughput, read from
    --prefill-speed-file.  The paper defines it as "a chunked-prefill speed c_i
    determined by the model that serves it" and its optimality argument assumes
    a single chunked-prefill stream of that throughput, so c_i is an ENGINE
    CAPACITY, not the reciprocal of a single request's measured TTFT.
B3. Every round is scored from scratch over the whole queue, matching
    Algorithm 2's `Require: a set of n requests R`.
"""

import json
import logging
import time
from collections import defaultdict
from typing import Dict, List

from sglang.multi_model.scheduling.gpu.moore_hodgson import MHJob, select

logger = logging.getLogger(__name__)

DEFAULT_PREFILL_SPEED = 2048.0   # released prototype's implicit constant


class MooreHodgsonMixin:
    # ------------------------------------------------------------ configuration
    def configure_moore_hodgson(self, enabled: bool, prefill_speed: Dict[str, float],
                                log_path: str = None):
        self._mh_enabled = bool(enabled)
        self._mh_speed = dict(prefill_speed or {})
        self._mh_round = 0
        self._mh_last_log = 0.0
        self._mh_zero_streak = 0
        self._mh_warned_at = 0.0
        self._mh_deferred_rids = set()
        self._mh_stats = defaultdict(float)
        self._mh_log_path = log_path
        self._mh_fh = None
        if log_path:
            try:
                self._mh_fh = open(log_path, "a", buffering=1)
            except OSError as e:                       # pragma: no cover
                logger.warning(f"[PAPER-ALG2] cannot open {log_path}: {e}")
        logger.info(
            f"[PAPER-ALG2] enabled={self._mh_enabled} "
            f"prefill_speed_entries={len(self._mh_speed)} log={log_path}"
        )

    def _c_i(self, model_name: str) -> float:
        return float(self._mh_speed.get(model_name, DEFAULT_PREFILL_SPEED))

    # ------------------------------------------------------------ Algorithm 2
    def admission_control_mh(
        self,
        available_resources: float,
        model_backend_queue_lens: Dict[str, int],
        model_states: Dict[str, str],
        allow_sending_when_activating: bool = False,
    ):
        import heapq

        admitted = defaultdict(list)
        self._mh_round += 1
        now = time.time()

        models_to_skip = {
            m for m, n in model_backend_queue_lens.items()
            if n > self._skip_model_threshold
        }

        with self._lock:
            queue_len = len(self._queue)
            if queue_len == 0:
                return admitted

            eligible, blocked = [], []
            for w in self._queue:
                state = model_states.get(w.model_name, "deactivated")
                if w.model_name in models_to_skip:
                    blocked.append(w); continue
                if state in ("deactivating", "deactivated"):
                    blocked.append(w); continue
                if state == "activating" and not allow_sending_when_activating:
                    blocked.append(w); continue
                eligible.append(w)

            if not eligible:
                self._queue = blocked
                heapq.heapify(self._queue)
                return admitted

            jobs = []
            for w in eligible:
                req = w.req
                plen = req.prompt_len or 0
                jobs.append(
                    MHJob(
                        key=req.rid,
                        deadline=req.arrival_time + req.slo,   # d_i = a_i + s_i
                        exec_time=plen / self._c_i(w.model_name),  # e_i = p_i / c_i
                        payload=w,
                    )
                )

            selected, deferred = select(jobs, now)             # Algorithm 2

            late = [j for j in deferred if j.deadline <= now]   # B1
            requeue = [j for j in deferred if j.deadline > now]

            dispatch = [j.payload for j in selected] + [j.payload for j in late]
            for w in dispatch:
                admitted[w.model_name].append(w.req)
                s = self._model_requests.get(w.model_name)
                if s is not None:
                    s.discard(w)
                    if not s:
                        del self._model_requests[w.model_name]

            self._queue = blocked + [j.payload for j in requeue]
            heapq.heapify(self._queue)

            est_cost = sum(j.exec_time for j in selected)
            n_sel, n_def, n_late, n_rq = len(selected), len(deferred), len(late), len(requeue)
            new_deferrals = sum(1 for j in deferred if j.key not in self._mh_deferred_rids)
            self._mh_deferred_rids.update(j.key for j in deferred)

        # ------------------------------------------------------- observability
        st = self._mh_stats
        st["rounds"] += 1
        st["eligible"] += len(eligible)
        st["selected"] += n_sel
        st["deferred"] += n_def
        st["late_dispatched"] += n_late
        st["requeued"] += n_rq
        st["max_queue_len"] = max(st["max_queue_len"], queue_len)
        if n_def:
            st["rounds_with_deferral"] += 1
            st["selected_in_deferral_rounds"] += n_sel

        pathological = bool(eligible) and n_sel == 0 and n_late == 0
        if pathological:
            self._mh_zero_streak += 1
            st["pathological_rounds"] += 1
        else:
            self._mh_zero_streak = 0

        should_log = (
            self._mh_fh is not None
            and (n_def or pathological or (now - self._mh_last_log) >= 0.5)
        )
        if should_log:
            self._mh_last_log = now
            rec = {
                "round": self._mh_round, "t": round(now, 4),
                "queue_length": queue_len, "eligible_requests": len(eligible),
                "selected_requests": n_sel, "deferred_requests": n_def,
                "requeued": n_rq, "late_dispatched": n_late,
                "new_deferrals": new_deferrals,
                "estimated_execution_cost_s": round(est_cost, 6),
                "earliest_deadline": round(min((j.deadline for j in jobs), default=0) - now, 4),
                "available_kv_gb": round(float(available_resources or 0), 4),
                "backend_queue_lens": model_backend_queue_lens,
                "blocked": len(blocked),
                "pathological": pathological,
                "zero_streak": self._mh_zero_streak,
            }
            self._mh_fh.write(json.dumps(rec) + "\n")

        # Loud, rate-limited warning for the exact pathology the brief asks about.
        if self._mh_zero_streak >= 20 and (now - self._mh_warned_at) > 5.0:
            self._mh_warned_at = now
            logger.warning(
                f"[PAPER-ALG2-WARN] under-admission: {self._mh_zero_streak} consecutive "
                f"rounds with eligible={len(eligible)} selected=0 late=0 "
                f"queue_len={queue_len} avail_kv={available_resources}"
            )
        if self._mh_round % 2000 == 0:
            logger.info("[PAPER-ALG2] " + json.dumps({k: v for k, v in st.items()}))
        return admitted
