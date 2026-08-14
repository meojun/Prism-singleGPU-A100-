"""Paper Algorithm 2 -- GPU-Local Request Scheduling (Moore-Hodgson).

Transcribed from *Prism: Cost-Efficient Multi-LLM Serving via GPU Memory
Ballooning*, Algorithm 2, verbatim:

    1: sort R ascending by deadline d_i = a_i + s_i
    2: S <- {}, current_time <- Timer.time()
    3: for k = 1..n:
    4:     r <- r_k, e_r <- p_r / c_r
    5:     append r to S
    6:     current_time <- current_time + e_r
    7:     if current_time > a_r + s_r:
    9:         r_max <- argmax_{r' in S} p_r' / c_r'
   10:         remove r_max from S
   11:         current_time <- current_time - p_rmax / c_rmax
   12: return S

Notes on fidelity, kept here because they matter for interpreting results:

* Line 2 initialises the clock to *wall time*, and line 6 accumulates
  ``e_r`` for every accepted request.  This is the classic single-machine
  ``1||sum U_j`` model: one job at a time.  The paper justifies it in its
  analysis paragraph ("prefill completion time ... d_ri = a_ri + sum p/c"),
  which is exact only if the engine runs one chunked-prefill stream whose
  aggregate token throughput is ``c``.  We keep it literal.  ``c_i`` must
  therefore be the model's *aggregate chunked-prefill throughput*, not a
  per-request latency reciprocal -- see exp/scripts/profile_prefill_speed_v2.py.
* The paper does not say what to do with the requests Moore-Hodgson excludes.
  This module returns them; the policy decision lives in request_queue.py.
"""

import dataclasses
from typing import Iterable, List, Sequence, Tuple


@dataclasses.dataclass
class MHJob:
    """One request as Algorithm 2 sees it.

    key       opaque identifier handed back to the caller
    deadline  absolute d_i = a_i + s_i (seconds, same clock as `now`)
    exec_time e_i = p_i / c_i (seconds)
    payload   caller's object, untouched
    """

    key: object
    deadline: float
    exec_time: float
    payload: object = None


def is_feasible(jobs: Sequence[MHJob], now: float) -> bool:
    """True if every job in `jobs` meets its deadline when run back-to-back
    in ascending-deadline order starting at `now`."""
    clock = now
    for j in sorted(jobs, key=lambda j: j.deadline):
        clock += j.exec_time
        if clock > j.deadline:
            return False
    return True


def select(jobs: Iterable[MHJob], now: float) -> Tuple[List[MHJob], List[MHJob]]:
    """Algorithm 2.

    Returns ``(selected, deferred)``.  ``selected`` is in dispatch order
    (ascending deadline) and is guaranteed feasible from ``now``.
    ``deferred`` holds everything Moore-Hodgson shed, in the order it was shed.
    """
    ordered = sorted(jobs, key=lambda j: j.deadline)
    selected: List[MHJob] = []
    deferred: List[MHJob] = []
    clock = float(now)

    for job in ordered:
        selected.append(job)                    # line 5
        clock += job.exec_time                  # line 6
        if clock > job.deadline:                # line 7
            # line 9: pop the request with the longest execution time
            worst = max(range(len(selected)), key=lambda i: selected[i].exec_time)
            dropped = selected.pop(worst)       # line 10
            clock -= dropped.exec_time          # line 11
            deferred.append(dropped)

    return selected, deferred
