#!/usr/bin/env python3
"""Test B — Moore-Hodgson (paper Algorithm 2) against hand-computed examples.

Run:  python exp/tests/test_moore_hodgson.py
"""
import sys

from sglang.multi_model.scheduling.gpu.moore_hodgson import MHJob, is_feasible, select

FAILURES = []


def check(name, cond, detail=""):
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        FAILURES.append(name)


def names(jobs):
    return [j.key for j in jobs]


def edf_on_time(jobs, now=0.0):
    """Plain EDF with no dropping: run all in deadline order, count on-time jobs.

    Used only as a contrast, to show the drop step is doing real work.
    """
    clock, on_time = now, 0
    for j in sorted(jobs, key=lambda j: j.deadline):
        clock += j.exec_time
        if clock <= j.deadline:
            on_time += 1
    return on_time


print("== 1. worked example: one infeasible job is shed ==")
# d/e:  A(4,3) B(5,2) C(6,4) D(8,1), now=0
#  A -> clock 3 <= 4 ok
#  B -> clock 5 <= 5 ok
#  C -> clock 9 >  6 : drop longest of {A,B,C} = C(e=4) -> clock 5
#  D -> clock 6 <= 8 ok
jobs = [MHJob("A", 4, 3), MHJob("B", 5, 2), MHJob("C", 6, 4), MHJob("D", 8, 1)]
sel, defd = select(jobs, 0.0)
check("selected == A,B,D", names(sel) == ["A", "B", "D"], f"got {names(sel)}")
check("deferred == C", names(defd) == ["C"], f"got {names(defd)}")
check("selection is feasible", is_feasible(sel, 0.0))

print("== 2. drop-the-longest, not drop-the-latest ==")
# A(6,6) B(7,1) C(7,1): adding C overflows. The evicted job must be A (longest,
# e=6), NOT C (the one that triggered it). This is exactly the step the released
# prototype lacks.
jobs = [MHJob("A", 6, 6), MHJob("B", 7, 1), MHJob("C", 7, 1)]
sel, defd = select(jobs, 0.0)
check("evicts the LONGEST job A", names(defd) == ["A"], f"got {names(defd)}")
check("keeps B and C", sorted(names(sel)) == ["B", "C"], f"got {names(sel)}")
check("selection is feasible", is_feasible(sel, 0.0))

print("== 3. beats plain EDF on number of on-time jobs ==")
# A long, early-deadline job would block three short ones under EDF.
#   EDF order A,B,C,D: clock 10(ok) 12(late) 14(late) 16(late) -> 1 on time
#   Moore-Hodgson sheds A -> B,C,D all on time                 -> 3 on time
jobs = [MHJob("A", 10, 10), MHJob("B", 11, 2), MHJob("C", 12, 2), MHJob("D", 13, 2)]
sel, defd = select(jobs, 0.0)
check("MH keeps 3 jobs", len(sel) == 3, f"got {names(sel)}")
check("MH sheds the blocker A", names(defd) == ["A"], f"got {names(defd)}")
check("MH selection is feasible", is_feasible(sel, 0.0))
check("plain EDF would only make 1 on time", edf_on_time(jobs) == 1, f"got {edf_on_time(jobs)}")

print("== 4. everything feasible -> nothing deferred ==")
jobs = [MHJob("A", 10, 1), MHJob("B", 20, 1), MHJob("C", 30, 1)]
sel, defd = select(jobs, 0.0)
check("all selected", len(sel) == 3 and not defd, f"sel={names(sel)} def={names(defd)}")

print("== 5. nothing feasible -> everything deferred ==")
# Each job alone already misses its deadline (now=100 is past every deadline).
jobs = [MHJob("A", 1, 5), MHJob("B", 2, 5)]
sel, defd = select(jobs, 100.0)
check("all deferred", sel == [] and len(defd) == 2, f"sel={names(sel)} def={names(defd)}")

print("== 6. `now` offset is honoured ==")
# Same jobs, absolute deadlines shifted by now=1000.
jobs = [MHJob("A", 1004, 3), MHJob("B", 1005, 2), MHJob("C", 1006, 4), MHJob("D", 1008, 1)]
sel, defd = select(jobs, 1000.0)
check("same result as example 1", names(sel) == ["A", "B", "D"], f"got {names(sel)}")

print("== 7. empty input ==")
sel, defd = select([], 0.0)
check("empty in, empty out", sel == [] and defd == [])

print("== 8. output is in dispatch (ascending deadline) order ==")
jobs = [MHJob("late", 100, 1), MHJob("early", 10, 1), MHJob("mid", 50, 1)]
sel, _ = select(jobs, 0.0)
check("ascending deadline", names(sel) == ["early", "mid", "late"], f"got {names(sel)}")

print("== 9. past-deadline jobs are separable (no starvation) ==")
# THE LIVELOCK REGRESSION. A request whose deadline has already passed can never
# be feasible, so select() evicts it every round forever. The scheduler must be
# able to tell those apart from jobs merely crowded out this round, or it holds
# them until the client gives up (observed: 328k rounds, benchmark hung).
now = 1000.0
jobs = [
    MHJob("expired-1", now - 5, 0.1),    # deadline already gone
    MHJob("expired-2", now - 1, 0.1),
    MHJob("feasible", now + 100, 0.1),   # plenty of headroom
]
sel, defd = select(jobs, now)
expired = [j for j in defd if j.deadline <= now]
requeued = [j for j in defd if j.deadline > now]
check("feasible job is selected", "feasible" in names(sel), f"got {names(sel)}")
check("both expired jobs are deferred", len(expired) == 2, f"got {names(expired)}")
check("nothing feasible is left requeued", requeued == [], f"got {names(requeued)}")
# Re-running with the same input must give the same verdict -- that is precisely
# why holding them back loops forever, and why they must be dispatched instead.
sel2, defd2 = select(jobs, now)
check("verdict is stable across rounds (hence the livelock)",
      names(defd2) == names(defd), f"{names(defd2)} vs {names(defd)}")

print("== 10. maximum cardinality vs brute force (randomised) ==")
# Moore-Hodgson is optimal for 1||sum U_j; verify against exhaustive search.
import itertools
import random

rng = random.Random(20260813)
worst = None
for trial in range(300):
    n = rng.randint(1, 9)
    jobs = [
        MHJob(i, round(rng.uniform(1, 20), 3), round(rng.uniform(0.5, 8), 3))
        for i in range(n)
    ]
    sel, _ = select(jobs, 0.0)
    if not is_feasible(sel, 0.0):
        worst = ("infeasible", trial, jobs)
        break
    best = 0
    for k in range(n, 0, -1):
        found = False
        for combo in itertools.combinations(jobs, k):
            if is_feasible(combo, 0.0):
                found = True
                break
        if found:
            best = k
            break
    if len(sel) != best:
        worst = ("suboptimal", trial, jobs, len(sel), best)
        break
check("optimal on 300 random instances", worst is None, f"counterexample: {worst}")

print()
if FAILURES:
    print(f"FAILED: {len(FAILURES)} check(s): {FAILURES}")
    sys.exit(1)
print("ALL MOORE-HODGSON TESTS PASSED")
