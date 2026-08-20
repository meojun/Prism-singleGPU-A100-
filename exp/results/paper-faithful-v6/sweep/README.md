# How to read this sweep

## What it is

`paper-faithful-v4` only, six runs (bursty/steady x seed 1-3, 8 req/s), launched
while the v6 arm was still unusable. It is the **control half** of the eventual
v4-vs-v6 comparison, banked in advance: the KV-migration work is flag-gated, so
these numbers stay valid however that flag's implementation changes.

When v6 works, run `SWEEP_ARMS=paper-faithful-v6 exp/scripts/run_v6_sweep.sh` --
finished runs are skipped by their DONE markers, so only the new arm executes.

## Do not compare these to the v4 report

Three things differ at once, and the sweep cannot separate them:

1. **A different box.** Compute is close (TTFT baselines within ~2% for four of
   six models) but the interconnect is not -- P2P migration runs 105 GB/s here
   against 72.9 there.
2. **P2P weight migration is ON.** The v4 sweep ran it *off*
   (`PRISM_V4_P2P_MIGRATION=0`) so that every run in that study shared one
   configuration after the IPC-leak OOM. V5 re-enabled and re-validated it, and
   `run_v4_case.sh` defaults it on, so these runs use it -- the first run already
   shows 8 gpu-to-gpu weight transfers where the v4 report shows 0.
3. **Traces rebuilt** against this box's SLO baselines, as they must be.

The first run lands at goodput 5.533, TTFT p50 64.2 ms, TPOT p50 25.3 ms, against
the v4 report's 1.296 / 118.7 / 53.4 for the same arm and condition. That gap is
large and it is **not attributable** -- it spans all three changes above. Report
it as "this box, this configuration", never as an improvement over the v4 study.

## What it can support

Comparisons *within* this sweep, against a v6 arm run on the same box with the
same configuration. That is what it exists for.

Three seeds per condition, because the v4 study measured seed-to-seed spread at
70-80% of the mean at these rates. `SUMMARY.txt` prints per-seed values and
refuses to call a winner when the gap sits inside the combined spread.
