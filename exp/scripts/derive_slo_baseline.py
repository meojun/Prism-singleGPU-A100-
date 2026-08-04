#!/usr/bin/env python3
"""Re-derive per-slot SLO baselines on THIS machine, for THIS workload.

Why this exists
---------------
`analyze_slo.py` (and `trace.py`) carry a hardcoded SLO_BASE table.  Those are
the Prism authors' numbers, measured on H100 with their workload.  On this
A100 the same slot measures noticeably slower -- model_1 came out at
TTFT p95 63.1 ms against the built-in 42.9 ms (1.47x) and TPOT p95 14.15 ms
against 11.46 ms (1.23x).  Swapping the dataset moves it again.

Comparing your system against Prism does not strictly require a correct
baseline -- both sides face the same threshold -- but an SLO that no longer
reflects unloaded performance makes "attainment" arbitrary: set it loose enough
and everything passes, tight enough and nothing does.  Re-derive it whenever
the hardware or the trace changes.

Method (paper §7.1): run each slot ALONE on the GPU, unloaded, over the trace
you will evaluate with, and take the p95 of TTFT and TPOT.  Case A of
run_sanity.sh already is exactly that for model_1.

Usage:
    # after a dedicated run, e.g. TAG=sharegpt_content ./run_sanity.sh A
    python exp/scripts/derive_slo_baseline.py \
        --run sharegpt_content:A:model_1 \
        --out exp/configs/slo_base_a100_sharegpt.json

    # then feed it back in
    SLO_BASE_FILE=exp/configs/slo_base_a100_sharegpt.json ./run_sanity.sh B

Each --run is TAG:CASE:SLOT.  Only slots that were alone on the GPU during that
run are valid; passing a colocated slot is refused, because a contended p95 is
not a baseline.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

RESULTS = "/workspace/prism-exp/exp/results"
# slots that share a GPU in each stock case -- used to reject contended inputs
CASE_SLOTS = {"A": ["model_1"], "B": ["model_1", "model_4"], "C": ["model_1", "model_2"]}


def load_requests(tag, case):
    pat = os.path.join(RESULTS, tag, "requests", f"{tag}_{case}_e2e_1gpu_*_output_requests.json")
    hits = sorted(glob.glob(pat))
    if not hits:
        raise SystemExit(f"no request dump for {tag}/{case} (looked for {pat})")
    return json.load(open(hits[0]))


def derive(tag, case, slot, allow_contended=False):
    slots = CASE_SLOTS.get(case)
    if slots and len(slots) > 1 and not allow_contended:
        raise SystemExit(
            f"case {case} colocates {slots} -- a contended p95 is not a baseline. "
            f"Run the slot alone (case A is the dedicated run for model_1), or pass "
            f"--allow-contended if you really mean it."
        )
    rows = [r for r in load_requests(tag, case)
            if r.get("success") and r.get("model") == slot
            and r.get("ttft") and r.get("tpot")]
    if not rows:
        # older dumps may not carry the model field when only one model ran
        rows = [r for r in load_requests(tag, case)
                if r.get("success") and r.get("ttft") and r.get("tpot")]
    if not rows:
        raise SystemExit(f"no successful requests for {slot} in {tag}/{case}")
    ttft = np.array([r["ttft"] for r in rows])
    tpot = np.array([r["tpot"] for r in rows])
    return {
        "slot": slot,
        "n": len(rows),
        "ttft_p95_s": float(np.percentile(ttft, 95)),
        "tpot_p95_ms": float(np.percentile(tpot, 95) * 1000.0),
        "source": f"{tag}/{case}",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True,
                    metavar="TAG:CASE:SLOT", help="repeatable")
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow-contended", action="store_true")
    args = ap.parse_args()

    table, detail = {}, []
    for spec in args.run:
        try:
            tag, case, slot = spec.split(":")
        except ValueError:
            raise SystemExit(f"--run must be TAG:CASE:SLOT, got {spec!r}")
        d = derive(tag, case, slot, args.allow_contended)
        table[slot] = [d["ttft_p95_s"], d["tpot_p95_ms"]]
        detail.append(d)

    # Report against the built-ins so the size of the correction is visible.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from analyze_slo import SLO_BASE as BUILTIN

    print(f"{'slot':10s} {'n':>5s} {'TTFT p95 (ms)':>15s} {'built-in':>10s} {'ratio':>7s}"
          f" | {'TPOT p95 (ms)':>15s} {'built-in':>10s} {'ratio':>7s}")
    print("-" * 96)
    for d in detail:
        b = BUILTIN.get(d["slot"])
        bt, bp = (b[0] * 1000, b[1]) if b else (float("nan"), float("nan"))
        print(f"{d['slot']:10s} {d['n']:5d} {d['ttft_p95_s']*1000:15.1f} {bt:10.1f}"
              f" {d['ttft_p95_s']*1000/bt:6.2f}x | {d['tpot_p95_ms']:15.2f} {bp:10.2f}"
              f" {d['tpot_p95_ms']/bp:6.2f}x")

    missing = [s for s in BUILTIN if s not in table]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(table, open(args.out, "w"), indent=2)
    print(f"\n{len(table)}개 슬롯 -> {args.out}")
    if missing:
        print(f"측정 안 된 슬롯은 built-in 값을 그대로 씀: {sorted(missing)}")


if __name__ == "__main__":
    main()
