#!/usr/bin/env python3
"""Turn one anti-affinity run's logs into raw per-cycle records plus a summary.

The aggregate is derived from the raw rows and never replaces them, which is
this project's rule and the reason its earlier findings survived re-reading.

What each cycle carries, and why:

``aa_violations``
    the unconstrained argmin wanted to put this shard on a GPU that already
    holds a part of the same model.  This is the counterfactual, and it is
    counted in **every** arm -- the OFF arm has to record the collisions it
    commits, or the ON/OFF comparison is an assertion rather than a measurement.
``aa_diverted``
    the constraint actually changed the choice.
``aa_second_also_collides``
    Appendix A.2.2 falls back to the *second-lowest* KVPR GPU and does not
    re-check it.  When that GPU also holds a part of the same model the paper's
    rule places there anyway.  Only reachable at tp_size >= 3.
``group_plan`` / ``current_groups``
    where the planner wanted each TP model, and where it actually was.

A run with every counter at zero is reported as "the constraint never bound",
not as "the constraint works".  Per the paper's own wording (A.2.2: the
decomposition "increases the likelihood" that parts land on different GPUs)
that is the expected common case.
"""

import argparse
import csv
import json
import re
from pathlib import Path

MARKER = "[PAPER-ALG1-TP] "


def read_cycles(run_dir: Path):
    """Every [PAPER-ALG1-TP] cycle record in the controller's log."""
    rows = []
    for path in sorted(run_dir.glob("server-logs/*.log")):
        try:
            text = path.read_text(errors="replace")
        except OSError:
            continue
        for line in text.splitlines():
            i = line.find(MARKER)
            if i < 0:
                continue
            payload = line[i + len(MARKER):].lstrip()
            if not payload.startswith("{"):
                continue          # the "init" and "group moved" lines
            try:
                rows.append(json.loads(payload))
            except json.JSONDecodeError:
                pass
    rows.sort(key=lambda r: r.get("timestamp", 0))
    return rows


def deltas(cycles):
    """Per-cycle increments of the cumulative audit counters.

    The engine reports totals; a per-cycle view is what makes "when did it
    bind" answerable, so both are kept.
    """
    keys = ("aa_violations", "aa_diverted", "aa_infeasible",
            "aa_second_also_collides", "shards_placed",
            "snapped_to_group", "group_unavailable")
    prev = {k: 0 for k in keys}
    out = []
    for c in cycles:
        audit = c.get("tp_audit", {})
        row = {f"d_{k}": int(audit.get(k, 0)) - prev[k] for k in keys}
        prev = {k: int(audit.get(k, 0)) for k in keys}
        out.append(row)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--rate", required=True)
    ap.add_argument("--seed", required=True)
    ap.add_argument("--outbase", required=True)
    ns = ap.parse_args()

    run = Path(ns.run)
    outbase = Path(ns.outbase)
    tag = f"tp-aa-{ns.arm}_r{ns.rate}_s{ns.seed}"
    raw = outbase / "raw"
    (raw / "alg1_tp").mkdir(parents=True, exist_ok=True)
    (raw / "placements").mkdir(parents=True, exist_ok=True)

    cycles = read_cycles(run)
    d = deltas(cycles)

    # 1. every cycle, verbatim -- the aggregate below is derived from this file
    with (raw / "alg1_tp" / f"{tag}.jsonl").open("w") as fh:
        for c, inc in zip(cycles, d):
            fh.write(json.dumps({**c, **inc}) + "\n")

    # 2. the placement each cycle produced, flat enough to diff between arms
    with (raw / "placements" / f"{tag}.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["cycle", "timestamp", "model", "tp_size",
                    "current_gpus", "planned_gpus", "planned_collides"])
        for c in cycles:
            cyc = c.get("tp_audit", {}).get("cycles")
            plan = c.get("group_plan") or {}
            cur = c.get("current_groups") or {}
            for model, gpus in plan.items():
                g = list(gpus)
                w.writerow([cyc, c.get("timestamp"), model, len(g),
                            "|".join(map(str, cur.get(model, []))),
                            "|".join(map(str, g)),
                            int(len(set(g)) != len(g))])

    # 3. requests, if the benchmark produced them
    req = {}
    for p in list(run.glob("requests/*.json")) + list(run.glob("*_output_requests.json")):
        try:
            data = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, list):
            req["total"] = req.get("total", 0) + len(data)
            req["ok"] = req.get("ok", 0) + sum(1 for r in data if r.get("success"))

    last = cycles[-1]["tp_audit"] if cycles else {}
    summary = {
        "tag": tag, "arm": ns.arm, "rate": ns.rate, "seed": ns.seed,
        "cycles_logged": len(cycles),
        "anti_affinity": cycles[-1].get("anti_affinity") if cycles else None,
        "anti_affinity_strict": cycles[-1].get("anti_affinity_strict") if cycles else None,
        "any_tp_group_active_ever": any(c.get("any_tp_group_active") for c in cycles),
        "totals": {k: last.get(k, 0) for k in
                   ("shards_placed", "aa_violations", "aa_diverted",
                    "aa_infeasible", "aa_second_also_collides",
                    "snapped_to_group", "group_unavailable")},
        "cycles_with_a_violation": sum(1 for r in d if r["d_aa_violations"] > 0),
        "cycles_with_a_diversion": sum(1 for r in d if r["d_aa_diverted"] > 0),
        "colliding_plans_emitted": sum(
            1 for c in cycles for g in (c.get("group_plan") or {}).values()
            if len(set(g)) != len(g)),
        "requests": req or None,
        "raw": {
            "cycles": str((raw / "alg1_tp" / f"{tag}.jsonl").relative_to(outbase)),
            "placements": str((raw / "placements" / f"{tag}.csv").relative_to(outbase)),
        },
    }
    if summary["totals"]["aa_violations"] == 0:
        summary["note"] = (
            "the constraint never bound in this run -- reported as such, not as "
            "evidence that it works. Paper A.2.2 expects this to be common: the "
            "1/tp_size decomposition already 'increases the likelihood' that "
            "parts land on different GPUs."
        )
    (outbase / f"{tag}_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
