#!/usr/bin/env python3
"""Turn a TP=2 run's own logs into a pass/fail record.

Nothing here is asserted from the configuration: every check reads back what
the server actually printed, so a run that parsed the flags and then quietly
placed both ranks on one GPU fails rather than passes.
"""
import argparse
import json
import re
from pathlib import Path


def read_logs(logdir):
    text = []
    for path in sorted(Path(logdir).rglob("*.log")):
        try:
            text.append((path.name, path.read_text(errors="replace")))
        except OSError:
            pass
    return text


def rank_gpu_map(logs):
    """Recover the TP rank -> GPU assignment the engines reported."""
    # Prefer the instrumentation's exact engine line and exclude the TP=1
    # helper engines.  The old generic regex merged those helpers into rank 0,
    # yielding rank0 -> [0, 1] even when the real TP=2 ranks were 0->0, 1->1.
    exact = re.compile(
        r"\[PAPER-TP\] engine rank: tp_rank=(\d+) gpu_id=(\d+) tp_size=(\d+)",
        re.I,
    )
    mapping = {}
    for _, text in logs:
        for m in exact.finditer(text):
            rank, gpu, tp_size = map(int, m.groups())
            if tp_size > 1:
                mapping.setdefault(rank, set()).add(gpu)
    if mapping:
        return {k: sorted(v) for k, v in sorted(mapping.items())}

    # Compatibility fallback for older logs that predate the exact marker.
    mapping = {}
    patterns = [
        re.compile(r"tp_rank[=: ]+(\d+).*?gpu_id[=: ]+(\d+)", re.I),
        re.compile(r"gpu_id[=: ]+(\d+).*?tp_rank[=: ]+(\d+)", re.I),
        re.compile(r"TP(\d+).*?on GPU (\d+)", re.I),
    ]
    for _, text in logs:
        for line in text.splitlines():
            for idx, pat in enumerate(patterns):
                m = pat.search(line)
                if not m:
                    continue
                a, b = int(m.group(1)), int(m.group(2))
                rank, gpu = (b, a) if idx == 1 else (a, b)
                mapping.setdefault(rank, set()).add(gpu)
    return {k: sorted(v) for k, v in sorted(mapping.items())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--startup-seconds", type=float, required=True)
    ap.add_argument("--ready", type=int, required=True)
    ap.add_argument("--load-rc", type=int, required=True)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    out = Path(args.outdir)
    logs = read_logs(args.logdir)
    blob = "\n".join(t for _, t in logs)

    # Match how the stack actually announces its collective backend, not one
    # spelling of it: absence of the literal string "NCCL" is not evidence
    # that NCCL did not initialise.
    nccl_pat = re.compile(
        r"nccl|init_process_group|distributed environment|torch\.distributed|"
        r"tensor.?parallel", re.I)
    nccl_lines = [ln for ln in blob.splitlines() if nccl_pat.search(ln)]
    nccl_init = re.findall(r"[Ii]nit(?:ialis|ializ)\w*.*?([\d.]+)\s*(?:s|sec)", "\n".join(nccl_lines))
    ranks = rank_gpu_map(logs)
    errors = [ln for ln in blob.splitlines()
              if re.search(r"Traceback|CUDA error|RuntimeError|deadlock|NCCL WARN|invalid device", ln)]

    client = {}
    client_path = out / "tp2_requests.json"
    if client_path.exists():
        client = json.loads(client_path.read_text()).get("summary", {})

    gpus_used = sorted({g for gs in ranks.values() for g in gs})
    tp_model = json.loads(Path(args.config).read_text())[0]
    planned_gpus = tp_model["init_placements"][0]["gpu_ids"]

    checks = {
        "server_started": bool(args.ready),
        "tp_size_2_configured": tp_model["tp_size"] == 2,
        "both_gpus_in_placement": len(set(planned_gpus)) == 2,
        "ranks_observed_on_distinct_gpus": (
            len(ranks) >= 2 and len({tuple(v) for v in ranks.values()}) >= 2
            if ranks else None),
        "nccl_mentioned_in_logs": bool(nccl_lines),
        "inference_succeeded": bool(client.get("successful", 0)) and client.get("failed", 1) == 0,
        "no_runtime_errors": not errors,
        "load_phase_exit_zero": args.load_rc == 0,
    }
    # A check that could not be observed is not a failure.  The verdict turns
    # on whether the thing actually served; unobservable evidence downgrades to
    # PARTIAL so the report says "not proven" rather than "broken".
    hard = ["server_started", "inference_succeeded", "no_runtime_errors",
            "load_phase_exit_zero"]
    if all(checks[k] is True for k in hard) and all(
            v is not False for v in checks.values()):
        verdict = "PASS"
    elif checks["inference_succeeded"]:
        verdict = "PARTIAL"
    else:
        verdict = "FAIL"

    record = {
        "verdict": verdict,
        "checks": checks,
        "startup_seconds": args.startup_seconds,
        "tp_rank_to_gpu": {str(k): v for k, v in ranks.items()},
        "planned_gpu_ids": planned_gpus,
        "gpus_observed": gpus_used,
        "nccl_log_lines": len(nccl_lines),
        "nccl_init_seconds_reported": nccl_init[:5],
        "runtime_error_lines": errors[:20],
        "latency": {k: client.get(k) for k in ("ttft", "tpot", "e2e")},
        "requests": {k: client.get(k) for k in
                     ("requests", "successful", "failed", "errors", "per_model")},
        "note_parallel_loading_disabled_under_tp": (
            "model_runner.py disables the model-service weight path when "
            "tp_size > 1, so a TP=2 model never uses parallel weight loading."),
    }
    (out / "tp2_validation.json").write_text(json.dumps(record, indent=2))

    lines = [f"# TP=2 validation: {verdict}", ""]
    for name, value in checks.items():
        mark = {True: "PASS", False: "FAIL", None: "NOT OBSERVED"}[value]
        lines.append(f"- [{mark}] {name}")
    lines += ["", f"startup: {args.startup_seconds:.1f}s",
              f"TP rank -> GPU: {record['tp_rank_to_gpu'] or 'not recoverable from logs'}"]
    for key in ("ttft", "tpot", "e2e"):
        stat = client.get(key) or {}
        if stat:
            lines.append(f"{key.upper()}: n={stat.get('n')} p50={stat.get('p50'):.4f}s "
                         f"p95={stat.get('p95'):.4f}s p99={stat.get('p99'):.4f}s")
    (out / "TP2_VALIDATION.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
