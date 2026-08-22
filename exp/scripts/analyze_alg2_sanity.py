#!/usr/bin/env python3
"""Analyze the paired short D0/D1 audit-off Algorithm-2 sanity run.

This script is analysis-only.  It reads the benchmark request dumps and writes
derived CSV/JSON artifacts; it does not change scheduler policy or runtime code.
"""

from __future__ import annotations

import csv
import argparse
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "exp/results/final-regression-diagnosis/alg2-root-cause"
SANITY = OUT / "sanity"
CI_FILE = ROOT / "exp/configs/v2/prefill_speed.json"

ARMS = {
    "D0": SANITY / "D0/raw/released-prototype/steady/rate_8/seed_1",
    "D1-audit-off": SANITY / "D1-audit-off/raw/paper-alg2-only/steady/rate_8/seed_1",
}


def pct(values: list[float]) -> dict[str, float | int | None]:
    a = np.asarray([v for v in values if v is not None and math.isfinite(v)], dtype=float)
    if not len(a):
        return {"n": 0, "mean": None, "p50": None, "p95": None, "p99": None, "max": None}
    return {
        "n": int(len(a)),
        "mean": float(np.mean(a)),
        "p50": float(np.percentile(a, 50)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
        "max": float(np.max(a)),
    }


def req_dump(path: Path) -> Path:
    files = sorted((path / "requests").glob("*_output_requests.json"))
    if len(files) != 1:
        raise RuntimeError(f"expected one request dump under {path}, found {files}")
    return files[0]


def metrics_dump(path: Path) -> Path:
    files = sorted(p for p in path.glob("*.json") if "output_requests" not in p.name)
    if len(files) != 1:
        raise RuntimeError(f"expected one metrics dump under {path}, found {files}")
    return files[0]


def positive_delta(a: float | None, b: float | None) -> float | None:
    if not a or not b:
        return None
    value = a - b
    return value if value >= -1e-6 else None


def correlation(a: list[float], b: list[float]) -> float | None:
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def analyze_arm(name: str, path: Path) -> tuple[dict, list[dict], list[dict]]:
    requests = json.load(req_dump(path).open())
    metrics = json.load(metrics_dump(path).open())
    successful = [r for r in requests if r.get("success")]

    stages: dict[str, list[float]] = {
        "frontend_wait": [],
        "local_scheduler_wait": [],
        "prefill_wait": [],
        "actual_prefill_service": [],
        "decode_start_wait": [],
        "ttft": [],
        "tpot": [],
        "e2e": [],
    }
    per_request = []
    for idx, r in enumerate(successful):
        arrival = r.get("arrival_time")
        queued = r.get("gpu_scheduler_queue_time")
        dispatched = r.get("gpu_scheduler_dispatch_time")
        out_queue = r.get("out_queue_time")
        prefill_done = r.get("prefill_finish_time")
        finish = r.get("finish_time")
        decodes = r.get("decode_timestamps") or []
        values = {
            "frontend_wait": positive_delta(queued, arrival),
            "local_scheduler_wait": positive_delta(dispatched, queued),
            "prefill_wait": positive_delta(out_queue, dispatched),
            "actual_prefill_service": positive_delta(prefill_done, out_queue),
            "decode_start_wait": positive_delta(decodes[0], prefill_done) if decodes else None,
            "ttft": r.get("ttft"),
            "tpot": r.get("tpot"),
            "e2e": r.get("latency_server") or r.get("latency"),
        }
        for stage, value in values.items():
            if value is not None and math.isfinite(value):
                stages[stage].append(value)
        per_request.append({
            "arm": name,
            "index": idx,
            "model": r.get("model"),
            "prompt_tokens": r.get("prompt_len"),
            **{f"{k}_s": v for k, v in values.items()},
        })

    ttft_met = [r["ttft"] <= r["slo_ttft"] for r in successful]
    tpot_met = [r["tpot"] <= r["slo_tpot"] for r in successful]
    summary = {
        "arm": name,
        "offered": len(requests),
        "accepted": len(requests),
        "completed": len(successful),
        "unfinished": 0,
        "rejected": 0,
        "aborted": len(requests) - len(successful),
        "request_throughput": metrics.get("request_throughput"),
        "ttft_slo": float(np.mean(ttft_met)) if ttft_met else None,
        "tpot_slo": float(np.mean(tpot_met)) if tpot_met else None,
        "joint_slo": float(np.mean(np.logical_and(ttft_met, tpot_met))) if ttft_met else None,
        "stages_seconds": {stage: pct(values) for stage, values in stages.items()},
    }
    stage_rows = []
    for stage, values in stages.items():
        stat = pct(values)
        stage_rows.append({"arm": name, "stage": stage, **stat})
    return summary, stage_rows, per_request


def prediction_analysis(per_request: list[dict]) -> tuple[dict, list[dict]]:
    ci = json.load(CI_FILE.open())
    rows = []
    for r in per_request:
        model = r["model"]
        actual = r.get("actual_prefill_service_s")
        if actual is None:
            continue
        predicted = float(r["prompt_tokens"]) / float(ci[model])
        signed = predicted - actual
        rows.append({
            "index": r["index"],
            "model": model,
            "prompt_tokens": r["prompt_tokens"],
            "c_i_tokens_per_s": ci[model],
            "predicted_prefill_s": predicted,
            "actual_prefill_service_s": actual,
            "signed_error_s": signed,
            "absolute_error_s": abs(signed),
            "relative_error": signed / actual if actual else None,
        })

    def group(rs: list[dict]) -> dict:
        pred = [r["predicted_prefill_s"] for r in rs]
        actual = [r["actual_prefill_service_s"] for r in rs]
        signed = [r["signed_error_s"] for r in rs]
        absolute = [r["absolute_error_s"] for r in rs]
        relative = [r["relative_error"] for r in rs if r["relative_error"] is not None]
        return {
            "n": len(rs),
            "predicted_seconds": pct(pred),
            "actual_seconds": pct(actual),
            "signed_error_seconds": pct(signed),
            "absolute_error_seconds": pct(absolute),
            "relative_error": pct(relative),
            "predicted_actual_correlation": correlation(pred, actual),
        }

    result = {"overall": group(rows), "by_model": {}}
    for model in sorted(ci):
        result["by_model"][model] = group([r for r in rows if r["model"] == model])
    return result, rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--d0", type=Path, default=ARMS["D0"])
    parser.add_argument("--d1", type=Path, default=ARMS["D1-audit-off"])
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--d1-label", default="D1-audit-off")
    args = parser.parse_args()
    arms = {"D0": args.d0, args.d1_label: args.d1}

    summaries, stage_rows, all_requests = [], [], []
    per_arm = {}
    for name, path in arms.items():
        summary, rows, requests = analyze_arm(name, path)
        summaries.append(summary)
        stage_rows.extend(rows)
        all_requests.extend(requests)
        per_arm[name] = requests

    prediction, prediction_rows = prediction_analysis(per_arm[args.d1_label])
    result = {"paired_sanity": summaries, "d1_prediction_vs_actual_prefill": prediction}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "sanity_analysis.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )

    for filename, rows in (
        ("queue_wait_comparison.csv", stage_rows),
        ("sanity_request_stages.csv", all_requests),
        ("sanity_prediction_audit.csv", prediction_rows),
    ):
        with (args.output_dir / filename).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
