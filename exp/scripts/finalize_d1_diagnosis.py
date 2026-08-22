#!/usr/bin/env python3
import argparse
import csv
import glob
import json
import os
import sys


def read_row(path):
    with open(path, newline="") as f:
        return next(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    args = ap.parse_args()
    root = os.path.abspath(args.root)
    result_root = os.path.join(root, "exp/results/final-regression-diagnosis")
    d0 = read_row(os.path.join(result_root, "D0/summary.csv"))
    d1 = read_row(os.path.join(result_root, "D1/summary.csv"))

    request_files = glob.glob(os.path.join(
        result_root, "D1/raw/paper-alg2-only/steady/rate_8/seed_1/requests/",
        "*output_requests.json"))
    if len(request_files) != 1:
        raise SystemExit(f"expected one D1 request dump, found {request_files}")
    with open(request_files[0]) as f:
        outputs = json.load(f)

    sys.path.insert(0, os.path.join(root, "exp/scripts"))
    from build_sharegpt_trace import _Unpickler
    with open(os.path.join(result_root, "workloads/steady_r8_s1.pkl"), "rb") as f:
        _, trace = _Unpickler(f).load()
    if len(outputs) != len(trace):
        raise SystemExit(f"output/trace mismatch: {len(outputs)} != {len(trace)}")

    events = {}
    alg2_glob = os.path.join(
        result_root, "D1/raw/paper-alg2-only/steady/rate_8/seed_1/server-logs/",
        "server.log.alg2_gpu*.jsonl")
    for path in glob.glob(alg2_glob):
        with open(path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("type") == "request_decision":
                    events.setdefault(str(rec["rid"]), []).append(rec)

    audit_path = os.path.join(result_root, "alg2_prediction_audit.csv")
    fields = [
        "request_id", "model", "prompt_tokens", "arrival_time", "deadline",
        "ttft_slo_s", "c_i_tokens_per_s", "predicted_prefill_time_s",
        "first_alg2_decision", "final_alg2_decision", "alg2_decision_transitions",
        "actual_queue_wait_s", "actual_prefill_start", "actual_prefill_end",
        "actual_prefill_service_time_s", "actual_ttft_s", "success", "error",
        "batch_size", "concurrent_prefill_count", "running_decode_count",
    ]
    with open(audit_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for req, out in zip(trace, outputs):
            ev = events.get(str(req.req_id), [])
            first = ev[0] if ev else {}
            arrival = out.get("arrival_time") or first.get("arrival_s")
            outq = out.get("out_queue_time")
            pf = out.get("prefill_finish_time")
            w.writerow({
                "request_id": req.req_id,
                "model": req.model,
                "prompt_tokens": req.prompt_len,
                "arrival_time": arrival,
                "deadline": first.get("deadline_s"),
                "ttft_slo_s": req.slo_ttft,
                "c_i_tokens_per_s": first.get("c_i_tokens_per_s"),
                "predicted_prefill_time_s": first.get("predicted_exec_s"),
                "first_alg2_decision": ev[0].get("decision") if ev else "",
                "final_alg2_decision": ev[-1].get("decision") if ev else "",
                "alg2_decision_transitions": len(ev),
                "actual_queue_wait_s": out.get("wait_time"),
                "actual_prefill_start": outq,
                "actual_prefill_end": pf,
                "actual_prefill_service_time_s": (pf - outq) if pf and outq else "",
                "actual_ttft_s": out.get("ttft"),
                "success": out.get("success"),
                "error": out.get("error", ""),
                "batch_size": "",
                "concurrent_prefill_count": "",
                "running_decode_count": "",
            })

    comparison = {
        "D0": d0,
        "D1": d1,
        "delta": {
            "joint_slo": float(d1["joint_slo"]) - float(d0["joint_slo"]),
            "goodput": float(d1["goodput"]) - float(d0["goodput"]),
            "ttft_slo": float(d1["ttft_slo"]) - float(d0["ttft_slo"]),
            "tpot_slo": float(d1["tpot_slo"]) - float(d0["tpot_slo"]),
        },
        "alg2_request_decisions": {
            "unique_requests": len(events),
            "transitions": sum(len(v) for v in events.values()),
        },
        "audit_csv": audit_path,
    }
    out_path = os.path.join(result_root, "D1/result.json")
    with open(out_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(json.dumps(comparison["delta"], sort_keys=True))
    print(f"wrote {out_path}")
    print(f"wrote {audit_path}")


if __name__ == "__main__":
    main()
