#!/usr/bin/env python3
"""서버 로그에서 진단 수치를 뽑아 processed/diagnostics.json 으로 저장한다.

REPORT.md 의 원인 분석 절이 인용하는 숫자를 손으로 세지 않고 재현하기 위한 것.
두 가지를 뽑는다.

1. 구간별(decile) TTFT p50 — 백로그가 쌓이기만 하는지, 아니면 회복하는지를 본다.
   집계 p50 하나로는 "일시적 스파이크"와 "영구 적체"가 구분되지 않는다.
2. Moore-Hodgson 라운드 통계 — 라운드당 실제로 몇 개나 admit 되었는지.
   과소 수용(under-admission) 가설의 직접 증거.

server-logs/ 와 requests/ 는 용량 때문에 git 에 올리지 않으므로, 이 스크립트는
실험을 돌린 머신에서만 완전한 결과를 낸다. 입력이 없으면 해당 항목을 조용히
비우고 넘어간다(부분 결과라도 남기는 편이 낫다).
"""
import argparse
import glob
import json
import os
import re

ROUND_RX = re.compile(
    r"round=(\d+) eligible=(\d+) selected=(\d+) requeued=(\d+) "
    r"late_dispatched=(\d+).*queue_len=(\d+)"
)
QLEN_RX = re.compile(r"queue_len[=:] ?(\d+)")


def ttft_deciles(rundir, nbins=10):
    """성공한 요청을 도착 순서로 nbins 등분하고 각 구간의 TTFT 중앙값(초)."""
    files = glob.glob(os.path.join(rundir, "requests", "*_output_requests.json"))
    if not files:
        return None
    reqs = json.load(open(files[0]))
    # 덤프는 트레이스 순서를 보존하므로 인덱스가 곧 도착 순서다.
    ok = [r for r in reqs if r.get("success")]
    if len(ok) < nbins:
        return None
    out = []
    n = len(ok)
    for i in range(nbins):
        seg = ok[int(n * i / nbins): int(n * (i + 1) / nbins)]
        t = sorted(r["ttft"] for r in seg)
        out.append(round(t[len(t) // 2], 3))
    return {"n_success": n, "p50_by_decile_s": out}


def mh_rounds(rundir):
    """deferral 이 발생한 라운드들의 집계. 없으면 None(= 프로토타입 arm)."""
    logs = glob.glob(os.path.join(rundir, "server-logs", "*gpu_scheduler*.log"))
    if not logs:
        return None
    agg = {
        "rounds_with_deferral": 0,
        "selected_total": 0,
        "requeued_total": 0,
        "rounds_with_requeue": 0,
        "late_dispatched_total": 0,
        "eligible_total": 0,
        "max_queue_len": 0,
    }
    for path in logs:
        with open(path, errors="ignore") as fh:
            for line in fh:
                m = ROUND_RX.search(line)
                if m:
                    _, elig, sel, req, late, qlen = (int(x) for x in m.groups())
                    agg["rounds_with_deferral"] += 1
                    agg["eligible_total"] += elig
                    agg["selected_total"] += sel
                    agg["requeued_total"] += req
                    agg["rounds_with_requeue"] += 1 if req else 0
                    agg["late_dispatched_total"] += late
                    agg["max_queue_len"] = max(agg["max_queue_len"], qlen)
                    continue
                q = QLEN_RX.search(line)
                if q:
                    agg["max_queue_len"] = max(agg["max_queue_len"], int(q.group(1)))
    if not agg["rounds_with_deferral"]:
        return None
    r = agg["rounds_with_deferral"]
    agg["selected_per_round"] = round(agg["selected_total"] / r, 3)
    agg["eligible_per_round"] = round(agg["eligible_total"] / r, 3)
    return agg


def proto_max_queue(rundir):
    """프로토타입 경로의 queue_len 최댓값(비교 기준선)."""
    logs = glob.glob(os.path.join(rundir, "server-logs", "*gpu_scheduler*.log"))
    best = None
    for path in logs:
        with open(path, errors="ignore") as fh:
            for line in fh:
                q = QLEN_RX.search(line)
                if q:
                    v = int(q.group(1))
                    best = v if best is None else max(best, v)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    a = ap.parse_args()

    out = {}
    for rundir in sorted(glob.glob(os.path.join(a.base, "raw", "*", "rate_*", "seed_*"))):
        if not os.path.exists(os.path.join(rundir, "DONE")):
            continue
        parts = rundir.split(os.sep)
        system, rate, seed = parts[-3], parts[-2].split("_")[1], parts[-1].split("_")[1]
        key = f"{system}|{rate}|{seed}"
        entry = {}
        d = ttft_deciles(rundir)
        if d:
            entry["ttft"] = d
        mh = mh_rounds(rundir)
        if mh:
            entry["moore_hodgson"] = mh
        else:
            q = proto_max_queue(rundir)
            if q is not None:
                entry["max_queue_len"] = q
        if entry:
            out[key] = entry

    path = os.path.join(a.base, "processed", "diagnostics.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(out, open(path, "w"), indent=2, ensure_ascii=False)
    print(f"wrote {path} ({len(out)} runs)")


if __name__ == "__main__":
    main()
