#!/usr/bin/env python3
"""V5 report: which algorithm costs the per-request tax, and does P2P hold up."""
import argparse, csv, math, statistics as st
from collections import defaultdict
from pathlib import Path


def read(p):
    try:
        with open(p) as fh: return list(csv.DictReader(fh))
    except FileNotFoundError: return []


def num(x, d=float("nan")):
    try:
        v = float(x); return d if math.isnan(v) else v
    except (TypeError, ValueError): return d


def cell(v, d=2):
    v = [x for x in v if not math.isnan(x)]
    if not v: return "—"
    return f"{st.fmean(v):.{d}f}" + (f" ± {st.stdev(v):.{d}f}" if len(v) > 1 else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v5", required=True); ap.add_argument("--v4", required=True)
    a = ap.parse_args()
    rows = read(Path(a.v5) / "summary.csv") + read(Path(a.v4) / "summary.csv")
    g = defaultdict(list)
    for r in rows:
        g[(r["workload"], int(r["request_rate"]), r["implementation"])].append(r)

    print("# Paper-Faithful Prism V5 — ablation and P2P re-validation\n")
    print("두 가지를 묻는다. 중부하에서 paper-faithful arm 이 잃는 것이 Algorithm 1 때문인지")
    print("Algorithm 2 때문인지, 그리고 고쳐 둔 P2P 경로가 부하에서 버티는지.\n")
    print("v4 스윕의 released-prototype / v3 / v4 행은 비교 기준으로 함께 싣는다.\n")

    order = ["released-prototype", "paper-faithful-v3-alg1only",
             "paper-faithful-v3-alg2only", "paper-faithful-v3", "paper-faithful-v4"]
    label = {"paper-faithful-v3-alg1only": "V3, Algorithm 1 만",
             "paper-faithful-v3-alg2only": "V3, Algorithm 2 만"}
    for wl, rate in sorted({(k[0], k[1]) for k in g}):
        present = [a_ for a_ in order if (wl, rate, a_) in g]
        if len(present) < 2: continue
        print(f"\n## {wl} {rate} req/s\n")
        print("| Arm | n | Goodput | Joint SLO | TTFT p50 (ms) | TPOT p50 (ms) | 처리율 | 마이그레이션 |")
        print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for arm in present:
            v = g[(wl, rate, arm)]
            print("| {} | {} | {} | {} | {} | {} | {} | {} |".format(
                label.get(arm, arm), len(v),
                cell([num(x["goodput"]) for x in v]),
                cell([num(x["joint_slo"]) for x in v], 3),
                cell([num(x["ttft_p50"]) for x in v], 1),
                cell([num(x["tpot_p50"]) for x in v], 1),
                cell([num(x["achieved_throughput"]) for x in v], 2),
                cell([num(x["migration_count"]) for x in v], 1)))

    print("\n## P2P 재활성화\n")
    print("| Workload | Rate | seed | 전송 | P2P | page-locked | 전송 대역폭 GB/s | 실패 요청 |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in sorted(read(Path(a.v5) / "summary.csv"),
                    key=lambda x: (x["workload"], x["seed"])):
        if r["implementation"] != "paper-faithful-v4": continue
        print(f"| {r['workload']} | {r['request_rate']} | {r['seed']} | "
              f"{num(r['weight_transfers']):.0f} | {num(r['p2p_weight_transfers']):.0f} | "
              f"{num(r['page_locked_transfers']):.0f} | {num(r['weight_transfer_mean_gbps']):.2f} | "
              f"{num(r['failed_requests']):.0f} |")
    print("\nP2P 열이 0 보다 크고 실패 요청이 0 이면, `ipc_collect()` 수정 후 GPU-to-GPU")
    print("경로가 부하에서 버틴다는 뜻이다. 실패가 있으면 수정이 불충분한 것이다.\n")


if __name__ == "__main__":
    main()
