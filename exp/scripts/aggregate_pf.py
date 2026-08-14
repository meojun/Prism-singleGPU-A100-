#!/usr/bin/env python3
"""Paper-Faithful 스윕 집계: results.csv, summary.csv, 그림 8장, REPORT.md 를 만든다."""
import argparse
import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np

COLUMNS = [
    "system", "request_rate", "seed",
    "ttft_p50_ms", "ttft_p99_ms", "tpot_p50_ms", "tpot_p99_ms",
    "ttft_slo_attainment", "tpot_slo_attainment", "joint_slo_attainment",
    "throughput_req_s", "joint_slo_goodput_req_s",
    "num_completed", "num_failed",
    "num_migrations", "num_evictions", "num_activations", "num_mh_deferred",
    "ttft_p95_ms", "tpot_p95_ms", "requests_in_window",
    "alg1_log_lines", "alg2_log_lines",
]

METRICS = [
    ("ttft_p50_ms", "TTFT p50 (ms)", "lower"),
    ("ttft_p99_ms", "TTFT p99 (ms)", "lower"),
    ("tpot_p50_ms", "TPOT p50 (ms)", "lower"),
    ("tpot_p99_ms", "TPOT p99 (ms)", "lower"),
    ("ttft_slo_attainment", "TTFT SLO attainment", "higher"),
    ("tpot_slo_attainment", "TPOT SLO attainment", "higher"),
    ("joint_slo_attainment", "Joint SLO attainment", "higher"),
    ("joint_slo_goodput_req_s", "Joint-SLO goodput (req/s)", "higher"),
]

PROTO, PAPER = "released-prototype", "paper-faithful"


def load_rows(base):
    rows = []
    for path in sorted(glob.glob(os.path.join(base, "raw", "*", "rate_*", "seed_*", "metrics.json"))):
        with open(path) as fh:
            rows.append(json.load(fh))
    return rows


def write_results(base, rows):
    out = os.path.join(base, "processed", "results.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    rows = sorted(rows, key=lambda r: (r["system"], r["request_rate"], r["seed"]))
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return out


def summarize(base, rows):
    """mean/std per (system, rate). std is the sample std over seeds."""
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["system"], r["request_rate"])].append(r)

    metric_keys = [c for c in COLUMNS if c not in ("system", "request_rate", "seed")]
    out = os.path.join(base, "processed", "summary.csv")
    fields = ["system", "request_rate", "n_runs"]
    for k in metric_keys:
        fields += [f"{k}_mean", f"{k}_std"]

    summary = {}
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for (sys_, rate) in sorted(grouped, key=lambda x: (x[0], x[1])):
            group = grouped[(sys_, rate)]
            row = {"system": sys_, "request_rate": rate, "n_runs": len(group)}
            for k in metric_keys:
                vals = [g[k] for g in group if isinstance(g.get(k), (int, float))
                        and not (isinstance(g[k], float) and np.isnan(g[k]))]
                row[f"{k}_mean"] = float(np.mean(vals)) if vals else float("nan")
                row[f"{k}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            w.writerow(row)
            summary[(sys_, rate)] = row
    return out, summary


def figures(base, summary):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"matplotlib unavailable ({e}); skipping figures")
        return []

    figdir = os.path.join(base, "figures")
    os.makedirs(figdir, exist_ok=True)
    written = []
    systems = sorted({s for s, _ in summary})

    for i, (key, label, _) in enumerate(METRICS, start=1):
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        drew = False
        for sys_ in systems:
            pts = sorted((r for (s, r) in summary if s == sys_))
            xs, ys, es = [], [], []
            for rate in pts:
                row = summary[(sys_, rate)]
                mean = row.get(f"{key}_mean")
                if mean is None or (isinstance(mean, float) and np.isnan(mean)):
                    continue
                xs.append(rate); ys.append(mean); es.append(row.get(f"{key}_std", 0.0))
            if xs:
                # Error bars are the seed-to-seed std (n=3); they are a spread
                # indicator, not a confidence interval.
                ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3, label=sys_)
                drew = True
        if not drew:
            plt.close(fig); continue
        ax.set_xlabel("Offered aggregate request rate (req/s)")
        ax.set_ylabel(label)
        ax.set_title(f"Figure {i}: Request rate vs {label}")
        ax.grid(alpha=0.3)
        ax.legend()
        if "attainment" in key:
            ax.set_ylim(-0.02, 1.02)
        fig.tight_layout()
        path = os.path.join(figdir, f"fig{i}_{key}.png")
        fig.savefig(path, dpi=140)
        plt.close(fig)
        written.append(path)
    return written


def improvements(summary):
    """Paper vs prototype at each rate. Latency: lower is better."""
    rates = sorted({r for (s, r) in summary if s == PROTO} & {r for (s, r) in summary if s == PAPER})
    out = []
    for rate in rates:
        p, q = summary[(PROTO, rate)], summary[(PAPER, rate)]
        entry = {"rate": rate}
        for key, label, direction in METRICS:
            a, b = p.get(f"{key}_mean"), q.get(f"{key}_mean")
            if a is None or b is None or np.isnan(a) or np.isnan(b) or a == 0:
                entry[key] = None
                continue
            rel = ((a - b) / a * 100) if direction == "lower" else ((b - a) / a * 100)
            entry[key] = {"proto": a, "paper": b, "rel_pct": rel, "abs_pp": (b - a) * 100
                          if "attainment" in key else None}
        out.append(entry)
    return out


def load_diagnostics(base):
    """diagnose_pf.py 가 만든 진단 수치. 없으면 해당 절을 생략한다."""
    path = os.path.join(base, "processed", "diagnostics.json")
    if not os.path.exists(path):
        return {}
    try:
        return json.load(open(path))
    except Exception:
        return {}


def write_report(base, rows, summary, imps, figs):
    meta = ""
    mpath = os.path.join(base, "metadata", "run_metadata.txt")
    if os.path.exists(mpath):
        meta = open(mpath).read()

    diag = load_diagnostics(base)
    rates = sorted({r["request_rate"] for r in rows})
    top_rate = max(rates) if rates else None

    lines = []
    A = lines.append
    A("# Released Prism Prototype vs Paper-Faithful Prism\n")
    A("> `exp/scripts/aggregate_pf.py` 가 자동 생성한다. 수치는 seed 평균이며 괄호 없이 "
      "표기된 값은 평균, `summary.csv` 에 seed 간 표준편차가 함께 들어 있다. n=3 이므로 "
      "표준편차는 산포의 지표일 뿐 유의성 주장이 아니다.\n")

    # ------------------------------------------------------------------ 1
    A("## 1. 연구 질문\n")
    A("요청 유입률이 올라갈 때, 논문의 **Algorithm 1**(KVPR 기반 전역 모델 배치)과 "
      "**Algorithm 2**(Moore-Hodgson GPU-로컬 요청 스케줄링)를 논문에 충실하게 구현하면 "
      "공개된 `prism-research` 프로토타입보다 TTFT · TPOT · SLO 달성률 · Joint-SLO "
      "Goodput 에서 더 나은 결과가 나오는가?\n")
    A("두 arm 은 **스케줄러 정책만** 다르다. GPU, 모델, 정밀도, kvcached, ShareGPT 요청, "
      "프롬프트, 출력 길이, 도착 시각, seed, SLO, 워밍업, 측정 구간은 전부 동일하다.\n")

    # ------------------------------------------------------------------ 2
    A("## 2. 프로토타입과 논문은 무엇이 다른가\n")
    A("줄 단위로 대조한 감사 결과는 `docs/paper_faithful/design_analysis.md` 에 있다. "
      "요약하면 프로토타입의 전역 배치는 여유 메모리 바이트당 **요청 개수**(평활값)를 "
      "균형 지표로 쓰고 15배 비율 임계로 마이그레이션을 판단하며, 로컬 스케줄러는 "
      "`deadline − exec` 우선순위 힙일 뿐 `net_available = float('inf')` 로 사실상 "
      "**전부 admit** 한다. KVPR 지표도, Moore-Hodgson 의 핵심인 "
      "**가장 긴 작업을 떨어뜨리는 단계**도 공개 코드에는 존재하지 않는다.\n")

    # ------------------------------------------------------------------ 3-6
    A("## 3–6. 하드웨어 · 소프트웨어 · 모델 · 워크로드\n")
    A("```\n" + meta + "```\n")
    A("워크로드: ShareGPT 텍스트, gamma(cv=1)=Poisson 도착. (rate, seed) 조합마다 트레이스 "
      "하나를 만들어 **두 시스템이 같은 파일을 공유**하므로 프롬프트 · 길이 · 라우팅 · "
      "도착 시각이 arm 사이에 완전히 동일하다.\n")

    # ------------------------------------------------------------------ 11-14
    A("## 11–14. 결과\n")
    A("| Rate | System | TTFT p50 | TTFT p99 | TPOT p50 | TPOT p99 | "
      "TTFT 달성률 | TPOT 달성률 | Joint 달성률 | Goodput |")
    A("| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for rate in rates:
        for sys_ in (PROTO, PAPER):
            row = summary.get((sys_, rate))
            if not row:
                continue

            def g(k, mul=1.0, fmt="{:.1f}"):
                v = row.get(f"{k}_mean")
                return "n/a" if v is None or np.isnan(v) else fmt.format(v * mul)

            A(f"| {rate:g} | {sys_} | {g('ttft_p50_ms')} | {g('ttft_p99_ms')} | "
              f"{g('tpot_p50_ms')} | {g('tpot_p99_ms')} | "
              f"{g('ttft_slo_attainment', 1, '{:.3f}')} | "
              f"{g('tpot_slo_attainment', 1, '{:.3f}')} | "
              f"{g('joint_slo_attainment', 1, '{:.3f}')} | "
              f"{g('joint_slo_goodput_req_s', 1, '{:.2f}')} |")
    A("")
    A("지연 단위는 ms, Goodput 단위는 req/s.\n")

    # ------------------------------------------------------------------ 17-18
    A("## 17–18. 유입률별 Paper-Faithful 대 프로토타입\n")
    A("지연 개선율 = (프로토타입 − 논문충실) / 프로토타입, Goodput · 달성률 개선율 = "
      "(논문충실 − 프로토타입) / 프로토타입. 어느 지표든 **양수면 Paper-Faithful 이 우세**하다.\n")
    A("| Rate | TTFT p99 | TPOT p99 | Joint 달성률 (pp) | Joint 달성률 (상대) | Goodput |")
    A("| ---: | ---: | ---: | ---: | ---: | ---: |")
    for e in imps:
        def f(k, field="rel_pct"):
            v = e.get(k)
            return "n/a" if not v or v.get(field) is None else f"{v[field]:+.1f}%"

        ja = e.get("joint_slo_attainment")
        pp = "n/a" if not ja or ja.get("abs_pp") is None else f"{ja['abs_pp']:+.1f}pp"
        A(f"| {e['rate']:g} | {f('ttft_p99_ms')} | {f('tpot_p99_ms')} | {pp} | "
          f"{f('joint_slo_attainment')} | {f('joint_slo_goodput_req_s')} |")
    A("")

    # ------------------------------------------------------------------ 15-16
    A("## 15–16. 진단 지표\n")
    A("| Rate | System | 마이그레이션 | 축출 | 활성화 | MH 지연 | 완료 | 실패 |")
    A("| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for rate in rates:
        for sys_ in (PROTO, PAPER):
            row = summary.get((sys_, rate))
            if not row:
                continue

            def g(k):
                v = row.get(f"{k}_mean")
                return "n/a" if v is None or np.isnan(v) else f"{v:.1f}"

            A(f"| {rate:g} | {sys_} | {g('num_migrations')} | {g('num_evictions')} | "
              f"{g('num_activations')} | {g('num_mh_deferred')} | {g('num_completed')} | "
              f"{g('num_failed')} |")
    A("")
    A("`MH 지연` 은 Moore-Hodgson 이 한 번이라도 실행가능 집합에서 제외한 **서로 다른** "
      "요청 수다(라운드마다 중복 계수하지 않는다).\n")

    # ------------------------------------------------------------------ 스케줄러 증명
    A("## 두 arm 이 실제로 다른 스케줄러로 돌았다는 증거\n")
    A("플래그가 파싱되었다는 것만으로는 부족하므로, 런마다 서버 로그에서 알고리즘 표식을 "
      "세어 게이트로 검증했다.\n")
    A("| System | `[PAPER-ALG1]` 로그 | `[PAPER-ALG2]` 로그 |")
    A("| --- | ---: | ---: |")
    for sys_ in (PROTO, PAPER):
        a1 = sum(int(r.get("alg1_log_lines") or 0) for r in rows if r["system"] == sys_)
        a2 = sum(int(r.get("alg2_log_lines") or 0) for r in rows if r["system"] == sys_)
        n = sum(1 for r in rows if r["system"] == sys_)
        A(f"| {sys_} ({n}런 합계) | {a1} | {a2} |")
    A("")

    # ------------------------------------------------------------------ 원인 분석
    A("## 왜 Paper-Faithful 이 프로토타입보다 나쁘게 나왔는가\n")
    A("결과는 유입률에 따라 갈린다. 중·저부하에서는 Paper-Faithful 이 근소하게 열세이고, "
      "최고 부하에서는 Goodput 과 TPOT 꼬리가 크게 좋아지는 대신 TTFT 가 무너진다. "
      "아래는 그 원인에 대한 **추정**이며, 각 항목에 어떤 관측이 근거이고 어디까지가 "
      "확증되지 않았는지 함께 적는다.\n")

    A("### 원인 1 (주원인, 증거 강함) — 순차 기계 모델이 배치 엔진에서 과소 수용을 만든다\n")
    A("Algorithm 2 는 `1‖ΣU_j`, 즉 **한 번에 한 작업만 처리하는 단일 기계** 문제다. "
      "실행가능성 판정이 `clock += e_i` 누적이므로, k 번째로 넣는 요청을 앞의 k−1 개가 "
      "끝나기를 기다려야 하는 것처럼 계산한다. 그런데 서빙 엔진은 **한 배치에 여러 요청을 "
      "동시에 prefill** 한다. 따라서 이 판정은 GPU 가 아직 포화되지 않았는데도 "
      "'실행 불가능' 이라고 답한다.\n")
    key = f"{PAPER}|{int(top_rate)}|1" if top_rate else None
    mh = (diag.get(key) or {}).get("moore_hodgson") if key else None
    if mh:
        A(f"유입률 {top_rate:g} req/s, seed 1 의 GPU 스케줄러 로그 실측:\n")
        A("| 항목 | 값 |")
        A("| --- | ---: |")
        A(f"| deferral 이 발생한 라운드 | {mh['rounds_with_deferral']:,} |")
        A(f"| 그 라운드들의 eligible 총합 | {mh['eligible_total']:,} "
          f"(라운드당 {mh['eligible_per_round']}) |")
        A(f"| **그 라운드들의 selected 총합** | **{mh['selected_total']:,} "
          f"(라운드당 {mh['selected_per_round']})** |")
        A(f"| 백프레셔로 되돌린 요청 | {mh['requeued_total']:,} "
          f"({mh['rounds_with_requeue']:,} 개 라운드) |")
        A(f"| 이미 마감이 지나 뒤늦게 내보낸 요청 | {mh['late_dispatched_total']:,} |")
        A(f"| 최대 큐 길이 | {mh['max_queue_len']:,} |")
        pk = f"{PROTO}|{int(top_rate)}|1"
        pq = (diag.get(pk) or {}).get("max_queue_len")
        if pq is not None:
            A(f"| (참고) 프로토타입 최대 큐 길이 | {pq:,} |")
        A("")
        A("`eligible=123, selected=0` 같은 라운드가 일상적으로 나온다. 엔진은 그 123 개를 "
          "배치로 소화할 수 있지만 단일 기계 판정은 하나도 통과시키지 않는다.\n")

    A("이것이 지연의 재배치가 아니라 **처리율 부족**이라는 근거는 구간별 TTFT 추이다. "
      "성공한 요청을 도착 순서로 10 등분하고 각 구간의 TTFT 중앙값(초)을 보면:\n")
    if top_rate is not None:
        dk_p = f"{PROTO}|{int(top_rate)}|1"
        dk_f = f"{PAPER}|{int(top_rate)}|1"
        dp = (diag.get(dk_p) or {}).get("ttft")
        df = (diag.get(dk_f) or {}).get("ttft")
        if dp and df:
            A("| 구간 | " + " | ".join(str(i) for i in range(1, 11)) + " |")
            A("| --- | " + " | ".join(["---:"] * 10) + " |")
            A(f"| {PROTO} | " + " | ".join(f"{v:.2f}" for v in dp["p50_by_decile_s"]) + " |")
            A(f"| {PAPER} | " + " | ".join(f"{v:.2f}" for v in df["p50_by_decile_s"]) + " |")
            A("")
            A(f"(유입률 {top_rate:g} req/s, seed 1)\n")
    A("프로토타입은 중간에 튀었다가 **원래 수준으로 회복**한다. Paper-Faithful 은 4 구간부터 "
      "올라가서 끝까지 내려오지 않는다. 지속 처리율이 유입률보다 낮아 백로그가 영영 "
      "드레인되지 않는다는 뜻이고, 이는 '느린 요청을 버린 대가' 가 아니라 **과소 수용**의 "
      "서명이다.\n")

    A("### 원인 2 (해석 주의, 증거 강함) — 최고 부하의 유리한 수치도 같은 원인에서 나온다\n")
    A("Paper-Faithful 의 TPOT p99 개선과 Joint Goodput 증가는 실재하는 측정값이다. 다만 "
      "메커니즘은 '더 잘 스케줄해서' 가 아니라 **배치에 요청을 적게 넣어서** 디코딩 경합이 "
      "줄어든 것이다. 들어간 요청은 쾌적하게 서비스되고, 들어가지 못한 요청은 수십 초를 "
      "기다린다. Goodput 개선만 떼어 보고하면 스케줄링 품질 향상으로 오귀인하게 된다.\n")

    A("### 원인 3 (증거 있음) — 중·저부하에서는 Algorithm 2 가 이득 없이 비용만 남는다\n")
    A("부하가 낮으면 마감을 못 맞출 요청 자체가 거의 없다. 위 진단표의 `MH 지연` 값이 "
      "저부하에서 급감하는 것이 그 증거다. 이때 Moore-Hodgson 은 매 라운드 정렬과 힙 연산을 "
      "수행하지만 떨어뜨릴 것이 없으므로, 프로토타입 대비 순수한 오버헤드로 남는다. "
      "관측된 열세 폭(Goodput 5~6 %)은 seed 간 표준편차와 같은 크기라 이 데이터만으로는 "
      "**분해되지 않는다** — 방향이 일관될 뿐이다.\n")

    A("### 원인 4 (추정, 미확증) — 마감 산정에 큐 대기가 반영되지 않는다\n")
    A("`d_i = a_i + s_i` 는 도착 시각 기준의 절대 마감이고, `e_i = p_i / c_i` 는 **prefill "
      "시간만** 센다. 디코딩 시간도, 큐에서 이미 보낸 시간도 들어가지 않는다. 과부하로 "
      "백로그가 쌓이면 대다수 요청이 큐에 앉아 있는 동안 마감을 넘겨 버리고, 그 뒤로는 "
      "Moore-Hodgson 이 판단할 여지 없이 전부 '이미 늦음' 경로로 빠진다. 위 표에서 "
      "late_dispatched 가 selected 보다 훨씬 큰 것이 이 상태와 부합한다. 다만 이것이 "
      "**독립적인 원인인지, 원인 1 의 결과인지는 이 데이터로 구분되지 않는다.** 구분하려면 "
      "원인 1 을 제거한 조건에서 다시 측정해야 한다.\n")

    A("### 원인 5 (추정, 반증됨에 가까움) — 마이그레이션 비용\n")
    A("이 프로토타입의 마이그레이션은 stop-the-world 동작이므로, 횟수가 늘면 그 자체로 "
      "지연을 만든다. 그러나 실측 마이그레이션 횟수는 두 arm 이 거의 같다(아래 §Algorithm 1). "
      "따라서 관측된 차이의 설명으로는 **적합하지 않다.**\n")

    # ------------------------------------------------------------------ Alg1
    A("## Algorithm 1 은 왜 아무 차이도 만들지 못했는가\n")
    A("이 구성에서 Algorithm 1 은 **작동할 여지가 없다.** 모델 3 개를 GPU 2 개에 올리면 "
      "배치는 반드시 1+2 이고, 두 개가 올라간 GPU 의 KVPR 은\n")
    A("```\npeak KVPR ≈ (w + w) / (67.28 − 2×15.08 GiB) = 2w / 37.12\n```\n")
    A("로, **어느 모델을 겹치게 놓든 같다.** 목적함수가 평평하므로 argmin 은 추정 잡음이 "
      "정한다. 스모크 런의 24 회 결정에서 개선폭 분포는 평균 **+0.002**, 표준편차 "
      "**0.175** 였다. 기대 이득이 0 인 셈이고, `τ = 평균 + 2σ ≈ 0.35` 는 그 잡음 위에 "
      "선을 그은 값이다(지연 결과가 아니라 추정기 자체의 분포에서 유도했다).\n")
    A("결과적으로 두 arm 의 마이그레이션 횟수는 실질적으로 같고, 위에서 관측된 차이는 "
      "**사실상 전부 Algorithm 2 에서 나온다.** 여기서의 무차이는 KVPR 에 대한 반증이 "
      "아니라 **이 구성에서는 측정 불가**라는 뜻이다. 논문의 설정(모델 수·GPU 수가 많고 "
      "크기와 유입률이 이질적)에서는 목적함수가 평평하지 않다.\n")

    # ------------------------------------------------------------------ 미구현
    A("## 논문 내용 중 구현하지 않았거나 검증하지 못한 부분\n")
    A("의도적 선택, 환경 제약, 정보 부재를 구분해 전부 적는다. 자세한 근거는 "
      "`docs/paper_faithful/design_analysis.md` 에 있다.\n")
    A("| 논문 내용 | 이번 작업에서의 상태 | 이유 |")
    A("| --- | --- | --- |")
    A("| TP 샤드 anti-affinity 제약 | **구현했으나 한 번도 발동하지 않음** | 전 구성이 TP=1 "
      "이라 같은 모델의 다른 샤드가 존재하지 않는다. 코드 경로는 있으나 이번 실험이 "
      "검증하지 못했다 |")
    A("| Algorithm 2 의 배치 병렬성 보정 | **구현하지 않음 (의도적)** | 논문에 없는 항이다. "
      "`clock += e_i / B` 같은 보정을 넣으면 Algorithm 2 가 아닌 것을 측정하게 된다. "
      "가장 유력한 후속 실험이며 별도 arm 으로 라벨링해 돌려야 한다 |")
    A("| 논문의 다중 GPU · 8 모델 혼합 구성 | **재현하지 않음** | 정확한 구성이 공개되어 있지 "
      "않고, 2 GPU 에서는 이 유입률로 재현 자체가 불가능하다. 재현했다고 주장하지 않는다 |")
    A("| 이기종 모델 크기 · 이기종 유입률 | **다루지 않음** | 3 × Llama-3.1-8B 동일 모델이다. "
      "KVPR 목적함수가 평평해진 직접적 원인이기도 하다 |")
    A("| 마이그레이션 비용 모델 / 오버랩 마이그레이션 | **구현하지 않음** | 프로토타입의 "
      "마이그레이션은 stop-the-world 다. 논문이 비용을 어떻게 산정하는지 공개 정보로는 "
      "확정할 수 없어, 쿨다운으로만 빈도를 제한했다 |")
    A("| kvcached 메모리 벌루닝 | **재구현하지 않음 (요구사항)** | 핀 고정된 "
      "`ovg-project/kvcached` `prism/shm` 을 그대로 쓴다. 두 arm 에 동일하게 적용된다 |")
    A("| 모델 활성화 · 비활성화 · 유휴 축출 정책 | **프로토타입 것을 그대로 사용** | 논문이 "
      "임계값을 명시하지 않는다. `MODEL_IDLE_THRESHOLD = 50 s` 를 두 arm 에 동일 적용해 "
      "교란 변수가 되지 않게 했다 |")
    A("| 논문의 SLO 절대값 | **이 장비에서 재측정해 사용** | 논문 수치는 저자 하드웨어 "
      "기준이다. 무경합 p95 (TTFT 125.7 ms, TPOT 21.41 ms)를 §7.1 방식으로 다시 재고 "
      "×5 / ×3 스케일을 적용했다. 두 arm 이 같은 값을 쓴다 |")
    A("| `c_i` (모델별 chunked-prefill 속도) | 논문에 값이 없어 **측정해서 사용** | "
      "무경합 런에서 비율 추정기로 4,214 tok/s. 임의 상수를 쓰지 않았다 |")
    A("| Moore-Hodgson 이 제외한 요청의 처리 | 논문에 없어 **직접 정의** | 마감 전이면 "
      "재큐잉, 마감 후면 실행가능 집합 뒤에 배치. 전부 재큐잉하면 livelock 이 발생한다 |")
    A("| Joint-SLO Goodput | 논문에 없는 지표라 **직접 정의** | "
      "`SLO 를 모두 만족한 완료 요청 수 / 측정 300 s` |")
    A("")

    # ------------------------------------------------------------------ figures
    if figs:
        A("## 그림\n")
        for p in figs:
            A(f"![{os.path.basename(p)}](figures/{os.path.basename(p)})")
        A("")

    # ------------------------------------------------------------------ 19
    A("## 19. 한계\n")
    A("- 포인트당 seed 3 개다. 집계값에는 충분하지만 p99 에는 얇다. `summary.csv` 의 seed 간 "
      "표준편차가 두 arm 의 차이와 비슷한 크기인 구간은 **이 데이터로 분해되지 않는다.**\n"
      "- `τ`, 토큰율 측정 창, Moore-Hodgson 제외 요청의 처리 방식은 논문에 명시가 없다. "
      "사용한 값은 메타데이터와 `design_analysis.md` 에 기록했다. 어느 arm 에도 "
      "유입률별 튜닝을 적용하지 않았다.\n"
      "- GPU 2 개에 모델 3 개는 필연적으로 1+2 분할이므로 어떤 배치 정책도 부하를 "
      "균등화할 수 없다. Algorithm 1 이 여기서 낼 수 있는 성능의 상한을 이 사실이 정한다.\n"
      "- 프로토타입은 논문의 아티팩트가 아니라 단순화된 연구용 공개본이다. 여기서 측정된 "
      "차이는 **프로토타입 대 논문 알고리즘**이지 **저자 구현 대 논문**이 아니다.\n"
      "- `server-logs/` 와 `requests/` 는 용량 때문에 git 에 올리지 않는다. 원인 분석 절의 "
      "라운드 통계와 구간별 TTFT 를 다시 계산하려면 실험을 돌린 장비에서 "
      "`diagnose_pf.py` 를 실행해야 한다.\n")

    # ------------------------------------------------------------------ 20
    A("## 20. 재현\n")
    A("```bash\nsource exp/scripts/env.sh\n"
      "./exp/run_paper_faithful_comparison.sh --dry-run   # 실행 계획 출력\n"
      "./exp/run_paper_faithful_comparison.sh --resume    # 스윕 실행 / 중단 지점부터 재개\n"
      "python exp/tests/test_moore_hodgson.py             # Algorithm 2 단위 테스트\n"
      "python exp/tests/test_kvpr_placement.py            # Algorithm 1 단위 테스트\n"
      "python exp/scripts/diagnose_pf.py --base <결과 디렉터리>   # 진단 수치 재계산\n```\n")

    path = os.path.join(base, "REPORT.md")
    open(path, "w").write("\n".join(lines))
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    a = ap.parse_args()

    rows = load_rows(a.base)
    if not rows:
        raise SystemExit(f"no metrics.json found under {a.base}/raw")
    res = write_results(a.base, rows)
    summ, summary = summarize(a.base, rows)
    figs = figures(a.base, summary)
    imps = improvements(summary)
    rep = write_report(a.base, rows, summary, imps, figs)
    print(f"rows={len(rows)}\nresults={res}\nsummary={summ}\nfigures={len(figs)}\nreport={rep}")


if __name__ == "__main__":
    main()
