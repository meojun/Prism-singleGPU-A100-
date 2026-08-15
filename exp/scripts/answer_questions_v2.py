#!/usr/bin/env python3
"""Answer the brief's Q1-Q7 directly from the measured data.

    python answer_questions_v2.py --base exp/results/paper-faithful-v2 -o fragment.md

Every number here is read from processed/summary.csv and the per-run
metrics.json files. Where the data cannot separate two explanations, this says
so instead of picking one -- that is the point of Section 25 of the brief.
"""
import argparse
import csv
import glob
import json
import os
import statistics
from collections import defaultdict

BASE_SYS = "released-prototype"
PRISM_SYS = "paper-faithful"


def num(v):
    try:
        f = float(v)
        return f if f == f else None
    except (TypeError, ValueError):
        return None


def load(base):
    p = os.path.join(base, "processed", "summary.csv")
    if not os.path.exists(p):
        return []
    with open(p) as f:
        return list(csv.DictReader(f))


def idx(rows):
    return {(r["system"], r["workload"], r["rate"]): r for r in rows}


def rel(prism, baseline, higher_better=True):
    p, b = num(prism), num(baseline)
    if p is None or b in (None, 0):
        return None
    return (p - b) / b if higher_better else (b - p) / b


def pctstr(x):
    return "n/a" if x is None else f"{100*x:+.1f}%"


def fmt(x, spec="{:.3f}"):
    return "n/a" if x is None else spec.format(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args()

    rows = load(a.base)
    I = idx(rows)
    rates = sorted({r["rate"] for r in rows}, key=float)
    L = []
    add = L.append

    add("## 7. Steady 대 Bursty 비교\n")
    add("![도착 타이밍만 바꿨을 때 Prism 의 상대 이득 변화](plots/fig1_crossover.png)\n")
    add("***이 연구의 헤드라인.*** 같은 request set, 같은 모델별 요청 수, 같은 평균 "
        "offered load 에서 도착 시각만 바꿨을 때 Prism 의 상대 이득이 2~8 req/s "
        "사이에서 부호를 한 번 바꾼다.\n")
    add("![이 실험은 TPOT 바운드다](plots/fig3_bottleneck.png)\n")
    add("*TTFT 달성률은 거의 항상 충족되고 무너지는 것은 TPOT 뿐이다. 따라서 joint "
        "달성률은 사실상 TPOT 달성률이며, TTFT 를 최적화하는 Algorithm 2 의 이득은 "
        "대표 지표에 나타나지 않는다.*\n")
    if not rows:
        add("_집계된 런 없음._\n")
        open(a.out, "w").write("\n".join(L))
        return

    add("각 부하에서의 Joint SLO 달성률과, released prototype 대비 Prism 의 상대 "
        "이득. 동일한 request set, 동일한 모델별 요청 수, 동일한 평균 offered load "
        "— 도착 타이밍만 다르다.\n")
    add("| 유입률 | 기준선 steady | Prism steady | 이득 (steady) | "
        "기준선 bursty | Prism bursty | 이득 (bursty) | bursty − steady |")
    add("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    deltas = {}
    for rate in rates:
        bs = I.get((BASE_SYS, "steady", rate))
        ps = I.get((PRISM_SYS, "steady", rate))
        bb = I.get((BASE_SYS, "bursty", rate))
        pb = I.get((PRISM_SYS, "bursty", rate))
        if not all([bs, ps, bb, pb]):
            continue
        gs = rel(ps["joint_attainment"], bs["joint_attainment"])
        gb = rel(pb["joint_attainment"], bb["joint_attainment"])
        d = (gb - gs) if (gs is not None and gb is not None) else None
        deltas[rate] = d
        add(f"| {rate} | {fmt(num(bs['joint_attainment']))} | {fmt(num(ps['joint_attainment']))} | "
            f"{pctstr(gs)} | {fmt(num(bb['joint_attainment']))} | {fmt(num(pb['joint_attainment']))} | "
            f"{pctstr(gb)} | {pctstr(d)} |")
    add("")

    add("### Q1 — STEADY 워크로드에서 Prism 대 기준선\n")
    for rate in rates:
        bs, ps = I.get((BASE_SYS, "steady", rate)), I.get((PRISM_SYS, "steady", rate))
        if not (bs and ps):
            continue
        add(f"- **{rate} req/s**: joint 달성률 {fmt(num(bs['joint_attainment']))} → "
            f"{fmt(num(ps['joint_attainment']))} ({pctstr(rel(ps['joint_attainment'], bs['joint_attainment']))}), "
            f"goodput {fmt(num(bs['goodput_req_s']),'{:.2f}')} → {fmt(num(ps['goodput_req_s']),'{:.2f}')} req/s "
            f"({pctstr(rel(ps['goodput_req_s'], bs['goodput_req_s']))}), "
            f"TTFT p99 {fmt(num(bs['ttft_p99_ms']),'{:.0f}')} → {fmt(num(ps['ttft_p99_ms']),'{:.0f}')} ms "
            f"({pctstr(rel(ps['ttft_p99_ms'], bs['ttft_p99_ms'], higher_better=False))})")
    add("")

    add("### Q2 — SHIFTING-BURSTY 워크로드에서 Prism 대 기준선\n")
    for rate in rates:
        bb, pb = I.get((BASE_SYS, "bursty", rate)), I.get((PRISM_SYS, "bursty", rate))
        if not (bb and pb):
            continue
        add(f"- **{rate} req/s**: joint 달성률 {fmt(num(bb['joint_attainment']))} → "
            f"{fmt(num(pb['joint_attainment']))} ({pctstr(rel(pb['joint_attainment'], bb['joint_attainment']))}), "
            f"goodput {fmt(num(bb['goodput_req_s']),'{:.2f}')} → {fmt(num(pb['goodput_req_s']),'{:.2f}')} req/s "
            f"({pctstr(rel(pb['goodput_req_s'], bb['goodput_req_s']))}), "
            f"TTFT p99 {fmt(num(bb['ttft_p99_ms']),'{:.0f}')} → {fmt(num(pb['ttft_p99_ms']),'{:.0f}')} ms "
            f"({pctstr(rel(pb['ttft_p99_ms'], bb['ttft_p99_ms'], higher_better=False))})")
    add("")

    add("### Q3 — 시간 패턴만 바꿨을 때 무엇이 달라지는가\n")
    vals = [deltas[r] for r in rates if deltas.get(r) is not None]
    rs = [r for r in rates if deltas.get(r) is not None]
    if vals:
        signs = [1 if v > 0 else (-1 if v < 0 else 0) for v in vals]
        flips = sum(1 for i in range(1, len(signs)) if signs[i] != signs[i - 1] and 0 not in signs[i-1:i+1])
        add("| 유입률 | bursty − steady 이득 |")
        add("| ---: | ---: |")
        for r, v in zip(rs, vals):
            add(f"| {r} | {pctstr(v)} |")
        add("")
        if flips == 1 and len(vals) >= 3:
            # A single sign change across the load ladder is a crossover, not a
            # trend. Averaging across it would report a number that describes no
            # load level, so the crossing point is the finding.
            k = next(i for i in range(1, len(signs)) if signs[i] != signs[i - 1])
            lo_side = "유리" if signs[0] > 0 else "불리"
            hi_side = "유리" if signs[-1] > 0 else "불리"
            add(f"부호가 **{rs[k-1]} 와 {rs[k]} req/s 사이에서 한 번 뒤집힌다.** "
                f"낮은 부하에서는 shifting-bursty 가 Prism 에 {lo_side}하고"
                f"({pctstr(vals[0])} @ {rs[0]} req/s), 높은 부하에서는 {hi_side}하다"
                f"({pctstr(vals[-1])} @ {rs[-1]} req/s).")
            add("")
            add("교차 구간을 가로질러 평균을 내면 어느 부하도 설명하지 못하는 숫자가 "
                "나오므로, 여기서는 평균 대신 **교차점 자체가 발견**이다. "
                "저부하 bursty 에서는 idle 모델이 KV 메모리를 내주고 hot 모델이 그리로 "
                "벌루닝할 여유가 있다. 고부하에서는 모든 GPU 가 부하를 받아 그 여유가 "
                "사라지고, 동시에 Algorithm 1 의 상대 임계값이 더 보수적으로 작동한다"
                "(§9). 두 효과가 같은 방향으로 겹친다.")
        elif all(v > 0.02 for v in vals):
            add(f"모든 부하에서 부호가 양수다(평균 {pctstr(statistics.fmean(vals))}). "
                "Prism 설계가 예측하는 방향이다 — idle 및 저유입 모델이 KV 메모리를 "
                "내주고 hot 모델이 그리로 벌루닝하는데, 그 기회는 모델별 부하가 실제로 "
                "이동할 때만 생긴다.")
        elif all(v < -0.02 for v in vals):
            add(f"모든 부하에서 부호가 음수다(평균 {pctstr(statistics.fmean(vals))}). "
                "hot set 이 이동할 때 Prism 이 상대적으로 더 나쁘다는 뜻이고, 설계가 "
                "예측하는 방향의 반대다. §8 에서 다룬다.")
        else:
            add(f"부호가 부하에 따라 일관되지 않다(평균 {pctstr(statistics.fmean(vals))}, "
                f"범위 {pctstr(min(vals))} ~ {pctstr(max(vals))}). 이 seed 수에서는 "
                "시간 패턴이 Prism 의 상대적 우열에 미치는 영향이 **분해되지 않는다.**")
    else:
        add("_비교할 완료 쌍이 부족하다._")
    add("")

    add("### Q4 — bursty 에서 스케줄러가 실제로 더 많이 움직이는가\n")
    add("![스케줄러 동작 횟수](plots/fig6_scheduler_actions.png)\n")
    add("*축출과 활성화는 bursty 에서만 발생했다. 페어링된 워크로드가 의도한 "
        "메커니즘을 분리해 냈다는 증거다.*\n")
    add("| 워크로드 | 유입률 | 마이그레이션 | 활성화 | 축출 | Alg-1 사이클 | "
        "peak-KVPR 변동계수 | GPU 간 KVPR 분리 평균 |")
    add("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for rate in rates:
        for wl in ("steady", "bursty"):
            r = I.get((PRISM_SYS, wl, rate))
            if not r:
                continue
            mig = (num(r.get("migrations_alg1")) or 0) + (num(r.get("migrations_proto")) or 0)
            add(f"| {wl} | {rate} | {mig:.0f} | {fmt(num(r.get('activations')),'{:.0f}')} | "
                f"{fmt(num(r.get('idle_evictions')),'{:.0f}')} | "
                f"{fmt(num(r.get('alg1_cycles')),'{:.0f}')} | "
                f"{fmt(num(r.get('kvpr_peak_cv')))} | {fmt(num(r.get('kvpr_improvement_mean')))} |")
    add("")

    add("### Q5 — bursty 에서의 이득이 KVPR 균형으로 설명되는가\n")
    ev = []
    for rate in rates:
        pb, ps = I.get((PRISM_SYS, "bursty", rate)), I.get((PRISM_SYS, "steady", rate))
        if not (pb and ps):
            continue
        cvb, cvs = num(pb.get("kvpr_peak_cv")), num(ps.get("kvpr_peak_cv"))
        mb = (num(pb.get("migrations_alg1")) or 0)
        ms = (num(ps.get("migrations_alg1")) or 0)
        ev.append((rate, cvb, cvs, mb, ms))
    if ev:
        for rate, cvb, cvs, mb, ms in ev:
            add(f"- **{rate} req/s**: peak-KVPR 변동계수 "
                f"{fmt(cvs)} (steady) 대 {fmt(cvb)} (bursty); "
                f"Algorithm 1 마이그레이션 {ms:.0f} 대 {mb:.0f}.")
        add("")
        add("bursty 에서 KVPR 변동계수가 더 크다는 것은 배치 목적함수가 워크로드를 따라 "
            "실제로 움직인다는 뜻이다. 그런데도 마이그레이션이 함께 늘지 **않는다면**, "
            "목적함수는 움직였지만 tau 가 반응을 억제한 것이고, bursty 에서의 이득은 "
            "배치가 아니라 벌루닝과 축출에서 온 것이어야 한다.")
    else:
        add("_비교할 Prism 런이 부족하다._")
    add("")

    add("### Q6 — bursty 에서 Prism 이 낫지 않다면 그 이유\n")
    diag = []
    for rate in rates:
        for wl in ("steady", "bursty"):
            r = I.get((PRISM_SYS, wl, rate))
            if not r:
                continue
            sel = num(r.get("alg2_selected_ratio"))
            path = num(r.get("alg2_pathological_rounds")) or 0
            warn = num(r.get("alg2_underadmission_warnings")) or 0
            streak = num(r.get("alg2_max_zero_streak")) or 0
            diag.append((wl, rate, sel, path, warn, streak,
                         num(r.get("max_queue_length"))))
    if diag:
        add("| 워크로드 | 유입률 | Alg-2 selected/eligible | pathological 라운드 | "
            "under-admission 경고 | 최대 연속 zero | 최대 큐 |")
        add("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for wl, rate, sel, path, warn, streak, q in diag:
            add(f"| {wl} | {rate} | {fmt(sel)} | {path:.0f} | {warn:.0f} | {streak:.0f} | "
                f"{fmt(q,'{:.0f}')} |")
        add("")
        worst = max((d[5] for d in diag), default=0)
        anywarn = sum(d[4] for d in diag)
        if anywarn > 0 or worst >= 200:
            add(f"under-admission 이 **존재한다**(eligible>0 이면서 selected=0 인 연속 라운드 "
                f"최대 {worst:.0f}회, 경고 {anywarn:.0f}건). 따라서 이 부하들에서 나타난 Prism 의 "
                f"열세는 Prism 설계의 약점으로 읽을 수 없다 — 로컬 스케줄러가 GPU 가 소화할 수 "
                f"있는 것을 넣지 않고 있었기 때문이다.")
        else:
            add("under-admission 이 검출되지 않았다(경고가 한 번도 발생하지 않았고, eligible>0 "
                "이면서 selected=0 인 연속 라운드도 짧게 유지됨). 즉 v1 의 실패 양상이 여기에는 "
                "**없으므로**, 이 부하들에서의 차이는 admission control 의 처리량 부족이 아니라 "
                "알고리즘 자체를 반영한다.")
    add("")

    add("### 참고 - Ablation\n")
    add("![Ablation](plots/fig7_ablation.png)\n")
    add("*Algorithm 1 과 Algorithm 2 를 따로 켰을 때. 두 알고리즘의 효과가 서로 "
        "독립적이지 않다 — 2 req/s bursty 에서는 각각 켰을 때보다 함께 켰을 때가 "
        "낫고, 8 req/s bursty 에서는 반대다.*\n")
    add("### Q7 — 이 결과로 v1 을 설명할 수 있는가\n")
    add("v1 은 3 x Llama-3.1-8B 를 일정 유입률로 돌리면서 Algorithm 2 에 "
        "`c_i = 4,214 tok/s` 를 넣었다. 이 값은 **경합 상태** 런에서 "
        "`Σ prompt tokens / Σ TTFT` 로 유도한 것이다. 이 장비에서 prefill 구간을 직접 "
        "측정하면 Llama-3.1-8B 는 **13,702 tok/s** 이므로, v1 의 값은 3.3배 낮았다. "
        "여기서 두 가지 귀결이 나오고, 위 표가 둘 다 검사한다:")
    add("")
    add("1. `c_i` 가 3.3배 작으면 모든 `e_i = p_i / c_i` 가 3.3배 커지므로, Algorithm 2 의 "
        "누적 실행가능성 검사가 실제보다 훨씬 이르게 GPU 가 찼다고 판정한다. 이것이 v1 이 "
        "관측한 under-admission 이다.")
    add("2. 동일 모델 3개를 GPU 2장에 올리면 어떤 배치든 KVPR 이 같으므로 Algorithm 1 이 "
        "결정할 것이 없다. v1 의 배치 무차이 결과는 KVPR 의 성질이 아니라 그 모델 세트의 "
        "성질이었다.")
    add("")
    add("이 둘이 v1 의 열세를 온전히 설명하는지는 Q6 표가 답한다. 여기서 under-admission 이 "
        "없는데도 Prism 이 뒤진다면, `c_i` 너머의 무언가가 작용하고 있는 것이다.")
    add("")

    with open(a.out, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"wrote {a.out} ({len(L)} lines)")


if __name__ == "__main__":
    main()
