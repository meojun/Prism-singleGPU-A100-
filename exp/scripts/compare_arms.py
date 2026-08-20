#!/usr/bin/env python3
"""Aggregate the prototype-vs-final comparison and answer its five questions.

Three arms, and the middle one exists to make the other two trustworthy:

    A  released-prototype, TP patch ABSENT   (/workspace/prism-base)
    B  released-prototype, TP patch present but off  (/workspace/prism-merge)
    C  paper-faithful-v6, everything on      (/workspace/prism-merge)

A vs B is the regression check.  If adding TP support changed the TP=1 path,
every A-vs-C number is contaminated and has to be read as "prototype vs a
different codebase" rather than "prototype vs the paper's mechanisms".  So the
gate runs first and can stop the sweep.

Two rules are fixed here before any number is seen:

* An improvement is only claimed when it clears the seed noise.  The v6 control
  sweep measured sd/mean = 17% across seeds, and this project has already been
  burned once by quoting a mean ratio whose seeds straddled it.  So every
  comparison carries n, sd, and a verdict that says "within noise" when the
  difference is smaller than the pooled spread.
* Anything with fewer than 2 successful seeds is reported as such and never
  aggregated into a headline.

Sign conventions, stated because they differ between metric families:
    goodput / attainment   (Final - Prototype) / Prototype * 100   higher better
    latency                (Prototype - Final) / Prototype * 100   higher better
"""

import argparse
import csv
import json
import math
import statistics as stats
from collections import defaultdict
from pathlib import Path

ARM_LABEL = {
    "A": "released-prototype (TP patch absent)",
    "B": "released-prototype (TP patch present, off)",
    "C": "paper-faithful-v6 (all mechanisms on)",
}


def read_summary(arm_dir: Path):
    """collect_v4_metrics.py writes summary.csv per results directory."""
    p = arm_dir / "summary.csv"
    if not p.exists():
        return []
    with p.open() as fh:
        return list(csv.DictReader(fh))


def fnum(row, *names):
    for n in names:
        v = row.get(n)
        if v not in (None, "", "None"):
            try:
                return float(v)
            except ValueError:
                pass
    return None


def key_of(row):
    wl = (row.get("workload") or row.get("trace") or "").strip()
    rate = fnum(row, "rate", "request_rate", "offered_rate")
    seed = fnum(row, "seed")
    return wl, (int(rate) if rate is not None else None), (int(seed) if seed is not None else None)


METRICS = [
    # (column candidates, label, higher_is_better)
    (("goodput",), "goodput", True),
    (("joint_slo_attainment", "joint_attainment"), "joint_slo", True),
    (("ttft_slo_attainment",), "ttft_slo", True),
    (("tpot_slo_attainment",), "tpot_slo", True),
    (("throughput", "achieved_throughput"), "throughput", True),
    (("ttft_p50",), "ttft_p50", False),
    (("ttft_p95",), "ttft_p95", False),
    (("ttft_p99",), "ttft_p99", False),
    (("tpot_p50",), "tpot_p50", False),
    (("tpot_p95",), "tpot_p95", False),
    (("tpot_p99",), "tpot_p99", False),
    (("e2e_p50",), "e2e_p50", False),
    (("e2e_p95",), "e2e_p95", False),
    (("e2e_p99",), "e2e_p99", False),
    (("migrations", "migration_count"), "migrations", True),
    (("kv_bytes",), "kv_bytes", True),
    (("weight_bytes",), "weight_bytes", True),
    (("failed", "failed_requests"), "failed", False),
]


def gather(out: Path):
    """arm -> (workload, rate) -> metric -> [per-seed values]"""
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    seeds = defaultdict(lambda: defaultdict(set))
    for arm in "ABC":
        for row in read_summary(out / "raw" / f"arm{arm}"):
            wl, rate, seed = key_of(row)
            if rate is None:
                continue
            seeds[arm][(wl, rate)].add(seed)
            for names, label, _ in METRICS:
                v = fnum(row, *names)
                if v is not None:
                    data[arm][(wl, rate)][label].append(v)
    return data, seeds


def agg(vals):
    vals = [v for v in vals if v is not None and not math.isnan(v)]
    if not vals:
        return None, None, 0
    return (stats.mean(vals),
            (stats.stdev(vals) if len(vals) > 1 else 0.0),
            len(vals))


def improvement(proto, final, higher_is_better):
    """Percent change, signed so that positive always means 'final is better'."""
    if proto in (None, 0) or final is None:
        return None
    if higher_is_better:
        return (final - proto) / abs(proto) * 100.0
    return (proto - final) / abs(proto) * 100.0


def verdict(pm, ps, pn, fm, fs, fn_):
    """Is the difference bigger than the seed noise?

    Deliberately conservative: pooled sd, and a difference must exceed it to be
    called anything.  The project's own control sweep sits at sd/mean = 17%, so
    a 10% mean gap across 3 seeds is not a result.
    """
    if pn < 2 or fn_ < 2:
        return f"insufficient seeds (n={pn},{fn_})"
    pooled = math.sqrt((ps ** 2 + fs ** 2) / 2.0)
    diff = abs((fm or 0) - (pm or 0))
    if pooled == 0:
        return "differs" if diff else "identical"
    return "within noise" if diff < pooled else "exceeds seed spread"


def write_gate(out: Path, data, seeds):
    """A vs B on the matched case: did adding TP support move the TP=1 path?"""
    rows = []
    stop = False
    for (wl, rate) in sorted(set(data["A"]) & set(data["B"])):
        for names, label, hib in METRICS:
            am, asd, an = agg(data["A"][(wl, rate)].get(label, []))
            bm, bsd, bn = agg(data["B"][(wl, rate)].get(label, []))
            if am is None or bm is None:
                continue
            delta = improvement(am, bm, hib)
            v = verdict(am, asd, an, bm, bsd, bn)
            # With one seed there is no spread to compare against, so fall back
            # to a flat tolerance and say which rule was used.
            if an < 2 or bn < 2:
                v = ("within 10%" if am and abs(bm - am) / abs(am) <= 0.10
                     else "differs >10%")
            rows.append({"workload": wl, "rate": rate, "metric": label,
                         "A_mean": am, "A_n": an, "B_mean": bm, "B_n": bn,
                         "delta_pct": None if delta is None else round(delta, 2),
                         "verdict": v})
            if label in ("goodput", "joint_slo", "throughput") and "differs" in v:
                stop = True
    (out / "aggregated").mkdir(parents=True, exist_ok=True)
    p = out / "aggregated" / "regression_check.csv"
    with p.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["workload", "rate", "metric", "A_mean",
                                           "A_n", "B_mean", "B_n", "delta_pct",
                                           "verdict"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
        fh.write(f"# GATE: {'STOP' if stop else 'PASS'}\n")
    print(f"regression gate -> {'STOP' if stop else 'PASS'}  ({p})")
    for r in rows:
        if r["metric"] in ("goodput", "joint_slo", "throughput", "failed"):
            print(f"  {r['workload']} r{r['rate']} {r['metric']:12s} "
                  f"A={r['A_mean']:.4g} B={r['B_mean']:.4g} "
                  f"delta={r['delta_pct']}%  {r['verdict']}")
    return not stop


def write_summaries(out: Path, data, seeds):
    agg_dir = out / "aggregated"
    agg_dir.mkdir(parents=True, exist_ok=True)

    with (agg_dir / "summary.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "arm_label", "workload", "rate", "metric",
                    "mean", "sd", "n", "seeds"])
        for arm in "ABC":
            for (wl, rate) in sorted(data[arm]):
                for _, label, _ in METRICS:
                    m, s, n = agg(data[arm][(wl, rate)].get(label, []))
                    if m is None:
                        continue
                    w.writerow([arm, ARM_LABEL[arm], wl, rate, label,
                                round(m, 6), round(s, 6), n,
                                "|".join(str(x) for x in sorted(seeds[arm][(wl, rate)]))])

    with (agg_dir / "improvement.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["comparison", "workload", "rate", "metric",
                    "baseline_mean", "baseline_sd", "baseline_n",
                    "final_mean", "final_sd", "final_n",
                    "improvement_pct", "verdict", "sign_rule"])
        for base_arm, fin_arm, name in (("A", "C", "prototype_vs_final"),
                                        ("B", "C", "tpoff_vs_final"),
                                        ("A", "B", "prototype_vs_tpoff")):
            for (wl, rate) in sorted(set(data[base_arm]) & set(data[fin_arm])):
                for _, label, hib in METRICS:
                    pm, ps, pn = agg(data[base_arm][(wl, rate)].get(label, []))
                    fm, fs, fn_ = agg(data[fin_arm][(wl, rate)].get(label, []))
                    if pm is None or fm is None:
                        continue
                    w.writerow([name, wl, rate, label,
                                round(pm, 6), round(ps, 6), pn,
                                round(fm, 6), round(fs, 6), fn_,
                                None if improvement(pm, fm, hib) is None
                                else round(improvement(pm, fm, hib), 2),
                                verdict(pm, ps, pn, fm, fs, fn_),
                                "higher-better" if hib else "lower-better"])

    # latency and migration get their own files, as the spec asks
    for fname, wanted in (("latency_summary.csv",
                           [l for _, l, _ in METRICS if l.startswith(("ttft", "tpot", "e2e"))]),
                          ("migration_summary.csv",
                           ["migrations", "kv_bytes", "weight_bytes"])):
        with (agg_dir / fname).open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["arm", "workload", "rate", "metric", "mean", "sd", "n"])
            for arm in "ABC":
                for (wl, rate) in sorted(data[arm]):
                    for label in wanted:
                        m, s, n = agg(data[arm][(wl, rate)].get(label, []))
                        if m is not None:
                            w.writerow([arm, wl, rate, label, round(m, 6), round(s, 6), n])
    print(f"wrote {agg_dir}/summary.csv, improvement.csv, latency_summary.csv, "
          f"migration_summary.csv")


def write_report(out: Path, data, seeds):
    agg_dir = out / "aggregated"
    L, A = [], None
    L.append("# Released Prototype vs Final Paper-Faithful Prism\n")
    L.append("생성: `exp/scripts/compare_arms.py`. 수치는 전부 `raw/*/summary.csv` 에서 "
             "파생된 것이고 그 반대가 아니다. 판정 규칙은 숫자를 보기 전에 코드에 "
             "고정돼 있다.\n")
    L.append("\n| arm | 무엇 |\n| --- | --- |")
    for a in "ABC":
        L.append(f"| {a} | {ARM_LABEL[a]} |")
    L.append("")
    L.append("증가가 좋은 지표는 `(Final-Proto)/Proto`, 지연은 `(Proto-Final)/Proto` 로 "
             "계산해 **양수면 언제나 Final 이 낫다**는 뜻이 되게 했다.")
    L.append("")
    L.append("판정은 보수적이다. 이 프로젝트의 대조군 스윕이 seed 분산 sd/mean=17% 를 "
             "기록했고, 과거에 seed 가 평균을 사이에 두고 갈라지는데도 평균비를 인용해 "
             "한 번 틀렸다. 그래서 차이가 pooled sd 를 넘지 못하면 **within noise** 로 "
             "적는다.")

    # Q1 -- regression
    L.append("\n## Q1. Prototype 과 Final-TP-OFF 가 E2E 에서 같게 동작하는가\n")
    gate = agg_dir / "regression_check.csv"
    if gate.exists():
        txt = gate.read_text()
        L.append(f"`aggregated/regression_check.csv` 참조. 게이트 판정: "
                 f"**{'STOP' if 'GATE: STOP' in txt else 'PASS'}**")
        L.append("")
        L.append("게이트가 STOP 이면 A-vs-C 수치는 '논문 메커니즘의 효과' 가 아니라 "
                 "'다른 코드베이스와의 비교' 로 읽어야 한다.")
    else:
        L.append("**게이트를 돌리지 않았다.**")

    # Q2/Q3 -- headline
    L.append("\n## Q2, Q3. 부하 구간별 차이, 그리고 bursty 에서 더 강한가\n")
    imp = agg_dir / "improvement.csv"
    if imp.exists():
        rows = list(csv.DictReader(imp.open()))
        head = [r for r in rows
                if r["comparison"] == "prototype_vs_final"
                and r["metric"] in ("goodput", "joint_slo", "ttft_p99", "e2e_p99")]
        if head:
            L.append("| workload | rate | metric | Proto | Final | 개선(%) | 판정 |")
            L.append("| --- | ---: | --- | ---: | ---: | ---: | --- |")
            for r in sorted(head, key=lambda x: (x["workload"], int(x["rate"]), x["metric"])):
                L.append(f"| {r['workload']} | {r['rate']} | {r['metric']} | "
                         f"{float(r['baseline_mean']):.4g} ± {float(r['baseline_sd']):.2g} | "
                         f"{float(r['final_mean']):.4g} ± {float(r['final_sd']):.2g} | "
                         f"{r['improvement_pct']} | {r['verdict']} |")
        else:
            L.append("**해당 수치 없음.**")
    else:
        L.append("**집계 파일이 없다.**")
    L.append("")
    L.append("20 req/s 는 일반 운영점이 아니라 **Extreme Load / Stress** 구간이다. "
             "이 구성의 포화점은 5~10 req/s 로 측정돼 있다.")

    # Q4 -- migration
    L.append("\n## Q4. Final 의 migration/KV 메커니즘이 실제로 얼마나 쓰이는가\n")
    any_mig = False
    for (wl, rate) in sorted(data.get("C", {})):
        m, s, n = agg(data["C"][(wl, rate)].get("migrations", []))
        kv, kvs, kvn = agg(data["C"][(wl, rate)].get("kv_bytes", []))
        if m:
            any_mig = True
            L.append(f"* {wl} r{rate}: 마이그레이션 {m:.1f} ± {s:.1f} (n={n}), "
                     f"KV {(kv or 0)/2**30:.2f} GiB")
    if not any_mig:
        L.append("**Final arm 에서 마이그레이션이 한 번도 일어나지 않았다.** "
                 "0 을 '메커니즘이 동작한다' 로 읽지 말 것 — 이 워크로드와 이 박스의 "
                 "tau 에서 Algorithm 1 이 이동을 낼 조건이 오지 않았다는 사실이다.")

    # Q5 -- capability
    L.append("\n## Q5. Prototype 이 못 하는 TP>1 / 대형 모델을 Final 은 하는가\n")
    L.append("| | Released Prototype | Final Paper-Faithful |")
    L.append("| --- | --- | --- |")
    L.append("| worker-pool 하 TP>1 | **Unsupported** | Supported |")
    L.append("| TP=2 / TP=4 | Unsupported | 검증됨 |")
    L.append("| anti-affinity | 미구현 | 논문 §A.2.2 구현 |")
    L.append("| Llama-3.1-70B | Unsupported | 서빙 확인 (TP=4, TP=2) |")
    L.append("")
    L.append("프로토타입은 worker-pool 경로에서 TP>1 을 아예 못 돌린다. 성능 arm 을 "
             "억지로 만들지 않고 **Unsupported** 로 적는다. 근거는 "
             "`exp/results/paper-faithful-v4/tp-validation/FINDING.md` 와 이 브랜치의 "
             "`paper-faithful-tp/REPORT.md`.")

    (out / "REPORT.md").write_text("\n".join(L) + "\n")
    print(f"wrote {out / 'REPORT.md'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--gate", action="store_true",
                    help="only run the A-vs-B regression gate and write its verdict")
    ns = ap.parse_args()
    out = Path(ns.out).resolve()
    data, seeds = gather(out)
    if ns.gate:
        write_gate(out, data, seeds)
        return
    write_gate(out, data, seeds)
    write_summaries(out, data, seeds)
    write_report(out, data, seeds)


if __name__ == "__main__":
    main()
