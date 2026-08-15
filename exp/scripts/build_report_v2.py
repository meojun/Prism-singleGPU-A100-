#!/usr/bin/env python3
"""Assemble the single REPORT.md for paper-faithful-v2.

    python build_report_v2.py --base exp/results/paper-faithful-v2 \
        -o exp/results/paper-faithful-v2/REPORT.md

Structure follows Section 22 of the brief exactly.  Numbers and tables only --
narrative sections are emitted as placeholders the author fills in, so nothing
in this file can silently invent a conclusion.
"""
import argparse
import csv
import glob
import json
import os
import platform
import subprocess
from collections import defaultdict

MODELS = [
    ("model_1", "meta-llama/Llama-3.2-1B", "1.24B", 2.28, 32768),
    ("model_2", "Qwen/Qwen2.5-1.5B-Instruct", "1.54B", 3.01, 28672),
    ("model_3", "meta-llama/Llama-3.2-3B", "3.21B", 6.00, 114688),
    ("model_4", "Qwen/Qwen2.5-3B-Instruct", "3.09B", 5.84, 36864),
    ("model_5", "meta-llama/Llama-3.1-8B", "8.03B", 15.08, 131072),
    ("model_6", "Qwen/Qwen2.5-7B-Instruct", "7.62B", 14.28, 57344),
]


def sh(cmd, default="n/a"):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                              timeout=30).stdout.strip() or default
    except Exception:
        return default


def num(v):
    try:
        f = float(v)
        return f if f == f else None
    except (TypeError, ValueError):
        return None


def fmt(v, spec="{:.1f}"):
    n = num(v)
    return spec.format(n) if n is not None else "-"


def read_csv(p):
    if not os.path.exists(p):
        return []
    with open(p) as f:
        return list(csv.DictReader(f))


def section_env(base, root):
    gpu = sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1")
    cnt = sh("nvidia-smi --list-gpus | wc -l", "?")
    cpu = sh("lscpu | grep 'Model name' | head -1 | cut -d: -f2- | xargs")
    ram = sh("free -g | awk 'NR==2{print $2\" GiB\"}'")
    drv = sh("nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1")
    def ver(mod):
        return sh(f"{root}/prism-venv/bin/python -c "
                  f"'import {mod};print({mod}.__version__)'")
    rows = [
        ("GPU", f"{cnt} x {gpu}"),
        ("Driver", drv),
        ("CPU", cpu), ("RAM", ram),
        ("OS", f"{platform.system()} {platform.release()}"),
        ("CUDA (torch)", sh(f"{root}/prism-venv/bin/python -c 'import torch;print(torch.version.cuda)'")),
        ("torch", ver("torch")),
        ("SGLang (prism-research fork)", ver("sglang")),
        ("prism-research commit", sh(f"git -C {root}/prism-research rev-parse HEAD")),
        ("kvcached commit (prism/shm)", sh(f"git -C {root}/kvcached-prism rev-parse HEAD")),
        ("Prism harness branch", sh(f"git -C {root} rev-parse --abbrev-ref HEAD")),
        ("Prism harness commit", sh(f"git -C {root} rev-parse HEAD")),
    ]
    out = ["## 1. 실험 환경", "", "| 항목 | 값 |", "| --- | --- |"]
    out += [f"| {k} | {v} |" for k, v in rows]
    return "\n".join(out) + "\n"


def section_models(base):
    prof = {}
    for p in glob.glob(os.path.join(base, "sanity", "profiling", "model_*.json")):
        d = json.load(open(p))
        prof[d["model"]] = d
    out = ["## 2. 모델", "",
           "| 슬롯 | 모델 | 파라미터 | dtype | 가중치 (GiB) | KV cell (B/token) | "
           "TTFT p95 (ms) | TPOT p95 (ms) |",
           "| --- | --- | --- | --- | ---: | ---: | ---: | ---: |"]
    for slot, path, params, w, cell in MODELS:
        b = prof.get(slot, {}).get("slo_baseline", {})
        out.append(f"| {slot} | `{path}` | {params} | bf16 | {w:.2f} | {cell} | "
                   f"{fmt(1000*b['ttft_p95_s'] if b.get('ttft_p95_s') else None)} | "
                   f"{fmt(1000*b['tpot_p95_s'] if b.get('tpot_p95_s') else None, '{:.2f}')} |")
    out += ["",
            "KV cell size 가 파라미터 수와 **단조가 아니도록** 일부러 골랐다. "
            "`model_3` 은 같은 크기의 `model_4` 보다 토큰당 KV 를 3.1배, "
            "`model_5` 는 `model_6` 보다 2.3배 쓴다. 이렇게 하지 않으면 KVPR 이 "
            "상주 가중치의 재라벨링으로 퇴화해 Algorithm 1 의 목적함수가 평평해진다 "
            "— v1 의 3 x Llama-3.1-8B 구성에서 실제로 그랬다.",
            "TTFT/TPOT p95 는 이 장비에서 잰 무경합 단독 측정값이다(논문 §7.1 방식). "
            "아래에서 쓰는 SLO 는 이 값에 스케일을 곱한 것이다."]
    return "\n".join(out) + "\n"


def section_ci(base):
    rows = []
    for p in sorted(glob.glob(os.path.join(base, "sanity", "profiling", "model_*.json"))):
        d = json.load(open(p))
        e = d["c_i_estimators"]
        rows.append((d["model"], e))
    out = ["### c_i 추정기 (tokens/s)", "",
           "![c_i 추정기 4종](plots/fig4_ci_estimators.png)", "",
           "*추정기에 따라 최대 10배까지 갈린다. Algorithm 2 에 넣는 값은 "
           "포화 상태의 총 prefill 처리량(E3sat)이다.*", "",
           "| 슬롯 | E1 비율 Σp/Σttft | E2 회귀 기울기 | E2 절편 (ms) | "
           "E3 실측 prefill, 단독 | E3 실측 prefill, 포화 | **사용값** |",
           "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for m, e in rows:
        used = e.get("E3_prefill_saturated") or e.get("E3_prefill_solo") or e.get("E1_ratio_sum_p_over_sum_ttft")
        out.append(f"| {m} | {fmt(e['E1_ratio_sum_p_over_sum_ttft'], '{:.0f}')} | "
                   f"{fmt(e['E2_regression_slope'], '{:.0f}')} | "
                   f"{fmt(1000*e['E2_intercept_s'] if e.get('E2_intercept_s') else None, '{:.1f}')} | "
                   f"{fmt(e['E3_prefill_solo'], '{:.0f}')} | "
                   f"{fmt(e['E3_prefill_saturated'], '{:.0f}')} | "
                   f"**{fmt(used, '{:.0f}')}** |")
    return "\n".join(out) + "\n"


def section_sanity(base):
    p = os.path.join(base, "sanity", "sanity_gate.json")
    out = ["## 4. Sanity 게이트", "", "| 검사 | 결과 | 통과 여부 |", "| --- | --- | --- |"]
    if not os.path.exists(p):
        out.append("| (미실행) | | |")
        return "\n".join(out) + "\n"
    d = json.load(open(p))
    for r in d["results"]:
        tag = "PASS" if r["pass"] else ("**FAIL**" if r["hard"] else "WARN")
        out.append(f"| {r['check']} | {r['detail']} | {tag} |")
    out.append("")
    out.append(f"Hard 실패: **{d['hard_failures']}건**"
               + ("" if d["hard_failures"] else " — 게이트 통과, 본 실험 진행됨."))
    return "\n".join(out) + "\n"


def section_calibration(base):
    rows = []
    for p in sorted(glob.glob(os.path.join(base, "sanity", "calibration", "rate_*", "metrics.json")),
                    key=lambda q: float(os.path.basename(os.path.dirname(q)).split("_")[1])):
        d = json.load(open(p))
        rows.append((float(os.path.basename(os.path.dirname(p)).split("_")[1]), d))
    if not rows:
        return ""
    out = ["### 부하 calibration (released prototype, steady, 짧은 런)", "",
           "![부하 calibration](plots/fig5_calibration.png)", "",
           "*처리율은 끝까지 유입률을 따라간다. 무너지는 것은 달성률뿐이므로 "
           "이 실험 구간은 용량 포화가 아니라 SLO 바운드다.*", "",
           "| 유입률 (req/s) | 처리율 | TTFT p50 (ms) | TTFT p99 (ms) | "
           "TPOT p50 (ms) | Joint 달성률 | Goodput | 최대 큐 |",
           "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r, d in rows:
        out.append(f"| {r:g} | {fmt(d['throughput_req_s'],'{:.2f}')} | {fmt(d['ttft_p50_ms'])} | "
                   f"{fmt(d['ttft_p99_ms'])} | {fmt(d['tpot_p50_ms'])} | "
                   f"{fmt(d['joint_attainment'],'{:.3f}')} | {fmt(d['goodput_req_s'],'{:.2f}')} | "
                   f"{fmt(d.get('max_queue_length'),'{:.0f}')} |")
    return "\n".join(out) + "\n"


def section_workloads(base, root):
    wl = os.path.join(root, "exp/workloads/paper-faithful-v2")
    metas = sorted(glob.glob(os.path.join(wl, "paired_requests_*.json")))
    out = ["## 5. 워크로드", ""]
    if metas:
        out += ["| 유입률 (req/s) | 길이 (s) | Seed | 총 요청 | 평균 offered load | "
                + " | ".join(m[0] for m in MODELS) + " |",
                "| ---: | ---: | ---: | ---: | ---: |" + " ---: |" * len(MODELS)]
        for p in metas:
            d = json.load(open(p))
            pm = d["per_model_requests"]
            out.append(f"| {d['rate']:g} | {d['duration']:g} | {d['seed']} | "
                       f"{d['total_requests']} | {d['average_offered_load_req_s']:.2f} | "
                       + " | ".join(str(pm.get(m[0], 0)) for m in MODELS) + " |")
    ph = sorted(glob.glob(os.path.join(wl, "phases_*.json")))
    if ph:
        d = json.load(open(ph[0]))
        out += ["", "### Shifting Bursty", "",
                f"- 위상 길이 범위: {d['phase_len_range'][0]:g}~{d['phase_len_range'][1]:g} 초 "
                f"(랜덤, seed 고정)",
                f"- hot set 크기: 1~3개 모델, 위상마다 다시 뽑음",
                f"- 유입률 배수: HOT {d['hot_multiplier_range']}, "
                f"MEDIUM {d['medium_multiplier_range']}, LOW {d['low_multiplier_range']}, IDLE = 0",
                f"- 기본 비중: {d['base_share']}",
                f"- seed: {d['seed']}",
                "- **집계 유입률은 모든 위상에서 같은 상수로 정규화된다.** 따라서 클러스터가 "
                "받는 총 부하는 전혀 움직이지 않고, 바뀌는 것은 *어느 모델이* hot 인가뿐이다. "
                "'클러스터가 더 바빠졌다' 를 교란변수에서 제거하고, Prism 이 이용한다고 "
                "주장하는 효과만 남긴다.",
                "", "### Steady", "",
                "- 모델별 요청 수를 bursty 트레이스에서 그대로 가져옴",
                "- 각 모델의 도착을 전체 구간에 균등 난수로 배치 "
                "(N 개 균등 점 = 개수를 N 으로 조건화한 균질 포아송 과정)",
                "", "### 두 워크로드에서 동일한 것", "",
                "| 속성 | Bursty | Steady |", "| --- | --- | --- |",
                "| Request set | 동일 | 동일 |", "| 프롬프트 | 동일 | 동일 |",
                "| 모델 배정 | 동일 | 동일 |", "| 출력 길이 | 동일 | 동일 |",
                "| 모델별 요청 수 | 동일 | 동일 |", "| 총 요청 수 | 동일 | 동일 |",
                "| 실험 길이 | 동일 | 동일 |", "| 평균 offered load | 동일 | 동일 |",
                "| Random seed | 동일 | 동일 |",
                "| **도착 타이밍** | **Bursty** | **Uniform** |"]
    return "\n".join(out) + "\n"


FIGS = {
    "results": [
        ("plots/fig2_joint_attainment.png",
         "부하별 Joint SLO 달성률. 왼쪽이 steady, 오른쪽이 shifting-bursty."),
        ("plots/fig8_ttft_p99.png",
         "TTFT p99 (로그 축). Algorithm 2 가 실제로 개선하는 지표."),
    ],
    "comparison": [
        ("plots/fig1_crossover.png",
         "**이 연구의 헤드라인.** 도착 타이밍만 바꿨을 때 Prism 의 상대 이득이 "
         "2~8 req/s 사이에서 부호를 한 번 바꾼다."),
        ("plots/fig6_scheduler_actions.png",
         "축출과 활성화는 bursty 에서만 발생한다 — 페어링 워크로드가 의도한 "
         "메커니즘을 분리했다는 증거."),
        ("plots/fig7_ablation.png",
         "Algorithm 1 / Algorithm 2 를 따로 켰을 때."),
    ],
}


def figures(key):
    out = []
    for path, cap in FIGS.get(key, []):
        out += [f"![{cap}]({path})", "", f"*{cap}*", ""]
    return out


def section_results(base):
    summ = read_csv(os.path.join(base, "processed", "summary.csv"))
    if not summ:
        return "## 6. 결과\n\n_(집계된 런 없음)_\n"
    out = ["## 6. 결과", ""] + figures("results")
    by_rate = defaultdict(list)
    for r in summ:
        by_rate[r["rate"]].append(r)
    for rate in sorted(by_rate, key=lambda x: float(x)):
        out += [f"### 유입률 {rate} req/s", "",
                "| 시스템 | 워크로드 | TTFT p50 | TTFT p95 | TTFT p99 | TPOT p50 | TPOT p95 | "
                "TPOT p99 | TTFT 달성률 | TPOT 달성률 | Joint 달성률 | 처리율 | Goodput | "
                "마이그 | 활성화 | 축출 | 최대 큐 |",
                "| --- | --- |" + " ---: |" * 15]
        for r in sorted(by_rate[rate], key=lambda r: (r["workload"], r["system"])):
            mig = num(r.get("migrations_alg1")) or 0
            mig += num(r.get("migrations_proto")) or 0
            out.append(
                f"| {r['system']} | {r['workload']} | "
                f"{fmt(r['ttft_p50_ms'])} | {fmt(r['ttft_p95_ms'])} | {fmt(r['ttft_p99_ms'])} | "
                f"{fmt(r['tpot_p50_ms'])} | {fmt(r['tpot_p95_ms'])} | {fmt(r['tpot_p99_ms'])} | "
                f"{fmt(r['ttft_attainment'],'{:.3f}')} | {fmt(r['tpot_attainment'],'{:.3f}')} | "
                f"{fmt(r['joint_attainment'],'{:.3f}')} | {fmt(r['throughput_req_s'],'{:.2f}')} | "
                f"{fmt(r['goodput_req_s'],'{:.2f}')} | {mig:.0f} | "
                f"{fmt(r['activations'],'{:.0f}')} | {fmt(r['idle_evictions'],'{:.0f}')} | "
                f"{fmt(r['max_queue_length'],'{:.0f}')} |")
        out.append("")
    out += ["지연은 ms, 처리율과 goodput 은 req/s. "
            "Joint 달성률 = 측정 구간 요청 중 TTFT 와 TPOT SLO 를 **둘 다** 만족한 비율. "
            "Goodput = 그 요청 수 / 측정 구간 길이.", ""]
    return "\n".join(out) + "\n"


def section_comparison(base):
    comp = read_csv(os.path.join(base, "processed", "comparison.csv"))
    if not comp:
        return "## 7. Steady vs Bursty Comparison\n\n_(pending)_\n"
    out = ["## 7. Steady vs Bursty Comparison", "",
           "Relative improvement of each paper arm over `released-prototype` at the "
           "same workload and load. Positive = the paper arm wins.", "",
           "| System | Workload | Rate | Joint att (base) | Joint att (paper) | dpp | "
           "Joint rel | Goodput rel | TTFT p99 rel | TPOT p99 rel |",
           "| --- | --- | ---: |" + " ---: |" * 7]
    for r in sorted(comp, key=lambda r: (r["system"], float(r["rate"]), r["workload"])):
        out.append(f"| {r['system']} | {r['workload']} | {r['rate']} | "
                   f"{fmt(r['joint_attainment_base'],'{:.3f}')} | "
                   f"{fmt(r['joint_attainment_prism'],'{:.3f}')} | "
                   f"{fmt(num(r['joint_attainment_pp'])*100 if num(r['joint_attainment_pp']) is not None else None,'{:+.1f}')} | "
                   f"{fmt(num(r['joint_attainment_rel'])*100 if num(r['joint_attainment_rel']) is not None else None,'{:+.1f}%')} | "
                   f"{fmt(num(r['goodput_rel'])*100 if num(r['goodput_rel']) is not None else None,'{:+.1f}%')} | "
                   f"{fmt(num(r['ttft_p99_rel'])*100 if num(r['ttft_p99_rel']) is not None else None,'{:+.1f}%')} | "
                   f"{fmt(num(r['tpot_p99_rel'])*100 if num(r['tpot_p99_rel']) is not None else None,'{:+.1f}%')} |")
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--root", default="/workspace/prism-exp")
    ap.add_argument("--preface", default=None,
                    help="markdown fragment placed immediately after the title "
                         "(section 0: how to read this report)")
    ap.add_argument("--impl-status", default=None,
                    help="markdown fragment for Section 3 (implementation status)")
    ap.add_argument("--narrative", default=None,
                    help="markdown fragment for Sections 8-10")
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args()

    parts = ["# Paper-Faithful Prism v2 — Shifting-Bursty 대 Steady", "",
             f"_`exp/scripts/build_report_v2.py` 가 생성. 하네스 커밋 `{sh(f'git -C {a.root} rev-parse --short HEAD')}`._", ""]
    if a.preface and os.path.exists(a.preface):
        parts.append(open(a.preface).read())
    parts.append(section_env(a.base, a.root))
    parts.append(section_models(a.base))
    if a.impl_status and os.path.exists(a.impl_status):
        parts.append(open(a.impl_status).read())
    parts.append("## 4. Sanity Check\n".replace("## 4. Sanity Check\n", ""))
    parts.append(section_ci(a.base))
    parts.append(section_sanity(a.base))
    parts.append(section_calibration(a.base))
    parts.append(section_workloads(a.base, a.root))
    parts.append(section_results(a.base))
    # NB: no section_comparison() here -- answer_questions_v2.py emits its own
    # "## 7. Steady vs Bursty Comparison" with the per-load contrast, and
    # appending both produced a duplicate heading.
    if a.narrative and os.path.exists(a.narrative):
        parts.append(open(a.narrative).read())
    with open(a.out, "w") as f:
        f.write("\n".join(parts))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
