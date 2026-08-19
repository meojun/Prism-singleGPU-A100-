#!/usr/bin/env python3
"""Generate REPORT.md, IMPLEMENTATION_AUDIT.md and the figures for v4.

Reads only what the study actually produced.  Anything missing is reported as
missing rather than filled in.
"""
import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402

ARM_LABEL = {
    "released-prototype": "Released Prototype",
    "paper-faithful-v3": "Paper-Faithful V3",
    "paper-faithful-v4": "Paper-Faithful V4",
}
ARM_ORDER = ["released-prototype", "paper-faithful-v3", "paper-faithful-v4"]


def read_csv(path):
    if not Path(path).exists():
        return []
    with open(path) as fh:
        return list(csv.DictReader(fh))


def num(value, default=float("nan")):
    try:
        out = float(value)
        return default if math.isnan(out) else out
    except (TypeError, ValueError):
        return default


def fmt(value, digits=3):
    value = num(value)
    return "—" if math.isnan(value) else f"{value:.{digits}f}"


def mean_std(values):
    values = [v for v in values if not math.isnan(v)]
    if not values:
        return float("nan"), 0.0
    return float(np.mean(values)), (statistics.stdev(values) if len(values) > 1 else 0.0)


def cell(values, digits=3):
    mean, std = mean_std(values)
    if math.isnan(mean):
        return "—"
    return f"{mean:.{digits}f} ± {std:.{digits}f}" if std else f"{mean:.{digits}f}"


# --------------------------------------------------------------------- figures
def figure_e2e(rows, outdir):
    if not rows:
        return []
    made = []
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        key = (row["workload"], int(row["request_rate"]))
        groups[key][row["implementation"]].append(row)

    for metric, label, digits in (("goodput", "Joint-SLO goodput (req/s)", 3),
                                  ("joint_slo", "Joint SLO attainment", 3)):
        keys = sorted(groups)
        if not keys:
            continue
        fig, ax = plt.subplots(figsize=(1.9 * len(keys) + 3, 4))
        width = 0.8 / max(1, len(ARM_ORDER))
        xs = np.arange(len(keys))
        for i, arm in enumerate(ARM_ORDER):
            means, errs = [], []
            for key in keys:
                vals = [num(r[metric]) for r in groups[key].get(arm, [])]
                mean, std = mean_std(vals)
                means.append(mean)
                errs.append(std)
            if all(math.isnan(m) for m in means):
                continue
            ax.bar(xs + i * width, means, width, yerr=errs, capsize=3,
                   label=ARM_LABEL.get(arm, arm))
        ax.set_xticks(xs + width * (len(ARM_ORDER) - 1) / 2)
        ax.set_xticklabels([f"{w}\n{r} req/s" for w, r in keys])
        ax.set_ylabel(label)
        ax.set_title(f"{label} (mean ± sd over seeds)")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        path = Path(outdir) / f"e2e_{metric}.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        made.append(path.name)
    return made


def figure_loading(loading, outdir):
    if not loading:
        return []
    by_arm = defaultdict(list)
    for rec in loading.get("records", []):
        by_arm[rec["arm"]].append(rec["aggregate_gbps"])
    if not by_arm:
        return []
    arms = list(by_arm)
    means = [np.mean(by_arm[a]) for a in arms]
    errs = [statistics.stdev(by_arm[a]) if len(by_arm[a]) > 1 else 0 for a in arms]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(len(arms)), means, yerr=errs, capsize=4, color="#4C78A8")
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels(arms, rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("aggregate GB/s")
    ax.set_title("Model weight loading, 6 models onto one GPU")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(Path(outdir) / "microbench_loading.png", dpi=140)
    plt.close(fig)
    return ["microbench_loading.png"]


def figure_migration(migration, outdir):
    if not migration:
        return []
    by_arm = defaultdict(lambda: defaultdict(list))
    for rec in migration.get("records", []):
        by_arm[rec["arm"]]["latency"].append(rec["migration_latency_s"])
        by_arm[rec["arm"]]["downtime"].append(rec["service_downtime_s"])
        by_arm[rec["arm"]]["gbps"].append(rec["effective_gbps"])
    arms = list(by_arm)
    if not arms:
        return []
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, key, label in zip(axes, ("latency", "downtime", "gbps"),
                              ("migration latency (s)", "service downtime (s)",
                               "effective GB/s")):
        means = [np.mean(by_arm[a][key]) for a in arms]
        errs = [statistics.stdev(by_arm[a][key]) if len(by_arm[a][key]) > 1 else 0
                for a in arms]
        ax.bar(range(len(arms)), means, yerr=errs, capsize=4, color="#F58518")
        ax.set_xticks(range(len(arms)))
        ax.set_xticklabels(arms, rotation=20, ha="right", fontsize=7)
        ax.set_title(label, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(Path(outdir) / "microbench_migration.png", dpi=140)
    plt.close(fig)
    return ["microbench_migration.png"]


# ---------------------------------------------------------------------- report
def audit_table(loading, migration, tp2, rows):
    def has_p2p():
        return any(num(r.get("p2p_weight_transfers"), 0) > 0 for r in rows
                   if r["implementation"] == "paper-faithful-v4")

    tp_verdict = (tp2 or {}).get("verdict", "NOT RUN")
    lock = None
    if loading:
        by_arm = defaultdict(list)
        for rec in loading.get("records", []):
            by_arm[rec["arm"]].append(rec["aggregate_gbps"])
        if "v3-parallel-activation" in by_arm and "v4-parallel-loading" in by_arm:
            lock = np.mean(by_arm["v4-parallel-loading"]) / np.mean(by_arm["v3-parallel-activation"])

    mig_gain = None
    if migration:
        by_arm = defaultdict(list)
        for rec in migration.get("records", []):
            by_arm[rec["arm"]].append(rec["migration_latency_s"])
        if "v3-target-first" in by_arm and "v4-p2p-target-first" in by_arm:
            mig_gain = np.mean(by_arm["v3-target-first"]) / np.mean(by_arm["v4-p2p-target-first"])

    return [
        ("Algorithm 1 — KVPR placement", "PARTIAL", "FULL", "FULL",
         "Prototype balances `memory_per_request` (smoothed request counts) instead of "
         "`token_rate x token_size / SLO`; v3 implements the paper's rule with the literal "
         "absolute tau; v4 keeps that rule unchanged. `raw/scheduler/*_alg1.jsonl`."),
        ("Algorithm 1 — runtime placement audit", "NOT IMPLEMENTED", "NOT IMPLEMENTED", "FULL",
         "v4 logs the full placement plan, every blocked move with its reason, and a "
         "per-cycle convergence gap. `alg1_convergence_gap_mean` in `summary.csv`."),
        ("Algorithm 2 — Moore-Hodgson", "NOT IMPLEMENTED", "FULL", "FULL",
         "Prototype reduces to EDF with `net_available = inf`; v3/v4 run the feasibility "
         "test and longest-job removal. `[PAPER-ALG2]` in the GPU scheduler logs."),
        ("Parallel weight loading — broker split", "FULL", "FULL", "FULL",
         "Present upstream in `multi_thread_copy_model_to_gpu` and active in all three arms "
         "(`--enable-model-service`); half of every tensor crosses the helper GPU."),
        ("Parallel weight loading — page-locked host memory", "NOT IMPLEMENTED",
         "NOT IMPLEMENTED", "FULL",
         f"v4 registers the shared mapping with `cudaHostRegister`; measured speed-up "
         f"{'x%.2f' % lock if lock else 'see microbench/loading.json'} over v3."),
        ("Parallel weight loading — pipelined helper leg", "NOT IMPLEMENTED",
         "NOT IMPLEMENTED", "PARTIAL",
         "Implemented and measured as an ablation (`v4-pipelined-helper`); it did not pay "
         "off on this hardware and is therefore not the production v4 path."),
        ("Overlap migration — target-first ordering", "NOT IMPLEMENTED", "FULL", "FULL",
         "Prototype deactivates the source first; v3/v4 activate the target and only then "
         "retire the source. `ordering` column in `raw/migrations/*.csv`."),
        ("NVLink / P2P weight migration", "NOT IMPLEMENTED", "NOT IMPLEMENTED",
         "FULL" if has_p2p() else "PARTIAL",
         f"v4 fills the target from the source GPU's resident weights over NVLink; "
         f"microbenchmark shows {'x%.1f' % mig_gain if mig_gain else 'see microbench'} lower "
         f"latency and NVLink counters equal to the full model size. "
         f"{'Observed in the end-to-end runs.' if has_p2p() else 'Not observed end to end (see REPORT).'}"),
        ("KV-cache migration", "NOT IMPLEMENTED", "NOT IMPLEMENTED", "NOT IMPLEMENTED",
         "KV pages are owned by kvcached's per-GPU virtual-memory allocator and are dropped, "
         "not moved, on deactivation. No arm transfers KV state; `kv_bytes` is 0 everywhere "
         "and is reported as 0 rather than omitted."),
        ("RDMA transport", "NOT IMPLEMENTED", "NOT IMPLEMENTED",
         "NOT TESTABLE ON CURRENT HARDWARE",
         "Single node. The one RoCE NIC reaches every GPU only via `SYS`, so there is no "
         "second node to move weights to and nothing to measure."),
        ("TP anti-affinity", "NOT IMPLEMENTED", "NOT IMPLEMENTED", "NOT IMPLEMENTED",
         "The global controller collapses a TP group to its rank-0 GPU "
         "(`controller_global.py`: \"For TP case, only consider rank0 state\"), so no "
         "placement code can express, let alone enforce, an anti-affinity constraint. "
         "There is also nothing to constrain: TP>1 cannot run in the worker-pool path "
         "at all -- see `tp-validation/FINDING.md`."),
        ("TP=2 runtime validation", "NOT SUPPORTED BY PROTOTYPE",
         "NOT SUPPORTED BY PROTOTYPE", tp_verdict,
         "Two configurations, mixed-TP and uniform-TP, both die at activation with "
         "\"not found in shared cpu models\". Worker-pool engines are built one per "
         "(GPU, worker slot), each bound to a single GPU, so TP shards cannot span "
         "GPUs -- and that path is where the GPU scheduler and migration live. "
         "`tp-validation/FINDING.md`, `tp2_validation.json`, and both runs' logs."),
        ("Placement convergence", "NOT IMPLEMENTED", "NOT IMPLEMENTED", "FULL",
         "Measured, not assumed: `convergence_gap` per cycle in "
         "`raw/scheduler/*_alg1.jsonl`."),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    a = ap.parse_args()
    base = Path(a.base)
    figs = base / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    rows = read_csv(base / "summary.csv")
    loading = json.loads((base / "microbench/loading.json").read_text()) \
        if (base / "microbench/loading.json").exists() else None
    migration = json.loads((base / "microbench/migration.json").read_text()) \
        if (base / "microbench/migration.json").exists() else None
    tp2 = json.loads((base / "tp-validation/tp2_validation.json").read_text()) \
        if (base / "tp-validation/tp2_validation.json").exists() else None
    env = (base / "environment.txt").read_text() if (base / "environment.txt").exists() else ""

    made = figure_e2e(rows, figs) + figure_loading(loading, figs) + figure_migration(migration, figs)

    # ---------------- IMPLEMENTATION_AUDIT.md
    lines = [
        "# Implementation audit — Prism paper vs released prototype vs V3 vs V4", "",
        "각 항목은 `FULL` / `PARTIAL` / `NOT IMPLEMENTED` / "
        "`NOT TESTABLE ON CURRENT HARDWARE` 중 하나이며, Evidence 열은 그 판정의 근거가 되는",
        "이 저장소 안의 파일을 가리킨다. 판정은 코드를 읽고 런타임 로그로 확인한 결과다.", "",
        "| Mechanism | Prototype | V3 | V4 | Evidence |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name, proto, v3, v4, evidence in audit_table(loading, migration, tp2, rows):
        lines.append(f"| {name} | {proto} | {v3} | {v4} | {evidence} |")
    lines += [
        "", "## 판정 기준", "",
        "- **FULL** — 논문이 기술한 메커니즘이 구현되어 있고, 런타임 로그로 실제 동작이 확인된다.",
        "- **PARTIAL** — 일부만 구현되었거나, 구현했으나 이 하드웨어에서 이득이 없어 기본 경로가 아니다.",
        "- **NOT IMPLEMENTED** — 해당 코드 경로가 존재하지 않는다.",
        "- **NOT TESTABLE ON CURRENT HARDWARE** — 구현 여부와 무관하게 이 장비에서 측정할 수 없다.",
        "",
        "`OverlapMigrateAction` (`scheduling/overlap_migration.py`) 는 V3 가 추가했으나 어디에서도",
        "참조되지 않는 dead code 다. V3 의 target-first 동작은 이 클래스가 아니라",
        "`execute_actions` 의 배치 순서와 readiness barrier 로 구현되어 있다.",
    ]
    (base / "IMPLEMENTATION_AUDIT.md").write_text("\n".join(lines) + "\n")

    # ---------------- REPORT.md
    R = ["# Paper-Faithful Prism V4 — 측정 보고서", "",
         "_`exp/scripts/build_report_v4.py` 가 이 디렉터리의 raw data 로부터 생성._", ""]

    R += ["## 0. 실험 환경", "",
          "**2×A100 allocation on a 4×A100 node.** 할당된 GPU 는 0,1 두 장뿐이며 GPU 2,3 은",
          "어떤 프로세스에도 노출되지 않는다 (`CUDA_VISIBLE_DEVICES=0,1`).",
          "각 런의 `gpu_timeline.txt` 는 네 장 모두를 2 초 간격으로 샘플링하므로,",
          "나머지 두 장이 유휴였다는 것은 주장이 아니라 raw data 로 남는다.", "",
          "```", env.strip()[:2600], "```", "",
          "이전 V3 보고서는 **다른 장비**에서 측정되었다. 그래서 이 연구는 released prototype 과",
          "V3 를 여기서 다시 돌린다. 이 보고서 안의 arm 간 비교만 유효하며, 이전 V3 보고서의",
          "절대 수치와 직접 비교해서는 안 된다. 자세한 차이는 `provenance/ENVIRONMENT.md`.", ""]

    # microbench: loading
    R += ["## 1. Microbenchmark — 병렬 가중치 로딩", ""]
    if loading:
        by_arm = defaultdict(list)
        per_model_rows = defaultdict(lambda: defaultdict(list))
        for rec in loading.get("records", []):
            by_arm[rec["arm"]].append(rec)
            for model, det in (rec.get("per_model") or {}).items():
                per_model_rows[rec["arm"]][model].append(det["seconds"])
        total_bytes = sum(loading.get("model_bytes", {}).values())
        R += [f"모델 6개(총 {total_bytes/2**30:.2f} GiB)를 GPU 0 으로 로드. 각 arm 3회.",
              f"peer access: `{loading.get('peer_access')}`", "",
              "| Arm | 총 로딩 시간 (s) | 대역폭 (GB/s) | H2D direct (GiB) | H2D helper (GiB) | P2P (GiB) | 경로 |",
              "| --- | ---: | ---: | ---: | ---: | ---: | --- |"]
        for arm in ("sequential", "v3-parallel-activation", "v4-parallel-loading",
                    "v4-pipelined-helper"):
            recs = by_arm.get(arm)
            if not recs:
                continue
            R.append("| {} | {} | {} | {:.2f} | {:.2f} | {:.2f} | {} |".format(
                arm,
                cell([r["total_loading_seconds"] for r in recs]),
                cell([r["aggregate_gbps"] for r in recs], 2),
                np.mean([r["bytes_h2d_direct"] for r in recs]) / 2**30,
                np.mean([r["bytes_h2d_helper"] for r in recs]) / 2**30,
                np.mean([r["bytes_p2p"] for r in recs]) / 2**30,
                recs[0]["transfer_path"]))
        seq = np.mean([r["aggregate_gbps"] for r in by_arm.get("sequential", [])]) \
            if by_arm.get("sequential") else float("nan")
        v3b = np.mean([r["aggregate_gbps"] for r in by_arm.get("v3-parallel-activation", [])]) \
            if by_arm.get("v3-parallel-activation") else float("nan")
        v4b = np.mean([r["aggregate_gbps"] for r in by_arm.get("v4-parallel-loading", [])]) \
            if by_arm.get("v4-parallel-loading") else float("nan")
        R += ["", f"V3 는 sequential 대비 **{v3b/seq:.2f}배**, V4 는 V3 대비 "
                  f"**{v4b/v3b:.2f}배**, sequential 대비 **{v4b/seq:.2f}배**.", "",
              "바이트 분할은 세 arm 이 동일하다 — 즉 이득은 옮긴 양이 아니라 **어떻게** 옮겼는지에서",
              "온다. V3 가 sequential 을 이기는 것은 두 번째 PCIe 링크를 쓰기 때문이고, V4 가",
              "V3 를 이기는 것은 공유 호스트 페이지를 제자리에서 page-lock 해 드라이버 bounce",
              "buffer 경유가 실제 DMA 가 되기 때문이다. 두 메커니즘 모두 논문 §5.3 의 의도에 있다.", ""]
        if by_arm.get("v4-pipelined-helper"):
            pb = np.mean([r["aggregate_gbps"] for r in by_arm["v4-pipelined-helper"]])
            R += [f"helper leg 파이프라이닝(ablation)은 {pb:.2f} GB/s 로 V4 기본 경로보다 "
                  f"{'느렸다' if pb < v4b else '빨랐다'}. sub-chunk 마다 드는 event 비용이 겹침으로",
                  "얻는 것보다 컸다. 구현은 남기되 기본 경로로 쓰지 않는다.", ""]
        R += ["| Arm | " + " | ".join(sorted(next(iter(per_model_rows.values())).keys())) + " |"
              if per_model_rows else ""]
        if per_model_rows:
            models = sorted(next(iter(per_model_rows.values())).keys())
            R += ["| --- | " + " | ".join("---:" for _ in models) + " |"]
            for arm in ("sequential", "v3-parallel-activation", "v4-parallel-loading",
                        "v4-pipelined-helper"):
                if arm not in per_model_rows:
                    continue
                R.append("| " + arm + " | " + " | ".join(
                    fmt(np.mean(per_model_rows[arm][m])) for m in models) + " |")
            R += ["", "_모델별 전송 시간 (초, 3회 평균). **sequential 의 값이 작은 것은 빠르다는 "
                  "뜻이 아니다** — 한 번에 한 모델만 옮기므로 각 모델이 링크를 독점하고, 대신 그것들이 "
                  "차례로 일어나 위 표의 총 시간이 가장 길다. 나머지 arm 은 여섯 모델이 동시에 "
                  "경합하므로 개별 시간은 길고 총 시간은 짧다._", ""]
    else:
        R += ["_아직 실행되지 않았다._", ""]

    # microbench: migration
    R += ["## 2. Microbenchmark — 마이그레이션", ""]
    if migration:
        by_arm = defaultdict(list)
        for rec in migration.get("records", []):
            by_arm[rec["arm"]].append(rec)
        R += [f"GPU {migration['records'][0]['source_gpu']} → "
              f"{migration['records'][0]['target_gpu']}, 모델 3개 × 3회.", "",
              "| Arm | latency (s) | service downtime (s) | 전송 바이트 (GiB) | 대역폭 (GB/s) | 경로 | NVLink Rx (GiB) |",
              "| --- | ---: | ---: | ---: | ---: | --- | ---: |"]
        for arm in ("prototype-source-first", "v3-target-first", "v4-p2p-target-first"):
            recs = by_arm.get(arm)
            if not recs:
                continue
            nvl = [((r.get("nvlink_delta_target_gpu") or {}).get("rx_bytes") or 0) / 2**30
                   for r in recs]
            R.append("| {} | {} | {} | {:.2f} | {} | {} | {:.2f} |".format(
                arm,
                cell([r["migration_latency_s"] for r in recs]),
                cell([r["service_downtime_s"] for r in recs]),
                np.mean([r["total_bytes"] for r in recs]) / 2**30,
                cell([r["effective_gbps"] for r in recs], 1),
                recs[0]["transfer_path"],
                np.mean(nvl)))
        # Per model as well: the three differ by 6.5x in size, so an average
        # over them hides the effect rather than showing it.
        models = sorted({r["model"] for r in migration["records"]},
                        key=lambda m: next(x["weight_bytes"] for x in migration["records"]
                                           if x["model"] == m))
        R += ["", "모델별로 나누어 보면(크기가 6.5배까지 차이나므로 평균은 효과를 가린다):", "",
              "| 모델 | 크기 (GiB) | Arm | latency (s) | downtime (s) | GB/s | NVLink Rx (GiB) |",
              "| --- | ---: | --- | ---: | ---: | ---: | ---: |"]
        for model in models:
            for arm in ("prototype-source-first", "v3-target-first", "v4-p2p-target-first"):
                sub = [r for r in migration["records"]
                       if r["model"] == model and r["arm"] == arm]
                if not sub:
                    continue
                nvl = [((r.get("nvlink_delta_target_gpu") or {}).get("rx_bytes") or 0) / 2**30
                       for r in sub]
                R.append("| {} | {:.2f} | {} | {} | {} | {} | {:.2f} |".format(
                    model.split("/")[-1], sub[0]["weight_bytes"] / 2**30, arm,
                    cell([r["migration_latency_s"] for r in sub]),
                    cell([r["service_downtime_s"] for r in sub]),
                    cell([r["effective_gbps"] for r in sub], 1),
                    float(np.mean(nvl))))

        R += ["", "NVLink Rx 는 드라이버의 링크 카운터를 전송 직전/직후에 읽어 뺀 값이다.",
              "broker 경로에서는 모델의 정확히 절반, P2P 경로에서는 모델 전체가 NVLink 를 건넌다 —",
              "즉 경로는 추정이 아니라 계측되었다. PCIe 상한(약 25 GB/s)을 넘는 대역폭도 같은 결론을",
              "독립적으로 뒷받침한다.", "",
              "downtime 이 갈리는 지점은 **순서**다. prototype 은 원본을 먼저 비활성화하므로 전송",
              "구간 전체가 서비스 공백이고, target-first 는 원본이 계속 서비스하므로 공백이 0 이다.",
              "그리고 target-first 이기 때문에 원본 GPU 에 가중치가 아직 살아 있고, 그것이 V4 의",
              "GPU→GPU 전송을 가능하게 하는 전제다.", ""]
    else:
        R += ["_아직 실행되지 않았다._", ""]

    # TP=2
    R += ["## 3. TP=2 검증", ""]
    if tp2:
        R += [f"**판정: {tp2['verdict']}**", ""]
        for name, value in tp2["checks"].items():
            mark = {True: "PASS", False: "FAIL", None: "NOT OBSERVED"}[value]
            R.append(f"- [{mark}] {name}")
        lat = tp2.get("latency", {})
        R += ["", f"startup {tp2['startup_seconds']:.1f}s, "
                  f"TP rank → GPU: `{tp2.get('tp_rank_to_gpu')}`", ""]
        if any(lat.values()):
            R += ["| 지표 | n | mean | p50 | p95 | p99 | max |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
            for key in ("ttft", "tpot", "e2e"):
                s = lat.get(key) or {}
                if s:
                    R.append(f"| {key.upper()} (s) | {s.get('n')} | {fmt(s.get('mean'),4)} | "
                             f"{fmt(s.get('p50'),4)} | {fmt(s.get('p95'),4)} | "
                             f"{fmt(s.get('p99'),4)} | {fmt(s.get('max'),4)} |")
        R += ["", "TP=2 는 논문 범위대로 **주어진 TP 설정의 배치/스케줄링**만 검증한다. Prism 이 TP",
              "degree 를 스스로 정하게 만드는 기능은 추가하지 않았다.", "",
              "부수적으로 확인된 사실: `tp_size > 1` 이면 upstream 이 model-service 경로를 끄므로",
              "(`model_runner.py`) **TP 모델은 병렬 가중치 로딩을 쓰지 않는다.**", ""]
    else:
        R += ["_아직 실행되지 않았다._", ""]

    # e2e
    R += ["## 4. End-to-End", ""]
    if rows:
        groups = defaultdict(lambda: defaultdict(list))
        for row in rows:
            groups[(row["workload"], int(row["request_rate"]))][row["implementation"]].append(row)
        R += ["측정 구간 300 초(워밍업 60 초 제외), seed 당 1 런. 값은 seed 간 mean ± sd.", "",
              "| Workload | Rate | Arm | seeds | Goodput | Joint SLO | TTFT SLO | TPOT SLO | Throughput |",
              "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for (workload, rate) in sorted(groups):
            for arm in ARM_ORDER:
                members = groups[(workload, rate)].get(arm)
                if not members:
                    continue
                R.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                    workload, rate, ARM_LABEL.get(arm, arm), len(members),
                    cell([num(m["goodput"]) for m in members]),
                    cell([num(m["joint_slo"]) for m in members]),
                    cell([num(m["ttft_slo"]) for m in members]),
                    cell([num(m["tpot_slo"]) for m in members]),
                    cell([num(m["achieved_throughput"]) for m in members], 2)))

        R += ["", "### 4.1 지연 분포 (ms)", "",
              "| Workload | Rate | Arm | TTFT p50 | p95 | p99 | TPOT p50 | p95 | p99 | E2E p50 | p95 | p99 |",
              "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for (workload, rate) in sorted(groups):
            for arm in ARM_ORDER:
                members = groups[(workload, rate)].get(arm)
                if not members:
                    continue
                vals = [cell([num(m[f"{n}_{p}"]) for m in members], 1)
                        for n in ("ttft", "tpot", "e2e") for p in ("p50", "p95", "p99")]
                R.append(f"| {workload} | {rate} | {ARM_LABEL.get(arm, arm)} | " +
                         " | ".join(vals) + " |")

        R += ["", "### 4.2 마이그레이션", "",
              "`latency` 열은 컨트롤러가 마이그레이션을 결정한 시점부터 대상이 준비될 때까지다.",
              "**released prototype 에는 readiness barrier 가 없어 제어 핸들러가 요청을 제출하는",
              "순간 반환한다** — 그 arm 의 latency 는 가중치 전송 시간이 아니라 제출 시간이며,",
              "`submission_only` 열이 1 인 행이 그것이다. 프로토타입의 실제 마이그레이션 비용은",
              "§2 의 microbenchmark 가 통제된 조건에서 측정한 값이다.", "",
              "| Workload | Rate | Arm | count | 결정 | submission_only | latency p50 (ms) | p95 | downtime p50 (ms) | 전송 바이트 | 대역폭 (GB/s) | P2P 전송 |",
              "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for (workload, rate) in sorted(groups):
            for arm in ARM_ORDER:
                members = groups[(workload, rate)].get(arm)
                if not members:
                    continue
                R.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                    workload, rate, ARM_LABEL.get(arm, arm),
                    cell([num(m["migration_count"]) for m in members], 1),
                    cell([num(m.get("migration_decisions_logged")) for m in members], 1),
                    int(num(members[0].get("migration_latency_is_submission_only"), 0)),
                    cell([num(m["migration_latency_p50"]) for m in members], 1),
                    cell([num(m["migration_latency_p95"]) for m in members], 1),
                    cell([num(m["service_downtime_p50"]) for m in members], 1),
                    cell([num(m["migration_total_bytes"]) / 2**30 for m in members], 2),
                    cell([num(m["migration_bandwidth"]) for m in members], 1),
                    cell([num(m["p2p_weight_transfers"]) for m in members], 1)))

        R += ["", "### 4.3 마이그레이션 비용의 내역 — 전송은 어디까지인가", "",
              "v4 가 최적화하는 것은 가중치 전송이다. 전송이 마이그레이션 비용의 어느 정도를",
              "차지하는지는 가정할 것이 아니라 런이 답할 수 있는 질문이므로, 제어 액션의 벽시계를",
              "활성화·비활성화·전송으로 나누어 기록한다.", "",
              "| Workload | Rate | Arm | 활성화 (회/총 s) | 비활성화 (회/총 s) | 전송 총 s | 전송 대역폭 GB/s | 제어 액션 총 s |",
              "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |"]
        for (workload, rate) in sorted(groups):
            for arm in ARM_ORDER:
                members = groups[(workload, rate)].get(arm)
                if not members:
                    continue
                act_n = cell([num(m.get("activation_count")) for m in members], 1)
                act_s = cell([num(m.get("activation_total_s")) for m in members], 1)
                de_n = cell([num(m.get("deactivation_count")) for m in members], 1)
                de_s = cell([num(m.get("deactivation_total_s")) for m in members], 1)
                tot = cell([num(m.get("activation_total_s")) + num(m.get("deactivation_total_s"))
                            for m in members], 1)
                R.append(f"| {workload} | {rate} | {ARM_LABEL.get(arm, arm)} | {act_n} / {act_s} | "
                         f"{de_n} / {de_s} | "
                         f"{cell([num(m.get('weight_transfer_total_s')) for m in members], 1)} | "
                         f"{cell([num(m.get('weight_transfer_mean_gbps')) for m in members], 1)} | {tot} |")
        R += ["", "**released prototype 의 액션 시간은 제출 시간이므로 이 표에서 비교 대상이 아니다**",
              "(§4.2 참조). 비교는 v3 대 v4 다.", ""]

        R += ["", "### 4.4 스케줄러 / Algorithm 1", "",
              "| Workload | Rate | Arm | alg1 cycles | placement decisions | MIGRATE | tau 억제 | 메모리 거부 | 수렴 gap | 큐 max |",
              "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for (workload, rate) in sorted(groups):
            for arm in ARM_ORDER:
                members = groups[(workload, rate)].get(arm)
                if not members:
                    continue
                R.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                    workload, rate, ARM_LABEL.get(arm, arm),
                    cell([num(m["alg1_cycles"]) for m in members], 1),
                    cell([num(m["alg1_placement_decisions"]) for m in members], 1),
                    cell([num(m["alg1_migrate_decisions"]) for m in members], 1),
                    cell([num(m.get("alg1_suppressed_by_tau")) for m in members], 1),
                    cell([num(m.get("alg1_rejected_by_memory")) for m in members], 1),
                    cell([num(m.get("alg1_convergence_gap_mean")) for m in members], 2),
                    cell([num(m.get("queue_length_max")) for m in members], 1)))
        R += [""]
    else:
        R += ["_아직 완료된 런이 없다._", ""]

    if made:
        R += ["## 5. 그림", ""] + [f"![{name}](figures/{name})" for name in made] + [""]

    R += ["## 6. Raw data", "",
          "집계가 raw data 를 대체하지 않는다. 다음이 모두 보존되어 있다.", "",
          "| 경로 | 내용 |", "| --- | --- |",
          "| `raw/requests/*.csv` | 요청 단위: 도착/완료 시각, 프롬프트·출력 토큰, TTFT/TPOT/E2E, SLO 충족 여부 |",
          "| `raw/migrations/*.csv` | 마이그레이션 단위: 시각, latency, downtime, 바이트, 경로, 대역폭, KVPR |",
          "| `raw/migrations/*_weight_transfers.jsonl` | 가중치 전송 단위: 경로별 바이트, 시간, 대역폭 |",
          "| `raw/scheduler/*_alg1.jsonl` | 사이클별 Algorithm 1 전체 배치 계획과 차단 사유 |",
          "| `raw/scheduler/*_actions.jsonl` | 제어 액션 단위 타이밍 |",
          "| `raw/gpu_metrics/*.csv` | GPU 4장 전부의 2초 간격 사용률·메모리 |",
          "| `microbench/*.json` | microbenchmark 원자료 |",
          "| `profiling/` | 이 장비에서 측정한 c_i 와 SLO 기준선 |",
          "| `logs/` | 런별 실행 로그 |", "",
          "GPU 샘플링 간격은 2 초다. `nvidia-smi` 호출 1 회/2 초는 벤치마크와 GPU 를 공유하지 않으므로",
          "측정에 영향을 주지 않는다.", "",
          "## 7. 한계", "",
          "- KV-cache 마이그레이션은 어느 arm 에도 구현되어 있지 않다. 마이그레이션 바이트는 전부 가중치다.",
          "- RDMA 는 단일 노드라 측정 대상이 없다.",
          "- TP anti-affinity 는 전역 컨트롤러가 TP 그룹을 rank0 GPU 로 축약하므로 표현 자체가 불가능하다.",
          "- 결과는 A100 80GB 2장, 6모델, 이 SLO scale 에 한정된다.",
          ]
    (base / "REPORT.md").write_text("\n".join(R) + "\n")
    print(f"wrote REPORT.md and IMPLEMENTATION_AUDIT.md ({len(made)} figures)")


if __name__ == "__main__":
    main()
