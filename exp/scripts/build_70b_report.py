#!/usr/bin/env python3
"""STABILITY_REPORT.md for the 70B run, from the run's own artefacts.

The PASS bar is fixed here before the numbers exist, and it is a *stability*
bar, not a performance one: startup, real inference, a completed sustained
window, no failed requests, no OOM, no CUDA/NCCL fatal, no rank crash, and no
memory drift.  A failure is never reported as a bare FAIL -- the stage is named,
because "70B does not work" is not actionable and "it dies during weight
loading" is.
"""

import argparse
import json
import re
from pathlib import Path

FATAL = ["CUDA error", "NCCL WARN", "NCCL error", "out of memory",
         "CUDA out of memory", "Segmentation fault"]


def load(p, default=None):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return default


def scan_logs(d: Path):
    counts = {k: 0 for k in FATAL}
    ranks = {}
    text_all = []
    for p in list(d.glob("**/server-logs/*.log")) + list(d.glob("logs/*.log")):
        try:
            t = p.read_text(errors="replace")
        except OSError:
            continue
        text_all.append(t)
        for k in FATAL:
            counts[k] += t.count(k)
        for m in re.finditer(r"\[PAPER-TP\] engine rank: tp_rank=(\d+) gpu_id=(\d+) tp_size=(\d+)", t):
            if int(m.group(3)) > 1:
                ranks.setdefault(m.group(1), set()).add(m.group(2))
    return counts, {k: sorted(v) for k, v in sorted(ranks.items())}, "\n".join(text_all)


def gpu_drift(path: Path):
    """First vs last memory sample per GPU -- a crude but honest leak check."""
    if not path.exists():
        return None
    first, last = None, None
    for line in path.read_text(errors="replace").splitlines():
        parts = line.strip().split(" ", 1)
        if len(parts) != 2:
            continue
        rows = [r for r in parts[1].split(";") if r.strip()]
        vals = {}
        for r in rows:
            f = [x.strip() for x in r.split(",")]
            if len(f) >= 2 and f[0].isdigit():
                vals[int(f[0])] = int(f[1])
        if vals:
            first = first or vals
            last = vals
    if not first or not last:
        return None
    return {g: {"first_mib": first.get(g), "last_mib": last.get(g),
                "drift_mib": (last.get(g, 0) - first.get(g, 0))} for g in sorted(last)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ns = ap.parse_args()
    base = Path(ns.base).resolve()

    meta = (base / "META.txt").read_text() if (base / "META.txt").exists() else ""
    s12 = load(base / "raw/stage12/tp2_requests.json", {}) or {}
    s12 = s12.get("summary", {})
    s3 = load(base / "raw/stage3/summary.json", {}) or {}
    counts, ranks, _ = scan_logs(base)
    drift = gpu_drift(base / "gpu_metrics/stage3_timeline.txt")

    startup_ok = bool(ranks)
    ranks_distinct = len(ranks) >= 2 and len({tuple(v) for v in ranks.values()}) >= 2
    basic_ok = bool(s12.get("successful")) and s12.get("failed", 1) == 0
    sustained_ran = bool(s3.get("completed"))
    sustained_clean = sustained_ran and s3.get("failed", 1) == 0 and s3.get("never_returned", 1) == 0
    fatal_total = sum(counts.values())
    drift_ok = True
    if drift:
        drift_ok = all(abs(v["drift_mib"]) < 8192 for v in drift.values())

    checks = [
        ("startup: TP ranks created", startup_ok),
        ("startup: ranks on distinct GPUs (anti-affinity held)", ranks_distinct),
        ("basic inference succeeded", basic_ok),
        ("sustained window completed", sustained_ran),
        ("no failed / abandoned requests in sustained", sustained_clean),
        ("no CUDA/NCCL/OOM fatal in logs", fatal_total == 0),
        ("no memory drift > 8 GiB per GPU", drift_ok),
    ]
    verdict = "PASS" if all(c for _, c in checks) else "FAIL"

    # Where it broke, if it did -- named stage, not a bare FAIL.
    stage = None
    if not startup_ok:
        stage = "startup / weight loading"
    elif not ranks_distinct:
        stage = "scheduler (rank -> GPU mapping / anti-affinity)"
    elif not basic_ok:
        stage = "request processing (basic inference)"
    elif not sustained_ran:
        stage = "sustained serving (window did not complete)"
    elif not sustained_clean:
        stage = "sustained serving (requests failed or never returned)"
    elif fatal_total:
        stage = "NCCL / CUDA / memory (fatal in logs)"
    elif not drift_ok:
        stage = "memory (drift over the sustained window)"

    L = []
    A = L.append
    A(f"# Llama 70B serving: **{verdict}**\n")
    A("Prototype 과의 성능 비교가 아니라 capability/stability 검증이다. "
      "프로토타입은 worker-pool 경로에서 TP>1 을 아예 못 돌리므로 비교 대상이 없다.\n")
    if verdict == "FAIL":
        A(f"\n**실패 단계: {stage}**\n")
        A("단순 FAIL 로 적지 않는 이유는 '70B 가 안 된다' 는 조치가 불가능하고 "
          "'가중치 로딩에서 죽는다' 는 가능하기 때문이다.\n")

    A("\n## 판정\n")
    A("| 항목 | 결과 |")
    A("| --- | --- |")
    for name, ok in checks:
        A(f"| {name} | {'PASS' if ok else 'FAIL'} |")

    A("\n## TP rank 배치\n")
    if ranks:
        A("| tp_rank | GPU |")
        A("| ---: | --- |")
        for r, gs in ranks.items():
            A(f"| {r} | {', '.join(gs)} |")
        A("")
        A("한 rank 가 여러 GPU 로 보이면 그 디렉터리에서 여러 번 실행된 로그의 "
          "합집합이다. 한 번의 실행 안에서는 1:1 이다.")
    else:
        A("**TP rank 로그가 없다** — 엔진이 뜨지 못했다는 뜻이다.")

    A("\n## Stage 2 — basic inference\n")
    A(f"```\n{json.dumps(s12, indent=2, ensure_ascii=False)}\n```")

    A("\n## Stage 3 — sustained\n")
    if s3:
        A(f"* 지속 시간: {s3.get('wall_seconds', 0)/60:.1f} 분")
        A(f"* 발송 {s3.get('sent')} / 완료 {s3.get('completed')} / 실패 "
          f"{s3.get('failed')} / 미반환 {s3.get('never_returned')}")
        A(f"* 처리량 {s3.get('achieved_throughput_rps', 0):.3f} req/s, "
          f"출력 토큰 {s3.get('output_token_throughput', 0):.1f} tok/s")
        for k in ("ttft", "tpot", "e2e"):
            b = s3.get(k) or {}
            if b.get("n"):
                A(f"* {k.upper()} p50 {b['p50']:.3f} / p95 {b['p95']:.3f} / "
                  f"p99 {b['p99']:.3f} / max {b['max']:.3f} (n={b['n']})")
        if s3.get("errors"):
            A(f"* 오류 표본: {s3['errors'][:5]}")
    else:
        A("**sustained 를 돌리지 않았거나 요약이 없다.**")

    A("\n## GPU 메모리 추이 (누수 징후)\n")
    if drift:
        A("| GPU | 시작(MiB) | 끝(MiB) | 변화 |")
        A("| ---: | ---: | ---: | ---: |")
        for g, v in drift.items():
            A(f"| {g} | {v['first_mib']} | {v['last_mib']} | {v['drift_mib']:+d} |")
        A("")
        A("KV 캐시가 부하에 따라 늘고 주는 것은 정상이다. 여기서 보는 것은 "
          "창 전체에 걸친 단조 증가다. 8 GiB 를 넘으면 FAIL 로 잡는다.")
    else:
        A("time-series 를 수집하지 못했다.")

    A("\n## 로그의 치명적 패턴\n")
    A("| 패턴 | 건수 |")
    A("| --- | ---: |")
    for k, v in counts.items():
        A(f"| `{k}` | {v} |")

    A("\n## 환경\n")
    A(f"```\n{meta.strip()[:2000]}\n```")

    (base / "STABILITY_REPORT.md").write_text("\n".join(L) + "\n")
    # a flat row for the summary table
    with (base / "summary.csv").open("w") as fh:
        fh.write("metric,value\n")
        fh.write(f"verdict,{verdict}\n")
        if stage:
            fh.write(f"failed_stage,{stage}\n")
        for k, v in (s3 or {}).items():
            if isinstance(v, (int, float)):
                fh.write(f"sustained_{k},{v}\n")
        for k, v in counts.items():
            fh.write(f"fatal_{k.replace(' ', '_')},{v}\n")
    print(f"70B verdict: {verdict}" + (f" (stage: {stage})" if stage else ""))
    print(f"wrote {base/'STABILITY_REPORT.md'}")


if __name__ == "__main__":
    main()
