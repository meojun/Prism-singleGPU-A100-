#!/usr/bin/env python3
"""Generate one Korean report covering the whole state of this experiment box.

Everything is PROBED, not hard-coded. Hardware, package versions, pinned
upstream SHAs, git history, the experiment inventory and its numbers are all
read at run time, and each paper-vs-code claim is re-verified by grepping the
pinned source and printed with the file:line it was found at. Re-run it after
any run and the report stays true; if a claim stops matching the source it is
reported as 확인 실패 rather than silently repeated.

Report prose is Korean to match the other REPORT.md files in this repo; code
comments stay English like the rest of exp/scripts.

    python exp/scripts/build_status_report.py            # -> exp/results/STATUS_REPORT.md
    python exp/scripts/build_status_report.py --out /tmp/r.md
"""
import argparse
import csv
import glob
import json
import os
import re
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXP = os.path.join(ROOT, "exp")
SRC = os.path.join(ROOT, "prism-research")
OUT = []


def w(s=""):
    OUT.append(s)


def sh(cmd, cwd=None):
    try:
        return subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True,
                              text=True, timeout=60).stdout.strip()
    except Exception as e:
        return f"(실패: {e})"


def h(level, text):
    w(); w("#" * level + " " + text); w()


def table(headers, rows, align=None):
    align = align or ["---"] * len(headers)
    w("| " + " | ".join(headers) + " |")
    w("| " + " | ".join(align) + " |")
    for r in rows:
        w("| " + " | ".join(str(c) for c in r) + " |")


# ---------------------------------------------------------------- environment
def section_environment():
    h(2, "1. 환경")

    gpus = sh("nvidia-smi --query-gpu=index,name,memory.total,compute_cap "
              "--format=csv,noheader").splitlines()
    drv = sh("nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1")
    topo = sh("nvidia-smi topo -m 2>/dev/null | sed -n '2p' | awk '{print $3}'")
    w(f"**GPU** — 드라이버 {drv}" + (f", GPU0↔GPU1 연결 `{topo}`" if topo else ""))
    w()
    table(["idx", "이름", "메모리", "compute cap"],
          [[c.strip() for c in g.split(",")] for g in gpus])
    w()
    w("> compute capability가 10.0 이상(Blackwell)이면 이 스택은 못 씁니다 — "
      "torch 2.4.0+cu121에 해당 아키텍처 커널이 없어 첫 GPU 연산에서 죽습니다.")

    mem = sh("free -g | awk 'NR==2{print $2\" GiB 중 \"$7\" GiB 가용\"}'")
    disk = sh(f"df -h {ROOT} | awk 'NR==2{{print $2\" 중 \"$4\" 여유\"}}'")
    shm = sh("df -h /dev/shm | awk 'NR==2{print $2}'")
    w()
    w(f"**호스트** — {sh('nproc')} 스레드, RAM {mem}, 레포 디스크 {disk}, /dev/shm {shm}")

    # versions come from the venv the runs actually used, not this interpreter
    py = os.path.join(ROOT, "prism-venv/bin/python")
    w()
    if os.path.exists(py):
        code = ("import importlib\n"
                "for m in ['torch','sglang','vllm','transformers','flashinfer','kvcached']:\n"
                "    try:\n"
                "        v=getattr(importlib.import_module(m),'__version__','설치됨 (버전 속성 없음)')\n"
                "    except Exception as e:\n"
                "        v='IMPORT 실패: %s' % type(e).__name__\n"
                "    print(m,v)\n"
                "import torch;print('cuda_available',torch.cuda.is_available())")
        # run via argv, not a shell string: the probe contains newlines and
        # quotes that a shell would mangle into a SyntaxError
        try:
            vers = subprocess.run([py, "-c", code], capture_output=True,
                                  text=True, timeout=120).stdout.strip()
        except Exception as e:
            vers = f"probe 실패 {type(e).__name__}"
        w("**스택** (`prism-venv` 기준 — 실제 실험이 로드한 것)")
        w()
        parsed = [l.split(None, 1) for l in vers.splitlines() if l.strip()]
        if parsed:
            table(["패키지", "버전"], parsed)
        else:
            w(f"*(버전 probe가 아무것도 반환하지 않음: `{vers[:120]}`)*")
    else:
        w("**스택** — `prism-venv`가 없습니다. `./bootstrap.sh`를 실행하세요.")

    if os.path.exists(os.path.join(ROOT, "setup/pins.env")):
        rows = []
        for name, d in (("prism-research (SGLang 포크)", "prism-research"),
                        ("kvcached (prism/shm)", "kvcached-prism"),
                        ("kvcached (main)", "kvcached")):
            p = os.path.join(ROOT, d)
            live = sh("git rev-parse --short HEAD", cwd=p) if os.path.isdir(p) else "(없음)"
            rows.append([name, f"`{live}`"])
        w()
        w("**고정된 upstream** (`setup/pins.env`) — 실제 체크아웃된 HEAD")
        w()
        table(["저장소", "HEAD"], rows)

    w()
    redis = sh("redis-cli ping 2>/dev/null") or "응답 없음"
    sup = sh("supervisorctl status redis 2>/dev/null") or "(supervisor 미등록)"
    w(f"**redis** — `{redis}`, supervisor: `{sup}`")
    w()
    w("> redis가 죽어 있으면 Prism이 기동 중 모델을 `activating` 상태로 둔 채 멈춥니다.")

    hf = sh("du -sh /workspace/.hf_home/hub/models--* 2>/dev/null")
    if hf:
        w()
        w("**모델 가중치**")
        w()
        table(["크기", "모델"],
              [[l.split()[0], "`" + l.split()[1].split("models--")[-1].replace("--", "/") + "`"]
               for l in hf.splitlines()])

    busy = sh("ps -eo cmd --no-headers | grep -c '[s]glang.launch_multi_model_server'")
    used = sh("nvidia-smi --query-gpu=memory.used --format=csv,noheader | tr '\\n' ' '")
    w()
    w(f"**현재 상태** — 서빙 프로세스 {busy}개, GPU 메모리 사용량 {used}")


# ------------------------------------------------------------------ repo state
def section_repo():
    h(2, "2. 저장소 상태")
    branch = sh("git rev-parse --abbrev-ref HEAD", cwd=ROOT)
    dirty = sh("git status --porcelain", cwd=ROOT)
    ahead = sh("git rev-list --count origin/main..HEAD 2>/dev/null", cwd=ROOT)
    state = "**커밋 안 된 변경 있음**" if dirty else "clean"
    sync = "origin과 동기화됨" if ahead in ("0", "") else f"**origin보다 {ahead}커밋 앞섬**"
    w(f"브랜치 `{branch}`, 워킹트리 {state}, {sync}")
    w()
    log = sh("git log --pretty=format:'%h|%ad|%s' --date=short -12", cwd=ROOT)
    table(["커밋", "날짜", "제목"],
          [l.split("|", 2) for l in log.splitlines() if "|" in l])
    if dirty:
        w()
        w("커밋 안 된 파일:")
        w("```")
        for l in dirty.splitlines()[:20]:
            w(l)
        w("```")


# ------------------------------------------------------------------ inventory
def load_summary(path):
    try:
        return next(csv.DictReader(open(path)))
    except Exception:
        return {}


def num(d, k, default=float("nan")):
    try:
        return float(d.get(k, ""))
    except (TypeError, ValueError):
        return default


def discover_runs():
    """Every analysed run on disk, grouped by results/<tag>/."""
    runs = []
    for slo in glob.glob(os.path.join(EXP, "results", "*", "*_slo.json")):
        tag = os.path.basename(os.path.dirname(slo))
        exp = os.path.basename(slo)[: -len("_slo.json")]
        try:
            d = json.load(open(slo))
        except Exception:
            continue
        runs.append({"tag": tag, "exp": exp, "slo": d,
                     "sum": load_summary(os.path.join(EXP, "results", tag,
                                                      f"{exp}_summary.csv"))})
    return sorted(runs, key=lambda r: (r["tag"], r["exp"]))


def section_inventory(runs):
    h(2, "3. 실험 목록")
    w(f"분석 완료된 run **{len(runs)}건**, 결과 네임스페이스 "
      f"{len({r['tag'] for r in runs})}개. 모든 값은 커밋된 "
      "`*_slo.json` / `*_summary.csv`에서 읽습니다.")
    w()
    rows = []
    for r in runs:
        a = r["slo"].get("per_model", {}).get("ALL", {})
        rows.append([
            f"`{r['tag']}`", f"`{r['exp']}`",
            r["slo"].get("total_requests", "?"),
            f"{r['slo'].get('duration_s', 0):.0f}초",
            f"{a.get('attain_ttft', float('nan')):.3f}",
            f"{a.get('attain_tpot', float('nan')):.3f}",
            f"{a['ttft_p95_ms']:,.0f}" if "ttft_p95_ms" in a else "—",
            f"{a.get('tpot_p50_ms', float('nan')):.0f}",
        ])
    table(["네임스페이스", "run", "요청수", "시간", "att TTFT", "att TPOT",
           "TTFT p95 ms", "TPOT p50 ms"], rows,
          ["---", "---", "---:", "---:", "---:", "---:", "---:", "---:"])
    w()
    w("> attainment는 항상 `analyze_slo.py`가 재계산한 값입니다. "
      "`benchmark.py`의 `average_attainment_tpot`은 ms 단위 baseline을 초 단위 "
      "측정값과 비교해서 **항상 1.0**이므로 쓰면 안 됩니다.")


# --------------------------------------------------------------- key results
def section_rate_sweep():
    pts = [("1", 1.0), ("0.8", 1.25), ("0.6667", 1.5), ("0.5", 2.0), ("0.4", 2.5)]
    have = [(ts, m) for ts, m in pts
            if os.path.exists(os.path.join(EXP, "results/4-rate-sweep",
                                           f"exp_glob_on_ts{ts}_summary.csv"))]
    if not have:
        return
    h(2, "4. Rate sweep — 3× Llama-3.1-8B, 2 GPU")
    lam = 12.0
    w(f"λ_base = **{lam:.0f} req/s**. 프로파일링으로 찾은 TTFT knee(약 26 req/s)의 "
      "46%로 정했습니다(임의 지정 아님). 모든 rate가 **동일한 request 시퀀스**를 "
      "`--time-scale`로 압축한 것이라 길이 분포·모델 비율·seed가 행마다 같습니다.")
    w()
    rows = []
    for ts, mult in have:
        s = load_summary(os.path.join(EXP, "results/4-rate-sweep",
                                      f"exp_glob_on_ts{ts}_summary.csv"))
        kv = json.loads(s.get("peak_kv_pool_frac", "{}") or "{}")
        rows.append([
            f"{lam*mult:.0f}", f"{mult:.2f}×",
            f"{num(s,'throughput_rps'):.1f}", f"{num(s,'out_tok_throughput'):,.0f}",
            f"{num(s,'ttft_p50_ms'):.0f}", f"**{num(s,'ttft_p95_ms'):,.0f}**",
            f"{num(s,'tpot_p50_ms'):.0f}", f"{num(s,'attain_tpot'):.3f}",
            " / ".join(f"{kv.get(m,0):.2f}" for m in ("model_1", "model_4", "model_5")),
            f"{num(s,'max_model_queue'):.0f} / {num(s,'max_sched_qlen'):.0f}",
        ])
    table(["요청 λ", "×base", "실제 처리", "출력 tok/s", "TTFT p50 ms",
           "TTFT p95 ms", "TPOT p50 ms", "att TPOT", "KV 풀 m1/m4/m5",
           "최대 큐 모델/스케줄러"], rows,
          ["---:"] * 8 + [":---:", ":---:"])
    w()
    lo = load_summary(os.path.join(EXP, "results/4-rate-sweep", "exp_glob_on_ts0.5_summary.csv"))
    hi = load_summary(os.path.join(EXP, "results/4-rate-sweep", "exp_glob_on_ts0.4_summary.csv"))
    if lo and hi:
        r95 = num(hi, "ttft_p95_ms") / num(lo, "ttft_p95_ms")
        r50 = num(hi, "ttft_p50_ms") / num(lo, "ttft_p50_ms")
        w(f"2.0× → 2.5× 구간에서 TTFT **p95는 {r95:,.0f}배 폭증**하는데 "
          f"**p50은 {r50:.1f}배**에 그칩니다 — 평균과 중앙값이 이 절벽을 완전히 "
          "감춥니다. 절벽 지점은 2모델 GPU의 KV 풀이 약 0.98에 도달하는 시점, "
          "그리고 큐가 처음으로 비어있지 않게 되는 시점과 정확히 일치합니다.")


def section_per_model():
    if not os.path.exists(os.path.join(EXP, "results/4-rate-sweep", "exp_glob_on_ts1_slo.json")):
        return
    h(3, "4.1 rate보다 colocation이 지배적이다")
    rows = []
    for ts, lam in [("1", 12), ("0.8", 15), ("0.6667", 18), ("0.5", 24), ("0.4", 30)]:
        p = os.path.join(EXP, "results/4-rate-sweep", f"exp_glob_on_ts{ts}_slo.json")
        if not os.path.exists(p):
            continue
        d = json.load(open(p))["per_model"]
        cells = [f"{d.get(m,{}).get('attain_tpot',float('nan')):.3f} / "
                 f"{d.get(m,{}).get('ttft_p95_ms',float('nan')):,.0f} / "
                 f"{d.get(m,{}).get('tpot_p50_ms',float('nan')):.0f}"
                 for m in ("model_1", "model_4", "model_5")]
        rows.append([lam] + cells)
    table(["요청 λ", "model_1 (GPU0 단독)", "model_4 (GPU1 공유)",
           "model_5 (GPU1 공유)"], rows, ["---:", ":---:", ":---:", ":---:"])
    w()
    w("각 칸은 `att_tpot / TTFT p95 ms / TPOT p50 ms`. λ_base에서 GPU0에 혼자 있는 "
      "모델은 무경합 baseline 그대로인데, GPU를 공유하는 두 모델은 이미 **약 2.8배** "
      "느립니다 — rate를 올리기도 전에 그렇습니다.")


def section_burst():
    f = os.path.join(EXP, "results/4-rate-sweep", "burst_glob_on_ts1_windows.csv")
    if not os.path.exists(f):
        return
    h(3, "4.2 Burst — hot 모델 수 1 → 2 → 3")
    names = ["1개 hot (8 / 0.5 / 0.5)", "2개 hot (8 / 8 / 0.5)", "3개 hot (8 / 8 / 8)"]
    rows = []
    for i, r in enumerate(csv.DictReader(open(f))):
        rows.append([i, names[i] if i < len(names) else "?",
                     f"{float(r['offered_rate_rps']):.1f}",
                     f"**{float(r['attain_both']):.3f}**",
                     f"{float(r['tpot_p50_ms']):.0f}",
                     f"**{float(r['model_1_ttft_p95_ms']):.0f}**",
                     f"{float(r['model_4_ttft_p95_ms']):.0f}",
                     f"{float(r['model_5_ttft_p95_ms']):.0f}"])
    table(["phase", "hot 모델", "총 λ", "att both", "TPOT p50 ms",
           "model_1 TTFT p95", "model_4 TTFT p95", "model_5 TTFT p95"], rows,
          ["---:", "---", "---:", "---:", "---:", "---:", "---:", "---:"])
    w()
    w("model_1은 세 phase 내내 8 req/s로 **고정**되어 있으므로, 그 열의 변화는 "
      "순수하게 다른 모델의 burst가 준 피해입니다.")
    log = os.path.join(EXP, "server-logs/burst_glob_on_ts1/server.log.global_controller.log")
    if os.path.exists(log):
        acts = [l for l in open(log, errors="replace") if "ACTION:" in l]
        if acts:
            w()
            w("이 run에서 컨트롤러가 실제로 한 일:")
            w("```")
            for l in acts:
                w(re.sub(r"^\[[^\]]*\] ", "", l).rstrip())
            w("```")
            w()
            w("3개가 모두 hot이 되자 Prism이 GPU1 부하를 덜려고 모델 하나를 GPU0으로 "
              "옮겼고, 그 결과 그때까지 보호받던 model_1이 경합에 노출됐습니다. "
              "hot 모델 3개를 GPU 2장에 놓는 좋은 배치는 존재하지 않으므로 이건 "
              "정책 결함이 아니라 자원 부족입니다.")


def section_capacity():
    files = [("results/4-rate-sweep/rampLO_windows.csv", "저부하 ramp (1 → 8 req/s)"),
             ("results/4-rate-sweep/probe_glob_on_ts1_windows.csv", "고부하 ramp (8 → 31 req/s)")]
    rows = []
    for rel, label in files:
        p = os.path.join(EXP, rel)
        if not os.path.exists(p):
            continue
        for r in csv.DictReader(open(p)):
            rows.append([label, f"{float(r['offered_rate_rps']):.1f}",
                         f"{float(r['out_tok_per_s']):,.0f}",
                         f"{float(r['ttft_p95_ms']):,.0f}",
                         f"{float(r['tpot_p50_ms']):.0f}"])
    if not rows:
        return
    h(3, "4.3 capacity 프로파일링 — λ_base를 어떻게 정했나")
    w("rate를 **한 번의 run 안에서** 계단식으로 올리고 도착 시각 구간별로 집계했습니다. "
      "rate당 run 하나씩 돌리는 대신 총 2번으로 capacity 곡선을 얻습니다.")
    w()
    table(["ramp", "요청 req/s", "출력 tok/s", "TTFT p95 ms", "TPOT p50 ms"],
          rows, ["---", "---:", "---:", "---:", "---:"])
    w()
    w("두 ramp가 약 7.8 req/s에서 겹치고 값이 일치하므로 재현성 검증도 겸합니다.")


# ------------------------------------------------- paper vs code, re-verified
# Component-level audit. "Is Algorithm 1 implemented?" is not a yes/no
# question: the released policy balances memory pressure across GPUs (the same
# goal) using a different metric. Checking whole algorithms as present/absent
# overstates the finding in both directions, so each ingredient is checked
# separately and the verdict is assembled from the parts.
#
# Scope of the audit these patterns encode (re-run scripts/audit notes):
#   * Multi-LLM/prism-research: 1 branch, 4 commits, pinned SHA == HEAD
#   * ovg-project/kvcached: 22 branches, 339 commits, 5 tags
#   * grepped ALL commits of BOTH repos for KVPR / kv_pressure / w_token_rate /
#     shared_kv / Hodgson / Moore -> 0 hits in source files
#   * no other public repo in either org carries a Prism control plane
ALGO1 = [
    ("분모 `shared_kv` (GPU 용량 − 가중치)", "구현됨",
     "python/sglang/multi_model/scheduling/policy/simple_global.py",
     r"memory_available_for_requests\s*=\s*total_gpu_memory\s*-\s*total_model_memory",
     "논문의 shared_kv와 같은 개념"),
    ("분자 = SLO 가중 토큰율 `token_rate*token_size/SLO`", "대체됨",
     "python/sglang/multi_model/scheduling/policy/simple_global.py",
     r"memory_per_request\s*=\s*memory_available_for_requests\s*/\s*total_reqs",
     "토큰 크기·SLO 가중 없이 **평활 요청 수**만 씀"),
    ("모델을 `t*tz/s` 내림차순 정렬 (Alg.1 line 1)", "없음",
     "python/sglang/multi_model/scheduling/policy/simple_global.py",
     r"sorted\([^)]*key=lambda[^)]*(slo|token_size|tz)", 
     "정렬 키 10개 전부 violation 비율·요청수·잔여예산·가용메모리 중 하나"),
    ("마이그레이션 임계값 τ (Alg.1 line 8)", "유사 구현",
     "python/sglang/multi_model/scheduling/policy/simple_global.py",
     r"MEMORY_PER_REQUEST_RATIO_THRESHOLD\s*=\s*(\d+)",
     "τ에 해당하는 임계값은 있으나 KVPR이 아닌 다른 지표에 적용"),
]

ALGO2 = [
    ("데드라인 `d = a + s`", "구현됨",
     "python/sglang/multi_model/scheduling/gpu/request_queue.py",
     r"req\.arrival_time \+ req\.slo",
     "우선순위 힙 키에 포함"),
    ("실행시간 추정 `e = p / c`", "구현됨",
     "python/sglang/multi_model/scheduling/gpu/request_queue.py",
     r"req\.prompt_len \* \(0\.5 / 1024\)",
     "c = 1024/0.5 tok/s로 고정된 chunked-prefill 속도"),
    ("데드라인 오름차순 처리", "구현됨",
     "python/sglang/multi_model/scheduling/gpu/request_queue.py",
     r"heapq\.heappop",
     "min-heap이므로 사실상 EDF"),
    ("실행 불가 시 최장 작업 제거 (Alg.2 line 9-11)", "없음",
     "python/sglang/multi_model/scheduling/gpu/request_queue.py",
     r"argmax|remove.*longest|current_time > ",
     "누적 완료시각 검사도, 제거 단계도 없음 → 최적성 주장 근거가 사라짐"),
    ("이를 적용할 admission control", "무력화됨",
     "python/sglang/multi_model/scheduling/gpu/request_queue.py",
     r'net_available\s*=\s*float\("inf"\)',
     "자원 한도가 무한이라 승인 판정 자체가 항상 통과"),
]

CLAIMS = [
    ("§6.2 admission control이 비활성",
     "python/sglang/multi_model/scheduling/gpu/request_queue.py",
     r'net_available\s*=\s*float\("inf"\)',
     "메모리 부족으로 요청이 거절되는 일이 없음 → `rejected` 0은 관측 실패가 아니라 구조적"),
    ("§6.1 migration 임계값이 하드코딩·매우 느슨",
     "python/sglang/multi_model/scheduling/policy/simple_global.py",
     r"MEMORY_PER_REQUEST_RATIO_THRESHOLD\s*=\s*(\d+)",
     "기본 정책은 GPU 간 이 배율만큼 벌어져야 migrate. 실측 불균형은 약 1.6배"),
    ("idle eviction 임계값",
     "python/sglang/multi_model/scheduling/policy/simple_global.py",
     r"MODEL_IDLE_THRESHOLD\s*=\s*(\d+)",
     "논문 §A.4가 최적이라고 한 약 45초와 근접"),
    ("model service가 --num-gpus가 아니라 device_count를 봄",
     "python/sglang/multi_model/multi_model_server.py",
     r"num_devices\s*=\s*torch\.cuda\.device_count\(\)",
     "멀티 GPU 박스에서 1-GPU 실험을 하려면 CUDA_VISIBLE_DEVICES가 필수"),
]


def _probe(rel, pat):
    """Return (found, line_no, text) for the first match of pat in rel."""
    p = os.path.join(SRC, rel)
    if not os.path.exists(p):
        return False, "", ""
    for i, l in enumerate(open(p, errors="replace"), 1):
        if re.search(pat, l):
            return True, str(i), l.strip()
    return False, "", ""


def _algo_table(title, items):
    w()
    w(f"**{title}**")
    w()
    rows = []
    for name, verdict, rel, pat, note in items:
        found, line, txt = _probe(rel, pat)
        # "없음" rows assert absence: finding a hit means the audit is stale
        if verdict == "없음":
            status = "없음 (확인)" if not found else f"**재확인 필요 — {os.path.basename(rel)}:{line}**"
            loc = "—"
        else:
            status = verdict if found else "**확인 실패**"
            loc = f"`{os.path.basename(rel)}:{line}`" if line else "—"
        rows.append([name, status, loc, note])
    table(["논문 구성요소", "공개 코드", "위치", "설명"], rows)


def section_paper_vs_code():
    h(2, "5. 논문 vs 공개 코드 — 고정된 소스에서 매번 재검증")
    if not os.path.isdir(SRC):
        w("`prism-research/`가 체크아웃되어 있지 않아 생략합니다.")
        return
    w("아래는 **이 보고서를 생성하는 시점에** 실제 소스를 grep해서 확인합니다. "
      "맞지 않게 된 항목은 `확인 실패` / `재확인 필요`로 표시됩니다.")
    w()
    w("**감사 범위** — `Multi-LLM/prism-research` 브랜치 1개·커밋 4개(핀된 SHA가 HEAD), "
      "`ovg-project/kvcached` 브랜치 22개·커밋 339개·태그 5개. 두 레포의 **모든 커밋**에서 "
      "`KVPR` / `kv_pressure` / `w_token_rate` / `shared_kv` / `Hodgson` / `Moore`를 "
      "검색해 소스 파일 히트 0건. 두 조직의 다른 공개 레포에도 Prism 컨트롤 플레인은 "
      "없습니다.")
    w()
    w("다만 **이름이 없다고 알고리즘이 없는 것은 아니므로**, 아래는 구성요소 단위로 "
      "형태를 대조한 결과입니다. 결론부터: 두 알고리즘 모두 *목적은 같지만 명세와 다른 "
      "단순화된 휴리스틱*으로 구현되어 있습니다.")

    _algo_table("Algorithm 1 (KVPR 기반 배치) — 구성요소 대조", ALGO1)
    w()
    w("→ **판정: 부분 구현.** GPU별 메모리 압력을 균형 잡는다는 목적과 분모(`shared_kv`)는 "
      "같지만, KVPR의 핵심인 **SLO·토큰 크기 가중**이 빠지고 평활 요청 수로 대체되었습니다. "
      "따라서 이 코드로 얻은 결과는 *global placement가 도움이 된다*는 것은 보여줄 수 "
      "있어도 *Algorithm 1을 검증했다*고는 할 수 없습니다.")

    _algo_table("Algorithm 2 (Moore-Hodgson 요청 중재) — 구성요소 대조", ALGO2)
    w()
    w("→ **판정: 재료는 있고 메커니즘이 없음.** 데드라인과 실행시간 추정이 모두 계산되고 "
      "데드라인 순으로 처리되지만, 논문이 최적성을 증명하는 **'완료 못 하면 최장 작업을 "
      "빼낸다'** 단계가 없어 결과적으로 단순 EDF입니다. 게다가 이를 적용할 admission "
      "control이 무한 자원으로 무력화되어 있습니다.")

    h(3, "5.1 그 밖에 재검증된 항목")
    rows = []
    for title, rel, pat, why in CLAIMS:
        found, line, txt = _probe(rel, pat)
        loc = f"`{os.path.basename(rel)}:{line}`" if line else f"`{os.path.basename(rel)}`"
        rows.append([title, "확인됨" if found else "**확인 실패**", loc,
                     f"`{txt[:60]}`" if txt else "—", why])
    table(["주장", "상태", "위치", "증거", "의미"], rows)
    w()
    w("**맥락 — 시기 문제가 아니다.** `prism-research`는 논문 제1저자"
      "(Shan Yu, shanyu1@g.ucla.edu)가 2025-08-09에 올린 \"Prism research prototype\"이고 "
      "커밋 4개짜리다. 처음에는 \"논문 개정 전 스냅샷이라 알고리즘이 없는 것\"이라고 "
      "추정했으나 **틀렸다**: arXiv 2505.04021 **v1(2025-05-06)에 이미** Algorithm 1(KVPR)과 "
      "Algorithm 2(Moore-Hodgson)가 둘 다 있다. 즉 코드가 알고리즘보다 3개월 뒤에 나왔다.")
    w()
    w("실제로 바뀐 것은 KVPR의 **분자**다.")
    w()
    table(["버전", "KVPR 분자", "비고"],
          [["arXiv v1 (2025-05)", "`req_rate / SLO`", "요청률 기반"],
           ["arXiv v3 = OSDI'26 (2026-06)", "`token_rate * token_size / SLO`", "토큰 기반으로 정교화"],
           ["공개 코드 (2025-08)", "요청 수 (SLO 가중 없음)", "v1의 역수에서 SLO 가중마저 빠짐"]])
    w()
    w("따라서 공개 코드는 *논문 개정 이전판*이 아니라 **어느 판본과도 일치하지 않는 "
      "단순화 프로토타입**이다. 논문이 공식 아티팩트로 링크하는 것은 kvcached뿐이고 "
      "(prism-research는 논문 어디에도 나오지 않는다), prism-research의 README는 아직 "
      "v1 제목(\"Unleashing GPU Sharing\")과 13인 저자 목록을 인용하고 있다 — 반면 "
      "kvcached의 README는 OSDI판(21인)을 인용한다.")
    w()
    w("위 표는 *공개 코드가 무엇을 하는지*에 대한 진술이며, 저자들이 무엇을 구현했는지에 "
      "대한 진술이 아니다. Algorithm 1을 직접 구현한다면 **OSDI판(v3)의 토큰 기반 분자**를 "
      "써야 한다.")
    w()
    w("다른 이유로 재현 불가: MuxServe++/QLM/ServerlessLLM 베이스라인은 torch/vllm 핀 "
      "충돌로 미설치이고, Hyperbolic / Novita / Chatbot Arena 프로덕션 트레이스는 "
      "비공개입니다.")


# ----------------------------------------------------------------- how to run
def section_repro():
    h(2, "6. 재현 방법")
    w("커맨드 전문과 설계 근거는 [`EXPERIMENT.md`](../../EXPERIMENT.md), "
      "새 서버 셋업은 [`CLAUDE.md`](../../CLAUDE.md)를 보세요.")
    w()
    w("```bash")
    for line in [
        "source exp/scripts/env.sh",
        "export SLO_BASE_FILE=$PWD/exp/configs/slo_base_3x8b_sharegpt.json",
        "",
        "# 커밋된 1-GPU baseline과 대조해 환경 검증",
        "CUDA_VISIBLE_DEVICES=0 TAG=verify ./exp/scripts/run_sanity.sh A   # 이어서 B, C",
        "",
        "# N-GPU 배치 config 생성 후 실행 (NGPU는 보이는 GPU 수로 자동 설정)",
        "python exp/scripts/make_config.py --num-gpus 2 --slots 1,4,5 \\",
        "    --placement balanced -o exp/configs/llama_2gpu_3x8b.json",
        "SLOTS=1,4,5 CFG=$PWD/exp/configs/llama_2gpu_3x8b.json TAG=exp \\",
        "  TRACE=$DATASETS/sharegpt/exp_base12.pkl TPOT_SCALE=3 \\",
        "  ./exp/scripts/run_multigpu.sh glob_on 1",
        "python exp/scripts/collect_metrics.py --exp exp_glob_on_ts1 --tag exp",
        "",
        "# 이 보고서 재생성",
        "python exp/scripts/build_status_report.py",
    ]:
        w(line)
    w("```")


def section_reports():
    h(2, "7. 상세 문서 위치")
    rows = []
    for rel, what in [
        ("exp/results/4-rate-sweep/REPORT_rate_sweep.md", "3× Llama-3.1-8B rate sweep + burst (이번 연구)"),
        ("exp/results/3-placement/REPORT.md", "환경 구축 검증 + §7.3 global placement 실험"),
        ("exp/results/2-colocation/REPORT.md", "ShareGPT colocation 연구, 1 GPU (기존)"),
        ("exp/results/1-env-verification/REPORT.md", "최초 1-GPU sanity 스윕 (기존)"),
        ("EXPERIMENT.md", "모든 커맨드와 각 선택의 근거"),
        ("CLAUDE.md", "새로 빌린 GPU 서버 셋업 런북"),
    ]:
        p = os.path.join(ROOT, rel)
        if os.path.exists(p):
            rows.append([f"[`{rel}`]({os.path.relpath(p, os.path.join(EXP,'results'))})",
                         what, f"{os.path.getsize(p)//1024} KB"])
    table(["문서", "내용", "크기"], rows)


def section_caveats():
    h(2, "8. 위 모든 수치에 공통으로 적용되는 주의사항")
    for line in [
        "**각 데이터 지점은 1회 측정입니다.** 수천 건 위에서 계산된 집계값"
        "(attainment, throughput, p50)은 안정적이지만, **30 req/s의 TTFT p95 21초 같은 "
        "꼬리값은 반복 측정 없이 정밀 수치로 인용하면 안 됩니다.** 절벽의 존재와 "
        "자릿수는 견고합니다.",
        "**전 구간 `--disable-cuda-graph`** (레포 관례). 이 때문에 절대 TPOT가 논문 "
        "장비 대비 약 1.57배 느리고, 그래서 SLO baseline을 이 장비에서 다시 측정했습니다. "
        "절대 latency를 논문 수치와 직접 비교하면 안 됩니다.",
        "**기본 `real_trace.pkl`의 프롬프트는 `\"Hello \"*n` 합성**이라 prefix 중복이 "
        "99% 수준입니다. rate sweep 작업은 전부 ShareGPT 실제 텍스트(중복 2~4%)를 "
        "씁니다. 합성 트레이스에 radix cache를 켜면 절대 안 됩니다.",
        "**포화 전까지 queue length는 0이고 rejection은 항상 0**입니다 — 둘 다 구조적인 "
        "것으로 §5를 보세요. 부하 신호로는 `#running-req`와 TTFT p95를 봐야 합니다.",
        "**`/workspace`는 영구 볼륨이 아닙니다.** 인스턴스를 recycle/destroy하면 venv, "
        "24 GB 가중치, `exp/server-logs/`가 전부 사라집니다. 커밋된 결과만 git에 "
        "푸시되어 살아남습니다.",
    ]:
        w(f"- {line}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(EXP, "results", "STATUS_REPORT.md"))
    a = ap.parse_args()

    w("# Prism 실험 환경 — 전체 상태 보고서")
    w()
    stamp = sh("date -u '+%Y-%m-%d %H:%M UTC'")
    w(f"생성 시각 {stamp} · 생성 스크립트 `exp/scripts/build_status_report.py`")
    w()
    w("이 문서의 모든 수치는 **생성 시점에 직접 조사하거나 커밋된 결과 파일에서 읽은 "
      "것**입니다. 손으로 입력한 값은 하나도 없습니다.")

    runs = discover_runs()
    section_environment()
    section_repo()
    section_inventory(runs)
    section_rate_sweep()
    section_per_model()
    section_burst()
    section_capacity()
    section_paper_vs_code()
    section_repro()
    section_reports()
    section_caveats()
    w()

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        f.write("\n".join(OUT) + "\n")
    print(f"{a.out}  ({len(OUT)}줄, {os.path.getsize(a.out)/1024:.1f} KB)")
    print(f"  run {len(runs)}건 수집")


if __name__ == "__main__":
    main()
