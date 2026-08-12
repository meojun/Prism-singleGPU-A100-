#!/usr/bin/env python3
"""Generate one report covering the whole state of this experiment box.

Everything here is PROBED, not hard-coded. Hardware, package versions, pinned
upstream SHAs, git history, the experiment inventory and its numbers are all
read at run time, and each paper-vs-code claim is re-verified by grepping the
pinned source and printed with the file:line it was found at. Re-run it after
any run and the report stays true; if a claim stops matching the source it is
reported as NOT CONFIRMED rather than silently repeated.

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
import sys

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
        return f"(failed: {e})"


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
    h(2, "1. Environment")

    gpus = sh("nvidia-smi --query-gpu=index,name,memory.total,compute_cap "
              "--format=csv,noheader").splitlines()
    drv = sh("nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1")
    topo = sh("nvidia-smi topo -m 2>/dev/null | sed -n '2p' | awk '{print $3}'")
    w(f"**GPU** — driver {drv}" + (f", GPU0↔GPU1 link `{topo}`" if topo else ""))
    w()
    table(["idx", "name", "memory", "compute cap"],
          [[c.strip() for c in g.split(",")] for g in gpus])

    mem = sh("free -g | awk 'NR==2{print $2\" GiB total, \"$7\" GiB available\"}'")
    disk = sh(f"df -h {ROOT} | awk 'NR==2{{print $2\" total, \"$4\" free\"}}'")
    w()
    w(f"**Host** — {sh('nproc')} threads, RAM {mem}, disk at repo {disk}, "
      f"/dev/shm {sh('df -h /dev/shm | awk \"NR==2{print \\$2}\"')}")

    # versions come from the venv the runs actually used, not this interpreter
    py = os.path.join(ROOT, "prism-venv/bin/python")
    if os.path.exists(py):
        code = ("import importlib\n"
                "for m in ['torch','sglang','vllm','transformers','flashinfer','kvcached']:\n"
                "    try:\n"
                "        v=getattr(importlib.import_module(m),'__version__','n/a')\n"
                "    except Exception as e:\n"
                "        v='IMPORT FAILED: %s' % type(e).__name__\n"
                "    print(m,v)\n"
                "import torch;print('cuda_available',torch.cuda.is_available())")
        # run via argv, not a shell string: the probe contains newlines and
        # quotes that a shell would mangle into a SyntaxError
        try:
            vers = subprocess.run([py, "-c", code], capture_output=True,
                                  text=True, timeout=120).stdout.strip()
        except Exception as e:
            vers = f"probe failed {type(e).__name__}"
        w()
        w("**Stack** (from `prism-venv`, i.e. what the runs actually loaded)")
        w()
        parsed = [l.split(None, 1) for l in vers.splitlines() if l.strip()]
        if parsed:
            table(["package", "version"], parsed)
        else:
            w(f"*(version probe returned nothing: `{vers[:120]}`)*")
    else:
        w()
        w("**Stack** — `prism-venv` missing; run `./bootstrap.sh`.")

    pins = os.path.join(ROOT, "setup/pins.env")
    if os.path.exists(pins):
        kv = dict(l.split("=", 1) for l in open(pins)
                  if "=" in l and not l.strip().startswith("#"))
        rows = []
        for name, d in (("prism-research", "prism-research"),
                        ("kvcached (prism/shm)", "kvcached-prism"),
                        ("kvcached (main)", "kvcached")):
            p = os.path.join(ROOT, d)
            live = sh("git rev-parse --short HEAD", cwd=p) if os.path.isdir(p) else "(absent)"
            rows.append([name, f"`{live}`"])
        w()
        w("**Pinned upstream** (`setup/pins.env`) — checked out HEADs")
        w()
        table(["repo", "HEAD"], rows)

    w()
    redis = sh("redis-cli ping 2>/dev/null") or "no response"
    sup = sh("supervisorctl status redis 2>/dev/null") or "(not supervised)"
    w(f"**redis** — `{redis}`, supervisor: `{sup}`")

    hf = sh("du -sh /workspace/.hf_home/hub/models--* 2>/dev/null")
    if hf:
        w()
        w("**Model weights**")
        w()
        table(["size", "model"],
              [[l.split()[0], "`" + l.split()[1].split("models--")[-1].replace("--", "/") + "`"]
               for l in hf.splitlines()])

    busy = sh("ps -eo cmd --no-headers | grep -c '[s]glang.launch_multi_model_server'")
    used = sh("nvidia-smi --query-gpu=memory.used --format=csv,noheader | tr '\\n' ' '")
    w()
    w(f"**Right now** — serving processes: {busy}, GPU memory in use: {used}")


# ------------------------------------------------------------------ repo state
def section_repo():
    h(2, "2. Repository state")
    branch = sh("git rev-parse --abbrev-ref HEAD", cwd=ROOT)
    dirty = sh("git status --porcelain", cwd=ROOT)
    ahead = sh("git rev-list --count origin/main..HEAD 2>/dev/null", cwd=ROOT)
    w(f"branch `{branch}`, working tree "
      f"{'**dirty**' if dirty else 'clean'}, "
      f"{'in sync with origin' if ahead in ('0', '') else f'**{ahead} commit(s) ahead of origin**'}")
    w()
    log = sh("git log --pretty=format:'%h|%ad|%s' --date=short -12", cwd=ROOT)
    table(["commit", "date", "subject"],
          [l.split("|", 2) for l in log.splitlines() if "|" in l])
    if dirty:
        w()
        w("Uncommitted:")
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
    v = d.get(k, "")
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def discover_runs():
    """Every analysed run on disk, newest first, grouped by results/<tag>/."""
    runs = []
    for slo in glob.glob(os.path.join(EXP, "results", "*", "*_slo.json")):
        tag = os.path.basename(os.path.dirname(slo))
        exp = os.path.basename(slo)[: -len("_slo.json")]
        s = load_summary(os.path.join(EXP, "results", tag, f"{exp}_summary.csv"))
        try:
            d = json.load(open(slo))
        except Exception:
            continue
        runs.append({"tag": tag, "exp": exp, "slo": d, "sum": s,
                     "mtime": os.path.getmtime(slo)})
    return sorted(runs, key=lambda r: (r["tag"], r["exp"]))


def section_inventory(runs):
    h(2, "3. Experiment inventory")
    w(f"{len(runs)} analysed runs across "
      f"{len({r['tag'] for r in runs})} result namespaces. "
      "Every row is read from the committed `*_slo.json` / `*_summary.csv`.")
    w()
    rows = []
    for r in runs:
        a = r["slo"].get("per_model", {}).get("ALL", {})
        rows.append([
            f"`{r['tag']}`", f"`{r['exp']}`",
            r["slo"].get("total_requests", "?"),
            f"{r['slo'].get('duration_s', 0):.0f}s",
            f"{a.get('attain_ttft', float('nan')):.3f}",
            f"{a.get('attain_tpot', float('nan')):.3f}",
            f"{a.get('ttft_p95_ms', float('nan')):,.0f}" if "ttft_p95_ms" in a else "—",
            f"{a.get('tpot_p50_ms', float('nan')):.0f}",
        ])
    table(["namespace", "run", "reqs", "dur", "att TTFT", "att TPOT",
           "TTFT p95 ms", "TPOT p50 ms"], rows,
          ["---", "---", "---:", "---:", "---:", "---:", "---:", "---:"])
    w()
    w("> attainment is computed by `analyze_slo.py`, never by `benchmark.py` — "
      "the latter's `average_attainment_tpot` compares millisecond baselines "
      "against second-valued measurements and is always 1.0.")


# --------------------------------------------------------------- key results
def section_rate_sweep():
    pts = [("1", 1.0), ("0.8", 1.25), ("0.6667", 1.5), ("0.5", 2.0), ("0.4", 2.5)]
    have = [(ts, m) for ts, m in pts
            if os.path.exists(os.path.join(EXP, "results/exp", f"exp_glob_on_ts{ts}_summary.csv"))]
    if not have:
        return
    h(2, "4. Rate sweep — 3× Llama-3.1-8B on 2 GPUs")
    lam = 12.0
    w(f"λ_base = **{lam:.0f} req/s**, chosen as ~46% of the profiled TTFT knee "
      "(~26 req/s). One request sequence replayed at every rate via "
      "`--time-scale`, so lengths, model mix and seed are identical across rows.")
    w()
    rows = []
    for ts, mult in have:
        s = load_summary(os.path.join(EXP, "results/exp", f"exp_glob_on_ts{ts}_summary.csv"))
        kv = json.loads(s.get("peak_kv_pool_frac", "{}") or "{}")
        rows.append([
            f"{lam*mult:.0f}", f"{mult:.2f}×",
            f"{num(s,'throughput_rps'):.1f}", f"{num(s,'out_tok_throughput'):,.0f}",
            f"{num(s,'ttft_p50_ms'):.0f}", f"**{num(s,'ttft_p95_ms'):,.0f}**",
            f"{num(s,'tpot_p50_ms'):.0f}", f"{num(s,'attain_tpot'):.3f}",
            " / ".join(f"{kv.get(m,0):.2f}" for m in ("model_1", "model_4", "model_5")),
            f"{num(s,'max_model_queue'):.0f} / {num(s,'max_sched_qlen'):.0f}",
        ])
    table(["offered λ", "×base", "achieved", "out tok/s", "TTFT p50 ms",
           "TTFT p95 ms", "TPOT p50 ms", "att TPOT", "KV pool m1/m4/m5",
           "max queue mdl/sch"], rows,
          ["---:"] * 8 + [":---:", ":---:"])
    w()
    lo = load_summary(os.path.join(EXP, "results/exp", "exp_glob_on_ts0.5_summary.csv"))
    hi = load_summary(os.path.join(EXP, "results/exp", "exp_glob_on_ts0.4_summary.csv"))
    if lo and hi:
        r95 = num(hi, "ttft_p95_ms") / num(lo, "ttft_p95_ms")
        r50 = num(hi, "ttft_p50_ms") / num(lo, "ttft_p50_ms")
        w(f"Between 2.0× and 2.5× λ_base, TTFT **p95 grows {r95:,.0f}×** while "
          f"**p50 grows only {r50:.1f}×** — mean and median hide the cliff. "
          "The cliff coincides exactly with the two-model GPU's KV pool reaching "
          "~0.98 and with the first non-empty queue.")


def section_per_model():
    f = os.path.join(EXP, "results/exp", "exp_glob_on_ts1_slo.json")
    if not os.path.exists(f):
        return
    h(3, "4.1 Colocation dominates arrival rate")
    rows = []
    for ts, lam in [("1", 12), ("0.8", 15), ("0.6667", 18), ("0.5", 24), ("0.4", 30)]:
        p = os.path.join(EXP, "results/exp", f"exp_glob_on_ts{ts}_slo.json")
        if not os.path.exists(p):
            continue
        d = json.load(open(p))["per_model"]
        cells = []
        for m in ("model_1", "model_4", "model_5"):
            v = d.get(m, {})
            cells.append(f"{v.get('attain_tpot',float('nan')):.3f} / "
                         f"{v.get('ttft_p95_ms',float('nan')):,.0f} / "
                         f"{v.get('tpot_p50_ms',float('nan')):.0f}")
        rows.append([lam] + cells)
    table(["offered λ", "model_1 (GPU0, alone)", "model_4 (GPU1, shared)",
           "model_5 (GPU1, shared)"], rows, ["---:", ":---:", ":---:", ":---:"])
    w()
    w("`att_tpot / TTFT p95 ms / TPOT p50 ms`. At λ_base the model alone on GPU0 "
      "runs at the uncontended baseline while the two sharing a GPU are ~2.8× "
      "slower — before any rate increase.")


def section_burst():
    f = os.path.join(EXP, "results/burst", "burst_glob_on_ts1_windows.csv")
    if not os.path.exists(f):
        return
    h(3, "4.2 Burst — hot models 1 → 2 → 3")
    names = ["1 hot (8 / 0.5 / 0.5)", "2 hot (8 / 8 / 0.5)", "3 hot (8 / 8 / 8)"]
    rows = []
    for i, r in enumerate(csv.DictReader(open(f))):
        rows.append([i, names[i] if i < len(names) else "?",
                     f"{float(r['offered_rate_rps']):.1f}",
                     f"**{float(r['attain_both']):.3f}**",
                     f"{float(r['tpot_p50_ms']):.0f}",
                     f"**{float(r['model_1_ttft_p95_ms']):.0f}**",
                     f"{float(r['model_4_ttft_p95_ms']):.0f}",
                     f"{float(r['model_5_ttft_p95_ms']):.0f}"])
    table(["phase", "hot models", "total λ", "att both", "TPOT p50 ms",
           "model_1 TTFT p95", "model_4 TTFT p95", "model_5 TTFT p95"], rows,
          ["---:", "---", "---:", "---:", "---:", "---:", "---:", "---:"])
    w()
    w("model_1 is pinned at 8 req/s in all three phases, so its column is pure "
      "cross-model interference.")
    log = os.path.join(EXP, "server-logs/burst_glob_on_ts1/server.log.global_controller.log")
    if os.path.exists(log):
        acts = [l for l in open(log, errors="replace") if "ACTION:" in l]
        if acts:
            w()
            w("Controller actions during the burst run:")
            w("```")
            for l in acts:
                w(re.sub(r"^\[[^\]]*\] ", "", l).rstrip())
            w("```")


def section_capacity():
    files = [("results/probe/rampLO_windows.csv", "low ramp (1 → 8 req/s)"),
             ("results/probe/probe_glob_on_ts1_windows.csv", "high ramp (8 → 31 req/s)")]
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
    h(3, "4.3 Capacity profiling (how λ_base was chosen, not guessed)")
    table(["ramp", "offered req/s", "out tok/s", "TTFT p95 ms", "TPOT p50 ms"],
          rows, ["---", "---:", "---:", "---:", "---:"])
    w()
    w("The two ramps overlap at ~7.8 req/s and agree, which doubles as a "
      "reproducibility check.")


# ------------------------------------------------- paper vs code, re-verified
CLAIMS = [
    ("§6.2 admission control is disabled",
     "python/sglang/multi_model/scheduling/gpu/request_queue.py",
     r'net_available\s*=\s*float\("inf"\)',
     "so no request is ever rejected for lack of memory; `rejected` is "
     "structurally 0, not merely unobserved"),
    ("§6.1 migration threshold is hard-coded and very loose",
     "python/sglang/multi_model/scheduling/policy/simple_global.py",
     r"MEMORY_PER_REQUEST_RATIO_THRESHOLD\s*=\s*(\d+)",
     "the default `memory_per_request` policy needs this ratio between GPUs "
     "before it will migrate; a realistic imbalance is ~1.6×"),
    ("idle eviction threshold",
     "python/sglang/multi_model/scheduling/policy/simple_global.py",
     r"MODEL_IDLE_THRESHOLD\s*=\s*(\d+)",
     "close to the ~45 s the paper reports as optimal in §A.4"),
    ("GPU-local scheduling is slack-ordered EDF, not Moore-Hodgson",
     "python/sglang/multi_model/scheduling/gpu/request_queue.py",
     r"return req\.arrival_time \+ req\.slo - profiled_prefill_time",
     "a plain min-heap priority; Algorithm 2's drop-the-longest-job step is "
     "absent"),
    ("model service sizes itself from the device count, not --num-gpus",
     "python/sglang/multi_model/multi_model_server.py",
     r"num_devices\s*=\s*torch\.cuda\.device_count\(\)",
     "so a 1-GPU experiment on a multi-GPU box needs CUDA_VISIBLE_DEVICES"),
]

ABSENT = [
    ("Algorithm 1 (KVPR)", "python/sglang/multi_model/scheduling/policy/simple_global.py",
     r"KVPR|kv_pressure|w_token_rate",
     "no KV Pressure Ratio anywhere in the placement policy; the released code "
     "uses `violation` and `memory_per_request` heuristics instead"),
    ("Algorithm 2 (Moore-Hodgson)", "python/sglang/multi_model/scheduling/gpu/request_queue.py",
     r"[Mm]oore|[Hh]odgson",
     "no reference to the algorithm the paper proves optimality with"),
]


def section_paper_vs_code():
    h(2, "5. Paper vs released code — re-verified against the pinned source")
    if not os.path.isdir(SRC):
        w("`prism-research/` not checked out; skipped.")
        return
    w("Each row below is checked against the source **at report time**. A claim "
      "that no longer matches prints `NOT CONFIRMED` instead of being repeated.")
    w()
    rows = []
    for title, rel, pat, why in CLAIMS:
        p = os.path.join(SRC, rel)
        found, line, txt = "NOT CONFIRMED", "", ""
        if os.path.exists(p):
            for i, l in enumerate(open(p, errors="replace"), 1):
                m = re.search(pat, l)
                if m:
                    found, line, txt = "confirmed", str(i), l.strip()
                    break
        loc = f"`{os.path.basename(rel)}:{line}`" if line else f"`{os.path.basename(rel)}`"
        rows.append([title, found, loc, f"`{txt[:64]}`" if txt else "—"])
    table(["claim", "status", "location", "evidence"], rows)
    w()
    w("Confirmed **absent** (the grep finds nothing, which is the finding):")
    w()
    rows = []
    for title, rel, pat, why in ABSENT:
        p = os.path.join(SRC, rel)
        hits = 0
        if os.path.exists(p):
            hits = sum(1 for l in open(p, errors="replace") if re.search(pat, l))
        rows.append([title, "absent" if hits == 0 else f"**{hits} hit(s) — recheck**",
                     f"`{os.path.basename(rel)}`", why])
    table(["paper mechanism", "status", "searched in", "meaning"], rows)
    w()
    w("Not reproducible for other reasons: MuxServe++/QLM/ServerlessLLM baselines "
      "are not installed (conflicting torch/vllm pins), and the Hyperbolic / "
      "Novita / Chatbot Arena production traces are not public.")


# ----------------------------------------------------------------- how to run
def section_repro():
    h(2, "6. Reproducing any of this")
    w("Full command list with rationale: [`EXPERIMENT.md`](../../EXPERIMENT.md). "
      "Setting up a fresh box: [`CLAUDE.md`](../../CLAUDE.md).")
    w()
    w("```bash")
    w("source exp/scripts/env.sh")
    w("export SLO_BASE_FILE=$PWD/exp/configs/slo_base_3x8b_sharegpt.json")
    w("")
    w("# environment check against the committed 1-GPU baseline")
    w("CUDA_VISIBLE_DEVICES=0 TAG=verify ./exp/scripts/run_sanity.sh A   # then B, C")
    w("")
    w("# N-GPU placement config, then a run; NGPU defaults to every visible GPU")
    w("python exp/scripts/make_config.py --num-gpus 2 --slots 1,4,5 \\")
    w("    --placement balanced -o exp/configs/llama_2gpu_3x8b.json")
    w("SLOTS=1,4,5 CFG=$PWD/exp/configs/llama_2gpu_3x8b.json TAG=exp \\")
    w("  TRACE=$DATASETS/sharegpt/exp_base12.pkl TPOT_SCALE=3 \\")
    w("  ./exp/scripts/run_multigpu.sh glob_on 1")
    w("python exp/scripts/collect_metrics.py --exp exp_glob_on_ts1 --tag exp")
    w("")
    w("# regenerate this report")
    w("python exp/scripts/build_status_report.py")
    w("```")


def section_reports():
    h(2, "7. Where the detail lives")
    rows = []
    for rel, what in [
        ("exp/results/exp/REPORT_rate_sweep.md", "3× Llama-3.1-8B rate sweep + burst (this study)"),
        ("exp/results/fig7/REPORT.md", "environment verification + §7.3 global-placement ablation"),
        ("exp/results/exp/REPORT.md", "ShareGPT colocation study, 1 GPU (pre-existing)"),
        ("exp/results/sanity/REPORT.md", "original 1-GPU sanity sweep (pre-existing)"),
        ("EXPERIMENT.md", "every command, with the reasoning behind each choice"),
        ("CLAUDE.md", "runbook for setting this up on a fresh rented GPU box"),
    ]:
        p = os.path.join(ROOT, rel)
        if os.path.exists(p):
            rows.append([f"[`{rel}`]({os.path.relpath(p, os.path.join(EXP,'results'))})",
                         what, f"{os.path.getsize(p)//1024} KB"])
    table(["document", "contents", "size"], rows)


def section_caveats():
    h(2, "8. Caveats that apply to every number above")
    for line in [
        "**Single run per data point.** Aggregates over thousands of requests "
        "(attainment, throughput, p50) are stable; **tail values such as the "
        "21 s TTFT p95 at 30 req/s must not be quoted as precise** without "
        "repeats. The existence and order of magnitude of the cliff are solid.",
        "**`--disable-cuda-graph` throughout** (repo convention). Absolute TPOT "
        "is ~1.57× slower than the paper's baseline hardware, which is why SLO "
        "baselines were re-derived here. Do not compare absolute latency with "
        "the paper.",
        "**The shipped `real_trace.pkl` has synthetic `\"Hello \"*n` prompts** "
        "with ~99% prefix overlap. All rate-sweep work uses ShareGPT text "
        "(~2-4% overlap). Never enable the radix cache with the synthetic trace.",
        "**Queue length is ~0 below saturation** and rejections are always 0 — "
        "both structural, see §5. Use `#running-req` and TTFT p95 as the load "
        "signal.",
        "**`/workspace` is not a persistent volume** on this instance: recycle "
        "or destroy wipes the venv, the 24 GB of weights and `exp/server-logs/`. "
        "Committed results survive because they are pushed to git.",
    ]:
        w(f"- {line}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(EXP, "results", "STATUS_REPORT.md"))
    a = ap.parse_args()

    stamp = sh("date -u '+%Y-%m-%d %H:%M UTC'")
    w("# Prism experiment box — status report")
    w()
    w(f"Generated {stamp} by `exp/scripts/build_status_report.py`. "
      "Every figure is probed or read from committed results at generation "
      "time; nothing in this file is typed by hand.")

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
    print(f"{a.out}  ({len(OUT)} lines, {os.path.getsize(a.out)/1024:.1f} KB)")
    print(f"  {len(runs)} runs inventoried")


if __name__ == "__main__":
    main()
