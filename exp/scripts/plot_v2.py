#!/usr/bin/env python3
"""paper-faithful-v2 의 그림. 표 대신 한눈에 읽히는 것이 목적이다.

    python plot_v2.py --base exp/results/paper-faithful-v2 -o <base>/plots

각 그림은 "독자가 무엇을 해야 하는가" 로 형태를 정했다:
  fig1  교차 (기준선 대비 이득의 bursty-steady 차) .... 기준선 위/아래 -> diverging bar
  fig2  joint 달성률 대 부하, 워크로드별 ............... 추세 + 계열 구분 -> multi-line
  fig3  TTFT 대 TPOT 달성률 ........................... 어느 것이 병목인가 -> multi-line
  fig4  c_i 추정기 4종 대 모델 ........................ 크기 비교 -> grouped bar
  fig5  calibration 곡선 .............................. 추세 2종 -> 2 panel (이중축 금지)
  fig6  스케줄러 동작 횟수 ............................ 크기 비교 -> grouped bar
  fig7  ablation ...................................... 크기 비교 -> grouped bar

색은 검증된 categorical 팔레트에서 고정 순서로만 배정한다
(validate_palette.js, light 모드 전 항목 PASS). aqua/yellow 는 표면 대비가
3:1 미만이라 "relief" 규칙에 따라 모든 막대에 값 라벨을 붙인다.
"""
import argparse
import csv
import json
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

# --- 검증된 팔레트 (references/palette.md, light) --------------------------
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"   # blue, orange, aqua, yellow
C7 = "#4a3aa7"                                                 # violet
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8983"
SURFACE, GRID = "#fcfcfb", "#e3e2dd"

SYS = {                      # 계열 -> 색은 엔티티에 고정, 순위에 따라 바뀌지 않음
    "released-prototype": (C1, "Released prototype"),
    "paper-faithful":     (C2, "Paper-faithful Prism"),
    "paper-faithful-v3":  (C2, "Paper-faithful Prism v3"),
    "paper-alg1-only":    (C3, "Algorithm 1 only"),
    "paper-alg2-only":    (C4, "Algorithm 2 only"),
}
PAPER_SYSTEM = "paper-faithful"

# 한글 라벨을 쓰므로 한글 글리프가 있는 폰트를 강제한다. 없으면 matplotlib 는
# 경고만 내고 모든 한글을 두부(□)로 그린다 -- 조용히 읽을 수 없는 그림이 된다.
#   apt-get install -y fonts-nanum && fc-cache -f
import matplotlib.font_manager as fm
_KO = None
for _cand in ("NanumGothic", "NanumBarunGothic", "Noto Sans CJK KR", "Malgun Gothic"):
    if any(_cand.lower() in f.name.lower() for f in fm.fontManager.ttflist):
        _KO = _cand
        break
if _KO is None:
    raise SystemExit("FATAL: 한글 폰트를 찾지 못했다. apt-get install -y fonts-nanum")

plt.rcParams.update({
    "font.family": _KO,
    "axes.unicode_minus": False,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "axes.edgecolor": GRID, "axes.linewidth": 1.0,
    "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
    "grid.color": GRID, "grid.linewidth": 1.0,
    "legend.frameon": False, "legend.fontsize": 9.5,
    "axes.spines.top": False, "axes.spines.right": False,
})


def style(ax, ylabel=None, xlabel=None, title=None):
    ax.grid(axis="y", alpha=0.9, zorder=0)
    ax.set_axisbelow(True)
    if ylabel: ax.set_ylabel(ylabel, color=INK2)
    if xlabel: ax.set_xlabel(xlabel, color=INK2)
    if title:  ax.set_title(title, color=INK, pad=9, loc="left", fontweight="bold")


def num(v):
    try:
        f = float(v)
        return f if f == f else None
    except (TypeError, ValueError):
        return None


def load_summary(base):
    p = os.path.join(base, "processed", "summary.csv")
    if not os.path.exists(p):
        return []
    with open(p) as f:
        return list(csv.DictReader(f))


def idx(rows):
    return {(r["system"], r["workload"], r["rate"]): r for r in rows}


def rates_of(rows, systems=None):
    systems = systems or ("released-prototype", PAPER_SYSTEM)
    rs = {r["rate"] for r in rows if r["system"] in systems}
    return sorted(rs, key=float)


# ------------------------------------------------------------------ fig 1
def fig_crossover(rows, out):
    """헤드라인. 기준선 대비 Prism 이득의 (bursty - steady). 0 이 기준."""
    I = idx(rows)
    xs, vals = [], []
    for rate in rates_of(rows):
        try:
            bs = num(I[("released-prototype", "steady", rate)]["joint_attainment"])
            ps = num(I[(PAPER_SYSTEM, "steady", rate)]["joint_attainment"])
            bb = num(I[("released-prototype", "bursty", rate)]["joint_attainment"])
            pb = num(I[(PAPER_SYSTEM, "bursty", rate)]["joint_attainment"])
        except KeyError:
            continue
        if None in (bs, ps, bb, pb) or not bs or not bb:
            continue
        xs.append(rate)
        vals.append(((pb - bb) / bb - (ps - bs) / bs) * 100)
    if not vals:
        return
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    # diverging: 두 극(따뜻/차가움) + 중립 0선. 순위가 아니라 부호가 색을 정한다.
    colors = [C2 if v > 0 else C1 for v in vals]
    b = ax.bar(range(len(xs)), vals, width=0.55, color=colors, zorder=3,
               linewidth=1.6, edgecolor=SURFACE)
    ax.axhline(0, color=INK2, lw=1.4, zorder=4)
    for i, (r, v) in enumerate(zip(xs, vals)):
        ax.annotate(f"{v:+.1f}%p", (i, v), ha="center",
                    va="bottom" if v > 0 else "top",
                    xytext=(0, 5 if v > 0 else -6), textcoords="offset points",
                    color=INK, fontsize=10, fontweight="bold")
    ax.set_xticks(range(len(xs)))
    ax.set_xticklabels([f"{x} req/s" for x in xs])
    lo, hi = min(vals), max(vals)
    pad = (hi - lo) * 0.24
    ax.set_ylim(lo - pad, hi + pad)
    style(ax, ylabel="bursty 이득 - steady 이득 (%p)", xlabel="유입률",
          title="도착 타이밍만 바꿨을 때 Prism 의 상대 이득 변화")
    ax.annotate("위쪽 = shifting-bursty 가 Prism 에 유리", xy=(0.015, 0.955),
                xycoords="axes fraction", color=C2, fontsize=9)
    ax.annotate("아래쪽 = 불리", xy=(0.015, 0.045), xycoords="axes fraction",
                color=C1, fontsize=9)
    fig.text(0.5, -0.02, "동일한 request set · 동일한 모델별 요청 수 · 동일한 평균 offered load. "
             "도착 시각만 다름.", ha="center", color=INK3, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


# ------------------------------------------------------------------ fig 2
def fig_panel_lines(rows, metric, title, ylabel, out, pct=False, logy=False,
                    systems=None):
    systems = systems or ("released-prototype", PAPER_SYSTEM)
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2), sharey=True)
    for ax, wl, wln in zip(axes, ("steady", "bursty"), ("Steady", "Shifting-bursty")):
        for sysname in systems:
            pts = sorted((float(r["rate"]), num(r[metric])) for r in rows
                         if r["system"] == sysname and r["workload"] == wl
                         and num(r[metric]) is not None)
            if not pts:
                continue
            color, label = SYS[sysname]
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "-o",
                    color=color, lw=2, ms=8, label=label, zorder=3,
                    markeredgecolor=SURFACE, markeredgewidth=1.6)
            x, y = pts[-1]
            ax.annotate(label, (x, y), xytext=(6, 0), textcoords="offset points",
                        color=color, fontsize=9, va="center")
        style(ax, xlabel="유입률 (req/s)", title=wln)
        if logy:
            ax.set_yscale("log")
    axes[0].set_ylabel(ylabel, color=INK2)
    if pct:
        axes[0].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0].legend(loc="upper right")
    fig.suptitle(title, color=INK, fontweight="bold", x=0.008, ha="left", y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


# ------------------------------------------------------------------ fig 3
def fig_bottleneck(rows, out):
    """무엇이 병목인가. TTFT 달성률은 평평하고 TPOT 이 무너진다."""
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2), sharey=True)
    for ax, wl, wln in zip(axes, ("steady", "bursty"), ("Steady", "Shifting-bursty")):
        for metric, color, label in (("ttft_attainment", C3, "TTFT 달성률"),
                                     ("tpot_attainment", C7, "TPOT 달성률")):
            pts = sorted((float(r["rate"]), num(r[metric])) for r in rows
                         if r["system"] == "released-prototype" and r["workload"] == wl
                         and num(r[metric]) is not None)
            if not pts:
                continue
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "-o", color=color,
                    lw=2, ms=8, label=label, zorder=3,
                    markeredgecolor=SURFACE, markeredgewidth=1.6)
            x, y = pts[-1]
            ax.annotate(label, (x, y), xytext=(6, 0), textcoords="offset points",
                        color=color, fontsize=9, va="center")
        ax.set_ylim(0, 1.05)
        style(ax, xlabel="유입률 (req/s)", title=wln)
    axes[0].set_ylabel("SLO 달성률", color=INK2)
    axes[0].yaxis.set_major_formatter(PercentFormatter(1.0))
    fig.suptitle("이 실험은 TPOT 바운드다 — TTFT 는 거의 항상 충족된다 (released prototype)",
                 color=INK, fontweight="bold", x=0.008, ha="left", y=1.02)
    fig.text(0.5, -0.03, "joint 달성률은 사실상 TPOT 달성률이다. "
             "그래서 TTFT 를 최적화하는 Algorithm 2 의 이득이 대표 지표에 나타나지 않는다.",
             ha="center", color=INK3, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


# ------------------------------------------------------------------ fig 4
SHORT = {"model_1": "Llama-3.2-1B", "model_2": "Qwen2.5-1.5B",
         "model_3": "Llama-3.2-3B", "model_4": "Qwen2.5-3B",
         "model_5": "Llama-3.1-8B", "model_6": "Qwen2.5-7B"}


def fig_ci(base, out):
    files = sorted(glob.glob(os.path.join(base, "sanity", "profiling", "model_*.json")))
    if not files:
        return
    models, series = [], {k: [] for k in ("E1", "E2", "E3solo", "E3sat")}
    for f in files:
        d = json.load(open(f))
        e = d["c_i_estimators"]
        models.append(d["model"])
        series["E1"].append(e["E1_ratio_sum_p_over_sum_ttft"] or 0)
        series["E2"].append(e["E2_regression_slope"] or 0)
        series["E3solo"].append(e["E3_prefill_solo"] or 0)
        series["E3sat"].append(e["E3_prefill_saturated"] or 0)
    labels = {"E1": "E1 비율 Σp/Σttft", "E2": "E2 회귀 기울기",
              "E3solo": "E3 실측 prefill (단독)", "E3sat": "E3 실측 prefill (포화) — 사용값"}
    colors = {"E1": C1, "E2": C2, "E3solo": C3, "E3sat": C7}
    n, w = len(models), 0.20
    fig, ax = plt.subplots(figsize=(10.4, 4.6))
    for i, k in enumerate(("E1", "E2", "E3solo", "E3sat")):
        xs = [j + (i - 1.5) * w for j in range(n)]
        ax.bar(xs, [v / 1000 for v in series[k]], width=w * 0.88, color=colors[k],
               label=labels[k], zorder=3, linewidth=1.4, edgecolor=SURFACE)
        # 표면 대비 3:1 미만인 슬롯이 있어 relief 규칙: 값 라벨을 붙인다
        for x, v in zip(xs, series[k]):
            ax.annotate(f"{v/1000:.0f}", (x, v / 1000), ha="center", va="bottom",
                        xytext=(0, 2), textcoords="offset points",
                        color=INK2, fontsize=7)
    ax.axhline(4.214, color=INK2, lw=1.6, ls="--", zorder=4)
    # 오른쪽에 여백을 만들어 기준선 라벨이 막대 위에 얹히지 않게 한다
    ax.set_xlim(-0.6, n - 1 + 0.95)
    ax.annotate("v1 이 쓴 값\n4,214 tok/s", (n - 1 + 0.42, 4.214), xytext=(0, 4),
                textcoords="offset points", color=INK, fontsize=8.5, ha="left",
                va="bottom", linespacing=1.3)
    ax.set_xticks(range(n))
    ax.set_xticklabels([f"{m}\n{SHORT.get(m, '')}" for m in models], fontsize=9)
    ax.legend(loc="upper right", ncol=2)
    style(ax, ylabel="c_i  (1,000 tok/s)", xlabel="",
          title="c_i 추정기 4종 — 무엇을 재느냐에 따라 10배까지 갈린다")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


# ------------------------------------------------------------------ fig 5
def fig_calibration(base, out):
    pts = []
    for p in sorted(glob.glob(os.path.join(base, "sanity", "calibration", "rate_*", "metrics.json")),
                    key=lambda q: float(os.path.basename(os.path.dirname(q)).split("_")[1])):
        d = json.load(open(p))
        pts.append((float(os.path.basename(os.path.dirname(p)).split("_")[1]), d))
    if not pts:
        return
    x = [p[0] for p in pts]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0))
    axes[0].plot(x, [p[1]["joint_attainment"] for p in pts], "-o", color=C1, lw=2, ms=8,
                 markeredgecolor=SURFACE, markeredgewidth=1.6, zorder=3)
    axes[0].set_ylim(0, 1.05)
    axes[0].yaxis.set_major_formatter(PercentFormatter(1.0))
    style(axes[0], ylabel="Joint SLO 달성률", xlabel="유입률 (req/s)", title="달성률")
    axes[1].plot(x, [p[1]["throughput_req_s"] for p in pts], "-o", color=C1, lw=2, ms=8,
                 markeredgecolor=SURFACE, markeredgewidth=1.6, zorder=3, label="완료 처리율")
    axes[1].plot(x, [p[1]["goodput_req_s"] for p in pts], "-o", color=C2, lw=2, ms=8,
                 markeredgecolor=SURFACE, markeredgewidth=1.6, zorder=3, label="Goodput (두 SLO 모두 만족)")
    axes[1].plot(x, x, ls=":", color=INK3, lw=1.4, zorder=2)
    axes[1].annotate("유입률 = 처리율", (x[-1], x[-1]), xytext=(-4, 6),
                     textcoords="offset points", color=INK3, fontsize=9, ha="right")
    axes[1].legend(loc="upper left")
    style(axes[1], ylabel="req/s", xlabel="유입률 (req/s)", title="처리율")
    fig.suptitle("부하 calibration — 처리율은 끝까지 유입률을 따라간다. 무너지는 것은 달성률뿐",
                 color=INK, fontweight="bold", x=0.008, ha="left", y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


# ------------------------------------------------------------------ fig 6
def fig_scheduler(rows, out):
    I = idx(rows)
    rates = rates_of(rows)
    cats = [("migrations", "마이그레이션", C1),
            ("activations", "활성화", C2),
            ("idle_evictions", "유휴 축출", C3)]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2), sharey=True)
    for ax, wl, wln in zip(axes, ("steady", "bursty"), ("Steady", "Shifting-bursty")):
        w = 0.26
        for i, (key, label, color) in enumerate(cats):
            vals = []
            for rate in rates:
                r = I.get((PAPER_SYSTEM, wl, rate))
                if key == "migrations":
                    v = (num(r.get("migrations_alg1")) or 0) + (num(r.get("migrations_proto")) or 0) if r else 0
                else:
                    v = (num(r.get(key)) or 0) if r else 0
                vals.append(v)
            xs = [j + (i - 1) * w for j in range(len(rates))]
            ax.bar(xs, vals, width=w * 0.88, color=color, label=label if wl == "steady" else None,
                   zorder=3, linewidth=1.4, edgecolor=SURFACE)
            for xx, v in zip(xs, vals):
                if v > 0:
                    ax.annotate(f"{v:.0f}", (xx, v), ha="center", va="bottom",
                                xytext=(0, 2), textcoords="offset points",
                                color=INK2, fontsize=8)
        ax.set_xticks(range(len(rates)))
        ax.set_xticklabels([f"{r}" for r in rates])
        style(ax, xlabel="유입률 (req/s)", title=wln)
    axes[0].set_ylabel("횟수 (런당)", color=INK2)
    _h, _l = axes[0].get_legend_handles_labels()
    fig.legend(_h, _l, loc="upper center", bbox_to_anchor=(0.5, 0.995),
               ncol=3, frameon=False)
    fig.suptitle("축출과 활성화는 bursty 에서만 일어난다 (Paper-faithful Prism)",
                 color=INK, fontweight="bold", x=0.008, ha="left", y=1.09)
    fig.text(0.5, -0.03, "페어링된 워크로드가 의도한 메커니즘을 분리해 냈다는 증거. "
             "steady 에는 회수할 유휴 모델이 없다.", ha="center", color=INK3, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


# ------------------------------------------------------------------ fig 7
def fig_ablation(rows, out):
    order = ["released-prototype", "paper-alg1-only", "paper-alg2-only", "paper-faithful"]
    I = idx(rows)
    rates = sorted({r["rate"] for r in rows if r["system"] == "paper-alg1-only"}, key=float)
    if not rates:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2), sharey=True)
    for ax, wl, wln in zip(axes, ("steady", "bursty"), ("Steady", "Shifting-bursty")):
        w = 0.20
        for i, sysname in enumerate(order):
            color, label = SYS[sysname]
            vals = [num((I.get((sysname, wl, rate)) or {}).get("joint_attainment")) or 0
                    for rate in rates]
            xs = [j + (i - 1.5) * w for j in range(len(rates))]
            ax.bar(xs, vals, width=w * 0.88, color=color,
                   label=label if wl == "steady" else None, zorder=3,
                   linewidth=1.4, edgecolor=SURFACE)
            for xx, v in zip(xs, vals):
                ax.annotate(f"{v:.2f}", (xx, v), ha="center", va="bottom",
                            xytext=(0, 2), textcoords="offset points",
                            color=INK2, fontsize=7.5)
        ax.set_xticks(range(len(rates)))
        ax.set_xticklabels([f"{r} req/s" for r in rates])
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        style(ax, xlabel="", title=wln)
    axes[0].set_ylabel("Joint SLO 달성률", color=INK2)
    # 범례를 축 안에 두면 막대와 겹친다. 그림 상단 바깥으로 뺀다.
    _h, _l = axes[0].get_legend_handles_labels()
    fig.legend(_h, _l, loc="upper center", bbox_to_anchor=(0.5, 0.995),
               ncol=4, frameon=False)
    fig.suptitle("Ablation — Algorithm 1 / Algorithm 2 를 따로 켰을 때",
                 color=INK, fontweight="bold", x=0.008, ha="left", y=1.09)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def main():
    global PAPER_SYSTEM
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--prism-system", default="paper-faithful")
    a = ap.parse_args()
    PAPER_SYSTEM = a.prism_system
    os.makedirs(a.out, exist_ok=True)
    rows = load_summary(a.base)
    if not rows:
        print("no summary.csv yet")
        return
    o = lambda n: os.path.join(a.out, n)
    fig_crossover(rows, o("fig1_crossover.png"))
    fig_panel_lines(rows, "joint_attainment",
                    "Joint SLO 달성률 — 두 SLO 를 모두 만족한 요청의 비율",
                    "Joint SLO 달성률", o("fig2_joint_attainment.png"), pct=True)
    fig_bottleneck(rows, o("fig3_bottleneck.png"))
    fig_ci(a.base, o("fig4_ci_estimators.png"))
    fig_calibration(a.base, o("fig5_calibration.png"))
    fig_scheduler(rows, o("fig6_scheduler_actions.png"))
    fig_ablation(rows, o("fig7_ablation.png"))
    fig_panel_lines(rows, "ttft_p99_ms", "TTFT p99 — Algorithm 2 가 실제로 개선하는 지표",
                    "TTFT p99 (ms)", o("fig8_ttft_p99.png"), logy=True)


if __name__ == "__main__":
    main()
