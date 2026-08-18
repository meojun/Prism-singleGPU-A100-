#!/usr/bin/env python3
"""Build the compact CSV, four requested figures, and validation report."""
import argparse, csv, glob, json, os, statistics
from collections import defaultdict

import matplotlib.pyplot as plt

METRICS = ["goodput_req_s", "joint_attainment", "ttft_p99_ms", "tpot_p99_ms",
           "throughput_req_s", "migration_count"]
RATES = [2, 4, 8, 10, 14, 20]
THREE = {4, 8, 20}

def load(base):
    rows = []
    pat = os.path.join(base, "raw", "*", "*", "rate_*", "seed_*", "metrics.json")
    for path in sorted(glob.glob(pat)):
        run = os.path.dirname(path)
        if not os.path.isfile(os.path.join(run, "DONE")):
            continue
        p = path.split(os.sep)
        d = json.load(open(path))
        system, workload = p[-5], p[-4]
        row = {"system": system, "workload": workload,
               "rate": int(p[-3][5:]), "seed": int(p[-2][5:])}
        for k in METRICS[:-1]: row[k] = float(d[k])
        row["migration_count"] = float(d["migrations_alg1"] if system == "paper-faithful-v3"
                                        else d["migrations_proto"])
        rows.append(row)
    return rows

def summarize(rows):
    groups = defaultdict(list)
    for r in rows: groups[(r["system"], r["workload"], r["rate"])].append(r)
    out = []
    for (system, workload, rate), rs in sorted(groups.items(), key=lambda x:(x[0][1],x[0][0],x[0][2])):
        z = {"system": system, "workload": workload, "rate": rate, "n_seeds": len(rs)}
        for m in METRICS:
            vals = [r[m] for r in rs]
            z[m + "_mean"] = statistics.fmean(vals)
            z[m + "_std"] = statistics.stdev(vals) if len(vals) > 1 else ""
        out.append(z)
    return out

def validate(rows):
    got = defaultdict(set)
    for r in rows: got[(r["system"], r["workload"], r["rate"])].add(r["seed"])
    errors=[]
    for s in ["released-prototype", "paper-faithful-v3"]:
      for w in ["steady", "bursty"]:
       for rate in RATES:
        want={1,2,3} if rate in THREE else {1}
        if got[(s,w,rate)] != want: errors.append(f"{s}/{w}/{rate}: {sorted(got[(s,w,rate)])} != {sorted(want)}")
    if errors: raise SystemExit("incomplete validation matrix:\n" + "\n".join(errors))

def plots(summary, outdir):
    os.makedirs(outdir, exist_ok=True)
    idx={(r["system"],r["workload"],r["rate"]):r for r in summary}
    colors={"released-prototype":"#4C78A8", "paper-faithful-v3":"#E45756"}
    styles={"steady":"-", "bursty":"--"}
    def metric_plot(metric, ylabel, name):
        fig, ax=plt.subplots(figsize=(7.2,4.4))
        for w in ["steady","bursty"]:
          for s in ["released-prototype","paper-faithful-v3"]:
            rs=[idx[(s,w,r)] for r in RATES]
            y=[x[metric+"_mean"] for x in rs]
            err=[(x[metric+"_std"] or 0) if x["rate"] in THREE else 0 for x in rs]
            ax.errorbar(RATES,y,yerr=err,color=colors[s],linestyle=styles[w],marker="o",capsize=3,
                        label=f"{s.replace('released-','').replace('paper-faithful-','')} / {'Shifting-Bursty' if w=='bursty' else 'Steady'}")
        ax.set(xlabel="Request rate (req/s)",ylabel=ylabel,xticks=RATES); ax.grid(alpha=.25); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(os.path.join(outdir,name),dpi=180); plt.close(fig)
    metric_plot("goodput_req_s","Goodput (req/s)","goodput_vs_rate.png")
    metric_plot("joint_attainment","Joint SLO attainment","joint_slo_vs_rate.png")
    metric_plot("ttft_p99_ms","TTFT p99 (ms)","ttft_p99_vs_rate.png")
    fig,ax=plt.subplots(figsize=(7.2,4.4))
    for w in ["steady","bursty"]:
        ys=[]; es=[]
        for rate in RATES:
            b=idx[("released-prototype",w,rate)]; v=idx[("paper-faithful-v3",w,rate)]
            ys.append(100*(v["goodput_req_s_mean"]-b["goodput_req_s_mean"])/b["goodput_req_s_mean"])
            if rate in THREE:
                # Propagated independent sample SD, adequate for the displayed seed variability.
                bm,vm=b["goodput_req_s_mean"],v["goodput_req_s_mean"]
                bs,vs=b["goodput_req_s_std"],v["goodput_req_s_std"]
                es.append(100*((vs/bm)**2 + (vm*bs/bm**2)**2)**.5)
            else: es.append(0)
        ax.errorbar(RATES,ys,yerr=es,linestyle=styles[w],marker="o",capsize=3,
                    label="Shifting-Bursty" if w=="bursty" else "Steady")
    ax.axhline(0,color="black",lw=.8); ax.set(xlabel="Request rate (req/s)",ylabel="V3 goodput improvement (%)",xticks=RATES)
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(os.path.join(outdir,"v3_goodput_improvement.png"),dpi=180); plt.close(fig)

def report(summary, path):
    idx={(r["system"],r["workload"],r["rate"]):r for r in summary}
    lines=["# Paper-Faithful V3 Validation", "", "| Workload | Rate | Prototype goodput | V3 goodput | Improvement | Prototype joint SLO | V3 joint SLO |", "|---|---:|---:|---:|---:|---:|---:|"]
    for w in ["steady","bursty"]:
      for rate in RATES:
        b,v=idx[("released-prototype",w,rate)],idx[("paper-faithful-v3",w,rate)]
        imp=100*(v["goodput_req_s_mean"]-b["goodput_req_s_mean"])/b["goodput_req_s_mean"]
        pm=lambda x,m: f'{x[m+"_mean"]:.3f}' + (f' ± {x[m+"_std"]:.3f}' if x[m+"_std"] != "" else "")
        lines.append(f"| {'Shifting-Bursty' if w=='bursty' else 'Steady'} | {rate} | {pm(b,'goodput_req_s')} | {pm(v,'goodput_req_s')} | {imp:+.1f}% | {pm(b,'joint_attainment')} | {pm(v,'joint_attainment')} |")
    b20,v20=idx[("released-prototype","bursty",20)],idx[("paper-faithful-v3","bursty",20)]
    imp20=100*(v20["goodput_req_s_mean"]-b20["goodput_req_s_mean"])/b20["goodput_req_s_mean"]
    lines += ["", "## 핵심 질문", "",
      f"1. 2–4 req/s 정상성: 위 표 및 Goodput/SLO 그래프 기준.",
      f"2. 8–10 req/s SLO knee: Joint SLO 및 TTFT p99 그래프 기준.",
      f"3. 20 req/s Shifting-Bursty 재현: V3 goodput 개선 {imp20:+.1f}% (각 arm n=3).",
      "4. workload별 이점: Improvement 그래프의 Steady/ Shifting-Bursty 곡선 비교.", "",
      "## Figures", "",
      "![Goodput](figures/goodput_vs_rate.png)", "![Joint SLO](figures/joint_slo_vs_rate.png)",
      "![TTFT p99](figures/ttft_p99_vs_rate.png)", "![Improvement](figures/v3_goodput_improvement.png)"]
    open(path,"w").write("\n".join(lines)+"\n")

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--base",required=True); a=ap.parse_args()
    rows=load(a.base); validate(rows); summary=summarize(rows)
    fields=list(summary[0])
    with open(os.path.join(a.base,"summary.csv"),"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator="\n")
        w.writeheader(); w.writerows(summary)
    plots(summary,os.path.join(a.base,"figures")); report(summary,os.path.join(a.base,"REPORT.md"))
    print(f"wrote {len(rows)} runs, {len(summary)} summary rows, 4 figures, REPORT.md")
if __name__ == "__main__": main()
