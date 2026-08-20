# This box, and what must not be carried off it

Full snapshot in `../environment.txt`. This records only what differs from the
v4 report's box and what follows from those differences.

| | v4 report box | this box |
| --- | --- | --- |
| GPU | 4×A100-SXM4-80GB, **2 allocated** | 2×A100-SXM4-80GB, dedicated |
| GPU interconnect | NV12 all-pairs via NVSwitch | NV12, `peer access {'0->1': True, '1->0': True}` |
| NVIDIA driver | 580.105.08 | 580.65.06 |
| Kernel | 6.8.0-90-generic | 5.15.0-151-generic |
| CPU | AMD EPYC 7532, **2 NUMA nodes** | AMD EPYC 7532, **8 NUMA nodes**, ~30.7 CPU cgroup quota |
| RAM | 1877 GiB | 2015 GB |
| CUDA toolkit present | — | 12.6 (driver advertises 13.0) |
| torch / SGLang / vLLM / flashinfer | 2.4.0+cu121 / 0.3.4.post2 / 0.6.3.post1 / 0.1.6 | identical, verified against the lockfile |

The pinned stack resolved without the index drift the v3 box hit, so
`bootstrap.sh` needed no workaround here.

`workspace_is_volume: false` — nothing on this instance survives a
recycle/destroy. Everything worth keeping is pushed to the branch.

## Re-derived here, and by how much it moved

`c_i` and the SLO baselines are properties of the GPU they were measured on.
Both were re-derived (`../profiling/`), and the committed values in
`exp/configs/v2/` were **not** used.

| slot | model | TTFT p95 this box | v4 box | Δ | c_i this box | v4 box | Δ |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| model_1 | Llama-3.2-1B | 29.0 ms | 29.5 | −1.4% | 62,004 | 54,978 | +12.8% |
| model_2 | Qwen2.5-1.5B | 36.8 | 45.3 | **−18.8%** | 47,146 | 40,585 | **+16.2%** |
| model_3 | Llama-3.2-3B | 64.5 | 65.6 | −1.7% | 26,270 | 24,198 | +8.6% |
| model_4 | Qwen2.5-3B | 63.9 | 65.5 | −2.4% | 26,058 | 24,423 | +6.7% |
| model_5 | Llama-3.1-8B | 127.8 | 129.0 | −0.9% | 12,795 | 12,440 | +2.9% |
| model_6 | Qwen2.5-7B | 117.8 | 120.3 | −2.1% | 13,843 | 13,345 | +3.7% |

**Compute is nearly identical; the interconnect is not.** Four of six TTFT
baselines land within 2.4% of the v4 box, yet P2P migration of a comparable
model runs at 105.3 GB/s here against 72.9 GB/s there (`../microbench-qwen/`).
So "the latencies match, so the migration numbers should carry over" is exactly
the wrong inference, and the raw data says so rather than an argument.

Re-deriving still mattered: model_2 moved 19% and `c_i` up to 16%. Both feed
SLO judgements and Algorithm 2's feasibility test.

## tau

Re-derived under suppressed migration (`../calibration/tau.json`), so the
estimator is not measuring a placement its own decisions changed.

```
tau = 0.15992   (mean + 2 sd of the line-8 delta, over cycles where a different
                 GPU was actually the argmin)
cycles 86, decisions 460, of which 200 had a move available
delta mean 0.0543, sd 0.0528, median 0.0346, p90 0.1285, p99 0.1635
admits 0.65% of all decisions; migrations during calibration: 0
```

Close to the 0.14606 the v4 box derived by the same rule — but far from the
0.00035 that sweep actually ran with, which was inherited from the v3 box and
sits two orders of magnitude below this distribution.

**A tau this size admits under 1% of decisions, i.e. it very nearly turns
migration off.** The v4 study found that suppressing migration *cost* goodput
(4.21 → 0.30 at bursty 20), so a study that wants to measure migration cannot
simply adopt this value. State which tau a run used and why; do not treat
0.15992 as the default.

## Values deliberately not committed to `exp/configs/v2/`

`slo_base.json` and `prefill_speed.json` are tracked at a shared path, so two
boxes committing their own values produce a merge conflict in which neither
side is right for the other. This box's values therefore live beside its
measurements (`../profiling/*_this_box.json`) and are selected at runtime via
`SLO_BASE_FILE` / `PREFILL_SPEED_FILE`, which every script already honours.
The `.pkl` workload traces are gitignored for the same reason and must be
rebuilt per box -- against *that box's* `slo_base.json`, because
`build_paired_workload.py` stamps the per-slot SLOs into the trace.

## A trap worth recording

`calibrate_tau_v4.sh` cannot be run standalone on a fresh box: it reads
`exp/workloads/paper-faithful-v4/bursty_r8_s1.pkl`, which is a gitignored build
product that `run_pipeline_v4.sh` STAGE 5 normally creates first. It died here
with `FileNotFoundError` after the profiling had already succeeded. Build the
traces first.
