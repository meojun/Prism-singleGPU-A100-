# Measurement environment, and how it differs from the v3 run

The v3 numbers committed in `exp/results/paper-faithful-v3/` were measured on a
different machine.  This study re-runs **both** arms here, so the v3-vs-v4
comparison inside this report is internally valid; the v3 numbers in the older
report are not directly comparable to these and are not compared against.

| | v3 report machine | this machine |
| --- | --- | --- |
| GPU | 2 x A100-SXM4-80GB | 4 x A100-SXM4-80GB, **2 allocated** |
| GPU interconnect | not recorded | NV12 all-pairs via 6 NVSwitches |
| CPU | AMD EPYC 7513 32-Core | AMD EPYC 7532 32-Core, 2 sockets, 128 threads |
| RAM | 1385 GiB | 1877 GiB |
| Kernel | 5.15.0-186-generic | 6.8.0-90-generic |
| NVIDIA driver | 570.211.01 | 580.105.08 |
| torch / SGLang / vLLM | 2.4.0+cu121 / 0.3.4.post2 / 0.6.3.post1 | identical (same pinned SHAs) |

## Why "2 of 4" is not the same as a 2-GPU node, and why it does not matter here

It is not the same node: the CPU, kernel and driver all differ, and the GPU-GPU
path runs through NVSwitch rather than whatever the older box used.  Absolute
latencies here should therefore not be read against the older v3 report.

What the allocation does *not* introduce is interference.  GPUs 2 and 3 are
never made visible to any process (`CUDA_VISIBLE_DEVICES=0,1` is exported from
`/workspace/.env`, which `exp/scripts/env.sh` sources, and every launcher pins
it again explicitly).  The per-run `gpu_timeline.txt` samples **all four** GPUs
every 2 s, so their idleness is evidence in the raw data rather than an
assertion.

## Machine-specific inputs re-derived here

`c_i` (Algorithm 2's feasibility test) and the TTFT/TPOT baselines (Algorithm
1's KVPR weighting, and every SLO in the report) are properties of the GPU they
were measured on.  The committed copies from the other box are preserved beside
this file as `*_committed_other_box.json` and were **not** used; both were
re-profiled here under `exp/results/paper-faithful-v4/profiling/`.
