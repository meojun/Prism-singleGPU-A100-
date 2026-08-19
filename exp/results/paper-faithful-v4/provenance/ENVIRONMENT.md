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

## flashinfer prefill workspace

Profiling Qwen2.5-7B (`model_6`) died three times with

    RuntimeError: Failed to allocate memory for batch_prefill_tmp_v
                  with size 454164480 ... in AlignedAllocator

This is flashinfer's fixed prefill scratch buffer, not GPU memory: the default
is 384 MiB and this model asks for roughly 420-450 MiB.  It is the only one of
the six that does — its GQA ratio is 28 query heads to 4 KV heads, against
32-to-8 for Llama-3.1-8B, which changes how flashinfer sizes the buffer.
Lowering the saturating concurrency from 48 to 24 did not help, because the
engine batches whatever is queued regardless.

`FLASHINFER_WORKSPACE_SIZE` exists for exactly this, but reading it back is
broken upstream: `global_config.py` takes it straight from the environment as a
**string**, and `torch.empty("1073741824", ...)` raises.  Setting the variable
therefore replaced one crash with another until the value was cast to `int`.
That one-line fix is part of the v4 patch set and applies to every arm.

The buffer is now 1 GiB for all runs and all arms.  It is a capacity ceiling
rather than a tuning knob: models 1-5 never came close to 384 MiB, so raising
it cannot change their numbers, and for model_6 it removes an artificial limit
instead of granting an advantage.  The cost is 1 GiB less KV pool per GPU out
of a 67.28 GiB budget, applied identically to every arm.
