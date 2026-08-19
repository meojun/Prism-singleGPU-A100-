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

## Algorithm 1's tau, and why it was left at the inherited value

`tau = 0.00035` was calibrated on the v3 report's machine, and it is not
obviously transferable: KVPR carries units of
(tokens/s x bytes/token / SLO) / GiB, and this study re-derived the SLO
baselines here, which rescales the numerator.  Measured on this box over a
bursty 8 req/s run, the line-8 deltas have mean 0.0153 and sd 0.0329, with 90%
of them below 0.0707 -- so 0.00035 sits two orders of magnitude under the
distribution and admits 28.9% of all decisions.  Applying the project's own
rule from `docs/paper_faithful/design_analysis.md` 5a (mean + 2 sd) to this
box's numbers would instead put tau near 0.13, which admits under 1% and
effectively turns migration off.

It was kept at 0.00035 anyway, for two reasons.

The migration rate it produces here -- 13 in a 300 s window -- is the rate the
v3 study itself reported (14), so the arms remain comparable both to that work
and to each other.  And a tau that suppresses migration would make this study
unable to measure the thing it is about: v4's contribution is that a migration
costs 2.2x less to load and up to 3.5x less to move, which is only observable
when migrations happen.  Every arm runs the same tau, so it cannot favour one.

What the first runs already show is that the cost is real.  At 8 req/s bursty
the prototype and v3 push identical throughput (8.04 req/s) and Algorithm 2 is
not under-admitting on either (v3 selects 1061 of 1102 eligible, 0 pathological
rounds, max queue 12).  The whole difference is latency -- TTFT p50 71 -> 115
ms, TPOT p50 31.7 -> 50.1 ms -- against v3's 13 migrations and 19
deactivations versus the prototype's 2 and 9.  Each migration reloads up to
14 GiB of weights onto a GPU already at 99% utilisation.  Whether v4's cheaper
transfers recover that is the question the sweep answers.

## What the v4 end-to-end arm actually ran

The v4 arm in the sweep runs with **page-locked host weights on and GPU-to-GPU
migration off** (`PRISM_V4_PAGELOCK=1`, `PRISM_V4_P2P_MIGRATION=0`).

The first v4 attempt ran with P2P on and it worked -- 9 of 19 weight transfers
went gpu-to-gpu -- but the run died of CUDA OOM on GPU 0 and lost 1300 of 3387
requests, so the harness refused it and no results were kept.  The cause is a
lifetime problem, not a transport one: to serve as a migration source the model
service holds a CUDA IPC mapping of the engine's weights, and dropping the
Python reference does not return that memory.  `empty_cache()` frees only the
calling process's own allocator cache; an IPC mapping needs `ipc_collect()`.
Twenty-four deactivations leaked their way into an OOM.

`ipc_collect()` is now called on release and when a migration displaces a
registry entry.  That fix is **not** validated end to end in the sweep: the
sweep continued with P2P off so every v4 run shares one configuration, rather
than mixing a fixed and an unfixed arm.  P2P migration's performance is
measured in the microbenchmark instead, where NVLink counters confirm the
whole model crosses the link (14.96 GiB for Llama-3.1-8B) at 72.9 GB/s against
20.8 for the host path.

None of this is taken on trust: every run's `weight_transfers.jsonl` records
`host_registered` and `transfer_path` per transfer, so what each arm actually
did is in its own raw data rather than in a configuration claim.
