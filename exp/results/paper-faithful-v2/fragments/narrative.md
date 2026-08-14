## 7. Steady vs Bursty Comparison

Joint SLO attainment, and Prism's relative gain over the released prototype, at each load. Same request set, same per-model request counts, same average offered load — only arrival timing differs.

| Rate | Baseline steady | Prism steady | gain (steady) | Baseline bursty | Prism bursty | gain (bursty) | bursty − steady gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.752 | 0.716 | -4.8% | 0.668 | 0.758 | +13.5% | +18.4% |
| 2 | 0.544 | 0.499 | -8.3% | 0.574 | 0.612 | +6.7% | +14.9% |
| 8 | 0.160 | 0.172 | +7.8% | 0.204 | 0.174 | -14.7% | -22.4% |
| 10 | 0.152 | 0.145 | -4.8% | 0.177 | 0.124 | -29.7% | -24.9% |

### Q1 — Prism vs baseline on the STEADY workload

- **1 req/s**: joint attainment 0.752 → 0.716 (-4.8%), goodput 0.76 → 0.72 req/s (-4.8%), TTFT p99 304 → 583 ms (-91.6%)
- **2 req/s**: joint attainment 0.544 → 0.499 (-8.3%), goodput 1.09 → 1.00 req/s (-8.3%), TTFT p99 380 → 346 ms (+8.8%)
- **8 req/s**: joint attainment 0.160 → 0.172 (+7.8%), goodput 1.29 → 1.39 req/s (+7.8%), TTFT p99 364 → 930 ms (-155.6%)
- **10 req/s**: joint attainment 0.152 → 0.145 (-4.8%), goodput 1.52 → 1.45 req/s (-4.8%), TTFT p99 336 → 413 ms (-22.7%)

### Q2 — Prism vs baseline on the SHIFTING-BURSTY workload

- **1 req/s**: joint attainment 0.668 → 0.758 (+13.5%), goodput 0.69 → 0.78 req/s (+13.5%), TTFT p99 3909 → 4228 ms (-8.2%)
- **2 req/s**: joint attainment 0.574 → 0.612 (+6.7%), goodput 1.20 → 1.28 req/s (+6.7%), TTFT p99 3967 → 4279 ms (-7.9%)
- **8 req/s**: joint attainment 0.204 → 0.174 (-14.7%), goodput 1.64 → 1.40 req/s (-14.7%), TTFT p99 1740 → 2765 ms (-58.9%)
- **10 req/s**: joint attainment 0.177 → 0.124 (-29.7%), goodput 1.80 → 1.26 req/s (-29.7%), TTFT p99 8370 → 2374 ms (+71.6%)

### Q3 — What changes when only the temporal pattern changes

Prism's relative joint-attainment gain is **-3.5%** larger under shifting-bursty than under steady, averaged over 4 load level(s) (range -24.9% to +18.4%).
The sign is **negative** — Prism does relatively worse when the hot set shifts. That is the opposite of the design's prediction and is examined in Section 8.

### Q4 — Does the scheduler actually act more under bursty?

| Workload | Rate | Migrations | Activations | Evictions | Alg-1 cycles | peak-KVPR cv | mean KVPR spread across GPUs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| steady | 1 | 8 | 0 | 0 | 88 | 0.424 | 0.151 |
| bursty | 1 | 2 | 8 | 9 | 86 | 0.384 | 0.153 |
| steady | 2 | 0 | 0 | 0 | 87 | 0.296 | 0.036 |
| bursty | 2 | 2 | 6 | 7 | 87 | 0.390 | 0.139 |
| steady | 8 | 0 | 0 | 0 | 87 | 0.220 | -0.012 |
| bursty | 8 | 0 | 6 | 7 | 88 | 0.441 | 0.158 |
| steady | 10 | 0 | 0 | 0 | 87 | 0.225 | -0.018 |
| bursty | 10 | 0 | 6 | 7 | 88 | 0.409 | 0.164 |

### Q5 — Is a bursty win traceable to KVPR balancing?

- **1 req/s**: peak-KVPR coefficient of variation 0.424 (steady) vs 0.384 (bursty); Algorithm 1 migrations 4 vs 1.
- **2 req/s**: peak-KVPR coefficient of variation 0.296 (steady) vs 0.390 (bursty); Algorithm 1 migrations 0 vs 1.
- **8 req/s**: peak-KVPR coefficient of variation 0.220 (steady) vs 0.441 (bursty); Algorithm 1 migrations 0 vs 0.
- **10 req/s**: peak-KVPR coefficient of variation 0.225 (steady) vs 0.409 (bursty); Algorithm 1 migrations 0 vs 0.

A higher KVPR cv under bursty means the placement objective is genuinely moving with the workload. If migrations do **not** rise with it, the objective moved but tau suppressed the response, and any bursty gain must come from ballooning and eviction rather than from placement.

### Q6 — If Prism is not better under bursty, why

| Workload | Rate | Alg-2 selected/eligible | pathological rounds | under-admission warnings | max zero-streak | max queue |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| steady | 1 | 1.000 | 0 | 0 | 0 | 3 |
| bursty | 1 | 0.970 | 0 | 0 | 0 | 3 |
| steady | 2 | 1.000 | 0 | 0 | 0 | 3 |
| bursty | 2 | 0.968 | 0 | 0 | 0 | 6 |
| steady | 8 | 1.000 | 0 | 0 | 0 | 3 |
| bursty | 8 | 0.955 | 0 | 0 | 0 | 16 |
| steady | 10 | 1.000 | 0 | 0 | 0 | 3 |
| bursty | 10 | 0.962 | 0 | 0 | 0 | 15 |

No under-admission was detected (no warning fired and the longest run of rounds with eligible>0 and selected=0 stayed short). The v1 failure mode is therefore **absent** here, so differences at these loads reflect the algorithms rather than a throughput shortfall in admission control.

### Q7 — Does this explain the v1 result?

v1 ran 3 x Llama-3.1-8B at a constant rate and fed Algorithm 2 `c_i = 4,214 tok/s`, derived as `sum(prompt tokens) / sum(TTFT)` over a **contended** run. Direct measurement of the prefill interval on this box puts Llama-3.1-8B at **13,702 tok/s** — v1's value was low by 3.3x. Two consequences follow, and the table above tests both:

1. A `c_i` that is 3.3x too small inflates every `e_i = p_i / c_i` by 3.3x, so Algorithm 2's cumulative feasibility test declares the GPU full far earlier than it is. That is the under-admission v1 observed.
2. With three identical models on two GPUs, KVPR is the same for every placement, so Algorithm 1 had nothing to decide. v1's null result for placement was a property of its model set, not of KVPR.

Whether those two together fully account for v1's deficit is answered by the Q6 table: if under-admission is absent here and Prism still trails, something beyond `c_i` is at work.

## 8. Root Cause

What the v1 study concluded, and what this study can now say about it. Only
claims backed by a measurement in this report appear here.

**`c_i` was low by 3.3x, and that alone changes Algorithm 2's verdict.**
v1 fed Algorithm 2 `c_i = 4,214 tok/s` for Llama-3.1-8B, obtained as
`sum(prompt tokens) / sum(TTFT)` over a *contended* run. Under contention TTFT
is dominated by queueing, so that ratio measures queue delay rather than prefill
speed. Measuring the prefill interval directly — the engine's own
`out_queue_timestamp -> prefill_finish_timestamp`, with no contention — puts the
same model at **13,702 tok/s** (Section 3.5). Since `e_i = p_i / c_i` scales
inversely with `c_i`, every execution estimate v1 fed to the feasibility test was
3.3x too large, so the cumulative check `clock += e_i` declared the GPU full at
roughly a third of the work it could actually absorb. v1's own diagnosis called
this "the sequential machine model is wrong for a batching engine"; the
measurement here supports a narrower and more actionable reading — the machine
model and the `c_i` estimator disagreed about what `c_i` means.

**The paper's own analysis says `c_i` is an engine capacity.** Section 6.2's
optimality argument holds "when chunked-prefill has prefill running at each
inference step", and derives prefill completion as
`d_ri = a_ri + sum_i p_ri / c_ri`. That summation is only coherent if `c` is the
throughput of one shared prefill pipeline. Read that way, requests genuinely do
queue behind one another for prefill capacity and `clock += p_i / c_i` is the
right model — no batch-parallelism correction is needed, and adding one (v1's
proposed `clock += e_i / B`) would be departing from the paper rather than
repairing it. Read as a per-request speed, the same test double-counts.

**v1's model set made Algorithm 1 unmeasurable, not ineffective.** Three
identical Llama-3.1-8B on two GPUs forces a 1+2 split whose peak KVPR is
`2w / (C - 2 x 15.08)` regardless of which pair is colocated: the objective is
flat, so the argmin is decided by estimator noise. v1 measured a migration
improvement distribution of mean +0.002 with standard deviation 0.175 — an
expected gain of zero. That is a property of the configuration, not evidence
about KVPR. This study replaces it with six models whose KV cell size is not
monotone in parameter count (Section 2), so `model_3` and `model_4` have nearly
equal prefill speed (27,057 vs 27,414 tok/s) but a 3.1x difference in KV bytes
per token — a placement question KVPR can actually answer.

**v1's workload had no reclaimable memory.** A constant-rate trace keeps every
model warm, so there is never an idle tenant whose KV pool a hot tenant could
balloon into. Prism's central mechanism was inactive by construction. The paired
workloads here hold the request set, per-model counts, prompts, output lengths,
duration and average offered load exactly equal and vary only arrival timing
(Section 5), which isolates that mechanism.

_The measured outcome of these four changes is in Sections 6, 7 and the Q1-Q7
answers; where the data does not separate two explanations, it says so._
## 10. Remaining Limitations

- **A100 80GB x2, not the paper's cluster.** Absolute latencies, the saturation
  point and the reachable model count all differ. Only two GPUs means every
  placement decision is binary, which bounds how much Algorithm 1 can express.
- **Production traces are not public.** Hyperbolic / Novita / Arena are not
  released, and the prototype's `--csv-trace` parses but is never used. Our
  arrivals are synthetic (a controlled shifting-bursty process and its exactly
  paired steady control) over real ShareGPT prompt/response content. The
  *shape* of the shifting hot set is modelled on the paper's description, not
  replayed from it.
- **Migration is stop-the-world.** The paper keeps the source instance serving
  until the destination is ready (Sec. 6.1); the prototype deactivates the
  source first, with `evict_waiting_requests=True`. NVLink / GPUDirect weight
  and KV transfer are absent. Migration therefore costs more here than the
  paper's design implies, which bounds Prism's upside in both arms equally.
- **The paper does not specify `c_i`'s profiling method,** nor `tau`'s units or
  value, nor the token-rate window, nor what happens to requests Algorithm 2
  excludes. Every such choice is listed in Section 3.2 with its rationale; none
  was tuned per load level or per arm.
- **`prism-research` is a simplified public prototype, not the paper's
  artifact.** Differences measured here are *prototype vs paper algorithm*, not
  *authors' implementation vs paper*.
- **Seeds.** Aggregate figures are stable at this seed count, but p99 is thin.
  Where the seed-to-seed spread is comparable to the arm-to-arm difference, the
  data does not resolve it and the report says so rather than ranking the arms.
- **TP = 1 throughout.** The TP anti-affinity constraint is implemented but
  never fires, so this study does not validate it.
