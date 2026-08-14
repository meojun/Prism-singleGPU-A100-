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

