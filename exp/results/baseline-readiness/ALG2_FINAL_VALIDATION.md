# Algorithm 2 final validation

## Verdict

`BASELINE-READY`

This verdict is limited to the Algorithm 2 runtime integration. Moore--Hodgson
selection, deadlines, `e_i = p_i/c_i`, workload, SLOs, Algorithm 1, migration,
and tau were not changed.

## Implemented runtime semantics

- Algorithm 2 returns a global ordered list instead of a model-keyed mapping.
- The GPU scheduler assigns a monotonic per-GPU `alg2_seq` and may pipeline a
  bounded prefix without waiting for each request's `prefill_start` ACK.
- Independent model engines stage fetched requests. A shared atomic per-GPU
  sequence token allows only the globally next request into actual prefill
  admission, and advances at its real prefill start.
- Dispatch-to-completion work remains in the outstanding map. Its full
  `p_i/c_i` is included in later Algorithm 2 feasibility calculations while the
  request is in Redis, staged in a backend, admitted, or running prefill.
- Backend admission, prefill start, and completion are checked fail-closed.

## Correctness gates

The forced unit/integration sequence A(model1), B(model2), C(model1) passed, and
the negative independent-backend reorder test failed closed as required. The
real same-GPU run then produced:

| request | model | seq | dispatch | backend admit | prefill start | complete |
|---|---|---:|---:|---:|---:|---:|
| A | model_1 | 4 | 1787384358.047087 | 1787384358.051000 | 1787384358.052595 | 1787384358.095195 |
| B | model_4 | 5 | 1787384358.047538 | 1787384358.057107 | 1787384358.061641 | 1787384358.114662 |
| C | model_1 | 6 | 1787384358.047816 | 1787384358.097819 | 1787384358.099035 | 1787384358.110928 |

All three were dispatched before A started, proving removal of per-request
start-ACK dispatch serialization. Dispatch, backend-admission, and prefill-start
orders were all A, B, C. There were no violations, losses, rejects, or aborts,
and outstanding work was zero at shutdown.

The steady8 D1 runtime audit recorded exact dispatch/admit/start/complete order
on both GPUs:

| GPU | events at each stage | max outstanding | remaining | order violations |
|---:|---:|---:|---:|---:|
| 0 | 442 | 3 | 0 | 0 |
| 1 | 522 | 3 | 0 | 0 |

## Steady8 seed1 result

The comparison uses the preregistered 60-second warmup and 60-second measurement
window. Both arms offered and completed 482 requests in-window; both full traces
completed 958/958 with zero aborts.

| metric | D0 | D1 fixed |
|---|---:|---:|
| achieved throughput (req/s) | 8.0333 | 8.0333 |
| goodput (req/s) | 0.2167 | 0.2167 |
| TTFT SLO | 85.48% | 75.93% |
| TPOT SLO | 4.36% | 5.19% |
| Joint SLO | 2.70% | 2.70% |
| completed / unfinished / rejected / aborted | 482 / 0 / 0 / 0 | 482 / 0 / 0 / 0 |

| latency | D0 P50/P95/P99 ms | D1 fixed P50/P95/P99 ms |
|---|---:|---:|
| TTFT | 126.09 / 482.26 / 614.99 | 161.75 / 612.25 / 814.33 |
| TPOT | 55.39 / 103.04 / 140.41 | 55.76 / 97.53 / 116.55 |
| E2E | 7405.75 / 25929.96 / 31666.38 | 7171.61 / 25474.63 / 32067.67 |

| wait/service | D0 mean/P50/P95/P99 ms | D1 fixed mean/P50/P95/P99 ms |
|---|---:|---:|
| frontend | 0.87 / 0.65 / 1.36 / 4.76 | 0.66 / 0.57 / 1.31 / 2.35 |
| local scheduler | 5.78 / 5.81 / 10.51 / 11.32 | 23.35 / 6.92 / 151.35 / 408.39 |
| engine/prefill wait | 102.63 / 55.21 / 355.79 / 527.55 | 144.85 / 82.33 / 500.36 / 694.46 |
| actual prefill service | 64.57 / 44.25 / 157.24 / 315.01 | 59.63 / 42.06 / 126.73 / 259.48 |
| prefill to first decode | 93.81 / 38.22 / 315.35 / 708.55 | 105.39 / 37.71 / 448.72 / 800.05 |

Relative to the superseded ACK-gated D1, local scheduler P95 fell from 630.39
to 151.35 ms and TTFT P99 from 2394 to 814.33 ms. D1 retains a moderate TTFT
penalty relative to D0, but the specific serialization blocker and its multi-
second tail are gone without changing the policy. Throughput, goodput, Joint
SLO, completion, and failure counts did not regress. This satisfies the stated
baseline-readiness gate; it is not a claim that D1 outperforms D0.

## Verification

- `python exp/tests/test_alg2_runtime_order.py`: PASS
- `python exp/tests/test_moore_hodgson.py`: all worked examples and 300
  randomized brute-force comparisons PASS
- Python compilation of the changed Algorithm 2/runtime/watchdog paths: PASS
- Same-GPU ordering pipeline: rc 0, monitor state COMPLETE
- Steady8 D1 pipeline: rc 0, monitor state COMPLETE
