# Algorithm 2 root-cause and paper-fidelity audit

Final classification: **IMPLEMENTATION BUG**

This diagnosis stops at D1. D2 and D3 were not run, and Algorithm 2 policy code
was not changed during the root-cause analysis.

## 1. Full D0 versus D1 result

The aggregate rows below use the 2,412-request measurement window. Latencies are
milliseconds.

| Metric | D0 | D1 | D1 - D0 |
|---|---:|---:|---:|
| TTFT SLO | 0.916667 | 0.873549 | -0.043118 (-4.70%) |
| TPOT SLO | 0.282338 | 0.061360 | -0.220978 (-78.27%) |
| Joint SLO | 0.267413 | 0.044362 | -0.223051 (-83.41%) |
| TTFT P50 | 92.495 | 109.385 | +16.890 |
| TTFT P95 | 410.635 | 467.975 | +57.340 |
| TTFT P99 | 677.671 | 629.188 | -48.483 |
| TPOT P50 | 42.085 | 50.766 | +8.682 |
| TPOT P95 | 88.528 | 93.169 | +4.641 |
| TPOT P99 | 113.459 | 117.644 | +4.185 |
| E2E P50 | 4,978.062 | 6,228.547 | +1,250.485 |
| E2E P95 | 24,276.853 | 27,334.221 | +3,057.368 |
| E2E P99 | 33,267.624 | 33,879.113 | +611.489 |
| achieved throughput | 8.04 req/s | 8.04 req/s | 0 |
| goodput | 2.15 req/s | 0.356667 req/s | -1.793333 |

The primary SLO collapse is TPOT, not throughput or request completion.

The complete trace contained 3,387 workload requests. Both arms completed all
3,387: unfinished=0, rejected=0, aborted=0. Their raw completion throughputs were
7.768155 req/s (D0) and 7.750939 req/s (D1). The aggregate evaluator's
offered/accepted/completed counts are 2,412/2,412/2,412 in both arms because it
excludes warm-up and drain from its measurement window.

## 2. Algorithm 2 request lifecycle

Runtime evaluated 3,393 unique requests: 3,387 workload requests plus six server
warm-ups. The workload-only lifecycle is:

| Event | Count |
|---|---:|
| evaluated | 3,387 |
| selected into the normal accepted subset | 3,386 |
| removed by Moore-Hodgson | 1 |
| deferred events | 1 |
| requeued | 0 |
| late-dispatched | 1 |
| permanently rejected | 0 |
| eventually completed | 3,387 |

The sole removed request was `model_5#582` (432 prompt tokens). It arrived at
1787377657.7274892 with deadline 1787377658.2767687, but was first evaluated at
1787377658.4774356, already 200.667 ms late. Its configured c_i was
14,249.064746 tokens/s and predicted execution was 30.318 ms. Moore-Hodgson
removed it, then the local B1 lifecycle rule immediately dispatched it as late.
It was not requeued or rejected and it completed successfully.

Therefore the D1 regression is not caused by request shedding or a smaller final
completion set: only one request was removed and even that request still ran.

## 3. Shared queue and arbitration semantics

There is one `RequestQueue` object per `GPUScheduler`, hence one per GPU. The
receiver polls every active model's Redis frontend queue and appends all requests
to that same object. Moore-Hodgson therefore arbitrates a shared per-GPU set,
not separate per-model sets.

Runtime confirms the shared set. Across 3,295 workload dispatch cycles, 49 cycles
contained requests from more than one model in the same Algorithm 2 round; a
round contained up to three models. For example, GPU 0 round 3519 jointly
evaluated and selected `model_4#21` and `model_5#64`.

Dispatch batch size was mean 1.0279, P50=1, P95=1, P99=2, max=3; 88 cycles
dispatched more than one request. Thus the GPU scheduler is not artificially
serializing accepted requests by sending only one per cycle.

## 4. Dispatch fidelity bug

`select()` returns the accepted subset in global ascending-deadline schedule
order. The caller then inserts those requests into a dictionary keyed by model.
The sender iterates each model and drains all of that model's requests to a
separate model backend queue. Consequently a valid global order such as
`A(model_1), B(model_2), C(model_1)` becomes `A, C, B` at dispatch.

All accepted requests are sent immediately in the same scheduler cycle; no
global execution reservation or single-machine schedule is carried into the
model engines. Furthermore, the MH calculation includes only requests still in
the GPU scheduler's frontend queue. Already-dispatched backend work and running
prefill/decode work are absent from its cumulative clock. Backend lengths only
block an entire model when its queue length is greater than 10.

This violates the requested/paper semantic that the final accepted subset be
dispatched in schedule order and that `sum(p_i/c_i)` represent the work ahead of
a request. The per-model engines also choose a new prefill before decode when one
is available, so the globally unordered immediate dispatch can increase decode
interference and TPOT.

## 5. Queue-stage comparison and polling

A paired 120-second prefix run was made only to disable per-request audit writes
and add timestamps. It used 958 identical requests per arm; both completed all
958 with no abort or rejection. Values below are milliseconds.

| Stage | D0 mean/P50/P95/P99 | D1 audit-off mean/P50/P95/P99 | Mean delta |
|---|---:|---:|---:|
| frontend wait | 0.783 / 0.648 / 1.230 / 2.966 | 0.735 / 0.658 / 1.272 / 1.995 | -0.048 |
| local scheduler wait | 5.766 / 5.734 / 10.495 / 11.163 | 5.947 / 5.918 / 10.793 / 11.580 | +0.181 |
| engine/prefill wait | 98.707 / 45.306 / 353.484 / 512.540 | 111.916 / 51.635 / 442.218 / 698.621 | +13.209 |
| actual prefill service | 67.736 / 43.589 / 193.442 / 429.462 | 69.244 / 43.827 / 196.861 / 434.925 | +1.508 |
| prefill finish to first decode event | 82.973 / 35.221 / 288.322 / 676.565 | 97.501 / 36.609 / 329.290 / 880.482 | +14.528 |

End metrics in this short paired run were:

| Metric | D0 | D1 audit-off |
|---|---:|---:|
| TTFT mean/P50/P95/P99 ms | 172.991 / 115.809 / 508.361 / 748.882 | 187.841 / 117.578 / 585.019 / 833.786 |
| TPOT mean/P50/P95/P99 ms | 55.656 / 51.630 / 92.762 / 127.310 | 59.234 / 53.144 / 102.940 / 156.009 |
| E2E mean/P50/P95/P99 ms | 9,539.608 / 6,945.228 / 26,668.945 / 31,089.875 | 9,874.336 / 6,967.740 / 26,658.710 / 32,345.342 |
| completion throughput req/s | 7.0772 | 6.8967 |

The 10 ms GPU-scheduler polling interval is visible as an approximately 5.8 ms
median local wait in both arms. It adds queueing latency, but it is common to D0
and D1 and changed by only 0.18 ms in D1. The D1-only increase appears after
dispatch: engine/prefill wait and decode-start wait, consistent with the lost
global order and cross-engine interference rather than Algorithm 2 CPU time.

The short-run absolute SLO values are startup-sensitive and do not reproduce the
full-run regression magnitude; they are used as a paired path decomposition and
instrumentation check, not as replacements for the full D0/D1 result.

## 6. Prediction accuracy and c_i

The original full-run `alg2_prediction_audit.csv` has all 3,387 decisions and
actual TTFT, but its queue/prefill columns are empty. The TP-mode serializer used
for that run discarded server timestamps. Therefore an actual prefill-service
error cannot honestly be computed from that CSV alone. Using TTFT as a clearly
labelled proxy gives absolute error mean/P50/P95/P99 =
147.603/92.564/461.593/639.150 ms and predicted-vs-TTFT correlation 0.31234.

The timestamped D1 audit-off sanity run gives direct `out_queue -> prefill_finish`
service time for 958 requests. With the exact c_i values used by D1, predicted
minus actual error is negative for every request:

| Model | n | abs error mean/P50/P95/P99 ms | correlation |
|---|---:|---:|---:|
| model_1 | 57 | 30.993 / 28.705 / 67.238 / 78.921 | 0.5241 |
| model_2 | 153 | 42.921 / 31.789 / 75.821 / 308.129 | 0.5416 |
| model_3 | 221 | 52.551 / 36.437 / 101.066 / 356.618 | 0.3995 |
| model_4 | 106 | 37.352 / 34.124 / 67.363 / 78.831 | 0.7721 |
| model_5 | 276 | 80.867 / 41.616 / 273.564 / 724.600 | 0.4251 |
| model_6 | 145 | 46.764 / 40.125 / 82.223 / 157.672 | 0.7531 |
| all | 958 | 55.331 / 36.780 / 127.451 / 405.509 | 0.47295 |

Overall predicted mean was 13.914 ms versus actual 69.244 ms; signed error mean
was -55.331 ms and relative error mean was -82.09%. This is a real
calibration/runtime-assumption mismatch, but it is not the primary D1 regression
cause because the full run selected 3,386/3,387 requests and eventually
dispatched all 3,387. The inaccurate estimate barely changed admission outcomes.

D1 loaded these values from `exp/configs/v2/prefill_speed.json`, confirmed by the
per-request runtime records:

| Model | c_i used by D1 (tokens/s) |
|---|---:|
| model_1 | 61,151.609351 |
| model_2 | 48,614.438491 |
| model_3 | 29,347.140690 |
| model_4 | 30,078.012379 |
| model_5 | 14,249.064746 |
| model_6 | 15,611.739433 |

The new model_1 saturated profile is 49,592.032389 tokens/s, 18.903% below the
value D1 used. Its new solo and regression-slope estimates are 31,211.139812 and
32,913.223539 tokens/s. The old same-box v6 value was 62,004.135496 tokens/s.

For model_6, the new saturated result is 11,431.443331 tokens/s, 26.777% below
the value D1 used; it had to use concurrency 8 after higher-concurrency trials
failed, so it is not directly saturation-comparable. Its solo and regression
slope estimates are 12,325.001758 and 14,672.000253 tokens/s; the old same-box
v6 value was 13,842.657451 tokens/s.

All runtime quantities are in compatible base units: prompt length is tokens,
c_i is tokens/second, execution time is seconds, and arrival/current/deadline are
absolute Unix seconds. For example, `model_2#0` used 76 / 48,614.438491 =
0.001563322 seconds and deadline = arrival + 0.1610327 seconds.

## 7. Instrumentation exclusion

The full D1 request audit did perform synchronous line-buffered file writes on
the admission hot path. It emitted 4,512 total JSONL lines (3,393 request records
plus periodic round records), 1.543 MB over about 420 seconds.

Loop timing across both GPU schedulers was:

| Arm | iterations | admission CPU | us/iteration | us/admitted request |
|---|---:|---:|---:|---:|
| D0 | 81,221 | 0.763865 s | 9.405 | 225.130 |
| D1 audit-on | 82,299 | 1.081339 s | 13.139 | 318.697 |

The D1 admission increment was 0.317474 CPU seconds over the full run. The paired
D1 sanity run had `PRISM_DIAG_ALG2` unset and zero `request_decision` records,
yet still showed downstream engine/prefill and decode-wait increases. Therefore
the regression is not classified as instrumentation-induced. Periodic 0.5-second
Algorithm 2 round logging remained enabled in the audit-off sanity, because it
is part of the existing D1 implementation rather than the added per-request
audit.

## 8. Verdict

**IMPLEMENTATION BUG** is the primary classification.

The Moore-Hodgson selector itself implements deadline sort, cumulative p_i/c_i,
longest-job removal, and clock subtraction correctly. The integration does not
preserve its accepted schedule: model grouping destroys global order, all work
is immediately fanned out to separate engine queues, and outstanding/running
work is absent from the selector's clock. This makes the paper's single-machine
completion-time guarantee false at runtime. The observed regression is dominated
by TPOT and appears downstream of the GPU scheduler, matching prefill/decode
interference from that integration defect.

Stale/optimistic c_i calibration is a secondary defect and should be corrected
after the dispatch integration is fixed, but it did not cause this run's large
regression through admission decisions. Per-request instrumentation is also not
the cause. D2/D3 remain blocked by this D1 stop gate.
