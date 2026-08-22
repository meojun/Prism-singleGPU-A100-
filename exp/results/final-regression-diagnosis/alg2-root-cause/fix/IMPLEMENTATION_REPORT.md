# Algorithm 2 integration fix validation

Status: integration fix implemented and validated; D2/D3 remain blocked.

## Minimal integration change

- `select()` output remains a global ordered list; model grouping no longer
  converts `A(model1), B(model2), C(model1)` into `A,C,B`.
- Only the head of the current global schedule is dispatched until its actual
  engine-side prefill-start ACK is received. A mismatched model/request start
  sets shutdown and fails closed.
- Started-but-incomplete prefills remain visible as outstanding
  `sum(p_i/c_i)` work in the next Algorithm 2 feasibility calculation. They are
  removed only by the engine-side final-prefill-completion ACK.
- The released-prototype/D0 path is unchanged.

## Ordering evidence

The forced unit/integration sequence passed for
`A(model1), B(model2), C(model1)`, including a negative test that attempts to
start B while A owns the admission token.

The real same-GPU mixed-model equivalent used
`A(model_1), B(model_4), C(model_1)`. Runtime events were:

1. dispatch A, prefill-start A
2. dispatch B, prefill-start B
3. dispatch C, prefill-start C
4. final prefill completions remove outstanding accounting

All nine A/B/C runtime events had `order_ok=true`. Across the steady sanity,
GPU 0 had 442 dispatches, 442 initial starts, and 442 completions; GPU 1 had
522/522/522. On both GPUs dispatch order exactly equalled actual admission
order, every event had `order_ok=true`, and zero outstanding prefills remained.
The maximum concurrently accounted outstanding count was four per GPU.

## Steady8 seed1 sanity versus D0

Paired trace: 958 requests. Standard window: discard 60 seconds, measure the
60--120 second arrival window (482 requests).

| Metric | D0 | D1 fixed | Delta |
|---|---:|---:|---:|
| Achieved throughput (req/s) | 8.0333 | 8.0333 | 0.0000 |
| Joint goodput (req/s) | 0.2167 | 0.2167 | 0.0000 |
| TTFT SLO | 85.48% | 75.31% | -10.17 pp |
| TPOT SLO | 4.36% | 4.56% | +0.21 pp |
| Joint SLO | 2.70% | 2.70% | 0.00 pp |
| Completed / unfinished / rejected / aborted | 482/0/0/0 | 482/0/0/0 | — |

The earlier full D1 absolute values (`goodput=0.357`, Joint `4.44%`, TPOT
`6.14%`) are from a different, longer measurement window and must not be
numerically compared to this 60-second sanity. The paired conclusion is that
the large D1-vs-D0 goodput/Joint/TPOT gap is absent in this sanity. TTFT and its
tail are not fully recovered, so this is not evidence to proceed to D2/D3.

### Latency and queue stages (milliseconds)

Each cell is mean / P50 / P95 / P99.

| Stage | D0 | D1 fixed |
|---|---:|---:|
| Frontend wait | 0.87 / 0.65 / 1.36 / 4.76 | 0.65 / 0.59 / 1.06 / 2.27 |
| Local scheduler wait | 5.78 / 5.81 / 10.51 / 11.32 | 140.59 / 9.90 / 630.39 / 2142.01 |
| Engine/prefill wait | 102.63 / 55.21 / 355.79 / 527.55 | 91.56 / 34.42 / 393.76 / 582.60 |
| Actual prefill service | 64.57 / 44.25 / 157.24 / 315.01 | 61.16 / 43.03 / 123.39 / 333.10 |
| Prefill to first decode | 93.81 / 38.22 / 315.35 / 708.55 | 119.98 / 40.49 / 418.27 / 1378.22 |
| TTFT | 173.85 / 126.09 / 482.26 / 614.99 | 293.95 / 155.79 / 934.17 / 2394.29 |
| TPOT | 60.22 / 55.39 / 103.04 / 140.41 | 66.41 / 58.74 / 123.43 / 233.05 |
| E2E | 9914.30 / 7405.75 / 25929.96 / 31666.38 | 10606.33 / 8274.89 / 27307.09 / 34579.24 |

## Gate decision

Do not run D2/D3, tau calibration, or a final/full sweep. The primary
goodput/Joint/TPOT regression is at D0 parity in the paired sanity, but the
remaining TTFT/local-wait tail requires separate evidence before any broader
experiment is authorized.
