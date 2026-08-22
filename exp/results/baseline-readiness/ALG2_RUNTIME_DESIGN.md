# Algorithm 2 runtime integration audit and design

## Scope and invariants

This change is runtime integration only. It does not change Moore--Hodgson
`select()`, deadlines, `e_i = p_i/c_i`, D0, Algorithm 1, migration, tau,
workload, or SLO definitions.

Required invariants:

1. The global order returned by Algorithm 2 is the order in which requests are
   admitted into SGLang prefill, across model processes on the same GPU.
2. Every dispatched-but-not-complete request remains visible to the next
   feasibility calculation as outstanding `p_i/c_i` work.
3. A bounded number of requests may be staged in independent model backends,
   but staging cannot grant prefill admission out of order.
4. Any missing sequence, duplicate admission, or out-of-order start fails
   closed and is recorded.

## Audited current path

The current path is:

```text
MooreHodgsonMixin.admission_control_mh()
  -> select(jobs, now + outstanding_work)
  -> bounded prefix of the global ordered schedule
GPUScheduler.run_scheduling_loop()
  -> assign dispatch_seq and outstanding record
  -> Redis backend_generate_request:<model>
Scheduler.recv_generation_requests() in each independent model process
  -> process_input_gen_requests()
  -> handle_generate_request()
  -> model-local waiting_queue
Scheduler.get_new_batch_prefill()
  -> BatchRunReq (prefill_start)
Scheduler.process_batch_result_prefill()
  -> PrefillCompleteReq
GPUScheduler
  -> validate start/completion and remove outstanding work
```

`select()` is global and ordered, but the runtime has one Redis queue and one
consumer process per model. Without additional metadata, `A(model1),
B(model2), C(model1)` can be consumed as `A,C` by model1 and `B` by model2.
Enqueue order alone therefore cannot establish GPU admission order.

The superseded repair kept only one dispatched-but-not-started request. The GPU
scheduler dispatched A, waited for A's engine-side `prefill_start` ACK, then
dispatched B. That was correct but added a scheduler-loop boundary per request
and created head-of-line wait in the frontend/shared queue. Its measured
consequence was local scheduler wait mean/P95 140.59/630.39 ms and TTFT P99
2394 ms. The implementation described below replaces that ACK gate.

## Alternatives considered

### Ordered enqueue only

Rejected. Independent per-model consumers do not preserve cross-key Redis
enqueue order.

### Dispatch everything and validate afterward

Rejected. This detects a violation after GPU admission and recreates invisible
backend work; it cannot guarantee correctness.

### Central per-request start-ACK gate (superseded)

This was the prior implementation. It is correct, but serializes dispatch on
the GPU scheduler's polling loop and fails the readiness latency gate.

### Backend sequence barrier (implemented)

Attach a monotonic per-GPU `alg2_seq` to every request in the global accepted
schedule. Pipeline a bounded number of ordered requests into model backends.
Each engine may fetch its own requests into a private staged list, but only the
engine holding the shared `next_alg2_seq` may move a request into its SGLang
waiting queue. The token advances only when that request is selected as a real
prefill batch, immediately after ordered backend-admit/start events are emitted.

This makes the admission decision at the component that actually creates the
prefill batch, removes the GPU scheduler's request-by-request ACK dependency,
and still produces:

```text
dispatch:       A(seq=101), B(seq=102), C(seq=103)
backend admit:  A(seq=101), B(seq=102), C(seq=103)
prefill start:  A(seq=101), B(seq=102), C(seq=103)
```

The shared sequence is advanced with an atomic Redis compare-and-advance. The
current engine emits its admit/start records before advancing, so a later
engine cannot report an earlier admission. A crash between the event and token
advance stalls and is terminated by the watchdog; it cannot silently reorder.

## Bounded pipeline and hidden-work accounting

The pipeline capacity is the number of active model backend consumers on the
GPU. This is a structural runtime bound, not a scheduling heuristic: it allows
the independent consumers that caused the integration gap to stage work while
preventing an unbounded backend backlog. Algorithm 2 still chooses the subset
and order.

Every pipelined request is inserted into the GPU scheduler's outstanding map at
dispatch, before it reaches Redis. Its full paper cost `p_i/c_i` is included in
subsequent feasibility calls while it is:

- in a model Redis queue,
- in an engine's private staged list,
- admitted to the engine waiting queue,
- running prefill.

Only final prefill completion removes it. Thus pipelining does not make backend
or running work invisible.

## Runtime evidence schema

Each request records:

```text
request_id, model, alg2_seq,
dispatch_ts, backend_admit_ts, prefill_start_ts, prefill_complete_ts,
order_ok
```

GPU scheduler validation maintains separate next-expected counters for backend
admission and initial prefill start. Completion may occur in a different order,
but must refer to a known, started outstanding request. Shutdown requires zero
outstanding records.

## Correctness gate before performance

1. Unit/integration: force A(model1), B(model2), C(model1); compare Algorithm 2,
   dispatch, backend admission, and prefill-start sequences.
2. Negative test: attempt to admit/start a later sequence and require
   fail-closed behavior.
3. Real same-GPU mixed-model test with the same sequence shape.
4. Write `alg2_runtime_events.csv` and require zero violations, zero loss,
   zero reject/abort, and zero outstanding at shutdown.

Only after all four pass may the steady8/seed1 sanity run. D2 is permitted only
if that sanity is baseline-ready. D3, tau calibration, and full sweeps remain
prohibited.

## Final validation result

The A(model1), B(model4), C(model1) same-GPU test dispatched all three before A
started, while backend admission and prefill start remained A, B, C. Across the
steady8 D1 run, GPU0 recorded 442 and GPU1 recorded 522 dispatch/admit/start/
complete events; both GPUs had exact order equality, zero violations, maximum
outstanding 3, and zero remaining outstanding work. The runtime integration is
therefore `BASELINE-READY`.
