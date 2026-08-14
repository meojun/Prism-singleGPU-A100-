## 3. Implementation Status

Audited line-by-line against arXiv **2505.04021v3** (the OSDI revision) and the
pinned prototype `Multi-LLM/prism-research @ 595ec1f`.

### 3.1 Paper-specified — implemented as written

| Paper | Where |
| --- | --- |
| Alg. 1 line 1: sort models by `t_j * tz_j / s_j` descending | `kvpr_global.py::_greedy_placement` |
| Alg. 1 lines 2-3: `shared_kv_i <- C`, `w_token_rate_i <- 0` | same |
| Alg. 1 line 6: pick the GPU minimising `w_token_rate_i / shared_kv_i` | same |
| Alg. 1 lines 9-11: assign, then update both accumulators | same |
| `tz_j` = KV bytes per token | `model_info.json::cell_size` |
| Alg. 2 line 1: sort ascending by `d_i = a_i + s_i` | `moore_hodgson.py::select` |
| Alg. 2 lines 4-6: `e_r = p_r / c_r`, append, `clock += e_r` | same |
| Alg. 2 lines 7-11: on overflow drop the **longest** job and rewind the clock | same |
| Alg. 2 line 12: dispatch `S` in schedule order | `request_queue_mh.py` |
| Shared per-GPU request queue (Sec. 6.2) | prototype's `RequestQueue`, reused |
| Idle model eviction / reactivation on arrival | prototype's `SimpleGlobalPolicy`, inherited |
| TP anti-affinity | implemented; never fires — every model here is TP=1 |
| GPU memory ballooning (Sec. 5) | pinned `kvcached` `prism/shm`, unmodified, identical in both arms |

`released-prototype` runs the upstream code path bit-for-bit: every new code
path sits behind `--policy kvpr-global` / `--enable-moore-hodgson`.

### 3.2 Our assumptions — the paper does not fix these

| Item | Paper | Our decision | Why |
| --- | --- | --- | --- |
| `tau` units | Alg. 1 line 8 tests `current_r - best_r > tau`, but KVPR is dimensioned and no value is given | dimensionless: relative reduction of the cluster's **peak** KVPR, default 0.35 | "Bounding the maximum KVPR across the cluster" is the paper's own stated objective for this greedy step (Analysis / App. A.2); a relative test is comparable across setups, an absolute one is not |
| Alg. 1 line 10 accumulator | printed as `+= r_k / s_k`, contradicting the line-1 key `t_j*tz_j/s_j` | `t_k*tz_k/s_k` for both | line 10 is a leftover from arXiv v1, where the numerator was a request rate |
| `s_j` in KVPR | "latency SLO" | **TPOT** SLO baseline x `--kvpr-tpot-slo-scale` | KVPR models memory pressure, and memory headroom is what governs TPOT (paper Sec. 6.2 Analysis) |
| `t_j` definition | "token rate" | admitted input tokens/s over a sliding window **+** engine-reported decode tokens/s | the rate at which the KV cache actually grows |
| token-rate window | unspecified | 30 s | matches the prototype's own `ModelRequestTracker` window, so window length is not a confound between arms |
| global scheduler period | unspecified | 5 s | the prototype's hard-coded `SCHEDULE_INTERVAL`; identical in both arms |
| migration cooldown | not in the paper at all | 30 s | decisions are made every 5 s but the rate estimate is a 30 s average, so consecutive passes share 25/30 of their input and are not independent observations |
| migrations per cycle | unspecified | at most 1 | matches the prototype; a migration is stop-the-world here |
| never empty a GPU | unspecified | enforced | the launcher only starts a GPU scheduler for GPUs in the initial placement, so an emptied GPU stays dead for the rest of the run |
| `c_i` | "a chunked-prefill speed determined by the model that serves it" — no value, no method | **measured**: aggregate prefill token throughput under a saturating prefill burst, from the engine's own `out_queue -> prefill_finish` timestamps | see Sec. 3.4 |
| fate of Alg-2-excluded requests | unspecified | `d_i > now` requeued; `d_i <= now` dispatched after `S` at lowest priority | Moore-Hodgson minimises the *number* of late jobs assuming all jobs still run. Holding back already-late requests is a livelock — the verdict is identical every round (regression-tested, `test_moore_hodgson.py` #9) |
| idle eviction threshold | "empirical", App. A.4 cites ~45 s | prototype's `MODEL_IDLE_THRESHOLD = 50 s`, unchanged | consistent with the paper and identical in both arms |
| SLO absolute values | authors' hardware | re-measured on this box, Sec. 7.1 method, x5 TTFT / x3 TPOT | the paper's numbers are not this machine's baseline |
| Joint-SLO goodput | not a paper metric | defined here: completed requests meeting both SLOs / measurement window | — |

### 3.3 Remaining mismatch — present in the paper, absent here

| Paper | Status | Note |
| --- | --- | --- |
| Overlapped migration (source keeps serving until destination is ready) | **not implemented** | the prototype deactivates the source *then* activates the destination, with `evict_waiting_requests=True`. Stop-the-world. |
| Reusable pre-initialised engine pool | partially — worker pool exists, contexts are not pre-warmed | |
| Parallel weight loading (Sec. 5.3) | **partial** | `model_sevice.py` computes `broker_gpu_id = (broker_id + target_gpu_id + 1) % num_gpus`; with 2 GPUs there is one non-target broker, so parallelism is 2-way at best |
| NVLink / GPUDirect weight+KV transfer | **not implemented** | |
| CPU DRAM eviction tier | not exercised | |
| Production traces (Hyperbolic / Novita / Arena) | **not public** | `--csv-trace` parses but is unused. Our workloads are ShareGPT content with synthetic arrivals. |
| Baselines MuxServe++ / QLM / ServerlessLLM | not installed | conflicting torch/vllm pins; each needs its own venv |
| Sec. 7.4 scale (58 models / 32 GPUs) | out of reach | 2 GPUs |
| TP > 1 | not exercised | every model here is TP=1 |

### 3.4 What changed from v1, and why

**`c_i`.** v1 used `sum(prompt_tokens) / sum(TTFT)` over a *contended* run and
got ~4,214 tok/s, on the argument that `e_i = p_i / c_i` has no intercept so
`c_i` must reproduce total prefill time. The flaw is the denominator: under
contention TTFT is mostly queueing, so that ratio measures queue delay, not
prefill speed. The paper calls `c_i` "a chunked-prefill speed determined by the
model", and its optimality argument (Sec. 6.2 Analysis) assumes prefill runs at
every inference step at rate `c` — that is an **engine throughput**, not the
reciprocal of one request's latency.

This matters because it decides whether the single-machine feasibility test
`clock += e_i` is even the right model. With `c_i` read as an *aggregate*
capacity, requests genuinely do share one prefill pipeline of that capacity and
the sequential accumulation is correct. With `c_i` read as a *per-request*
speed, the test charges each request the full serial cost and under-admits.
v1's under-admission is therefore consistent with a `c_i`/machine-model
mismatch rather than with a defect in Algorithm 2 as written.

v2 measures the prefill interval directly, from the engine's own
`out_queue_timestamp -> prefill_finish_timestamp`, and reports four estimators
side by side (Sec. 3.5). The value fed to Algorithm 2 is the saturated
aggregate.

**Model set.** v1 used 3 x Llama-3.1-8B, which makes KVPR identical for every
placement (`2w / (C - 2*15.08)` whichever pair you colocate) and leaves
Algorithm 1 nothing to decide. v2 uses six models whose KV cell size is not
monotone in parameter count.

**Workload.** v1 used a constant-rate ShareGPT trace, so no model was ever idle
and there was never reclaimable memory to move. v2 adds the shifting-bursty
workload and its exactly-paired steady control.

**Instrumentation.** Both algorithms now emit a structured record per decision,
and Algorithm 2 raises `[PAPER-ALG2-WARN]` after 20 consecutive rounds with
`eligible > 0 and selected = 0` — the exact pathology v1 discovered by hand.
