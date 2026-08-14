# Paper-Faithful Prism — design analysis

Source of truth for the **paper** side: *Prism: Cost-Efficient Multi-LLM Serving via
GPU Memory Ballooning*, Algorithm 1 (KVPR-based global model placement) and
Algorithm 2 (Moore-Hodgson GPU-local request scheduling).

Source of truth for the **prototype** side: `Multi-LLM/prism-research`
@ `595ec1f170e75a43897a7a2ad58ac5a9820aa2e8` (the SHA pinned in `setup/pins.env`),
read directly — not inferred from the paper.

Naming, per the project brief: the released code is the **Released Prism Prototype**,
never "original Prism". Our implementation is **Paper-Faithful Prism**, never a claim
to have recovered the authors' unreleased implementation.

---

## 1. What the released prototype actually does

### 1.1 Global placement — `python/sglang/multi_model/scheduling/policy/simple_global.py`

Entry point `SimpleGlobalPolicy.gen_actions()` (line 779), driven by
`GlobalController.run_scheduling_loop()` with a hard-coded `SCHEDULE_INTERVAL = 5`
seconds (`controller_global.py:283`). Each pass does, in order:

1. idle-instance eviction (`MODEL_IDLE_THRESHOLD = 50` s, line 181)
2. migration search — `_find_optimal_migrations()` (line 661)
3. activation of inactive models that have pending requests
4. emit `DeactivateAction` / `ActivateAction`, deactivate-first

The migration policy is selected by an **instance attribute, not a flag**:

```python
self.migrate_policy = "memory_per_request"        # simple_global.py:185
assert self.migrate_policy in ["violation", "memory_per_request"]
```

`_find_optimal_migrations_by_memory()` (line 547) is therefore the live path. Its
metric is, per GPU (`_calculate_memory_per_request`, line 341):

```
memory_per_request(g) = (gpu_mem - Σ_{m on g} model_size(m)) / Σ_{m on g} smoothed_req_count(m)
```

where `smoothed_req_count` is `ModelRequestTracker.get_model_request_stats()` — the
mean over a **30 s window** of `len(running_reqs) + num_waiting_reqs`, i.e. a
**count of requests in flight**, not a rate, not tokens, not SLO-weighted.

A placement is "unstable" only when some GPU pair satisfies

```
max(mem_per_req) / min(mem_per_req) > MEMORY_PER_REQUEST_RATIO_THRESHOLD   # = 15, line 183
```

and a migration is emitted only if it strictly reduces the number of unstable pairs.
The model chosen to move is the one with the **fewest** requests (line 619).

### 1.2 GPU-local scheduling — `python/sglang/multi_model/scheduling/gpu/request_queue.py`

`GPUScheduler.run_scheduling_loop()` (`gpu_scheduler.py:128`) calls
`RequestQueue.admission_control()` (line 155) every ~10 ms. The queue is a min-heap
keyed by (`RequestWrapper._calculate_priority`, line 21):

```python
profiled_prefill_time = clamp(prompt_len * (0.5 / 1024), 0.2, 2)   # e_i
return req.arrival_time + req.slo - profiled_prefill_time          # d_i - e_i
```

`req.slo` is the **TTFT SLO in seconds** — `trace.py` sends `slo=slo_ttft`
(lines 334, 409), so `arrival_time + slo` is exactly the paper's deadline `d_i`.

`admission_control()` then pops in that order and admits essentially everything:

```python
net_available = float("inf")        # request_queue.py:137
```

with the only real filter being `models_to_skip` — models whose Redis backend queue
exceeds `_skip_model_threshold = 10`.

---

## 2. Difference vs paper Algorithm 1

| Paper Algorithm 1 | Released prototype | Same? |
| --- | --- | --- |
| per-model weight `token_rate × token_size / SLO` | smoothed **count** of in-flight requests | ✗ |
| `token_size` = KV bytes per token | not used anywhere | ✗ |
| SLO weighting (TPOT SLO in the denominator) | not used anywhere | ✗ |
| models processed in **descending weighted token rate** | no sort by any such quantity | ✗ |
| `KVPR = Σ weighted_token_rate / shared_kv` | `memory_per_request = shared_kv / req_count` | partially — same *goal*, different metric |
| pick the GPU giving the **lowest resulting KVPR** | pick a GPU that reduces the count of "unstable pairs" | ✗ |
| migrate when improvement > `τ` | migrate when the 15× ratio test flips | ✗ |

The denominator is the one component that genuinely matches: `gpu_mem - Σ model_size`
(line 365) is the paper's `shared_kv`. The numerator, the ordering, the selection rule
and the threshold semantics all differ.

Note the two metrics are *reciprocal in spirit*: KVPR is pressure-per-byte (higher =
worse), `memory_per_request` is bytes-per-request (higher = better). The prototype
migrates from the GPU with the lowest `memory_per_request` to the highest, which is
directionally the same balancing intent — this matters when interpreting results, since
a null result would mean "the cheap heuristic captured what KVPR captures", not "the
prototype ignores balance entirely".

**In practice the prototype's migration path almost never fires.** A 15× per-GPU
imbalance is required; a realistic one on this box is ~1.6×. The committed 2-GPU
rate sweep (`exp/results/4-rate-sweep/`) records exactly **1 migration** at every
offered load from 12 to 30 req/s.

## 3. Difference vs paper Algorithm 2

| Paper Algorithm 2 (Moore-Hodgson) | Released prototype | Same? |
| --- | --- | --- |
| deadline `d_i = a_i + s_i` | computed (`arrival_time + slo`) | ✓ |
| execution estimate `e_i = p_i / c_i` | `clamp(prompt_len·0.5/1024, 0.2, 2)` → fixed `c = 2048` tok/s | partially |
| sort by deadline ascending | min-heap on `d_i − e_i` | ✗ (that is LST/EDF-with-offset, not EDF) |
| add jobs, tracking cumulative completion time | no cumulative time is tracked | ✗ |
| on violation, **drop the longest-`e` job already selected** | absent | ✗ |
| dispatch the feasible set `S` | dispatches everything (`net_available = inf`) | ✗ |

Both ingredients (`d_i`, `e_i`) exist; the mechanism the optimality proof rests on —
feasibility check plus drop-the-longest — does not. The released path reduces to a
priority order, with no admission decision at all.

---

## 4. Files: changed vs reused

**Changed (all new files except where noted):**

| File | Role |
| --- | --- |
| `prism-research/python/sglang/multi_model/scheduling/policy/kvpr_global.py` | **new** — paper Algorithm 1 |
| `prism-research/python/sglang/multi_model/scheduling/gpu/moore_hodgson.py` | **new** — paper Algorithm 2, pure function |
| `prism-research/python/sglang/multi_model/scheduling/gpu/request_queue.py` | +opt-in Moore-Hodgson branch; prototype path untouched by default |
| `prism-research/python/sglang/multi_model/scheduling/controller_global.py` | register `kvpr-global` policy |
| `prism-research/python/sglang/multi_model/multi_model_server_args.py` | add `kvpr-global` choice + Alg-2 flags |
| `prism-research/python/sglang/multi_model/scheduling/gpu/gpu_scheduler.py` | pass Alg-2 config into `RequestQueue` |
| `bootstrap.sh` | one-line index fix (see §7) |

**Reused unchanged:** `kvcached-prism` (all ballooning), the engine, `benchmark.py`,
`trace.py`, `build_sharegpt_trace.py`, `make_config.py`, `analyze_slo.py`,
`collect_metrics.py`, `derive_slo_baseline.py`, `run_multigpu.sh` (the new runner is
additive and models itself on it).

The prototype's behaviour is preserved by construction: every new code path is behind
`--policy kvpr-global` / `--enable-moore-hodgson`, and the defaults
(`policy="simple-global"`, Moore-Hodgson off) reproduce the released prototype exactly.

---

## 5. What the paper does not pin down, and what we chose

| Paper item | Publicly unambiguous? | Our implementation | Why |
| --- | --- | --- | --- |
| token-rate measurement window | **Not specified** | 30 s sliding window | Matches the prototype's own `ModelRequestTracker` window, so window length is not a confound between the two arms |
| rate smoothing method | **Not specified** | arithmetic mean over the window | Simplest estimator; no tuning knob that could be tilted toward either arm |
| global scheduler interval | **Not specified** | 5 s | The prototype's hard-coded `SCHEDULE_INTERVAL`; keeps decision cadence identical across arms |
| migration threshold `τ` | **Not specified numerically** | `0.35`, `--kvpr-tau` | Set to `mean + 2σ` of the measured improvement estimator (§5a), i.e. above the sampling noise, so migrations fire only on imbalance that is not noise. Derived from the estimator's own distribution, **not** from any latency outcome |
| minimum interval between migrations | **Not specified** (paper has no such term) | `30 s`, `--kvpr-migration-cooldown` | Decisions run every 5 s but the rate estimate is a 30 s sliding mean, so consecutive passes share 25/30 of their input and are not independent observations. One full window means each migration is justified by substantially fresh data |
| `token_rate` definition | Partially — "newly admitted input tokens + running decode tokens" | `input_tok/s` over arrivals in the window **+** `output_tok/s` over decoding requests in the window | Follows the brief's reading: the rate at which KV cache actually grows |
| `token_size` | Yes (KV bytes/token) | `model_info.json["cell_size"]` | Already profiled upstream; Llama-3.1-8B = 131072 B/token |
| which SLO weights KVPR | Yes — TPOT SLO | per-slot TPOT baseline × `--tpot-slo-scale`, from `SLO_BASE_FILE` | The TPOT SLO is *not* plumbed to the controller (`trace.py` sends only `slo_ttft`), so it is loaded controller-side from the same baseline file the analysis uses |
| `shared_kv` | Partially | `max_mem_usage − Σ active model_size` on that GPU | `max_mem_usage` is the per-GPU budget the harness already hands the scheduler; using measured free memory would make the metric depend on transient allocator state |
| fate of requests Moore-Hodgson excludes | **Not specified** | kept in the queue with original `a_i`/`d_i`, reconsidered next round (~10 ms later) | The brief's preferred option. Dropping would change the completed-request set between arms and make TTFT/TPOT incomparable |
| per-model chunked-prefill speed `c_i` | **Not specified** | measured, `exp/configs/prefill_speed.json` (§6) | The prototype's `c = 2048 tok/s` is an unexplained constant; we profile it on this box |
| eviction threshold | **Not specified** | prototype's `MODEL_IDLE_THRESHOLD = 50 s`, unchanged | Paper §A.4 quotes ~45 s optimum; the prototype's 50 s is already consistent. Identical in both arms |
| paper's exact 2-GPU model configuration | **Not reproducible from public info** | 3 × Llama-3.1-8B in slots `model_1/4/5` | See §6.2 |

### 5a. The KVPR objective is flat in this configuration — how `τ` was set

Running Algorithm 1 with `τ = 0.10` on a 90 s smoke workload produced **8 migrations in
38 placement passes**, and they oscillated: `model_4` 1→0, `model_1` 0→1, `model_5` 1→0,
`model_4` 0→1, `model_4` 1→0, … Each one is a *stop-the-world* move in this prototype
(deactivate source, then activate target, `evict_waiting_requests=True`; §6.1's
overlapped migration is not implemented), so the arm would have been measuring
thrashing rather than placement quality.

The cause is structural, and it is a result in its own right. With three
similar-rate models on two GPUs the placement is forced to be 1+2, and the doubled GPU's
KVPR is

```
peak KVPR ≈ (w + w) / (67.28 − 2×15.08 GiB) = 2w / 37.12
```

which is **the same whichever model is doubled**. The objective is flat, so the argmin is
decided by estimator noise. Measured over the smoke run's 24 decisions:

```
improvement:  mean +0.002   std 0.175   median +0.013   range [−0.411, +0.396]
τ = 0.10 → migrates on 33 % of passes      τ = 0.35 → 4 %
```

The expected gain from migrating is **zero** (+0.2 %); everything above it is sampling
noise of ±17.5 %. So the paper's rule, applied with a `τ` above the noise, correctly
decides *not* to move — and `τ = mean + 2σ ≈ 0.35` is that threshold. This is derived
from the estimator's distribution, not from any TTFT/goodput outcome; no per-rate tuning
was applied to either arm.

Two things follow for the report. Algorithm 1 has **no lever** in this 3-model/2-GPU
configuration, so a null placement result here is expected and is not evidence against
KVPR. And in the paper's own setting — more models, more GPUs, heterogeneous sizes and
rates — the objective is *not* flat, so `τ` there separates real imbalance from noise
rather than suppressing everything.

**Anti-affinity / TP:** the paper constrains TP shards of one model away from sharing a
GPU. Our configuration is TP=1 throughout, so the constraint is implemented
(a candidate GPU already holding a shard of the same model is rejected) but is never
exercised. Stated so the omission is not mistaken for a silent simplification.

---

## 6. Experiment-side decisions

### 6.1 `c_i` is measured, not assumed

The brief forbids an arbitrary constant. There is no per-model prefill-speed table in
`prism-research`, `model_info.json`, or `kvcached`. So `exp/scripts/profile_prefill_speed.py`
derives it from a solo, uncontended run and persists it to
`exp/configs/prefill_speed.json`, which the server reads at start-up so `c_i` cannot
drift between runs of the sweep.

**Estimator choice matters more than it looks.** Two defensible estimators differ by 5×
on this box (525 uncontended Llama-3.1-8B samples):

| estimator | value | meaning |
| --- | ---: | --- |
| regression slope of `ttft = a + p/c` | 20,775 tok/s (intercept **29 ms**) | marginal cost of one extra prompt token |
| ratio `Σ prompt_len / Σ ttft` | **4,214 tok/s** | speed that reproduces the whole prefill time |

The paper's `e_i = p_i / c_i` has **no intercept term**, so `c_i` must be the quantity
that reproduces total prefill time — the ratio estimator. Using the slope would put
every `e_i` in the 3–30 ms range, an order of magnitude below real prefill time, and
Moore-Hodgson's feasibility test would essentially never fire; the algorithm would be
present but inert. The slope and intercept are still recorded in
`prefill_speed_detail.json` as diagnostics. Prompts here are short (p50 = 70 tokens,
p99 = 608), which is why the fixed overhead dominates and the two estimators diverge so
far; on a workload with longer prompts they would converge.

For reference the released prototype's implied constant is 2048 tok/s
(`clamp(prompt_len*0.5/1024, 0.2, 2)`), within 2× of the measured ratio value.

### 6.1a Measured SLO baseline

Re-derived on this box the paper-§7.1 way (uncontended p95 over 702 solo requests):
**TTFT p95 = 125.7 ms, TPOT p95 = 21.41 ms**. With the study's ×5 / ×3 scales the SLOs
are **628.7 ms TTFT** and **64.23 ms TPOT**. The built-in table in `trace.py` is the
authors' hardware and is not this machine's baseline; both arms use the re-derived
values, and `--slo-base-file` feeds the same numbers to Algorithm 1's KVPR weighting.

### 6.2 Model configuration

3 × `meta-llama/Llama-3.1-8B` in slots `model_1`, `model_4`, `model_5`.

* Slots are not free choices — `trace.py::generate_e2e_benchmark_reqs` hard-codes a
  per-slot SLO baseline measured for one specific model; 1/4/5 are the three 8B slots.
* 3 models on 2 GPUs is necessarily a 1+2 split, so a placement decision genuinely
  exists and migration has somewhere to go. This is the same configuration as the
  committed `exp/results/4-rate-sweep/`, which keeps our numbers comparable to it.
* The paper's 8-model mix is not reproducible on 2 GPUs at these rates, and its exact
  composition is not public. We do not claim to reproduce it.

### 6.3 Request rate semantics

`build_sharegpt_trace.py --variant rate --phase-rates` takes **per-slot** rates.
This study is specified in **aggregate** offered rate, so each aggregate `R` is emitted
as `R/3` per slot. Recorded in every run's metadata as
`request_rate_semantics: "aggregate over all models"`.

Rates swept: **2.5, 5, 7.5, 10, 15, 20, 25, 30 req/s** × seeds 1,2,3 × 2 systems = 48 runs.
The 15–30 range was added because the committed capacity profile puts this box's TTFT
knee at ~26 req/s: at ≤10 req/s the queue is empty and the KV pool is under 20 %, so
neither Moore-Hodgson nor KVPR has anything to act on. Both ranges are reported.

### 6.4 Warm-up and measurement

No existing convention in the repo covers this (the committed sweeps measure the whole
trace), so the brief's fallback is used: traces are 360 s, the first **60 s is warm-up
and excluded**, the following **300 s is the measurement window**. Requests are
assigned to the window by trace arrival time, and the same window is applied to both
arms.

### 6.5 TPOT definition

`tpot = (finish_time − prefill_finish_time) / (output_len − 1)` as computed by the
harness' `benchmark.py`. We do **not** substitute mean-ITL or e2e/output_len. Note the
harness' own `average_attainment_tpot` field is unusable (ms-vs-s unit bug documented in
`CLAUDE.md`); all TPOT attainment here is recomputed from the raw per-request dump.

### 6.6 Joint-SLO Goodput

Not a metric the paper defines, so it is defined here:

```
Joint-SLO Goodput [req/s] = |{completed requests meeting BOTH TTFT and TPOT SLO}| / 300 s
```

Plain completed throughput is recorded separately.

---

## 7. Environment deviation from the committed baseline

`bootstrap.sh` step 3 failed on this box: `torch==2.4.0+cu121` pins
`nvidia-cudnn-cu12==9.1.0.70` exactly, and `download.pytorch.org/whl/cu121` has since
pruned that file (it serves 9.0.0.312, then jumps to 9.2x). PyPI still carries it. Fixed
by adding `--extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match`
so torch still resolves to `2.4.0+cu121` from the pytorch index while its `nvidia-*`
dependencies come from PyPI. Verified: torch 2.4.0+cu121, sglang 0.3.4.post2,
vllm 0.6.3.post1, transformers 4.45.2, flashinfer 0.1.6, kvcached+vmm_ops, CUDA available.
Original failure log preserved. This is upstream index drift, not a repo defect, and it
affects both arms identically.
