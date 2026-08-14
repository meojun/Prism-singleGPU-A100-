## 7. Steady vs Bursty Comparison

_No aggregated runs yet._
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
