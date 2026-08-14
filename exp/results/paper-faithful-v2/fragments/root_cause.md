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
