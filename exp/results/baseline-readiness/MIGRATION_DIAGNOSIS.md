# D2 migration-only diagnosis

## Verdict

`IMPLEMENTATION STALL`

The single allowed D2 run used Algorithm 2 OFF, migration ON, tau 0.07, bursty
rate 20, seed 1. Tau was diagnostic only. No D3, calibration, or sweep was run.
There were four migration decisions: three completed with a full timeline; the
fourth exposed a deterministic residency-accounting defect and terminated the
server by GPU OOM. Percentiles below therefore use the three completed events
(`n=3`), while the failure verdict uses the fourth event's code and runtime
evidence.

## Completed migration phase timing

All values are seconds in mean / P50 / P95 / P99 / max order.

| phase | mean | P50 | P95 | P99 | max |
|---|---:|---:|---:|---:|---:|
| target prepare | 2.109238 | 1.642962 | 3.016925 | 3.139055 | 3.169587 |
| target ready to quiesce | 0.049931 | 0.026101 | 0.103084 | 0.109927 | 0.111638 |
| quiesce control | 0.045904 | 0.021346 | 0.099517 | 0.106466 | 0.108203 |
| request drain | 0.915252 | 1.063470 | 1.414122 | 1.445291 | 1.453083 |
| KV stash | 1.877916 | 1.551167 | 2.772343 | 2.880892 | 2.908029 |
| weight transfer | 0.940571 | 0.422774 | 2.024597 | 2.166981 | 2.202577 |
| KV transfer | 8.712372 | 7.347063 | 12.469107 | 12.924400 | 13.038223 |
| target inject total | 12.071137 | 12.441610 | 15.636620 | 15.920621 | 15.991621 |
| KV inject exclusive | 3.358765 | 2.953398 | 4.880432 | 5.051724 | 5.094547 |
| routing switch | 0.00001184 | 0.00000954 | 0.00001576 | 0.00001631 | 0.00001645 |
| routing to first target request | 0.367798 | 0.056608 | 0.915898 | 0.992279 | 1.011374 |
| first target request to first decode | 5.848749 | 4.244419 | 10.529469 | 11.088140 | 11.227808 |
| exposed service downtime | 16.634287 | 19.069322 | 19.264199 | 19.281521 | 19.285852 |
| total migration wall | 24.644921 | 24.970129 | 31.427454 | 32.001438 | 32.144934 |

Routing itself is approximately 12 microseconds and is not the stall. Among
the completed events, target injection dominates exposed downtime: KV transfer
averages 8.71 s and exclusive target-side inject/rebuild another 3.36 s. The
post-routing first-decode interval averages 5.85 s but is separated from the
exposed-downtime definition.

## Fourth migration failure and root cause

The first three weight moves correctly used live GPU residency:

- model6 GPU1 to GPU0: `src=1`, GPU-to-GPU P2P, 2.2026 s
- model4 GPU0 to GPU1: `src=0`, GPU-to-GPU P2P, 0.4228 s
- model1 GPU0 to GPU1: `src=0`, GPU-to-GPU P2P, 0.1964 s

After target-first preparation, the source release handler executes
`self.v4_resident.pop(released_key, None)` without checking which GPU that
record names. Since target preparation has already replaced the entry with the
new target `(gpu_id, state_dict)`, releasing the old source removes the new
target residency record. The log makes this explicit: after model4 GPU0 to GPU1
completed, source release logged `gpu 0 held=1`--it popped the GPU1 target.

The reverse model4 migration GPU1 to GPU0 then reported `src=None` and used a
CPU cold-load path even though GPU1 was the active source. The controller also
reported `blocked=[]` and `rejected_by_memory=0`. GPU0 was already around
74,040 MiB; the unnecessary 6.794 GB cold-load drove it to about 81,152 MiB,
after which kvcached failed `cuMemCreate` with CUDA out-of-memory and server PID
179433 was killed. The benchmark stopped at 3,914 successful responses and the
watchdog finalized the pipeline as FAIL (rc 143); no result was fabricated.

This is not evidence that migration's intrinsic transfer cost alone caused the
run failure. It is a concrete implementation/accounting stall at the residency
lifecycle boundary.

## Minimal fix target (not applied in this diagnostic task)

Keep Moore--Hodgson, Algorithm 1, tau, and migration policy unchanged. Make
source release conditional on the stored residency still belonging to the GPU
being released (or represent source and target residency separately during the
handoff), so a late source release cannot delete the committed target mapping.
Target preparation must also fail closed against actual target memory headroom
before allocating a cold copy. After that narrow fix, rerun only this single D2
diagnostic gate before considering any wider experiment.

The fix is deliberately not included here: this session's specified endpoint
was migration root-cause diagnosis and minimal fix identification.
