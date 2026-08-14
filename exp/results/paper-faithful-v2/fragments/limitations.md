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
