# Implementation audit — Prism paper vs released prototype vs V3 vs V4

각 항목은 `FULL` / `PARTIAL` / `NOT IMPLEMENTED` / `NOT TESTABLE ON CURRENT HARDWARE` 중 하나이며, Evidence 열은 그 판정의 근거가 되는
이 저장소 안의 파일을 가리킨다. 판정은 코드를 읽고 런타임 로그로 확인한 결과다.

| Mechanism | Prototype | V3 | V4 | Evidence |
| --- | --- | --- | --- | --- |
| Algorithm 1 — KVPR placement | PARTIAL | FULL | FULL | Prototype balances `memory_per_request` (smoothed request counts) instead of `token_rate x token_size / SLO`; v3 implements the paper's rule with the literal absolute tau; v4 keeps that rule unchanged. `raw/scheduler/*_alg1.jsonl`. |
| Algorithm 1 — runtime placement audit | NOT IMPLEMENTED | NOT IMPLEMENTED | FULL | v4 logs the full placement plan, every blocked move with its reason, and a per-cycle convergence gap. `alg1_convergence_gap_mean` in `summary.csv`. |
| Algorithm 2 — Moore-Hodgson | NOT IMPLEMENTED | FULL | FULL | Prototype reduces to EDF with `net_available = inf`; v3/v4 run the feasibility test and longest-job removal. `[PAPER-ALG2]` in the GPU scheduler logs. |
| Parallel weight loading — broker split | FULL | FULL | FULL | Present upstream in `multi_thread_copy_model_to_gpu` and active in all three arms (`--enable-model-service`); half of every tensor crosses the helper GPU. |
| Parallel weight loading — page-locked host memory | NOT IMPLEMENTED | NOT IMPLEMENTED | FULL | v4 registers the shared mapping with `cudaHostRegister`; measured speed-up see microbench/loading.json over v3. |
| Parallel weight loading — pipelined helper leg | NOT IMPLEMENTED | NOT IMPLEMENTED | PARTIAL | Implemented and measured as an ablation (`v4-pipelined-helper`); it did not pay off on this hardware and is therefore not the production v4 path. |
| Overlap migration — target-first ordering | NOT IMPLEMENTED | FULL | FULL | Prototype deactivates the source first; v3/v4 activate the target and only then retire the source. `ordering` column in `raw/migrations/*.csv`. |
| NVLink / P2P weight migration | NOT IMPLEMENTED | NOT IMPLEMENTED | PARTIAL | v4 fills the target from the source GPU's resident weights over NVLink; microbenchmark shows see microbench lower latency and NVLink counters equal to the full model size. Not observed end to end (see REPORT). |
| KV-cache migration | NOT IMPLEMENTED | NOT IMPLEMENTED | NOT IMPLEMENTED | KV pages are owned by kvcached's per-GPU virtual-memory allocator and are dropped, not moved, on deactivation. No arm transfers KV state; `kv_bytes` is 0 everywhere and is reported as 0 rather than omitted. |
| RDMA transport | NOT IMPLEMENTED | NOT IMPLEMENTED | NOT TESTABLE ON CURRENT HARDWARE | Single node. The one RoCE NIC reaches every GPU only via `SYS`, so there is no second node to move weights to and nothing to measure. |
| TP anti-affinity | NOT IMPLEMENTED | NOT IMPLEMENTED | NOT IMPLEMENTED | The global controller collapses a TP group to its rank-0 GPU (`controller_global.py`: "For TP case, only consider rank0 state"), so no placement code can express, let alone enforce, an anti-affinity constraint. |
| TP=2 runtime validation | NOT IMPLEMENTED | NOT IMPLEMENTED | NOT RUN | `tp-validation/tp2_validation.json` — startup, rank-to-GPU mapping, NCCL, inference and per-GPU memory, each read back from the run's own logs. |
| Placement convergence | NOT IMPLEMENTED | NOT IMPLEMENTED | FULL | Measured, not assumed: `convergence_gap` per cycle in `raw/scheduler/*_alg1.jsonl`. |

## 판정 기준

- **FULL** — 논문이 기술한 메커니즘이 구현되어 있고, 런타임 로그로 실제 동작이 확인된다.
- **PARTIAL** — 일부만 구현되었거나, 구현했으나 이 하드웨어에서 이득이 없어 기본 경로가 아니다.
- **NOT IMPLEMENTED** — 해당 코드 경로가 존재하지 않는다.
- **NOT TESTABLE ON CURRENT HARDWARE** — 구현 여부와 무관하게 이 장비에서 측정할 수 없다.

`OverlapMigrateAction` (`scheduling/overlap_migration.py`) 는 V3 가 추가했으나 어디에서도
참조되지 않는 dead code 다. V3 의 target-first 동작은 이 클래스가 아니라
`execute_actions` 의 배치 순서와 readiness barrier 로 구현되어 있다.
