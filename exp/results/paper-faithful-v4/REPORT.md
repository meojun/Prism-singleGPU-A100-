# Paper-Faithful Prism V4 — 측정 보고서

_`exp/scripts/build_report_v4.py` 가 이 디렉터리의 raw data 로부터 생성._

## 0. 실험 환경

**2×A100 allocation on a 4×A100 node.** 할당된 GPU 는 0,1 두 장뿐이며 GPU 2,3 은
어떤 프로세스에도 노출되지 않는다 (`CUDA_VISIBLE_DEVICES=0,1`).
각 런의 `gpu_timeline.txt` 는 네 장 모두를 2 초 간격으로 샘플링하므로,
나머지 두 장이 유휴였다는 것은 주장이 아니라 raw data 로 남는다.

```

```

이전 V3 보고서는 **다른 장비**에서 측정되었다. 그래서 이 연구는 released prototype 과
V3 를 여기서 다시 돌린다. 이 보고서 안의 arm 간 비교만 유효하며, 이전 V3 보고서의
절대 수치와 직접 비교해서는 안 된다. 자세한 차이는 `provenance/ENVIRONMENT.md`.

## 1. Microbenchmark — 병렬 가중치 로딩

_아직 실행되지 않았다._

## 2. Microbenchmark — 마이그레이션

_아직 실행되지 않았다._

## 3. TP=2 검증

_아직 실행되지 않았다._

## 4. End-to-End

_아직 완료된 런이 없다._

## 6. Raw data

집계가 raw data 를 대체하지 않는다. 다음이 모두 보존되어 있다.

| 경로 | 내용 |
| --- | --- |
| `raw/requests/*.csv` | 요청 단위: 도착/완료 시각, 프롬프트·출력 토큰, TTFT/TPOT/E2E, SLO 충족 여부 |
| `raw/migrations/*.csv` | 마이그레이션 단위: 시각, latency, downtime, 바이트, 경로, 대역폭, KVPR |
| `raw/migrations/*_weight_transfers.jsonl` | 가중치 전송 단위: 경로별 바이트, 시간, 대역폭 |
| `raw/scheduler/*_alg1.jsonl` | 사이클별 Algorithm 1 전체 배치 계획과 차단 사유 |
| `raw/scheduler/*_actions.jsonl` | 제어 액션 단위 타이밍 |
| `raw/gpu_metrics/*.csv` | GPU 4장 전부의 2초 간격 사용률·메모리 |
| `microbench/*.json` | microbenchmark 원자료 |
| `profiling/` | 이 장비에서 측정한 c_i 와 SLO 기준선 |
| `logs/` | 런별 실행 로그 |

GPU 샘플링 간격은 2 초다. `nvidia-smi` 호출 1 회/2 초는 벤치마크와 GPU 를 공유하지 않으므로
측정에 영향을 주지 않는다.

## 7. 한계

- KV-cache 마이그레이션은 어느 arm 에도 구현되어 있지 않다. 마이그레이션 바이트는 전부 가중치다.
- RDMA 는 단일 노드라 측정 대상이 없다.
- TP anti-affinity 는 전역 컨트롤러가 TP 그룹을 rank0 GPU 로 축약하므로 표현 자체가 불가능하다.
- 결과는 A100 80GB 2장, 6모델, 이 SLO scale 에 한정된다.
