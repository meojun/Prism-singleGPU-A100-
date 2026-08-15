## 3. 구현 상태

arXiv **2505.04021v3** (OSDI 개정판) 원문과 핀 고정된 프로토타입
`Multi-LLM/prism-research @ 595ec1f` 를 줄 단위로 대조했다.

### 3.1 논문에 명시된 것 — 쓰인 그대로 구현

| 논문 | 구현 위치 |
| --- | --- |
| Alg. 1 line 1: 모델을 `t_j * tz_j / s_j` 내림차순 정렬 | `kvpr_global.py::_greedy_placement` |
| Alg. 1 lines 2-3: `shared_kv_i <- C`, `w_token_rate_i <- 0` | 동일 |
| Alg. 1 line 6: `w_token_rate_i / shared_kv_i` 를 최소화하는 GPU 선택 | 동일 |
| Alg. 1 lines 9-11: 배정 후 두 누적기 갱신 | 동일 |
| `tz_j` = 토큰당 KV 바이트 | `model_info.json::cell_size` |
| Alg. 2 line 1: `d_i = a_i + s_i` 오름차순 정렬 | `moore_hodgson.py::select` |
| Alg. 2 lines 4-6: `e_r = p_r / c_r`, 추가, `clock += e_r` | 동일 |
| Alg. 2 lines 7-11: 초과 시 **가장 긴** 작업 제거 후 시계 되감기 | 동일 |
| Alg. 2 line 12: `S` 를 스케줄 순서로 디스패치 | `request_queue_mh.py` |
| GPU 별 공유 요청 큐 (§6.2) | 프로토타입 `RequestQueue` 재사용 |
| 유휴 모델 축출 / 요청 도착 시 재활성화 | 프로토타입 `SimpleGlobalPolicy` 상속 |
| TP anti-affinity | 구현했으나 한 번도 발동 안 함 — 전 구성이 TP=1 |
| GPU 메모리 벌루닝 (§5) | 핀 고정 `kvcached` `prism/shm`, 무수정, 두 arm 동일 |

`released-prototype` 은 상류 코드 경로를 그대로 실행한다. 우리가 추가한 모든 경로는
`--policy kvpr-global` / `--enable-moore-hodgson` 뒤에 있어 그 arm 에서는 실행되지 않는다.
런마다 서버 로그의 `[PAPER-ALG1]` / `[PAPER-ALG2]` 표식 개수로 이를 검증했다(§4 검사 0).

### 3.2 우리의 가정 — 논문이 정하지 않은 것들

| 항목 | 논문 | 우리 결정 | 이유 |
| --- | --- | --- | --- |
| `tau` 의 단위 | Alg. 1 line 8 이 `current_r - best_r > tau` 로 절대 비교하는데, KVPR 은 차원을 가지며 단위도 값도 제시하지 않음 | 무차원. 클러스터 **peak** KVPR 의 상대 감소분, 기본 0.35 | "클러스터 최대 KVPR 을 억제한다" 가 이 greedy 근사에 대한 논문 자신의 목표 서술(Analysis / App. A.2)이다. 상대 기준은 구성이 달라도 비교 가능하지만 절대 기준은 그렇지 않다 |
| Alg. 1 line 10 누적항 | `+= r_k / s_k` 로 인쇄되어 line 1 의 정렬 키 `t_j*tz_j/s_j` 와 모순 | 양쪽 모두 `t_k*tz_k/s_k` | line 10 은 분자가 요청률이던 arXiv v1 의 잔재다 |
| KVPR 의 `s_j` | "latency SLO" | **TPOT** SLO 기준선 × `--kvpr-tpot-slo-scale` | KVPR 은 메모리 압력을 모델링하고, 메모리 여유가 지배하는 것은 TPOT 이다(논문 §6.2 Analysis) |
| `t_j` 정의 | "token rate" | 슬라이딩 창 기준 admit 된 입력 tok/s **+** 엔진 보고 디코드 tok/s | KV 캐시가 실제로 자라는 속도 |
| 토큰율 측정 창 | 미지정 | 30초 | 프로토타입의 `ModelRequestTracker` 창과 맞춰, 창 길이가 두 arm 사이 교란변수가 되지 않게 함 |
| 전역 스케줄러 주기 | 미지정 | 5초 | 프로토타입 하드코딩 `SCHEDULE_INTERVAL`. 두 arm 동일 |
| 마이그레이션 쿨다운 | 논문에 항목 자체가 없음 | 30초 | 결정은 5초마다지만 비율 추정은 30초 평균이라, 연속 패스는 입력의 25/30 을 공유해 독립 관측이 아니다 |
| 사이클당 마이그레이션 수 | 미지정 | 최대 1회 | 프로토타입과 동일. 여기서 마이그레이션은 stop-the-world 다 |
| GPU 를 비우지 않음 | 미지정 | 강제 | 런처가 초기 배치에 있는 GPU 에만 스케줄러를 띄우므로, 비워진 GPU 는 런 내내 죽어 있다 |
| `c_i` | "모델이 결정하는 chunked-prefill 속도" — 값도 방법도 없음 | **측정값**: 포화 상태 prefill burst 에서의 총 prefill 토큰 처리량. 엔진의 `out_queue -> prefill_finish` 타임스탬프 기준 | §3.4 참조 |
| Alg-2 가 제외한 요청의 처리 | 미지정 | `d_i > now` 재큐잉, `d_i <= now` 는 `S` 뒤 최하위로 디스패치 | Moore-Hodgson 은 모든 작업이 실행된다는 전제에서 *늦은 작업의 개수*를 최소화한다. 이미 늦은 것을 붙잡아 두면 livelock 이며 판정이 매 라운드 동일하다(회귀 테스트 `test_moore_hodgson.py` #9) |
| 유휴 축출 임계 | "경험적", App. A.4 가 약 45초 인용 | 프로토타입의 `MODEL_IDLE_THRESHOLD = 50초` 그대로 | 논문과 정합적이고 두 arm 에 동일 적용 |
| SLO 절대값 | 저자 하드웨어 기준 | 이 장비에서 §7.1 방식으로 재측정, TTFT ×5 / TPOT ×3 | 논문 수치는 이 장비의 기준선이 아니다 |
| Joint-SLO goodput | 논문 지표가 아님 | 여기서 정의: 두 SLO 를 모두 만족한 완료 요청 수 / 측정 구간 | — |

### 3.3 남은 불일치 — 논문에 있으나 여기에 없는 것

| 논문 | 상태 | 비고 |
| --- | --- | --- |
| 오버랩 마이그레이션 (대상이 준비될 때까지 원본이 계속 서비스) | **미구현** | 프로토타입은 원본을 비활성화한 *뒤* 대상을 활성화하며 `evict_waiting_requests=True` 다. stop-the-world |
| 재사용 가능한 사전 초기화 엔진 풀 | **부분** | worker pool 은 있으나 컨텍스트 사전 예열은 없음 |
| 병렬 가중치 로딩 (§5.3) | **부분** | `model_sevice.py` 가 `broker_gpu_id = (broker_id + target_gpu_id + 1) % num_gpus` 를 계산하므로, GPU 2장에서는 비대상 broker 가 하나뿐이라 병렬도가 최대 2 |
| NVLink / GPUDirect 가중치·KV 전송 | **미구현** | — |
| CPU DRAM 축출 계층 | 사용 안 함 | — |
| 프로덕션 트레이스 (Hyperbolic / Novita / Arena) | **비공개** | `--csv-trace` 는 파싱만 되고 쓰이지 않는다. 여기 워크로드는 ShareGPT 내용 + 합성 도착 시각이다 |
| 기준선 MuxServe++ / QLM / ServerlessLLM | 미설치 | torch/vllm 핀이 충돌해 각각 자기 venv 가 필요하다 |
| §7.4 규모 (모델 58개 / GPU 32장) | 불가능 | GPU 2장 |
| TP > 1 | 사용 안 함 | 전 구성이 TP=1 |

### 3.4 v1 에서 무엇이 왜 바뀌었나

**`c_i`.** v1 은 `Σ prompt tokens / Σ TTFT` 를 **경합 상태** 런에서 계산해
`c_i = 4,214 tok/s` 를 얻었고, `e_i = p_i / c_i` 에 절편 항이 없으니 `c_i` 는 전체
prefill 시간을 재현하는 양이어야 한다는 논리를 근거로 삼았다. 문제는 분모다. 경합
상태의 TTFT 는 대부분 큐 대기이므로 그 비율은 prefill 속도가 아니라 큐 지연을 잰다.
논문은 `c_i` 를 "모델이 결정하는 chunked-prefill 속도" 라고 부르고, 최적성 논증
(§6.2 Analysis)은 prefill 이 매 inference step 에서 속도 `c` 로 돈다고 전제한다 —
이는 **엔진 처리량**이지 한 요청 지연의 역수가 아니다.

이 구분이 중요한 이유는, 단일 기계 실행가능성 검사 `clock += e_i` 가 옳은 모델인지를
그것이 결정하기 때문이다. `c_i` 를 **총 용량**으로 읽으면 요청들은 실제로 그 용량의
단일 prefill 파이프라인을 두고 줄을 서므로 누적 방식이 맞고, 배치 병렬도 보정은
필요 없다. **요청당 속도**로 읽으면 같은 검사가 이중 계산을 하고 과소 수용한다.
v1 의 under-admission 은 Algorithm 2 의 결함이 아니라 `c_i` 와 기계 모델의 불일치와
정합적이다.

v2 는 prefill 구간을 엔진의 `out_queue_timestamp -> prefill_finish_timestamp` 로
직접 재고, 네 가지 추정기를 나란히 보고한다(§3.5). Algorithm 2 에 넣는 값은 포화
상태의 총 처리량이다.

**모델 세트.** v1 은 동일한 Llama-3.1-8B 3개를 썼는데, 이러면 어느 쌍을 겹치든
KVPR 이 `2w / (C - 2×15.08)` 로 같아 Algorithm 1 이 결정할 것이 없다. v2 는 KV cell
size 가 파라미터 수와 단조가 아닌 6모델을 쓴다.

**워크로드.** v1 은 일정 유입률 ShareGPT 트레이스라 어떤 모델도 유휴가 되지 않고
회수할 메모리가 애초에 없었다. v2 는 shifting-bursty 와 정확히 페어링된 steady
대조군을 추가한다.

**계측.** 두 알고리즘이 이제 결정마다 구조화된 기록을 남기고, Algorithm 2 는
`eligible > 0 이면서 selected = 0` 이 20라운드 연속되면 `[PAPER-ALG2-WARN]` 을
띄운다 — v1 이 손으로 찾아낸 바로 그 병리다.
