# Paper-Faithful Prism — 설계 분석

**논문** 쪽 기준: *Prism: Cost-Efficient Multi-LLM Serving via GPU Memory Ballooning*,
Algorithm 1 (KVPR 기반 전역 모델 배치)과 Algorithm 2 (Moore-Hodgson GPU-로컬 요청
스케줄링).

**프로토타입** 쪽 기준: `Multi-LLM/prism-research`
@ `595ec1f170e75a43897a7a2ad58ac5a9820aa2e8` (`setup/pins.env` 에 핀 고정된 SHA).
논문에서 유추하지 않고 코드를 직접 읽었다.

명명 규칙(프로젝트 지침): 공개된 코드는 **Released Prism Prototype** 이며 "Original
Prism" 이라고 부르지 않는다. 우리 구현은 **Paper-Faithful Prism** 이며, 저자들의
미공개 구현을 복원했다는 주장이 아니다.

---

## 1. 공개 프로토타입은 실제로 무엇을 하는가

### 1.1 전역 배치 — `python/sglang/multi_model/scheduling/policy/simple_global.py`

진입점은 `SimpleGlobalPolicy.gen_actions()` (779행)이고,
`GlobalController.run_scheduling_loop()` 이 하드코딩된 `SCHEDULE_INTERVAL = 5` 초
(`controller_global.py:283`) 주기로 호출한다. 매 패스는 순서대로:

1. 유휴 인스턴스 축출 (`MODEL_IDLE_THRESHOLD = 50` s, 181행)
2. 마이그레이션 탐색 — `_find_optimal_migrations()` (661행)
3. 대기 요청이 있는 비활성 모델의 활성화
4. `DeactivateAction` / `ActivateAction` 방출, 비활성화 먼저

마이그레이션 정책은 **플래그가 아니라 인스턴스 속성**으로 선택된다:

```python
self.migrate_policy = "memory_per_request"        # simple_global.py:185
assert self.migrate_policy in ["violation", "memory_per_request"]
```

따라서 실제로 살아 있는 경로는 `_find_optimal_migrations_by_memory()` (547행)이다.
지표는 GPU 당 (`_calculate_memory_per_request`, 341행):

```
memory_per_request(g) = (gpu_mem - Σ_{m on g} model_size(m)) / Σ_{m on g} smoothed_req_count(m)
```

여기서 `smoothed_req_count` 는 `ModelRequestTracker.get_model_request_stats()`, 즉
**30초 창**에서의 `len(running_reqs) + num_waiting_reqs` 평균이다. 다시 말해
**처리 중인 요청의 개수**이며, 비율(rate)도 아니고 토큰도 아니고 SLO 가중도 아니다.

배치가 "불안정" 하다고 판정되는 조건은 어떤 GPU 쌍이 다음을 만족할 때뿐이다:

```
max(mem_per_req) / min(mem_per_req) > MEMORY_PER_REQUEST_RATIO_THRESHOLD   # = 15, 183행
```

그리고 불안정 쌍의 **개수를 엄격히 줄이는** 경우에만 마이그레이션이 방출된다.
옮길 모델로는 요청이 **가장 적은** 모델이 선택된다(619행).

### 1.2 GPU-로컬 스케줄링 — `python/sglang/multi_model/scheduling/gpu/request_queue.py`

`GPUScheduler.run_scheduling_loop()` (`gpu_scheduler.py:128`)이 약 10 ms 주기로
`RequestQueue.admission_control()` (155행)을 호출한다. 큐는 다음 키의 최소 힙이다
(`RequestWrapper._calculate_priority`, 21행):

```python
profiled_prefill_time = clamp(prompt_len * (0.5 / 1024), 0.2, 2)   # e_i
return req.arrival_time + req.slo - profiled_prefill_time          # d_i - e_i
```

`req.slo` 는 **초 단위 TTFT SLO** 다 — `trace.py` 가 `slo=slo_ttft` 를 보낸다
(334, 409행). 따라서 `arrival_time + slo` 는 논문의 마감 `d_i` 와 정확히 같다.

`admission_control()` 은 그 순서대로 pop 하면서 사실상 전부 admit 한다:

```python
net_available = float("inf")        # request_queue.py:137
```

실질적인 필터는 `models_to_skip` 하나뿐이다 — Redis 백엔드 큐가
`_skip_model_threshold = 10` 을 넘은 모델을 건너뛴다.

---

## 2. 논문 Algorithm 1 과의 차이

| 논문 Algorithm 1 | 공개 프로토타입 | 같은가? |
| --- | --- | --- |
| 모델별 가중치 `token_rate × token_size / SLO` | 처리 중 요청 **개수**의 평활값 | ✗ |
| `token_size` = 토큰당 KV 바이트 | 어디에도 쓰이지 않음 | ✗ |
| SLO 가중(분모의 TPOT SLO) | 어디에도 쓰이지 않음 | ✗ |
| 가중 토큰율 **내림차순**으로 모델 처리 | 그런 양으로 정렬하지 않음 | ✗ |
| `KVPR = Σ weighted_token_rate / shared_kv` | `memory_per_request = shared_kv / req_count` | 부분적 — *목표*는 같고 지표가 다름 |
| 결과 KVPR 이 **가장 낮아지는** GPU 선택 | "불안정 쌍" 개수를 줄이는 GPU 선택 | ✗ |
| 개선폭 > `τ` 일 때 마이그레이션 | 15배 비율 검사가 뒤집힐 때 마이그레이션 | ✗ |

진짜로 일치하는 구성요소는 분모 하나다. `gpu_mem - Σ model_size` (365행)가 논문의
`shared_kv` 다. 분자, 정렬, 선택 규칙, 임계값 의미는 전부 다르다.

두 지표가 *정신적으로는 역수 관계* 라는 점에 유의해야 한다. KVPR 은 바이트당 압력
(높을수록 나쁨)이고 `memory_per_request` 는 요청당 바이트(높을수록 좋음)다.
프로토타입은 `memory_per_request` 가 가장 낮은 GPU 에서 가장 높은 GPU 로 옮기므로
방향성으로는 같은 균형 의도를 갖는다. 결과 해석에 중요한 지점이다 — 차이가 없다는
결과가 나오면 그것은 "프로토타입이 균형을 완전히 무시한다" 가 아니라 "값싼 휴리스틱이
KVPR 이 포착하는 것을 이미 포착했다" 를 뜻하기 때문이다.

**실제로 프로토타입의 마이그레이션 경로는 거의 발동하지 않는다.** GPU 간 15배
불균형이 필요한데 이 장비에서 현실적인 값은 약 1.6배다. 커밋된 2-GPU 유입률 스윕
(`exp/results/4-rate-sweep/`)은 12~30 req/s 전 구간에서 정확히 **마이그레이션 1회**를
기록한다.

## 3. 논문 Algorithm 2 와의 차이

| 논문 Algorithm 2 (Moore-Hodgson) | 공개 프로토타입 | 같은가? |
| --- | --- | --- |
| 마감 `d_i = a_i + s_i` | 계산함 (`arrival_time + slo`) | ✓ |
| 실행시간 추정 `e_i = p_i / c_i` | `clamp(prompt_len·0.5/1024, 0.2, 2)` → 고정 `c = 2048` tok/s | 부분적 |
| 마감 오름차순 정렬 | `d_i − e_i` 최소 힙 | ✗ (이는 EDF 가 아니라 오프셋이 붙은 LST) |
| 누적 완료 시각을 추적하며 작업 추가 | 누적 시간을 추적하지 않음 | ✗ |
| 위반 시 **이미 선택된 것 중 `e` 가 가장 긴 작업을 떨어뜨림** | 없음 | ✗ |
| 실행가능 집합 `S` 를 디스패치 | 전부 디스패치 (`net_available = inf`) | ✗ |

두 재료(`d_i`, `e_i`)는 존재하지만, 최적성 증명이 기대고 있는 메커니즘 —
실행가능성 검사 + 가장 긴 작업 제거 — 이 없다. 공개 경로는 admission 판단이 전혀
없는 단순 우선순위 정렬로 환원된다.

---

## 4. 파일: 변경한 것과 그대로 쓴 것

**변경(따로 표시한 것 외에는 전부 신규 파일):**

| 파일 | 역할 |
| --- | --- |
| `prism-research/python/sglang/multi_model/scheduling/policy/kvpr_global.py` | **신규** — 논문 Algorithm 1 |
| `prism-research/python/sglang/multi_model/scheduling/gpu/moore_hodgson.py` | **신규** — 논문 Algorithm 2, 순수 함수 |
| `prism-research/python/sglang/multi_model/scheduling/gpu/request_queue.py` | Moore-Hodgson 분기 추가(opt-in). 기본값에서는 프로토타입 경로 그대로 |
| `prism-research/python/sglang/multi_model/scheduling/controller_global.py` | `kvpr-global` 정책 등록 |
| `prism-research/python/sglang/multi_model/multi_model_server_args.py` | `kvpr-global` 선택지 + Alg-2 플래그 추가 |
| `prism-research/python/sglang/multi_model/scheduling/gpu/gpu_scheduler.py` | Alg-2 설정을 `RequestQueue` 로 전달 |
| `bootstrap.sh` | 인덱스 한 줄 수정 (§7 참조) |

**그대로 재사용:** `kvcached-prism`(벌루닝 전부), 엔진, `benchmark.py`, `trace.py`,
`build_sharegpt_trace.py`, `make_config.py`, `analyze_slo.py`, `collect_metrics.py`,
`derive_slo_baseline.py`, `run_multigpu.sh`(새 러너는 이를 본떠 만든 추가물이다).

프로토타입의 동작은 구조적으로 보존된다. 새로 추가한 모든 코드 경로는
`--policy kvpr-global` / `--enable-moore-hodgson` 뒤에 있고, 기본값
(`policy="simple-global"`, Moore-Hodgson 꺼짐)은 공개 프로토타입을 그대로 재현한다.

---

## 5. 논문이 확정하지 않은 것과 우리의 선택

| 논문 내용 | 공개적으로 명확한가 | 우리 구현 | 선택 이유 |
| --- | --- | --- | --- |
| 토큰율 측정 창 | **명시 없음** | 30초 슬라이딩 창 | 프로토타입 자체의 `ModelRequestTracker` 창과 일치시켜, 창 길이가 두 arm 사이의 교란 변수가 되지 않게 함 |
| 비율 평활 방법 | **명시 없음** | 창 내 산술평균 | 가장 단순한 추정기. 어느 한쪽으로 기울일 수 있는 튜닝 손잡이를 만들지 않음 |
| 전역 스케줄러 주기 | **명시 없음** | 5초 | 프로토타입의 하드코딩 `SCHEDULE_INTERVAL`. 두 arm 의 결정 주기를 동일하게 유지 |
| 마이그레이션 임계 `τ` | **수치 명시 없음** | `0.35`, `--kvpr-tau` | 측정된 개선폭 추정기의 `평균 + 2σ` (§5a). 즉 샘플링 잡음 위에 선을 그어, 잡음이 아닌 불균형에만 마이그레이션이 발동하게 함. 지연 결과가 **아니라** 추정기 자체의 분포에서 유도 |
| 마이그레이션 최소 간격 | **명시 없음** (논문에 해당 항 자체가 없음) | `30초`, `--kvpr-migration-cooldown` | 결정은 5초마다 내려지지만 비율 추정은 30초 슬라이딩 평균이므로, 연속된 패스는 입력의 25/30 을 공유하는 독립 관측이 아니다. 창 하나만큼 띄우면 각 마이그레이션이 충분히 새로운 데이터에 근거하게 된다 |
| `token_rate` 정의 | 부분적 — "새로 admit 된 입력 토큰 + 실행 중인 디코드 토큰" | 창 내 도착 기준 `input_tok/s` **+** 창 내 디코딩 요청 기준 `output_tok/s` | 지침의 해석을 따름: KV 캐시가 실제로 자라는 속도 |
| `token_size` | 명확함 (토큰당 KV 바이트) | `model_info.json["cell_size"]` | 상류에서 이미 프로파일됨. Llama-3.1-8B = 131072 B/token |
| KVPR 을 가중하는 SLO | 명확함 — TPOT SLO | `SLO_BASE_FILE` 의 슬롯별 TPOT 기준선 × `--tpot-slo-scale` | TPOT SLO 는 컨트롤러로 전달되지 않으므로(`trace.py` 는 `slo_ttft` 만 보낸다), 분석이 쓰는 것과 같은 기준선 파일에서 컨트롤러 쪽이 직접 읽는다 |
| `shared_kv` | 부분적 | 해당 GPU 의 `max_mem_usage − Σ 활성 model_size` | `max_mem_usage` 는 하네스가 이미 스케줄러에 건네주는 GPU 당 예산이다. 실측 여유 메모리를 쓰면 지표가 일시적인 할당기 상태에 의존하게 된다 |
| Moore-Hodgson 이 제외한 요청의 처리 | **명시 없음** | 마감 경과 여부로 분리: `d_i > now` → 원래 `a_i`/`d_i` 를 유지한 채 재큐잉, `d_i ≤ now` → 실행가능 집합 뒤에 디스패치 | 전부 재큐잉하는 것이 지침의 선호안이지만 **livelock** 이 발생한다(§5b). 버리면 두 arm 의 완료 요청 집합이 달라져 TTFT/TPOT 비교가 불가능해진다 |
| 모델별 chunked-prefill 속도 `c_i` | **명시 없음** | 측정값, `exp/configs/prefill_speed.json` (§6) | 프로토타입의 `c = 2048 tok/s` 는 근거가 설명되지 않은 상수다. 이 장비에서 직접 프로파일한다 |
| 축출 임계 | **명시 없음** | 프로토타입의 `MODEL_IDLE_THRESHOLD = 50 s` 그대로 | 논문 §A.4 는 최적값 약 45초를 인용하며, 프로토타입의 50초는 이미 이와 정합적이다. 두 arm 에 동일 적용 |
| 논문의 정확한 2-GPU 모델 구성 | **공개 정보로 재현 불가** | 슬롯 `model_1/4/5` 에 Llama-3.1-8B 3개 | §6.2 참조 |
| 실행가능성 검사의 기계 모델 | **명시 없음** — Algorithm 2 는 `1‖ΣU_j`, 즉 한 번에 한 작업만 처리하는 *단일* 기계로 서술됨 | 문자 그대로 채택: `clock += e_i`, 병렬성 항 없음 | 충실성이 우선이다. 문자 그대로의 해석이 우리가 측정하기로 한 대상이고, §8 의 결과가 보고하는 것도 그것이다. 동시에 이것이 고부하에서 **측정 가능한 지배적 효과**이기도 하다(§5c). 이 표에서 가장 중대한 모호성 항목이다 |

### 5a. 이 구성에서 KVPR 목적함수는 평평하다 — `τ` 를 정한 방법

90초 스모크 워크로드에서 `τ = 0.10` 으로 Algorithm 1 을 돌리면 **38회 배치 패스 중
8회 마이그레이션**이 발생했고, 그것들은 진동했다: `model_4` 1→0, `model_1` 0→1,
`model_5` 1→0, `model_4` 0→1, `model_4` 1→0, … 이 프로토타입에서 마이그레이션 한 번은
*stop-the-world* 동작이므로(원본 비활성화 후 대상 활성화, `evict_waiting_requests=True`.
§6.1 의 오버랩 마이그레이션은 구현되어 있지 않다), 이 상태로는 배치 품질이 아니라
스래싱을 측정하게 된다.

원인은 구조적이며, 그 자체가 하나의 결과다. 유입률이 비슷한 모델 3개를 GPU 2개에
올리면 배치는 반드시 1+2 가 되고, 두 개가 올라간 GPU 의 KVPR 은

```
peak KVPR ≈ (w + w) / (67.28 − 2×15.08 GiB) = 2w / 37.12
```

로 **어느 모델을 겹치게 놓든 같다.** 목적함수가 평평하므로 argmin 은 추정 잡음이
결정한다. 스모크 런의 24회 결정에서 측정된 값:

```
개선폭:  평균 +0.002   표준편차 0.175   중앙값 +0.013   범위 [−0.411, +0.396]
τ = 0.10 → 패스의 33 % 에서 마이그레이션      τ = 0.35 → 4 %
```

마이그레이션의 기대 이득은 **0** 이고(+0.2 %), 그 위의 모든 것은 ±17.5 % 의 샘플링
잡음이다. 따라서 논문의 규칙을 잡음보다 큰 `τ` 로 적용하면 **옮기지 않는 것이 올바른
결정**이며, `τ = 평균 + 2σ ≈ 0.35` 가 그 경계다. 이는 TTFT/goodput 결과가 아니라
추정기의 분포에서 유도한 값이고, 어느 arm 에도 유입률별 튜닝을 적용하지 않았다.

보고서에 따라오는 함의는 두 가지다. 이 3모델/2GPU 구성에서 Algorithm 1 은 **작동할
여지가 없으므로**, 여기서 배치에 관한 무차이 결과가 나오는 것은 예상된 일이며 KVPR 에
대한 반증이 아니다. 그리고 논문 자신의 설정 — 모델 수와 GPU 수가 많고 크기와 유입률이
이질적인 환경 — 에서는 목적함수가 평평하지 않으므로, 거기서의 `τ` 는 모든 것을
억제하는 것이 아니라 진짜 불균형을 잡음에서 분리하는 역할을 한다.

### 5b. 제외된 요청을 그냥 재큐잉할 수 없는 이유

지침의 선호안 — 제외된 요청을 원래 도착 시각과 마감을 유지한 채 큐에 남겨 다음
라운드에 재고 — 은 **livelock** 이며, 스모크 런에서 실제로 잡혔다:

```
round=328763  eligible=2  selected=0  deferred=2  (cumulative 651754)  queue_len=2
bench: Waiting for task req_105_model_1        <- 끝내 디스패치되지 않음
```

`d_i ≤ now` 가 되는 순간, 그 작업을 넣으면 완료 시계가 즉시 마감을 초과하므로
Moore-Hodgson 은 그것을 제거한다. 그리고 입력이 도움이 될 방향으로 전혀 바뀌지 않으므로
이후 모든 라운드에서 똑같이 제거한다. 이 상태에서 빠져나오는 경로가 없다. 스케줄러는
클라이언트가 그 두 요청에 블록된 채 32만 8천 라운드를 돌며 같은 요청을 재지연했다.

해결책은 Moore-Hodgson 이 실제로 최적화하는 대상에서 나온다. 이 알고리즘은 **늦은
작업의 개수**를 최소화하며, 그 전제는 *모든 작업이 실행된다* 는 것이다 — 늦은 것들은
정시 작업 뒤에 실행될 뿐이다. 따라서 제외된 요청을 둘로 나눈다:

* `d_i > now` — 이번 라운드에 밀렸지만 원리상 아직 실행 가능. 재큐잉한다. 이것이
  진짜 백프레셔이고, 백로그가 빠지면 다시 기회를 얻는다.
* `d_i ≤ now` — 이미 늦었고 영구히 실행 불가능. 실행가능 집합 뒤, 가장 낮은 우선순위로
  디스패치한다. 이미 SLO 위반으로 계수된 요청이므로, 굶기면 늦은 응답이 무응답으로
  바뀔 뿐이고 두 arm 의 완료 요청 집합이 달라진다.

`exp/tests/test_moore_hodgson.py` 의 케이스 9 가 이에 대한 회귀 테스트다. 두 부류가
분리 가능하다는 것과, 라운드가 바뀌어도 판정이 동일하다는 것을 검증한다. 후자가 바로
붙잡아 두는 방식이 통하지 않는 이유다.

**Anti-affinity / TP:** 논문은 한 모델의 TP 샤드들이 같은 GPU 를 공유하지 않도록
제약한다. **이 제약은 구현되어 있지 않다.**

> **정정 (V6).** 이 문단은 원래 "제약은 구현되어 있으나 TP=1 이라 한 번도 발동하지
> 않는다" 고 적고 있었다. 코드에 그런 분기는 없다 — `kvpr_global{,_v3,_v4}.py`
> 어디에도 같은 모델의 샤드를 이미 가진 후보 GPU 를 거부하는 코드가 없고,
> `patches/` 전체에 `tp_size` / `tp_rank` 참조가 0 건이다.
> `exp/results/paper-faithful-v4/IMPLEMENTATION_AUDIT.md` 의 `NOT IMPLEMENTED` 가
> 맞고 이 문단이 틀렸다.

구현하지 않은 이유는 게으름이 아니라 **표현 자체가 불가능**하기 때문이며, 그 근거는
`exp/results/paper-faithful-v4/tp-validation/FINDING.md` 에 있다. 요약하면 세 층에서
막힌다:

* `launch_worker_pool_engines` 가 `(GPU, worker slot)` 당 엔진 하나를 `[gpu_id]`
  **단일 GPU** 에 바인딩한다. 샤드가 GPU 를 가로지르는 구조가 없다.
* `controller_global.py` 가 TP 그룹을 rank0 로 축약한다
  (`gpu_ids = set([mod.gpu_ids[0] for mod in models])`). 배치 코드가 TP 그룹을
  멀티-GPU 객체로 볼 수 없다.
* `model_runner.py::_get_cpu_model_ref` 가 가중치를 `(model_path, tp_size)` 로
  조회하므로 엔진과 모델의 `tp_size` 가 어긋나면 만나지 못한다. TP=2 검증이
  `not found in shared cpu models` 로 FAIL 한 실제 원인이다.

그리고 `--enable-worker-pool` 은 선택이 아니다 — GPU 스케줄러와 마이그레이션 전부가
그 경로에 산다. 즉 이 프로토타입에서 **TP>1 과 Prism 스케줄링은 상호 배타적**이고,
논문이 TP 샤드에 거는 제약은 제약할 대상 자체를 갖지 못한다.

**검증에 필요한 환경.** GPU 2 장으로는 TP=2 의 배치 선택지가 `{0,1}` 하나뿐이라
제약이 자동 만족되어 검증할 것이 없다. 제약이 구속력을 갖는(= 위반 배치가 표현
가능하고 제약이 argmin 을 실제로 바꾸는) 최소 조건은 **GPU 4 장 이상 + NVLink
all-pairs** 이며, 논문의 TP=4/8 을 재현하려면 8 장이다. 다만 장비를 갖춰도 위
세 층을 먼저 고쳐야 하므로 이는 패치가 아니라 별도 연구 규모다.

### 5c. 순차 기계 모델은 배치 엔진에서 과소 수용을 만든다

이번 스윕의 지배적 발견이며, 우리의 옮겨쓰기가 아니라 **작성된 그대로의 Algorithm 2**
가 가진 성질이다.

Algorithm 2 는 `1‖ΣU_j` 를 푼다. 기계 하나, 한 번에 작업 하나, 실행가능성은
`clock += e_i` 를 누적해 `d_i` 와 비교하는 방식으로 판정한다. 서빙 엔진은 그런 기계가
아니다 — **한 배치에 여러 요청을 prefill** 한다. 그래서 이 판정은 k 번째로 admit 하는
요청의 비용을 마치 앞의 k−1 개가 끝나기를 기다려야 하는 것처럼 매기지만, 실제로는 같은
배치에 함께 실린다. 결과적으로 GPU 가 아직 포화와 거리가 먼데도 "실행 불가능" 을
보고한다.

30 req/s, seed 1 의 GPU 스케줄러 로그 실측:

| 항목 | 값 |
| --- | ---: |
| deferral 이 발생한 라운드 | 1,083 |
| **그 라운드들에서 선택된 요청 총합** | **246 (라운드당 0.23)** |
| 백프레셔로 되돌린 요청 | 1,024 (1,013개 라운드에서) |
| 이미 마감이 지난 채 디스패치된 요청 | 4,459 |
| 최대 큐 길이 | 211 (프로토타입 178, 중앙값 0) |

`eligible=123, selected=0` 인 라운드가 일상적으로 나온다. 엔진은 그 123개를 배치로
처리할 수 있지만, 단일 기계 판정은 하나도 통과시키지 않는다.

그 결과는 단순한 지연 재배치가 아니라 처리율 부족이다. 실행 구간(decile)별 TTFT p50
(초, 30 req/s, seed 1):

| 구간 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| released prototype | 0.09 | 0.10 | 0.12 | 0.52 | 0.19 | 0.14 | 3.63 | 7.01 | 0.18 | 0.14 |
| paper-faithful | 0.09 | 0.10 | 0.10 | 6.66 | 9.84 | 20.45 | 20.09 | 26.90 | 29.03 | 23.32 |

프로토타입은 튀었다가 **회복한다.** Paper-Faithful 은 4구간부터 올라가 끝내 돌아오지
않는다 — 드레인되지 않는 백로그이며, 지속 처리율(26.7 req/s)이 유입 30 req/s 아래에
있기 때문이다. 이는 요청을 떨어뜨려서 생긴 서명이 아니라 과소 수용의 서명이다.

이 사실은 **유리하게 나온** 고부하 수치도 다시 해석하게 만든다. Paper-Faithful 의 더
나은 TPOT p99(243 ms 대 404 ms)와 더 높은 joint goodput(7.86 대 4.53 req/s)은 실재하는
값이지만, 메커니즘은 배치에 요청을 적게 넣어 그 안에 들어간 요청들의 디코딩 경합을
줄인 것이다. 들어간 요청은 잘 서비스되고, 들어가지 못한 요청은 약 30초를 기다린다. 이
단서 없이 goodput 이득만 보고하면 더 나은 스케줄링으로 오귀인하게 된다.

**여기서 고치지 않은 것은 의도적이다.** 명백한 보정 — 측정된 배치 폭으로 나누는
`clock += e_i / B` — 은 *논문에 없다.* 이를 적용하면 Algorithm 2 가 아닌 것을 보고하게
된다. 자연스러운 후속 실험으로 기록해 두며(§8), Paper-Faithful arm 에 섞지 않고 명확히
라벨링된 세 번째 arm 으로 돌려야 한다.

---

## 6. 실험 쪽 결정

### 6.1 `c_i` 는 가정하지 않고 측정한다

지침은 임의 상수를 금지한다. `prism-research`, `model_info.json`, `kvcached` 어디에도
모델별 prefill 속도 표가 없다. 그래서 `exp/scripts/profile_prefill_speed.py` 가 경합
없는 단독 실행에서 이를 유도해 `exp/configs/prefill_speed.json` 에 저장하고, 서버가
시작 시 읽어서 스윕의 런들 사이에 `c_i` 가 흔들리지 않게 한다.

**추정기 선택이 보기보다 중요하다.** 이 장비의 무경합 Llama-3.1-8B 샘플 525개에서,
방어 가능한 두 추정기가 5배 차이가 난다:

| 추정기 | 값 | 의미 |
| --- | ---: | --- |
| `ttft = a + p/c` 회귀 기울기 | 20,775 tok/s (절편 **29 ms**) | 프롬프트 토큰 1개 추가의 한계 비용 |
| 비율 `Σ prompt_len / Σ ttft` | **4,214 tok/s** | 전체 prefill 시간을 재현하는 속도 |

논문의 `e_i = p_i / c_i` 에는 **절편 항이 없다.** 따라서 `c_i` 는 전체 prefill 시간을
재현하는 양이어야 하고, 그것이 비율 추정기다. 기울기를 쓰면 모든 `e_i` 가 3~30 ms 대에
들어가 실제 prefill 시간보다 한 자릿수 작아지고, Moore-Hodgson 의 실행가능성 검사는
사실상 발동하지 않는다 — 알고리즘이 존재하되 불활성이 된다. 기울기와 절편은
`prefill_speed_detail.json` 에 진단용으로 함께 남긴다. 여기 프롬프트는 짧고
(p50 = 70 토큰, p99 = 608) 그래서 고정 오버헤드가 지배해 두 추정기가 이렇게 크게
갈린다. 프롬프트가 긴 워크로드에서는 둘이 수렴한다.

참고로 공개 프로토타입의 암묵 상수는 2048 tok/s
(`clamp(prompt_len*0.5/1024, 0.2, 2)`)이며, 측정된 비율 값과 2배 이내다.

### 6.1a 측정된 SLO 기준선

논문 §7.1 방식(무경합 단독 요청 702개의 p95)으로 이 장비에서 다시 유도했다:
**TTFT p95 = 125.7 ms, TPOT p95 = 21.41 ms.** 본 연구의 ×5 / ×3 스케일을 적용하면
SLO 는 **TTFT 628.7 ms, TPOT 64.23 ms** 다. `trace.py` 에 내장된 표는 저자들의
하드웨어 값이며 이 장비의 기준선이 아니다. 두 arm 모두 재유도한 값을 쓰고,
`--slo-base-file` 이 같은 수치를 Algorithm 1 의 KVPR 가중에도 공급한다.

### 6.2 모델 구성

슬롯 `model_1`, `model_4`, `model_5` 에 `meta-llama/Llama-3.1-8B` 3개.

* 슬롯은 자유 선택이 아니다 — `trace.py::generate_e2e_benchmark_reqs` 가 특정 모델에
  대해 측정된 슬롯별 SLO 기준선을 하드코딩하고 있고, 1/4/5 가 8B 슬롯 셋이다.
* GPU 2개에 모델 3개는 필연적으로 1+2 분할이므로 배치 결정이 실제로 존재하고
  마이그레이션이 갈 곳이 있다. 커밋된 `exp/results/4-rate-sweep/` 과 같은 구성이라
  수치를 그것과 비교할 수 있다.
* 논문의 8모델 혼합은 이 유입률에서 GPU 2개로 재현 불가능하며, 정확한 구성도 공개되어
  있지 않다. 재현했다고 주장하지 않는다.

### 6.3 요청 유입률의 의미

`build_sharegpt_trace.py --variant rate --phase-rates` 는 **슬롯별** 유입률을 받는다.
본 연구는 **총합** 유입률로 규정되어 있으므로, 총합 `R` 을 슬롯당 `R/3` 으로 방출한다.
모든 런의 메타데이터에
`request_rate_semantics: "aggregate over all models"` 로 기록된다.

스윕한 유입률: **2.5, 5, 7.5, 10, 15, 20, 25, 30 req/s** × seed 1,2,3 × 시스템 2개
= 48런. 15~30 구간을 추가한 이유는, 커밋된 용량 프로파일이 이 장비의 TTFT 무릎을 약
26 req/s 로 잡고 있기 때문이다. 10 req/s 이하에서는 큐가 비어 있고 KV 풀도 20 % 미만이라
Moore-Hodgson 도 KVPR 도 작용할 대상이 없다. 두 구간 모두 보고한다.

> 실제 실행: 결과를 빨리 보기 위해 유입률을 **30 / 20 / 10 req/s** 로 줄여
> 18런(2 시스템 × 3 유입률 × 3 seed)을 돌렸다. 축소 자체는 두 arm 에 동일하게
> 적용되므로 비교의 공정성에는 영향이 없다.

### 6.4 워밍업과 측정

레포에 이를 규정한 기존 관례가 없으므로(커밋된 스윕들은 트레이스 전체를 측정한다)
지침의 대안을 따른다. 트레이스는 360초이고 앞의 **60초는 워밍업으로 제외**, 이어지는
**300초가 측정 구간**이다. 요청은 트레이스 도착 시각으로 구간에 배정하며, 같은 구간을
두 arm 에 동일하게 적용한다.

### 6.5 TPOT 정의

하네스의 `benchmark.py` 가 계산하는 그대로
`tpot = (finish_time − prefill_finish_time) / (output_len − 1)` 를 쓴다. 평균 ITL 이나
e2e/output_len 으로 **대체하지 않는다.** 하네스 자체의 `average_attainment_tpot`
필드는 쓸 수 없다는 점에 유의(`CLAUDE.md` 에 기록된 ms-대-s 단위 버그). 여기의 TPOT
달성률은 전부 요청 단위 원시 덤프에서 다시 계산한 값이다.

### 6.6 Joint-SLO Goodput

논문이 정의하는 지표가 아니므로 여기서 정의한다:

```
Joint-SLO Goodput [req/s] = |{TTFT 와 TPOT SLO 를 모두 만족한 완료 요청}| / 300 s
```

단순 완료 처리율은 따로 기록한다.

---

## 7. 커밋된 기준 환경과의 차이

이 장비에서 `bootstrap.sh` 3단계가 실패했다. `torch==2.4.0+cu121` 이
`nvidia-cudnn-cu12==9.1.0.70` 을 정확히 핀 고정하는데,
`download.pytorch.org/whl/cu121` 에서 그 파일이 제거되었다(9.0.0.312 를 제공한 뒤
9.2x 로 건너뛴다). PyPI 에는 여전히 있다.
`--extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match` 를
추가해, torch 는 pytorch 인덱스에서 `2.4.0+cu121` 로 그대로 해석되고 그 `nvidia-*`
의존성은 PyPI 에서 오도록 했다. 검증 완료: torch 2.4.0+cu121, sglang 0.3.4.post2,
vllm 0.6.3.post1, transformers 4.45.2, flashinfer 0.1.6, kvcached+vmm_ops, CUDA 사용
가능. 원본 실패 로그는 보존했다. 이는 상류 인덱스 변동이지 레포의 결함이 아니며, 두
arm 에 동일하게 영향을 준다.

---

## 8. 결과와 후속 실험

측정 결과와 원인 분석은 `exp/results/paper-faithful-comparison/REPORT.md` 에 있다.
요약하면, 이 구성에서 Algorithm 1 은 작동할 여지가 없었고(§5a), 관측된 차이는 사실상
전부 Algorithm 2 에서 나왔으며, 고부하에서의 TTFT 붕괴는 §5c 의 과소 수용으로 설명된다.

가장 유력한 후속 실험은 실행가능성 검사에 배치 병렬도를 반영하는 것이다
(`clock += e_i / B`, `B` 는 실측 동시 prefill 수). 이는 논문에 없는 항이므로
Paper-Faithful arm 을 수정하는 방식이 **아니라**, 별도의 세 번째 arm 으로 명확히
라벨링해 돌려야 한다.
