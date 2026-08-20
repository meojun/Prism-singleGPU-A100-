# IMPLEMENTATION_AUDIT 갱신분 — TP 관련 행

> `exp/results/paper-faithful-v4/IMPLEMENTATION_AUDIT.md` 의 해당 행을 이것으로 대체할 것. 판정 기준은 그 문서의 것을 그대로 쓴다.

| Mechanism | V4 | **이 브랜치** | Evidence |
| --- | --- | --- | --- |
| TP=2 runtime validation | FAIL | **PASS** | `boot-tp2-g4/tp2_validation.json` 8개 체크 전부 PASS, 요청 4/4. 엔진 자신의 로그가 `tp_rank=0 gpu_id=0` / `tp_rank=1 gpu_id=1`. V4 의 실패 원인은 가중치 조회 결함이 아니라 `tp_size` 가 `keys_to_remove`(server_args.py:265)에서 버려져 기본값 1 로 떨어진 것. |
| TP>1 under worker pool | NOT SUPPORTED | **FULL** | TP=2/TP=4 부팅·서빙, 활성화-비활성화 순환에서 슬롯 반환 확인 (`cycle-tp2/cycle_verdict.json`), Llama-3.1-70B 을 TP=4 와 TP=2 로 서빙. |
| TP anti-affinity | NOT IMPLEMENTED | **FULL** | 구현되었고 런타임에서 구속했다 — 위반 110회, 우회 75회. `step4/`, `raw/alg1_tp/`. 논문 §A.2.2 를 문자 그대로 구현한 `--enable-tp-anti-affinity` 와 그보다 강한 `--enable-tp-anti-affinity-strict` 를 분리했다. |

## 바뀌지 않은 것

* **KV migration** — 이 브랜치는 손대지 않았다. 2xA100 쪽 작업이다.
* **RDMA transport** — 단일 노드. 여전히 측정 불가.
* **parallel weight loading / P2P migration (TP 경로)** — `NOT APPLICABLE UNDER TP`. `model_runner.py:133-136` 이 `tp_size > 1` 에서 model service 를 끄므로 TP 모델은 두 메커니즘을 쓰지 못한다. 비-TP 경로의 FULL 판정은 그대로 유효하다.
