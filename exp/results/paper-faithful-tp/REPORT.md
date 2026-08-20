# TP 지원과 anti-affinity — 측정 결과

생성: `exp/scripts/build_report_tp.py`. 모든 수치는 `raw/` 에서 파생된 것이고 그 반대가 아니다.


## 0. 이 박스

* GPU 4 x NVIDIA A100-SXM4-80GB
* peer access 전 쌍: `True` (NV12, NVSwitch)
* 이 박스에서 재유도한 tau = **0.17108602804407128** (커밋된 기본값 0.35, 이전 박스 0.146)
* SLO 기준선과 c_i 도 이 박스에서 다시 쟀다. 커밋된 값은 다른 박스 것이다.

## 1. 메커니즘 — TP 가 worker-pool 경로에서 도는가

V4 의 판정은 FAIL 이었다. 아래는 이 브랜치의 런타임 증거다.

| 구성 | 판정 | rank -> GPU | 요청 |
| --- | --- | --- | --- |
| `boot-tp2` | PASS | 0->0, 1->1 | 4/4 성공 |
| `boot-tp2-g4` | PASS | 0->0,1,2, 1->1,2,3 | 4/4 성공 |
| `boot-tp4-g4` | PARTIAL | 0->0, 1->1, 2->2, 3->3 | 4/4 성공 |
| `serve-70b-tp4` | PARTIAL | 0->0, 1->1, 2->2, 3->3 | 4/4 성공 |
| `serve-70b-tp2` | PASS | 0->0, 1->1 | 4/4 성공 |
| `serve-70b-tp4-kvpr` | PARTIAL | 0->0, 1->1, 2->2, 3->3 | 4/4 성공 |
| `cycle-tp2` | PASS | 0->0,1,2, 1->1,2,3 | - |

`boot-tp4-g4` 등의 판정이 PARTIAL 로 찍히는 것은 실패가 아니라 판정기 아티팩트다 — `collect_tp2_evidence.py` 가 `tp_size == 2` 와 `len(planned_gpus) == 2` 를 하드코딩한 TP=2 전용 체커다. 그 두 줄만 FAIL 이고 나머지 6개는 PASS 다. rank->GPU 열이 실제 증거다.

rank 하나가 여러 GPU 로 보이는 칸은 그 디렉터리에서 **여러 번 실행된 로그의 합집합**이다(재실행 시 서버 로그가 누적된다). 한 번의 실행 안에서는 rank 와 GPU 가 1:1 이며, 그것이 `boot-tp4-g4` 의 `0->0, 1->1, 2->2, 3->3` 처럼 깨끗하게 나오는 칸이다.

## 2. anti-affinity ON / OFF

**측정하지 않았다.** `step4/` 에 요약 파일이 없다. 아래 판정은 메커니즘 검증까지만 근거를 갖는다.

### 논문 규칙과 strict 규칙은 이 장비에서 구분할 수 없다

논문 §A.2.2 는 충돌 시 second-lowest KVPR 로 물러나되 그 GPU 를 재검사하지 않는다. `--enable-tp-anti-affinity-strict` 는 충돌하지 않는 후보 중 최소를 고른다. 두 규칙이 갈리려면 second-lowest 자체가 이미 같은 모델의 샤드를 갖고 있어야 하고, 그러려면 **둘 다** 필요하다:

* `k >= 3` — k=2 면 이미 놓인 샤드가 하나뿐이라 second-lowest 는 반드시 비충돌
* `n > k` — n == k 면 클러스터 전체가 한 그룹이라 물러날 후보가 없다

4 GPU 에서는 `k = 3` 만 남는데, **이 실험의 어느 모델도 TP=3 을 지원하지 않는다** — `num_key_value_heads` 가 전부 8, 4, 2 이고 3 으로 나뉘지 않는다 (Llama-3.2-1B/3B, Llama-3.1-8B/70B = 8, Qwen2.5-7B = 4, Qwen2.5-1.5B/3B = 2).

따라서 **두 플래그는 이 하드웨어에서 동작상 동일하며, 위 표의 paper 와 strict 가 같게 나오는 것은 실패가 아니라 구조적 필연이다.** 논증이 아니라 테스트로 남겼다 (`test_tp_anti_affinity.py` case 9). 이 차이를 측정하려면 GPU 8장이 필요하다 (k=4, n=8 -> 후보 그룹 70개, 4번째 샤드를 놓기 전에 3개가 이미 놓여 있다).

## 3. 수치 해석 시 반드시 반영할 것

`model_runner.py:133-136` 이 `tp_size > 1` 에서 model service 경로를 끈다(런타임 로그의 `model_service=False` 로 확인). 즉 TP arm 은 V4 의 **병렬 가중치 로딩도 P2P 마이그레이션도 쓰지 않는다.** 논문 §5.3 이 기술하는 parallel weight loading 이 TP 경로에서 빠져 있다는 뜻이기도 하다. **TP arm 의 시간 수치를 non-TP arm 이나 논문 Figure 10 과 나란히 놓지 말 것.**

전 k-부분집합을 후보 그룹으로 연 것은 제약이 구속력을 가질 수 있게 하려는 실험 설계이지 프로덕션 권고가 아니다. 대가는 물리 메모리가 아니라 슬롯 가용성이다 (논문 §5.2 대로 kvcached 가 물리 페이지를 on-demand 로 잡으므로 유휴 엔진은 가상 주소 공간과 CUDA 컨텍스트만 점유한다). 자세한 것은 `DESIGN_DECISIONS.md`.
