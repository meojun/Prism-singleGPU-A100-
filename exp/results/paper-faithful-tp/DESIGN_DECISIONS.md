# TP 배치 — 논문이 정한 것과 정하지 않은 것

> **정정 이력.** 이 문서의 첫 판은 "worker pool 은 논문 메커니즘이 아니고 TP
> 슬롯 예약은 전적으로 우리 선택" 이라고 적었다. **틀렸다.** 논문 원문을 확보해
> 확인한 결과 engine pool 은 §5.3 이고 TP 배치는 §A.2.2 에 완전한 알고리즘으로
> 있다. 아래는 원문 기준으로 다시 쓴 것이다.
>
> 첫 판이 틀린 이유는 저장소에 논문 원문이 없었고 `design_analysis.md:233` 의
> 한 줄 요약("논문은 TP 샤드가 같은 GPU 를 공유하지 않도록 제약한다")을 논문의
> 전부로 읽었기 때문이다. 그 한 줄은 §A.2.2 의 마지막 문장만 옮긴 것이다.

## 1. 논문이 정한 것 — 재현 대상

### 1.1 Engine pool (§5.3)

> "it maintains an **engine pool on each GPU**, where engines are pre-initialized
> with virtual address space and distributed contexts. Upon model activation,
> Prism selects an available engine from the pool and starts model loading
> directly. When a model is evicted, its physical memory is released, but its
> engine with virtual address space is **returned to the engine pool** for
> future reuse."

프로토타입의 `--enable-worker-pool` 이 이것이다. 이번 측정이 논문의 동기를
그대로 재현한다:

| | 소요 |
| --- | ---: |
| 엔진 기동 (프로세스 + CUDA 컨텍스트 + 메모리 풀) | **41 s** |
| 이미 뜬 슬롯에 모델 활성화 (Llama-3.2-1B, TP=2) | **0.91 s** |
| 같은 것, Llama-3.1-70B TP=4 | 5.9 ~ 7.0 s |

논문 Figure 10 의 70B(TP=8) 활성화 1.5 s 와 자릿수가 다른데, 그쪽은 H100 이고
parallel loading 이 켜져 있다. 우리 TP arm 은 `model_runner.py:133-136` 때문에
model service 경로가 꺼져 parallel loading 을 쓰지 못한다(§3 참조).

### 1.2 GPU group 은 불가분 단위 (§4)

> "a **GPU group** represents the strict scheduling boundary—a set of GPUs
> tightly coupled to jointly serve one model instance. Prism's global scheduler
> treats each of these groups as a **distinct, indivisible resource unit**."

`tp_slots.py` 의 TP 그룹이 이것이다. 슬롯을 그룹 단위로 타이핑한 것은 우리
발명이 아니라 이 문장의 구현이다.

### 1.3 rank0 이 받고 broadcast (§7)

> "For tensor-parallel models, the GPU-local scheduler runs only on the **first
> rank**, and the resulting scheduling decisions are **broadcast to all other
> ranks** to ensure consistency."

상류 SGLang 코드가 이미 이렇게 되어 있다(`scheduler.py:157-168`, `:628-646`).
런타임에서 두 rank 가 동일한 rid 로 활성화 요청을 받는 것을 확인했다.

### 1.4 TP 배치 알고리즘 (§A.2.2) — 저장소 요약보다 훨씬 구체적

> "We conceptualize a TP model requiring tp_size GPUs as being composed of
> **tp_size distinct parts**. For scheduling purposes, we create tp_size entries
> in the sorted model list for such a model, assigning each entry **1/tp_size of
> the original weight and request rate**. A beneficial property emerges from
> this decomposition: since these entries have identical rk/sk values, they
> remain **adjacent after sorting**. This adjacency **increases the likelihood**
> that, as the algorithm iterates, these parts are initially assigned to
> different GPUs due to rising KVPRs. To ensure the distribution, if assigning a
> TP part to the GPU with the minimum KVPR would result in collocating it with
> another part of the same original model, we instead assign it to the GPU
> exhibiting the **second-lowest KVPR**."

구현 대조:

| 논문 §A.2.2 | 구현 | |
| --- | --- | --- |
| tp_size 개의 distinct parts | 샤드 단위 argmin | 일치 |
| 각 entry 에 1/tp_size 의 weight 와 request rate | `model_size/k`, `weighted_token_rate/k` | 일치 |
| 정렬 후 인접 → 대개 저절로 분산 | 측정으로 확인 (아래 §2) | 일치 |
| 충돌 시 **second-lowest KVPR** | `enable_tp_anti_affinity` 기본 경로 | 일치 (수정 후) |

**주의 — 논문 규칙은 anti-affinity 를 보장하지 않는다.** "second-lowest KVPR"
는 그 GPU 가 같은 모델의 다른 part 를 이미 갖고 있는지 재검사하지 않는다.
tp_size=2 에서는 차이가 없다(이미 놓인 part 가 하나뿐이라 second-lowest 는
반드시 비충돌). **tp_size≥3 부터 갈린다.** 그래서 두 경로를 나눴다:

* `--enable-tp-anti-affinity` — 논문 문자 그대로. second-lowest 로 물러난다.
  그 GPU 도 충돌하면 그대로 두고 `aa_second_also_collides` 로 센다.
* `--enable-tp-anti-affinity-strict` — 충돌하지 않는 후보 중 최소 KVPR.
  논문보다 강한 규칙이므로 **별도 플래그로 분리**했다. 조용히 강화하면
  재현한 것이 무엇인지 잘못 보고하게 된다.

## 2. 논문이 스스로 인정하는 것 — 제약은 자주 발동하지 않는다

§A.2.2 의 "increases the **likelihood**" 가 정확한 표현이다. 단위테스트로
측정한 것도 같다(`test_tp_anti_affinity.py`):

| 시나리오 | OFF | ON | 구속? |
| --- | --- | --- | --- |
| 균형 잡힌 클러스터 | `(0, 1)` | `(0, 1)` | 안 함 |
| GPU 3 만 한가한 불균형 | `(3, 3)` | `(3, 1)` | **함** |

샤드를 배치할 때마다 그 GPU 에 1/k 부하를 물리므로 KVPR 이 올라가고, 두 번째
샤드는 대개 저절로 다른 GPU 로 간다. 규칙이 발동하는 것은 한 GPU 가 나머지보다
압도적으로 한가할 때뿐이다 — 막 드레인된 GPU 가 그 모양이다.

**4단계 워크로드는 이 상태를 만들어야 한다.** 균등 부하로 돌리면
`aa_violations = 0` 이 나오고, 그것은 "제약이 동작했다" 가 아니라 "제약이 한
번도 구속하지 않았다" 로 보고해야 하는 결과다.

## 3. 논문이 정하지 않은 것 — 우리 선택

### 3.1 TP=k 그룹을 몇 개나 미리 띄우는가

논문은 "engine pool on each GPU" 라고만 하고, TP 그룹용 엔진을 몇 개 준비하는지
어떤 GPU 조합으로 준비하는지는 쓰지 않는다. 우리는 **k-부분집합 전부**를 열었다
(4 GPU, TP=2 → 6개 쌍).

**이유는 실험 설계다.** 후보 그룹이 하나뿐이면 §A.2.2 의 second-lowest 규칙이
고를 대상 자체가 없어 제약이 구성상 자동 만족된다. 측정할 것이 없어진다.

**대가 — 슬롯 가용성.** 4 GPU / `workers_per_gpu=1` / TP=2 전 쌍에서 GPU 당
슬롯은 이렇다:

```
GPU0: [0, 1, 2, 3]   GPU1: [0, 1, 4, 5]
GPU2: [0, 2, 4, 6]   GPU3: [0, 3, 5, 6]
```

슬롯 0 만 TP=1 이고 나머지 3 개는 TP 그룹 몫이라, TP 모델이 유휴여도 TP=1
모델이 그 슬롯을 쓸 수 없다.

**물리 GPU 메모리 비용은 이와 별개이고 작다.** 논문 §5.2 대로 kvcached 가
가상/물리를 분리해 "physical memory is allocated and mapped only on demand"
하므로, 유휴 pool 엔진은 가상 주소 공간과 CUDA 컨텍스트만 잡고 가중치도 KV
물리 페이지도 잡지 않는다. 비용은 물리 메모리가 아니라 **스케줄링 슬롯**이다.

프로덕션이라면 `--tp-max-groups` 로 줄이는 것이 맞다. 드롭된 그룹은 조용히
빠지지 않고 로그에 남는다.

### 3.2 그룹 정보를 어디에 두는가

논문은 GPU group 을 불가분 단위로 다루라고만 하고 자료구조를 지정하지 않는다.
`ModelInstanceState.gpu_ids` 를 그룹 전체로 넓히는 것이 자연스러워 보이지만,
그러면 컨트롤러가 첫 사이클에 죽는다(측정된 회귀, `aa-on` 4/4 → 0/4):

```
simple_global.py:239-240
    for gpu_id in instance.gpu_ids: model_active_instances[gpu_id] = ...
    assert len(model_active_instances) <= 1
```

그래서 공유 표현은 그대로 두고 그룹을 `KVPRGlobalPolicyTP` 안에서 추적한다.
설정된 배치에서 seed 하고 이 정책이 이동을 낼 때만 갱신한다.

## 4. 수치 해석 시 주의 (논문과의 차이)

`model_runner.py:133-136` 이 `tp_size > 1` 에서 model service 경로를 끈다
(런타임 로그의 `model_service=False` 로 확인). 즉 우리 TP arm 은 논문 §5.3 의
**parallel model weight loading 을 쓰지 못한다.** 논문 Figure 10 이 보고하는
70B 활성화 1.5 s 와 우리 5.9~7.0 s 의 차이에는 이것과 H100/A100 차이가 함께
들어 있다. **TP arm 의 시간 수치를 non-TP arm 이나 논문 수치와 나란히 놓지 말 것.**
