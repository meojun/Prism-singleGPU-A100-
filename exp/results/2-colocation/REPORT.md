# Prism colocation 실험 — ShareGPT · slowdown SLO · 1× A100-80G

측정일: 2026-08-04 · 목적: **Prism (OSDI'26)을 baseline으로 삼는 후속 연구**(대역폭 기준을
추가한 스케줄링)를 위해, 세 가지 colocation 구성에서 경합이 어느 지표를 얼마나 망가뜨리는지
정량화한다.

이전 sanity 스윕(`../sanity/REPORT.md`)과의 차이는 두 가지다. 데이터셋이 합성 프롬프트에서
**ShareGPT 실제 텍스트**로 바뀌었고, SLO가 논문의 하드코딩 절대값에서 **무경합 대비 slowdown
배수**로 바뀌었다.

---

## 1. 한 줄 결론

경합은 **decode만** 무너뜨린다. TTFT는 모든 슬롯·모든 백분위에서 SLO를 통과했고, 위반은
전부 TBT·E2E의 P50·P90에서만 나왔다. 8B 두 개를 colocate하면 **TBT가 정확히 2.0배**
느려지고, 3B+8B에서는 **8B는 거의 무영향(1.00×)인데 3B만 1.77배** 손해를 본다.

---

## 2. 환경

| 구성요소 | 값 |
| --- | --- |
| GPU | 1× NVIDIA A100-SXM4-80GB, cc 8.0, driver 595.71.05 (`driver_max_cuda` 13.2) |
| CUDA | cu121 휠 (minor-version compatibility로 동작), fp16 matmul 241 TFLOP/s (peak 312의 77%) |
| torch / sglang / vllm | 2.4.0+cu121 / 0.3.4.post2 (Prism fork) / 0.6.3.post1 |
| transformers / flashinfer | 4.45.2 / 0.1.6+cu121torch2.4 |
| kvcached | branch `prism/shm`, `vmm_ops` 확장 빌드됨 |
| redis | 127.0.0.1:6379, `PING → PONG` |
| 모델 | `meta-llama/Llama-3.1-8B` (15 GB), `meta-llama/Llama-3.2-3B` (6 GB) |

환경 재구축은 `../../../SETUP.md` 참고. 이 실험은 `bootstrap.sh`로 재현한 환경에서
서버 기동·생성 요청까지 실제로 검증한 뒤 수행했다.

### 서버 설정 (전 케이스 동일, full Prism 모드)

```
--disable-cuda-graph --disable-radix-cache
--enable-elastic-memory --use-kvcached-v0 --enable-cpu-share-memory
--max-mem-usage 67.28
--enable-gpu-scheduler --enable-controller --policy simple-global
--enable-model-service --enable-worker-pool
--workers-per-gpu <N> --num-model-service-workers <N> --num-gpus 1
```

`--disable-radix-cache`가 켜져 있다. **이걸 끄면 결과 해석이 달라진다** — §7 참고.

### 데이터셋

`anon8231489123/ShareGPT_Vicuna_unfiltered` / `ShareGPT_V3_unfiltered_cleaned_split.json`
(673 MB, ungated, 2턴 이상 대화 92,886개).

Prism의 멀티모델 e2e 경로에는 ShareGPT 로더가 없다. `trace.py::generate_e2e_benchmark_reqs`가
pkl에서 `req.prompt`를 그대로 읽으므로, 같은 구조의 pkl을 새로 만들었다
(`exp/scripts/build_sharegpt_trace.py`). 쓴 것은 **`content` 변형**이다:

- 도착 시각·라우팅·`prompt_len`·`output_len`을 원본 트레이스에서 **그대로 보존**
- 프롬프트 텍스트만 ShareGPT로 교체, **같은 토큰 수로 정확히 절단** (1500/1500 정확 일치)
- 케이스 A/B/C 어디서도 prefill 토큰 총합이 원본과 **1토큰도 다르지 않다**

즉 부하는 원본과 동일하고 내용만 실제 텍스트다. `prompt_len` 필드를 원본대로 둔 것은
Prism의 GPU-local 스케줄러가 `request_queue.py`에서 이 값을 쓰기 때문이다.

재생성:
```bash
source exp/scripts/env.sh
hf download anon8231489123/ShareGPT_Vicuna_unfiltered \
    ShareGPT_V3_unfiltered_cleaned_split.json --repo-type dataset \
    --local-dir $DATASETS/sharegpt
python exp/scripts/build_sharegpt_trace.py     # seed 42 고정, 약 15초
```

---

## 3. SLO 정의 — 무경합 대비 slowdown

절대 지연 임계값이 아니라 **같은 요청이 무경합 A100에서 냈던 지연 대비 배수**다.

| | P50 | P90 | P99 |
| --- | --- | --- | --- |
| TTFT | 2× | 3× | 6× |
| TBT | 1.25× | 1.5× | 5× |
| E2E | 1.25× | 1.5× | 5× |

**왜 요청별로 계산하는가.** 하네스가 같은 트레이스를 같은 순서로 재생하므로 무경합 실행과
경합 실행에 동일한 요청이 같은 인덱스에 있다(`output_len` 수열로 검증, 어긋나면 분석기가
거부). 그래서 요청 *i* 의 지연을 그 요청 **자신의** 무경합 지연으로 나눈다. p99끼리 나누면
서로 다른 요청을 비교하게 되어 정의와 맞지 않는다.

교차검증용으로 백분위 비율(측정 pX ÷ baseline pX)도 계산하지만 **꼬리에서는 쓸 수 없다.**
baseline의 p99가 드문 사건에 지배되면 비율이 1 아래로 내려가(실측 0.11) "경합이 빨라지게
했다"는 무의미한 값이 나온다.

**attainment / goodput / 위반율**은 tier별로 계산한다. tier P*x* 는 TTFT·TBT·E2E 세 조건을
그 백분위의 배수로 **동시에** 만족하는 요청의 비율이다. 기본은 **P50 tier**를 쓴다 —
P99 tier는 6×/5×/5×로 느슨해 attainment가 1.000으로 포화되어 변별력이 없다.

---

## 4. 실행 구성

총 6회, 케이스당 600초, replication 1, time_scale 1. **실패 0건.**

### 무경합 baseline (slowdown의 분모)

| 실행 | config | 슬롯 | 모델 | 요청 |
| --- | --- | --- | --- | --- |
| `base/M1` | `llama_1x8b.json` | `model_1` | Llama-3.1-8B | 296 |
| `base/M2` | `llama_1x3b_m2.json` | `model_2` | Llama-3.2-3B | 22 |
| `base/M4` | `llama_1x8b_m4.json` | `model_4` | Llama-3.1-8B | 262 |

### 실험

| 케이스 | config | GPU 0에 올린 것 | 슬롯 | 모델별 KV pool 상한 |
| --- | --- | --- | --- | --- |
| **A** | `llama_1x8b.json` | 8B 1개 | `model_1` | 45 GiB |
| **B** | `llama_2x8b.json` | 8B 2개 | `model_1`, `model_4` | 22 GiB |
| **C** | `llama_8b_3b.json` | 8B + 3B | `model_1`, `model_2` | 25 GiB |

### 슬롯이란

`model_N`은 모델이 아니라 **슬롯**이다. 원본이 S-LoRA 계열 멀티어댑터 트레이스라 요청마다
어댑터가 붙어 있고, `trace.py`가 어댑터 8개를 `model_1..model_8`로 매핑한다. 슬롯은
**어떤 요청 묶음을 받는지**만 결정하고, 실제로 올라가는 모델은 config JSON이 정한다.

| 슬롯 | 어댑터 | 요청 | 입력 p50 | 출력 p50 |
| --- | --- | --- | --- | --- |
| `model_1` | rank-2 | 296 | 72 | 252 |
| `model_2` | rank-14 | 22 | 192 | 147 |
| `model_4` | rank-10 | 262 | 30 | 317 |

`model_1`과 `model_4`는 둘 다 8B 슬롯이지만 트래픽 성격이 다르다 — `model_4`가 출력이 25%
길어 더 decode-heavy하다. 케이스 B가 `model_1 + model_2`가 아니라 `model_1 + model_4`인
이유는, 8B 두 장을 올릴 때 양쪽 다 8B급 트래픽을 받게 하기 위해서다.

### 무경합 baseline 실측값

| 슬롯 | 건수 | TTFT p50 | TTFT p90 | TTFT p99 | TBT p50 | TBT p99 | E2E p50 | E2E p99 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `model_1` | 296 | 33.6 | 60.3 | **1268.9** | 13.88 | 40.62 | 3960.4 | 11490.0 |
| `model_2` | 22 | 34.7 | 47.1 | 60.2 | 11.86 | 12.60 | 1790.7 | 6991.2 |
| `model_4` | 262 | 33.0 | 54.1 | 105.8 | 13.74 | 15.28 | 4537.9 | 7561.5 |

(ms) `model_1`의 TTFT p99만 유독 크다. 실행 중반 424초 지점의 **트레이스 버스트**
(8건이 3.5초 안에 도착) 때문이며 cold start가 아니다. 이 버스트는 `model_1` 스트림에만 있다.

---

## 5. 결과

### 5.1 TABLE VI 준수 여부 (요청별 slowdown, 관측/허용)

| 케이스 / 슬롯 | TTFT p50 | p90 | p99 | TBT p50 | p90 | p99 | E2E p50 | p90 | p99 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **A** `model_1` | 1.02 ✅ | 1.47 ✅ | 1.92 ✅ | 1.01 ✅ | 1.06 ✅ | 1.16 ✅ | 1.01 ✅ | 1.06 ✅ | 1.16 ✅ |
| **B** `model_1` | 1.69 ✅ | 2.49 ✅ | 3.25 ✅ | **2.03 ❌** | **2.15 ❌** | 2.23 ✅ | **2.02 ❌** | **2.14 ❌** | 2.21 ✅ |
| **B** `model_4` | 1.80 ✅ | 2.57 ✅ | 3.79 ✅ | **2.08 ❌** | **2.15 ❌** | 2.28 ✅ | **2.08 ❌** | **2.15 ❌** | 2.27 ✅ |
| **C** `model_1` | 1.04 ✅ | 1.46 ✅ | 2.07 ✅ | 1.00 ✅ | 1.26 ✅ | 1.61 ✅ | 1.00 ✅ | 1.25 ✅ | 1.60 ✅ |
| **C** `model_2` | 1.44 ✅ | 1.89 ✅ | 2.14 ✅ | **1.77 ❌** | **1.84 ❌** | 1.85 ✅ | **1.76 ❌** | **1.83 ❌** | 1.84 ✅ |

허용치: TTFT 2/3/6, TBT·E2E 1.25/1.5/5.

### 5.2 처리량 · attainment · goodput · 위반율 (P50 tier)

| 케이스 / 슬롯 | tput req/s | tput tok/s | attain | goodput req/s | goodput tok/s | 위반율 |
| --- | --- | --- | --- | --- | --- | --- |
| **A** `model_1` | 0.470 | 140.6 | **0.986** | 0.463 | 138.6 | 0.014 |
| **B** `model_1` | 0.475 | 141.3 | **0.344** | 0.163 | 42.8 | 0.656 |
| **B** `model_4` | 0.430 | 132.0 | **0.050** | 0.022 | 4.9 | 0.950 |
| **C** `model_1` | 0.472 | 140.6 | **0.883** | 0.417 | 127.8 | 0.117 |
| **C** `model_2` | 0.030 | 6.1 | **0.222** | 0.007 | 1.3 | 0.778 |

다른 tier의 attainment:

| 케이스 / 슬롯 | P50 | P90 | P99 |
| --- | --- | --- | --- |
| **A** `model_1` | 0.986 | 0.996 | 1.000 |
| **B** `model_1` | 0.344 | 0.344 | 1.000 |
| **B** `model_4` | 0.050 | 0.066 | 0.996 |
| **C** `model_1` | 0.883 | 0.940 | 0.996 |
| **C** `model_2` | 0.222 | 0.273 | 1.000 |

### 5.3 절대 지연 (ms)

| 케이스 / 슬롯 | TTFT p50 | TTFT p99 | TPOT p50 | TPOT p99 | E2E p50 | E2E p99 |
| --- | --- | --- | --- | --- | --- | --- |
| **A** `model_1` | 34.6 | 82.9 | 14.13 | 15.86 | 4102.1 | 12029.7 |
| **B** `model_1` | 57.4 | 144.7 | 28.21 | 30.99 | 5784.5 | 22969.8 |
| **B** `model_4` | 60.9 | 180.2 | 28.50 | 30.98 | 8711.5 | 15466.8 |
| **C** `model_1` | 35.5 | 92.6 | 13.89 | 22.26 | 4250.1 | 13436.9 |
| **C** `model_2` | 46.8 | 80.3 | 20.84 | 21.91 | 2801.7 | 10712.7 |

TPOT = TBT.

---

## 6. 해석

**TTFT는 안 막히고 decode만 막힌다.** 경합이 있는 네 슬롯 전부에서 TTFT는 모든 백분위를
통과했고, 위반은 예외 없이 TBT·E2E의 P50·P90이다. prefill은 compute-bound라 여유가 있고
decode는 memory-bandwidth-bound라 먼저 포화된다는 가설과 일치한다.

**8B 두 개 = TBT 2.0배.** `model_1` 2.03, `model_4` 2.08. 대역폭을 정확히 반으로 나눠 쓴
형태다. E2E도 2.02/2.08로 TBT를 그대로 따라가는데, 출력이 250~320토큰이라 E2E가 decode에
지배되기 때문이다.

**3B + 8B는 비대칭이다.** 8B(`model_1`)는 TBT 1.00 / E2E 1.00으로 **사실상 무영향**인데
3B(`model_2`)만 1.77배 손해를 본다. 작은 모델이 일방적으로 밀린다. 대역폭 기준 스케줄링이
개입할 여지가 가장 큰 지점이다.

**케이스 B의 슬롯 간 attainment 격차(0.344 vs 0.050)는 성능 차이가 아니다.** `model_1`의
TBT slowdown이 이봉형이다 — p25에서 1.00, p50에서 2.03. `model_4`가 활성화되기 전에 도착한
35%가 혼자 돌았기 때문이다. `model_4`는 처음부터 끝까지 경합한다. 활성화 시점 차이지
슬롯 우열이 아니다.

---

## 7. 데이터 처리 — 반드시 읽을 것

원시 수치에는 경합과 무관한 사건 두 종류가 섞여 있어 제외했다. 제외 전/후 JSON을 모두
남겨두었다(`*_slowdown.json` / `*_slowdown_clean.json`). 위 표는 **처리 후**다.

### 7.1 cold start (`--warmup 4`)

colocate 실행에서는 모델이 **첫 요청이 도착할 때** 로드되지만, 단독 baseline에서는 서버
기동 시 이미 올라가 있다. 그래서 케이스 B `model_4`의 첫 4건이:

```
인덱스        0     1     2     3
baseline     68    70    27    32 ms
측정       5969  4380  4249  2123 ms
slowdown   88.0  62.3 155.0  65.6
```

Prism §5.3 모델 로딩 경로이며 정상 동작이지만 **정상상태 경합이 아니다.** 슬롯별 앞 4건을
제외했다. 영향:

| `model_4` TTFT | 제외 전 | 제외 후 |
| --- | --- | --- |
| slowdown p99 | **63.58 ❌** | **3.79 ✅** |
| p99 절대값 | 2952.1 ms | 180.2 ms |

**이 처리 하나로 유일했던 TTFT 위반이 사라진다.** 제외하지 않으면 "TTFT도 P99에서 깨진다"는
잘못된 결론이 나온다.

### 7.2 baseline 오염 (`--min-slowdown 0.5`)

slowdown이 1보다 **한참 아래**인 요청은 경합이 빨라지게 한 것이 아니라 **분모가 이상했던**
것이다. 이 트레이스는 버스트가 있고, baseline 실행에서만 터진 버스트는 부풀려진 분모를
남긴다:

```
인덱스        203   204   205   206   207
baseline     3467  2166  1954  1233   938 ms
케이스 A       36    54    38    40    31 ms
slowdown     0.01  0.02  0.02  0.03  0.03
```

버스트는 **측정 실행에서 재현되지 않았다.** 그대로 두면 이 요청들이 어떤 SLO든 무조건
통과해 attainment를 최대 1.7%p 부풀린다. slowdown 0.5 미만 쌍을 제외했다. `model_1`
스트림에서 7~10건 잡혔고 `model_2`·`model_4`는 0건이다(버스트가 `model_1` 스트림에만 있음).

두 처리 모두 §5.1의 다른 셀은 5% 미만으로만 움직인다. 결론은 바뀌지 않는다.

---

## 8. 한계 — 이 실험이 증명하지 못하는 것

**노이즈 바닥.** 케이스 A는 무경합 설정을 두 번 돌려 비교한 것이므로 이상적으로 slowdown
1.00이어야 한다. 실측은:

| 케이스 A | p50 | p90 | p99 |
| --- | --- | --- | --- |
| TTFT | 1.02 | **1.47** | **1.92** |
| TBT | 1.01 | 1.06 | 1.16 |
| E2E | 1.01 | 1.06 | 1.16 |

- **TTFT의 P90·P99는 신뢰할 수 없다.** 무경합끼리도 1.47×/1.92×가 나온다. TTFT는 P50만 쓸 것.
- TBT·E2E는 p99에서도 1.16이라 안정적이다. 대역폭 연구가 TBT를 주 지표로 삼는 것은
  통계적으로도 유리하다 — TTFT는 단일 시점 측정이라 큐잉 타이밍에 민감하지만 TBT는 수백
  스텝 평균이라 평활화된다.
- attainment도 P50 tier에서 0.986이므로 **1.4%p는 노이즈다.** 개선 주장은 이보다 커야 한다.

**케이스당 1회, error bar 없음.** 별도로 측정한 실행 간 분산은 작지 않다 — 원본 트레이스로
케이스 C를 두 번 돌렸을 때 전체 attainment가 0.959 ↔ 0.912로 약 5%p 움직였다
(`../sanity_repeat/`). 케이스 A·B는 소수점 셋째 자리까지 재현됐다. **희소 모델이 섞인 구성
(케이스 C)이 특히 불안정하다.** 논문용으로는 3~5회 반복이 필요하다.

**`model_2`는 22건뿐이다.** P90은 상위 2~3건이, P99는 최댓값 1건이 결정한다. 케이스 C에서
`model_2`의 꼬리 수치는 통계적 근거가 약하다. 이번 방법론은 baseline을 직접 측정하므로
`trace.py`의 하드코딩 SLO가 쓰이지 않는다 — 즉 3B를 `model_5` 슬롯(120건)에 올려도 되며,
그러면 5배 표본을 얻는다.

**GPU 1장.** Prism §7.2(8모델/2GPU), global placement의 부하 분산 효과(≥2 GPU 필요),
§7.4(58모델/32GPU), TP 실험은 이 장비에서 불가능하다.

**`prism` 모드만.** `static`(S-Partition), `elastic`은 돌리지 않았으므로 **Prism 내부
baseline 대비 delta가 없다.** 이 보고서는 "경합이 무엇을 얼마나 망가뜨리는가"의 측정이지
Prism이 이긴다는 증명이 아니다.

**부하가 낮다.** 트레이스가 설계상 희소하다(가장 바쁜 슬롯도 600초에 296건, 0.49 req/s).
프로덕션 형태의 replay이지 saturation sweep이 아니다. 부하 곡선이 필요하면
`exp/scripts/run_load_sweep.sh`로 `--time-scale`을 쓸어야 한다.

**radix cache를 껐다.** `--disable-radix-cache`가 켜져 있어 prefix 재사용이 없다. 이걸 끄면
데이터셋 선택이 결정적으로 중요해진다 — 원본 합성 트레이스(`"Hello "*n`)는 짧은 프롬프트가
긴 프롬프트의 완전한 prefix라서 prefill의 **99.3%**가 캐시 히트가 된다. ShareGPT 변형은
2.0%다. **prefix caching을 실험 축으로 쓸 계획이면 원본 트레이스는 못 쓴다.**

---

## 9. 재현

```bash
source exp/scripts/env.sh

# 1) 데이터셋
hf download anon8231489123/ShareGPT_Vicuna_unfiltered \
    ShareGPT_V3_unfiltered_cleaned_split.json --repo-type dataset \
    --local-dir $DATASETS/sharegpt
python exp/scripts/build_sharegpt_trace.py

# 2) 무경합 baseline
for C in M1 M2 M4; do
  TRACE=$SHAREGPT_CONTENT TAG=base ./exp/scripts/run_sanity.sh $C
done

# 3) 실험
for C in A B C; do
  TRACE=$SHAREGPT_CONTENT TAG=exp ./exp/scripts/run_sanity.sh $C
done

# 4) 분석 (warmup + baseline 오염 제외)
python exp/scripts/analyze_slowdown.py --baseline base --measurement exp \
    --case A --slots model_1 --base-case M1 \
    --warmup 4 --min-slowdown 0.5 --out exp/results/2-colocation/exp_A_slowdown_clean.json
python exp/scripts/analyze_slowdown.py --baseline base --measurement exp \
    --case B --slots model_1 model_4 --base-case M1 M4 \
    --warmup 4 --min-slowdown 0.5 --out exp/results/2-colocation/exp_B_slowdown_clean.json
python exp/scripts/analyze_slowdown.py --baseline base --measurement exp \
    --case C --slots model_1 model_2 --base-case M1 M2 \
    --warmup 4 --min-slowdown 0.5 --out exp/results/2-colocation/exp_C_slowdown_clean.json

# 5) 표
python exp/scripts/report_slowdown.py --measurement exp --suffix _clean --tier 50
```

`TAG`는 출력 네임스페이스라 필수다. 안 주면 `results/1-env-verification/`의 커밋된 baseline을 덮어쓴다.

실행 시각 기록 (총 62분):

```
base M1  06:51:51 → 07:03:04      exp A  07:19:48 → 07:31:00
base M2  07:03:04 → 07:08:29      exp B  07:31:00 → 07:42:21
base M4  07:08:29 → 07:19:48      exp C  07:42:21 → 07:53:33
```

---

## 10. 산출물

| 경로 | 내용 |
| --- | --- |
| `exp/results/1-env-verification/` | 무경합 baseline 3슬롯 (원시 지표 + per-request 덤프) |
| `exp/results/2-colocation/exp_{A,B,C}_slowdown_clean.json` | **최종 결과** (warmup·오염 제외) |
| `exp/results/2-colocation/exp_{A,B,C}_slowdown.json` | 처리 전 (비교용) |
| `exp/results/2-colocation/requests/` | per-request 원시 덤프 |
| `exp/scripts/build_sharegpt_trace.py` | ShareGPT → Prism 트레이스 pkl |
| `exp/scripts/analyze_slowdown.py` | slowdown SLO 분석 |
| `exp/scripts/report_slowdown.py` | 표 출력 |
| `exp/scripts/derive_slo_baseline.py` | 절대 SLO 방식용 baseline 추출 (이번엔 미사용) |
| `exp/scripts/run_load_sweep.sh` | `--time-scale` 부하 스윕 |
| `exp/configs/llama_1x3b_m2.json`, `llama_1x8b_m4.json` | 단독 baseline용 config |

### 참고 — 방법론 검증에 쓴 보조 데이터

| 경로 | 왜 있는가 |
| --- | --- |
| `exp/results/2-colocation/` | 데이터셋 교체가 부하 중립임을 확인. 원본 트레이스와 케이스 A attainment 1.000 동일, 케이스 B 0.224 동일 |
| `exp/results/1-env-verification/` | 실행 간 분산 측정 (§8). 원본 트레이스 케이스 C를 재실행해 0.959 → 0.912 관측 |

---

## 11. 다음 단계 제안

1. **`model_4` 관점 정리** — 케이스 B의 슬롯 간 격차가 활성화 시점 차이임을 감안해, 정상상태
   구간만 잘라 비교하면 더 깨끗한 2.0× 신호를 얻는다.
2. **부하 스윕** — `run_load_sweep.sh B 2.0 1.0 0.5 0.25` (약 42분). 지금 동작점은 Prism이
   TTFT를 여유롭게 통과하는 구간이라, 개선 여지가 큰 지점을 찾으려면 부하 곡선이 필요하다.
3. **반복 실행** — 케이스당 3~5회로 error bar 확보. 특히 케이스 C.
4. **`model_5` 슬롯으로 3B 재실험** — `model_2`(22건) 대신 120건 표본.
5. **`static` / `elastic` 모드** — Prism 내부 baseline 대비 delta 확보.
