# 3x Llama-3.1-8B rate-sweep on 2x A100-80G

측정일: 2026-08-12 · 목적: 모델 크기·아키텍처 차이를 변수에서 제거하고, **request
arrival rate와 그로 인한 GPU/KV-cache contention만**이 Prism에 미치는 영향을 본다.
재현 커맨드 전문은 [`../../../EXPERIMENT.md`](../../../EXPERIMENT.md).

---

## 1. 한 줄 결론

throughput은 24 req/s에서 부드럽게 포화하지만 **TTFT p95는 24 -> 30 req/s 구간에서
239 ms -> 21,325 ms로 89배 폭증**한다. 같은 지점에서 2모델 GPU의 KV 풀이 98%로 차고
큐가 처음 생긴다. **평균과 p50은 이 절벽을 완전히 감춘다** (p50은 92 -> 132 ms,
1.4배). 그리고 부하와 무관하게 지배적인 요인은 **colocation 자체**다: 무부하 대비
2.8배 느린 TPOT가 lambda_base에서 이미 발생한다.

## 2. 구성

3x `meta-llama/Llama-3.1-8B` = 슬롯 model_1 / model_4 / model_5 (trace.py가 하드코딩한
8B 슬롯). GPU0 = model_1, GPU1 = model_4 + model_5. **3모델/2GPU는 필연적으로 1+2**이고
모델별 rate가 같으므로 GPU1이 부하의 2/3를 받는다. 슬롯 선택 근거와 `-Instruct`를 쓰지
않은 이유는 EXPERIMENT.md 1절.

SLO는 논문 7.1 방식으로 이 장비에서 재측정한 무경합 p95
(**TTFT 76.1 ms / TPOT 18.04 ms**, built-in 42.9/11.46 대비 1.77x/1.57x 느림)에
scale 5 / 3 을 곱한 **380 ms / 54.1 ms**.

lambda_base = **12 req/s** — ramp 프로파일링으로 찾은 knee(약 26 req/s)의 46%.
rate sweep은 동일한 5,090건 시퀀스를 `--time-scale`로만 압축해 만들었으므로
길이 분포·모델 비율·seed가 모든 지점에서 동일하다.

## 3. Experiment 1 + 2 — rate sweep

| offered λ | ×λ_base | achieved | out tok/s | TTFT p50 | **p95** | p99 | TPOT p50 | p95 | e2e p95 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 12 | 1.00× | 11.5 | 2360 | 70 ms | **165 ms** | 225 ms | 44 ms | 58 ms | 29 s |
| 15 | 1.25× | 14.1 | 2885 | 75 ms | **177 ms** | 245 ms | 57 ms | 74 ms | 36 s |
| 18 | 1.50× | 16.4 | 3365 | 79 ms | **191 ms** | 268 ms | 68 ms | 93 ms | 43 s |
| 24 | 2.00× | 21.0 | 4304 | 92 ms | **239 ms** | 423 ms | 89 ms | 144 ms | 58 s |
| 30 | 2.50× | 24.3 | 4983 | 132 ms | **21,325 ms** | 23,555 ms | 111 ms | 173 ms | 69 s |

**절벽은 TTFT에만 있다.** TPOT는 44 -> 111 ms로 단조롭게(2.5배) 나빠지고 throughput은
24.3 req/s에서 포화한다(offered 30 대비 19% 부족). 반면 TTFT p95만 89배 튄다.

| offered λ | att TTFT | att TPOT | att both | KV pool m1 / m4 / m5 | max queue (model / sched) | GPU0 / GPU1 util | migrations |
| ---: | ---: | ---: | ---: | :---: | :---: | :---: | ---: |
| 12 | 1.000 | 0.872 | 0.872 | 0.08 / 0.16 / 0.16 | 0 / 0 | 80% / 95% | 1 |
| 15 | 1.000 | 0.447 | 0.447 | 0.09 / 0.21 / 0.20 | 0 / 0 | 79% / 95% | 1 |
| 18 | 0.999 | 0.379 | 0.379 | 0.11 / 0.28 / 0.26 | 0 / 0 | 78% / 94% | 1 |
| 24 | 0.987 | 0.307 | 0.306 | 0.23 / 0.69 / 0.58 | 0 / 0 | 85% / 88% | 1 |
| 30 | 0.707 | 0.291 | 0.271 | 0.29 / 0.98 / 0.98 | 38 / 184 | 84% / 92% | 1 |

절벽의 원인은 표에서 바로 읽힌다. **KV 풀이 98%에 도달하는 지점과 큐가 처음 생기는
지점과 TTFT가 터지는 지점이 정확히 일치한다** (30 req/s). 24 req/s까지는 KV 풀이
0.69/0.58로 여유가 있고 큐는 0이며 TTFT p95는 239 ms에 머문다. 즉 이 구성에서 관측되는
degradation은 24 req/s까지는 순수 compute contention이고, **KV-cache contention은 마지막
한 단계에서만 등장하며 그 순간 성능이 선형에서 절벽으로 바뀐다.**

TPOT attainment의 최대 낙폭은 오히려 1.0x -> 1.25x 구간이다(0.872 -> 0.447). TPOT SLO
54.1 ms를 p50이 44 -> 57 ms로 넘어서는 지점이라 그렇다. 즉 **TPOT는 saturation보다 훨씬
이른 시점에 먼저 무너진다.**

## 4. 모델별 분해 — colocation이 rate보다 지배적이다

`att_tpot / TTFT p95 (ms) / TPOT p50 (ms)`:

| offered λ | model_1 (GPU0 alone) | model_4 (GPU1 shared) | model_5 (GPU1 shared) |
| ---: | :---: | :---: | :---: |
| 12 | 1.000 / 89 / 17 | 0.835 / 170 / 47 | 0.787 / 183 / 49 |
| 15 | 1.000 / 89 / 19 | 0.189 / 185 / 62 | 0.177 / 199 / 62 |
| 18 | 1.000 / 95 / 20 | 0.093 / 200 / 76 | 0.072 / 216 / 74 |
| 24 | 0.854 / 153 / 24 | 0.056 / 247 / 115 | 0.037 / 277 / 101 |
| 30 | 0.826 / 177 / 30 | 0.062 / 23,054 / 133 | 0.009 / 12,411 / 127 |

lambda_base에서 이미 **GPU0에 혼자 있는 model_1은 TPOT p50 17 ms** — 무경합 baseline
18.04 ms와 사실상 동일하다(GPU0가 경합 없음을 교차 확인해준다). 같은 순간 GPU1의
model_4/5는 **47-49 ms로 2.8배** 느리다. rate를 2.5배 올려도 model_1은 17 -> 30 ms에
그치는 반면 model_4/5는 47 -> 133 ms가 되고 attainment는 0.8 -> 0.01로 붕괴한다.

**colocation 여부가 rate보다 큰 변수다.** Prism의 global scheduler는 이걸 고칠 수 없다 —
모델을 쪼갤 수 없으므로 어떤 배치를 해도 한 GPU는 2모델을 갖는다.

## 5. Burst 시나리오 — hot 모델 수 1 -> 2 -> 3

model_1을 **세 phase 내내 8 req/s로 고정**하고 나머지를 켜 나간다. model_1의 변화가
곧 다른 모델의 burst가 준 피해다.

| phase | hot models | total λ | att both | TTFT p95 | TPOT p50 | m1 TTFT p95 | m4 TTFT p95 | m5 TTFT p95 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1 hot (8 / 0.5 / 0.5) | 8.8 | **0.986** | 108 ms | 22 ms | **100 ms** | 155 ms | 141 ms |
| 1 | 2 hot (8 / 8 / 0.5) | 17.3 | **0.489** | 190 ms | 41 ms | **111 ms** | 217 ms | 155 ms |
| 2 | 3 hot (8 / 8 / 8) | 24.3 | **0.207** | 286 ms | 97 ms | **271 ms** | 250 ms | 334 ms |


attainment가 0.986 -> 0.489 -> 0.207로 단조 붕괴한다. hot 모델이 늘수록 Prism이
재분배할 여유 메모리가 사라지기 때문이다.

**결정적 관측: phase 2에서 model_1의 TTFT p95가 111 -> 271 ms로 2.4배 악화된다.**
model_1의 rate는 변하지 않았고 GPU0에 혼자 있었는데도 그렇다. 원인은 컨트롤러 로그에
있다:

```
trace t~387s : ACTION: deactivate model_5 on GPU 1 and activate model_5 on GPU 0
               Reason: migrate model
trace t~480s : (원위치)
```

세 모델이 모두 hot이 되자 Prism이 GPU1의 부하를 덜려고 **model_5를 GPU0으로
migrate했고, 그 결과 그때까지 보호받던 model_1이 경합에 노출됐다.** 3개 hot 모델을
2개 GPU에 놓는 좋은 배치는 존재하지 않으므로 이건 정책 버그가 아니라 자원 부족이다.
논문이 말하는 "idle 모델의 메모리를 활성 모델에 넘겨주는" 이점이 **모두가 hot일 때
사라지는** 양상 그 자체다.

## 6. 산출물

| 파일 | 내용 |
| --- | --- |
| `exp_glob_on_ts*_slo.json` | 모델별 attainment, mean/p50/p95/p99 TTFT·TPOT, e2e |
| `exp_glob_on_ts*_summary.csv` | rate-vs-X 표 한 줄 |
| `exp_glob_on_ts*_timeseries.csv` | 1초 bin — 모델별 arrivals/running/queue/KV tokens/KV 풀 점유/decode throughput, GPU별 큐·메모리·util |
| `../probe/rampLO_windows.csv`, `../probe/probe_*_windows.csv` | capacity 곡선 |
| `../burst/burst_*_windows.csv` | phase별 결과 |
| `../ref/ref_*_slo.json` | 무경합 baseline |

## 7. 한계

- **rejected는 항상 0이고 이는 관측 실패가 아니다.** `request_queue.py:137`이
  `net_available = float("inf")`라 admission control이 코드상 동작하지 않는다.
  런타임 로그도 매초 `net_available: inf`를 찍는다.
- **queue length는 포화 전까지 구조적으로 0**이다. 전부 즉시 admit되어 back-pressure가
  `#running-req`와 TTFT로 나타난다. 부하 신호로는 큐가 아니라 running/TTFT p95를 봐야
  한다.
- 각 rate 지점은 **1회씩만** 측정했다. 집계 지표(attainment, throughput, p50)는 5,090건
  위에서 계산되어 안정적이지만, **30 req/s의 TTFT p95 21초 같은 꼬리값은 반복 측정
  없이 정밀한 수치로 인용하면 안 된다.** 절벽의 존재와 자릿수는 견고하다.
- migration은 모든 rate에서 1회씩 발생했다. 이 구성에서 migration 경로가 살아있다는
  것은 확인되지만(fig7 실험에서는 0회였다), 횟수가 적어 성능 영향은 분리 측정하지
  않았다.
- `--disable-cuda-graph`로 실행했다(레포 관례). 이 때문에 절대 TPOT가 논문 baseline보다
  1.57배 느리다. 재측정한 baseline을 쓰므로 attainment 비교는 유효하지만, 절대
  latency를 논문 수치와 직접 비교하면 안 된다.
