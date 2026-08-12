# Prism 실험 환경 — 전체 상태 보고서

생성 시각 2026-08-12 11:51 UTC · 생성 스크립트 `exp/scripts/build_status_report.py`

이 문서의 모든 수치는 **생성 시점에 직접 조사하거나 커밋된 결과 파일에서 읽은 것**입니다. 손으로 입력한 값은 하나도 없습니다.

## 1. 환경

**GPU** — 드라이버 595.71.05, GPU0↔GPU1 연결 `NV12`

| idx | 이름 | 메모리 | compute cap |
| --- | --- | --- | --- |
| 0 | NVIDIA A100-SXM4-80GB | 81920 MiB | 8.0 |
| 1 | NVIDIA A100-SXM4-80GB | 81920 MiB | 8.0 |

> compute capability가 10.0 이상(Blackwell)이면 이 스택은 못 씁니다 — torch 2.4.0+cu121에 해당 아키텍처 커널이 없어 첫 GPU 연산에서 죽습니다.

**호스트** — 128 스레드, RAM 2003 GiB 중 1888 GiB 가용, 레포 디스크 512G 중 478G 여유, /dev/shm 250G

**스택** (`prism-venv` 기준 — 실제 실험이 로드한 것)

| 패키지 | 버전 |
| --- | --- |
| torch | 2.4.0+cu121 |
| sglang | 0.3.4.post2 |
| vllm | 0.6.3.post1 |
| transformers | 4.45.2 |
| flashinfer | 0.1.6+cu121torch2.4 |
| kvcached | 설치됨 (버전 속성 없음) |
| cuda_available | True |

**고정된 upstream** (`setup/pins.env`) — 실제 체크아웃된 HEAD

| 저장소 | HEAD |
| --- | --- |
| prism-research (SGLang 포크) | `595ec1f` |
| kvcached (prism/shm) | `d78649d` |
| kvcached (main) | `ce76a12` |

**redis** — `PONG`, supervisor: `redis                            RUNNING   pid 3178, uptime 6:35:16`

> redis가 죽어 있으면 Prism이 기동 중 모델을 `activating` 상태로 둔 채 멈춥니다.

**모델 가중치**

| 크기 | 모델 |
| --- | --- |
| 15G | `meta-llama/Llama-3.1-8B` |
| 2.4G | `meta-llama/Llama-3.2-1B` |
| 6.0G | `meta-llama/Llama-3.2-3B` |

**현재 상태** — 서빙 프로세스 0개, GPU 메모리 사용량 0 MiB 0 MiB

## 2. 저장소 상태

브랜치 `main`, 워킹트리 **커밋 안 된 변경 있음**, origin과 동기화됨

| 커밋 | 날짜 | 제목 |
| --- | --- | --- |
| 1f48aa0 | 2026-08-12 | Collapse results/ from ten directories to four studies |
| 6969c0b | 2026-08-12 | Audit both public repos properly and soften the Algorithm 1/2 claim |
| 60bc285 | 2026-08-12 | Status report generator now emits Korean |
| 07e03fb | 2026-08-12 | Add a status report generator that probes rather than asserts |
| ddfb5d7 | 2026-08-12 | Rate-sweep and burst experiments on 3x Llama-3.1-8B, 2 GPUs |
| a5f676b | 2026-08-12 | Generalise the runner to N GPUs and add an agent runbook |
| 12462c5 | 2026-08-12 | Two-GPU support: environment re-verification and §7.3 placement ablation |
| ddff762 | 2026-08-04 | ShareGPT colocation experiment with slowdown-based SLO on 1x A100 |
| eabf72e | 2026-08-03 | Reproducible single-GPU Prism (OSDI'26) baseline environment + A100 sanity sweep |

커밋 안 된 파일:
```
M CLAUDE.md
 M exp/results/3-placement/REPORT.md
 M exp/scripts/build_status_report.py
```

## 3. 실험 목록

분석 완료된 run **28건**, 결과 네임스페이스 4개. 모든 값은 커밋된 `*_slo.json` / `*_summary.csv`에서 읽습니다.

| 네임스페이스 | run | 요청수 | 시간 | att TTFT | att TPOT | TTFT p95 ms | TPOT p50 ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `1-env-verification` | `base_M1` | 296 | 603초 | 0.983 | 0.986 | — | 14 |
| `1-env-verification` | `base_M2` | 22 | 272초 | 1.000 | 1.000 | — | 12 |
| `1-env-verification` | `base_M4` | 262 | 607초 | 1.000 | 1.000 | — | 14 |
| `1-env-verification` | `sanity_A` | 296 | 603초 | 1.000 | 1.000 | — | 14 |
| `1-env-verification` | `sanity_B` | 558 | 612초 | 0.991 | 0.224 | — | 28 |
| `1-env-verification` | `sanity_C` | 318 | 603초 | 1.000 | 0.959 | — | 14 |
| `1-env-verification` | `sanity_repeat_C` | 318 | 603초 | 1.000 | 0.912 | — | 15 |
| `1-env-verification` | `verify_A` | 296 | 603초 | 0.997 | 0.997 | — | 14 |
| `1-env-verification` | `verify_B` | 558 | 612초 | 0.987 | 0.219 | — | 29 |
| `1-env-verification` | `verify_C` | 318 | 604초 | 0.987 | 0.934 | — | 14 |
| `2-colocation` | `exp_A` | 296 | 603초 | 1.000 | 1.000 | — | 14 |
| `2-colocation` | `exp_B` | 558 | 611초 | 0.991 | 0.229 | — | 28 |
| `2-colocation` | `exp_C` | 318 | 603초 | 0.997 | 0.959 | — | 14 |
| `2-colocation` | `sharegpt_content_A` | 296 | 603초 | 1.000 | 1.000 | — | 14 |
| `2-colocation` | `sharegpt_content_B` | 558 | 613초 | 0.991 | 0.224 | — | 29 |
| `2-colocation` | `sharegpt_content_C` | 318 | 603초 | 0.997 | 0.918 | — | 16 |
| `3-placement` | `fig7_glob_off_ts0.5` | 754 | 316초 | 0.932 | 0.212 | — | 34 |
| `3-placement` | `fig7_glob_off_ts1` | 754 | 616초 | 0.954 | 0.259 | — | 34 |
| `3-placement` | `fig7_glob_on_ts0.5` | 754 | 310초 | 0.960 | 0.387 | — | 24 |
| `3-placement` | `fig7_glob_on_ts1` | 754 | 608초 | 0.966 | 0.569 | — | 20 |
| `4-rate-sweep` | `burst_glob_on_ts1` | 7569 | 497초 | 0.990 | 0.483 | 231 | 63 |
| `4-rate-sweep` | `exp_glob_on_ts0.4` | 5090 | 210초 | 0.707 | 0.291 | 21,325 | 111 |
| `4-rate-sweep` | `exp_glob_on_ts0.5` | 5090 | 243초 | 0.987 | 0.307 | 239 | 89 |
| `4-rate-sweep` | `exp_glob_on_ts0.6667` | 5090 | 311초 | 0.999 | 0.379 | 191 | 68 |
| `4-rate-sweep` | `exp_glob_on_ts0.8` | 5090 | 362초 | 1.000 | 0.447 | 177 | 57 |
| `4-rate-sweep` | `exp_glob_on_ts1` | 5090 | 443초 | 1.000 | 0.872 | 165 | 44 |
| `4-rate-sweep` | `probe_glob_on_ts1` | 5375 | 352초 | 0.894 | 0.206 | 3,868 | 56 |
| `4-rate-sweep` | `ref_glob_on_ts1` | 702 | 193초 | 1.000 | 0.994 | 76 | 15 |

> attainment는 항상 `analyze_slo.py`가 재계산한 값입니다. `benchmark.py`의 `average_attainment_tpot`은 ms 단위 baseline을 초 단위 측정값과 비교해서 **항상 1.0**이므로 쓰면 안 됩니다.

## 4. Rate sweep — 3× Llama-3.1-8B, 2 GPU

λ_base = **12 req/s**. 프로파일링으로 찾은 TTFT knee(약 26 req/s)의 46%로 정했습니다(임의 지정 아님). 모든 rate가 **동일한 request 시퀀스**를 `--time-scale`로 압축한 것이라 길이 분포·모델 비율·seed가 행마다 같습니다.

| 요청 λ | ×base | 실제 처리 | 출력 tok/s | TTFT p50 ms | TTFT p95 ms | TPOT p50 ms | att TPOT | KV 풀 m1/m4/m5 | 최대 큐 모델/스케줄러 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | :---: |
| 12 | 1.00× | 11.5 | 2,360 | 70 | **165** | 44 | 0.872 | 0.08 / 0.16 / 0.16 | 0 / 0 |
| 15 | 1.25× | 14.1 | 2,885 | 75 | **177** | 57 | 0.447 | 0.09 / 0.21 / 0.20 | 0 / 0 |
| 18 | 1.50× | 16.4 | 3,365 | 79 | **191** | 68 | 0.379 | 0.11 / 0.28 / 0.26 | 0 / 0 |
| 24 | 2.00× | 21.0 | 4,304 | 92 | **239** | 89 | 0.307 | 0.23 / 0.69 / 0.58 | 0 / 0 |
| 30 | 2.50× | 24.3 | 4,983 | 132 | **21,325** | 111 | 0.291 | 0.29 / 0.98 / 0.98 | 38 / 184 |

2.0× → 2.5× 구간에서 TTFT **p95는 89배 폭증**하는데 **p50은 1.4배**에 그칩니다 — 평균과 중앙값이 이 절벽을 완전히 감춥니다. 절벽 지점은 2모델 GPU의 KV 풀이 약 0.98에 도달하는 시점, 그리고 큐가 처음으로 비어있지 않게 되는 시점과 정확히 일치합니다.

### 4.1 rate보다 colocation이 지배적이다

| 요청 λ | model_1 (GPU0 단독) | model_4 (GPU1 공유) | model_5 (GPU1 공유) |
| ---: | :---: | :---: | :---: |
| 12 | 1.000 / 89 / 17 | 0.835 / 170 / 47 | 0.787 / 183 / 49 |
| 15 | 1.000 / 89 / 19 | 0.189 / 185 / 62 | 0.177 / 199 / 62 |
| 18 | 1.000 / 95 / 20 | 0.093 / 200 / 76 | 0.072 / 216 / 74 |
| 24 | 0.854 / 153 / 24 | 0.056 / 247 / 115 | 0.037 / 277 / 101 |
| 30 | 0.826 / 177 / 30 | 0.062 / 23,054 / 133 | 0.009 / 12,411 / 127 |

각 칸은 `att_tpot / TTFT p95 ms / TPOT p50 ms`. λ_base에서 GPU0에 혼자 있는 모델은 무경합 baseline 그대로인데, GPU를 공유하는 두 모델은 이미 **약 2.8배** 느립니다 — rate를 올리기도 전에 그렇습니다.

### 4.2 Burst — hot 모델 수 1 → 2 → 3

| phase | hot 모델 | 총 λ | att both | TPOT p50 ms | model_1 TTFT p95 | model_4 TTFT p95 | model_5 TTFT p95 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1개 hot (8 / 0.5 / 0.5) | 8.8 | **0.986** | 22 | **100** | 155 | 141 |
| 1 | 2개 hot (8 / 8 / 0.5) | 17.3 | **0.489** | 41 | **111** | 217 | 155 |
| 2 | 3개 hot (8 / 8 / 8) | 24.3 | **0.207** | 97 | **271** | 250 | 334 |

model_1은 세 phase 내내 8 req/s로 **고정**되어 있으므로, 그 열의 변화는 순수하게 다른 모델의 burst가 준 피해입니다.

이 run에서 컨트롤러가 실제로 한 일:
```
ACTION: deactivate model_5 on GPU 1 and activate model_5 on GPU 0. Reason: migrate model
ACTION: deactivate model_5 on GPU 0 and activate model_5 on GPU 1. Reason: migrate model
```

3개가 모두 hot이 되자 Prism이 GPU1 부하를 덜려고 모델 하나를 GPU0으로 옮겼고, 그 결과 그때까지 보호받던 model_1이 경합에 노출됐습니다. hot 모델 3개를 GPU 2장에 놓는 좋은 배치는 존재하지 않으므로 이건 정책 결함이 아니라 자원 부족입니다.

### 4.3 capacity 프로파일링 — λ_base를 어떻게 정했나

rate를 **한 번의 run 안에서** 계단식으로 올리고 도착 시각 구간별로 집계했습니다. rate당 run 하나씩 돌리는 대신 총 2번으로 capacity 곡선을 얻습니다.

| ramp | 요청 req/s | 출력 tok/s | TTFT p95 ms | TPOT p50 ms |
| --- | ---: | ---: | ---: | ---: |
| 저부하 ramp (1 → 8 req/s) | 1.1 | 262 | 154 | 29 |
| 저부하 ramp (1 → 8 req/s) | 1.9 | 373 | 128 | 27 |
| 저부하 ramp (1 → 8 req/s) | 2.9 | 576 | 132 | 29 |
| 저부하 ramp (1 → 8 req/s) | 3.9 | 794 | 146 | 31 |
| 저부하 ramp (1 → 8 req/s) | 5.9 | 1,191 | 137 | 33 |
| 저부하 ramp (1 → 8 req/s) | 7.8 | 1,680 | 157 | 38 |
| 고부하 ramp (8 → 31 req/s) | 7.8 | 1,612 | 151 | 36 |
| 고부하 ramp (8 → 31 req/s) | 11.5 | 2,336 | 151 | 41 |
| 고부하 ramp (8 → 31 req/s) | 16.5 | 3,525 | 186 | 58 |
| 고부하 ramp (8 → 31 req/s) | 23.0 | 4,811 | 228 | 97 |
| 고부하 ramp (8 → 31 req/s) | 30.8 | 6,085 | 5,552 | 109 |

두 ramp가 약 7.8 req/s에서 겹치고 값이 일치하므로 재현성 검증도 겸합니다.

## 5. 논문 vs 공개 코드 — 고정된 소스에서 매번 재검증

아래는 **이 보고서를 생성하는 시점에** 실제 소스를 grep해서 확인합니다. 맞지 않게 된 항목은 `확인 실패` / `재확인 필요`로 표시됩니다.

**감사 범위** — `Multi-LLM/prism-research` 브랜치 1개·커밋 4개(핀된 SHA가 HEAD), `ovg-project/kvcached` 브랜치 22개·커밋 339개·태그 5개. 두 레포의 **모든 커밋**에서 `KVPR` / `kv_pressure` / `w_token_rate` / `shared_kv` / `Hodgson` / `Moore`를 검색해 소스 파일 히트 0건. 두 조직의 다른 공개 레포에도 Prism 컨트롤 플레인은 없습니다.

다만 **이름이 없다고 알고리즘이 없는 것은 아니므로**, 아래는 구성요소 단위로 형태를 대조한 결과입니다. 결론부터: 두 알고리즘 모두 *목적은 같지만 명세와 다른 단순화된 휴리스틱*으로 구현되어 있습니다.

**Algorithm 1 (KVPR 기반 배치) — 구성요소 대조**

| 논문 구성요소 | 공개 코드 | 위치 | 설명 |
| --- | --- | --- | --- |
| 분모 `shared_kv` (GPU 용량 − 가중치) | 구현됨 | `simple_global.py:93` | 논문의 shared_kv와 같은 개념 |
| 분자 = SLO 가중 토큰율 `token_rate*token_size/SLO` | 대체됨 | `simple_global.py:370` | 토큰 크기·SLO 가중 없이 **평활 요청 수**만 씀 |
| 모델을 `t*tz/s` 내림차순 정렬 (Alg.1 line 1) | 없음 (확인) | — | 정렬 키 10개 전부 violation 비율·요청수·잔여예산·가용메모리 중 하나 |
| 마이그레이션 임계값 τ (Alg.1 line 8) | 유사 구현 | `simple_global.py:183` | τ에 해당하는 임계값은 있으나 KVPR이 아닌 다른 지표에 적용 |

→ **판정: 부분 구현.** GPU별 메모리 압력을 균형 잡는다는 목적과 분모(`shared_kv`)는 같지만, KVPR의 핵심인 **SLO·토큰 크기 가중**이 빠지고 평활 요청 수로 대체되었습니다. 따라서 이 코드로 얻은 결과는 *global placement가 도움이 된다*는 것은 보여줄 수 있어도 *Algorithm 1을 검증했다*고는 할 수 없습니다.

**Algorithm 2 (Moore-Hodgson 요청 중재) — 구성요소 대조**

| 논문 구성요소 | 공개 코드 | 위치 | 설명 |
| --- | --- | --- | --- |
| 데드라인 `d = a + s` | 구현됨 | `request_queue.py:30` | 우선순위 힙 키에 포함 |
| 실행시간 추정 `e = p / c` | 구현됨 | `request_queue.py:27` | c = 1024/0.5 tok/s로 고정된 chunked-prefill 속도 |
| 데드라인 오름차순 처리 | 구현됨 | `request_queue.py:154` | min-heap이므로 사실상 EDF |
| 실행 불가 시 최장 작업 제거 (Alg.2 line 9-11) | 없음 (확인) | — | 누적 완료시각 검사도, 제거 단계도 없음 → 최적성 주장 근거가 사라짐 |
| 이를 적용할 admission control | 무력화됨 | `request_queue.py:137` | 자원 한도가 무한이라 승인 판정 자체가 항상 통과 |

→ **판정: 재료는 있고 메커니즘이 없음.** 데드라인과 실행시간 추정이 모두 계산되고 데드라인 순으로 처리되지만, 논문이 최적성을 증명하는 **'완료 못 하면 최장 작업을 빼낸다'** 단계가 없어 결과적으로 단순 EDF입니다. 게다가 이를 적용할 admission control이 무한 자원으로 무력화되어 있습니다.

### 5.1 그 밖에 재검증된 항목

| 주장 | 상태 | 위치 | 증거 | 의미 |
| --- | --- | --- | --- | --- |
| §6.2 admission control이 비활성 | 확인됨 | `request_queue.py:137` | `net_available = float("inf")` | 메모리 부족으로 요청이 거절되는 일이 없음 → `rejected` 0은 관측 실패가 아니라 구조적 |
| §6.1 migration 임계값이 하드코딩·매우 느슨 | 확인됨 | `simple_global.py:183` | `self.MEMORY_PER_REQUEST_RATIO_THRESHOLD = 15` | 기본 정책은 GPU 간 이 배율만큼 벌어져야 migrate. 실측 불균형은 약 1.6배 |
| idle eviction 임계값 | 확인됨 | `simple_global.py:181` | `self.MODEL_IDLE_THRESHOLD = 50  # seconds` | 논문 §A.4가 최적이라고 한 약 45초와 근접 |
| model service가 --num-gpus가 아니라 device_count를 봄 | 확인됨 | `multi_model_server.py:579` | `num_devices = torch.cuda.device_count()` | 멀티 GPU 박스에서 1-GPU 실험을 하려면 CUDA_VISIBLE_DEVICES가 필수 |

**맥락 — 시기 문제가 아니다.** `prism-research`는 논문 제1저자(Shan Yu, shanyu1@g.ucla.edu)가 2025-08-09에 올린 "Prism research prototype"이고 커밋 4개짜리다. 처음에는 "논문 개정 전 스냅샷이라 알고리즘이 없는 것"이라고 추정했으나 **틀렸다**: arXiv 2505.04021 **v1(2025-05-06)에 이미** Algorithm 1(KVPR)과 Algorithm 2(Moore-Hodgson)가 둘 다 있다. 즉 코드가 알고리즘보다 3개월 뒤에 나왔다.

실제로 바뀐 것은 KVPR의 **분자**다.

| 버전 | KVPR 분자 | 비고 |
| --- | --- | --- |
| arXiv v1 (2025-05) | `req_rate / SLO` | 요청률 기반 |
| arXiv v3 = OSDI'26 (2026-06) | `token_rate * token_size / SLO` | 토큰 기반으로 정교화 |
| 공개 코드 (2025-08) | 요청 수 (SLO 가중 없음) | v1의 역수에서 SLO 가중마저 빠짐 |

따라서 공개 코드는 *논문 개정 이전판*이 아니라 **어느 판본과도 일치하지 않는 단순화 프로토타입**이다. 논문이 공식 아티팩트로 링크하는 것은 kvcached뿐이고 (prism-research는 논문 어디에도 나오지 않는다), prism-research의 README는 아직 v1 제목("Unleashing GPU Sharing")과 13인 저자 목록을 인용하고 있다 — 반면 kvcached의 README는 OSDI판(21인)을 인용한다.

위 표는 *공개 코드가 무엇을 하는지*에 대한 진술이며, 저자들이 무엇을 구현했는지에 대한 진술이 아니다. Algorithm 1을 직접 구현한다면 **OSDI판(v3)의 토큰 기반 분자**를 써야 한다.

다른 이유로 재현 불가: MuxServe++/QLM/ServerlessLLM 베이스라인은 torch/vllm 핀 충돌로 미설치이고, Hyperbolic / Novita / Chatbot Arena 프로덕션 트레이스는 비공개입니다.

## 6. 재현 방법

커맨드 전문과 설계 근거는 [`EXPERIMENT.md`](../../EXPERIMENT.md), 새 서버 셋업은 [`CLAUDE.md`](../../CLAUDE.md)를 보세요.

```bash
source exp/scripts/env.sh
export SLO_BASE_FILE=$PWD/exp/configs/slo_base_3x8b_sharegpt.json

# 커밋된 1-GPU baseline과 대조해 환경 검증
CUDA_VISIBLE_DEVICES=0 TAG=verify ./exp/scripts/run_sanity.sh A   # 이어서 B, C

# N-GPU 배치 config 생성 후 실행 (NGPU는 보이는 GPU 수로 자동 설정)
python exp/scripts/make_config.py --num-gpus 2 --slots 1,4,5 \
    --placement balanced -o exp/configs/llama_2gpu_3x8b.json
SLOTS=1,4,5 CFG=$PWD/exp/configs/llama_2gpu_3x8b.json TAG=exp \
  TRACE=$DATASETS/sharegpt/exp_base12.pkl TPOT_SCALE=3 \
  ./exp/scripts/run_multigpu.sh glob_on 1
python exp/scripts/collect_metrics.py --exp exp_glob_on_ts1 --tag exp

# 이 보고서 재생성
python exp/scripts/build_status_report.py
```

## 7. 상세 문서 위치

| 문서 | 내용 | 크기 |
| --- | --- | --- |
| [`exp/results/4-rate-sweep/REPORT_rate_sweep.md`](4-rate-sweep/REPORT_rate_sweep.md) | 3× Llama-3.1-8B rate sweep + burst (이번 연구) | 8 KB |
| [`exp/results/3-placement/REPORT.md`](3-placement/REPORT.md) | 환경 구축 검증 + §7.3 global placement 실험 | 14 KB |
| [`exp/results/2-colocation/REPORT.md`](2-colocation/REPORT.md) | ShareGPT colocation 연구, 1 GPU (기존) | 18 KB |
| [`exp/results/1-env-verification/REPORT.md`](1-env-verification/REPORT.md) | 최초 1-GPU sanity 스윕 (기존) | 10 KB |
| [`EXPERIMENT.md`](../../EXPERIMENT.md) | 모든 커맨드와 각 선택의 근거 | 8 KB |
| [`CLAUDE.md`](../../CLAUDE.md) | 새로 빌린 GPU 서버 셋업 런북 | 13 KB |

## 8. 위 모든 수치에 공통으로 적용되는 주의사항

- **각 데이터 지점은 1회 측정입니다.** 수천 건 위에서 계산된 집계값(attainment, throughput, p50)은 안정적이지만, **30 req/s의 TTFT p95 21초 같은 꼬리값은 반복 측정 없이 정밀 수치로 인용하면 안 됩니다.** 절벽의 존재와 자릿수는 견고합니다.
- **전 구간 `--disable-cuda-graph`** (레포 관례). 이 때문에 절대 TPOT가 논문 장비 대비 약 1.57배 느리고, 그래서 SLO baseline을 이 장비에서 다시 측정했습니다. 절대 latency를 논문 수치와 직접 비교하면 안 됩니다.
- **기본 `real_trace.pkl`의 프롬프트는 `"Hello "*n` 합성**이라 prefix 중복이 99% 수준입니다. rate sweep 작업은 전부 ShareGPT 실제 텍스트(중복 2~4%)를 씁니다. 합성 트레이스에 radix cache를 켜면 절대 안 됩니다.
- **포화 전까지 queue length는 0이고 rejection은 항상 0**입니다 — 둘 다 구조적인 것으로 §5를 보세요. 부하 신호로는 `#running-req`와 TTFT p95를 봐야 합니다.
- **`/workspace`는 영구 볼륨이 아닙니다.** 인스턴스를 recycle/destroy하면 venv, 24 GB 가중치, `exp/server-logs/`가 전부 사라집니다. 커밋된 결과만 git에 푸시되어 살아남습니다.

