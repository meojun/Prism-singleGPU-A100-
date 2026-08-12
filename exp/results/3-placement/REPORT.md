# 2× A100-80G 환경 구축 검증 + §7.3 global placement 실험

측정일: 2026-08-12 · 목적: (1) `bootstrap.sh`로 재구축한 스택이 커밋된 1-GPU baseline을
재현하는지 확인하고, (2) 1-GPU에서는 원천적으로 불가능했던 **§7.3 / Figure 7 (global model
placement on/off)** 를 두 장짜리 GPU에서 실제로 돌린다.

---

## 1. 한 줄 결론

환경은 재현된다. 그리고 global placement를 켜면 **TPOT attainment가 2.2배(1× 부하) /
1.8배(2× 부하)**, goodput이 각각 **2.2배 / 1.9배** 올라간다 — 논문 Figure 7의 정성적
결론과 같다. 다만 **그 이득의 출처가 논문과 다르다**: 이득은 전부 idle eviction +
수요 기반 재활성화에서 나왔고, 논문 §6.1이 강조하는 **model migration은 두 부하 지점
모두에서 정확히 0회 발동**했다.

---

## 2. 환경

| 구성요소 | 값 |
| --- | --- |
| GPU | 2× NVIDIA A100-SXM4-80GB, cc 8.0, **NV12 NVLink**, driver 595.71.05 |
| CPU / RAM | 2× 64코어 (128 thread), 2003 GB |
| torch / sglang / vllm | 2.4.0+cu121 / 0.3.4.post2 (Prism fork) / 0.6.3.post1 |
| transformers / flashinfer | 4.45.2 / 0.1.6+cu121torch2.4 |
| kvcached | branch `prism/shm`, `vmm_ops` 확장 빌드됨 |
| redis | supervisor 서비스로 등록, 127.0.0.1:6379 |
| 모델 | Llama-3.1-8B (15.08 GB), Llama-3.2-3B (6.00), Llama-3.2-1B (2.28) |

`bootstrap.sh` 검증 단계 전부 OK (torch/sglang/vllm/transformers/flashinfer/kvcached+vmm_ops/cuda).

---

## 3. 1-GPU 재현 검증 (`results/1-env-verification/`)

`CUDA_VISIBLE_DEVICES=0 TAG=verify ./exp/scripts/run_sanity.sh {A,B,C}`.
커밋된 `results/1-env-verification/` 와 대조:

| case | model | n | att_ttft ref→new | att_tpot ref→new | tpot_p50 ms ref→new |
| --- | --- | ---: | --- | --- | --- |
| A | model_1 (8B) | 296 | 1.000 → 0.997 | 1.000 → 0.997 | 13.56 → 13.69 |
| B | model_1 (8B) | 296 | 1.000 → 0.993 | 0.348 → 0.345 | 28.05 → 28.36 |
| B | model_4 (8B) | 262 | 0.981 → 0.981 | 0.084 → 0.076 | 28.28 → 28.78 |
| C | model_1 (8B) | 296 | 1.000 → 0.986 | 1.000 → 0.976 | 14.26 → 14.16 |
| C | model_2 (3B) | 22 | 1.000 → 1.000 | 0.409 → 0.364 | 20.69 → 20.76 |

3케이스 1,172 요청 전부 완료, 실패 0건. **decode 지표(tpot_p50)가 1% 이내로 일치**하고,
`REPORT.md`가 환경 건전성 지표로 지목한 두 패턴 — B의 TPOT 붕괴, C의 높은 attainment —
가 그대로 재현된다.

케이스 A가 1.000이 아닌 0.997인 이유는 조사했다. 위반 2건 모두 고립된 이상치다:
`idx=156` TTFT 246 ms 단발 스파이크(레퍼런스 최대는 86 ms), `idx=64`는 TTFT 45.9 ms로
정상인데 **출력이 7토큰뿐이라 TPOT 분모가 6개**인 노이즈. 나머지 분포는 레퍼런스와
겹친다(상위 TTFT 88.6/86.6/84.9 ms vs 레퍼런스 86.2/85.9/83.1 ms). 체제 변화가 아니다.

---

## 4. §7.3 / Figure 7 — 2 GPU, 8 모델

### 4.1 왜 이 믹스인가

임의로 고른 게 아니다. `trace.py::generate_e2e_benchmark_reqs`가 슬롯별 SLO baseline을
**모델 이름 주석과 함께 하드코딩**하고 있고, 그게 논문 §7.2/§7.3의 "eight models on two
shared GPUs" 믹스다:

| 슬롯 | 모델 | `real_trace.pkl` 요청 수 |
| --- | --- | ---: |
| model_1 | Llama-3.1-8B | **296** |
| model_2 | Llama-3.2-3B | 22 |
| model_3 | Llama-3.2-1B | 22 |
| model_4 | Llama-3.1-8B | **262** |
| model_5 | Llama-3.1-8B | **120** |
| model_6 / 7 / 8 | Llama-3.2-1B | 19 / 11 / 2 |

상위 3개가 요청의 90%를 차지한다. 초기 배치는 **인덱스 순 블록 분할**(GPU0 = model_1~4,
GPU1 = model_5~8) 로 고정했다 — 트레이스를 모르는 상태에서 누구나 쓸 배치이고, 그 결과
GPU0가 요청의 **80%**(602/754)를 받는다. 두 arm 모두 동일한 초기 배치에서 출발한다.
재생성: `python exp/scripts/make_config.py --num-gpus 2 --placement blocks -o <경로>`
(같은 스크립트의 `roundrobin`은 60/40, `balanced`는 49/51이 된다).

### 4.2 arm 정의

| arm | 플래그 | 의미 |
| --- | --- | --- |
| `glob_on` | `--enable-controller --policy simple-global` | 전체 Prism (§5 + §6) |
| `glob_off` | (controller 제외) | 초기 배치 고정, ballooning + GPU-local 스케줄러만 |

GPU-local 스케줄러는 끌 수 없다. worker pool 요청 핸들러가
`"GPU scheduler must be enabled when using worker pool"` 로 죽는다. 따라서
`--enable-controller` 가 global placement를 분리하는 유일한 깨끗한 손잡이다.

### 4.3 결과 (Figure 7a 대응)

`ALL` 기준, 754 요청 전부 완료 / 실패 0건:

| 부하 | | TTFT attain | TPOT attain | goodput (req/s) | goodput (tok/s) | TPOT p99 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1× (ts=1.0) | w/o global | 0.954 | 0.259 | 0.31 | 65 | 50.4 ms |
| | **w/ global** | 0.966 | **0.569** | **0.68** | **182** | 39.3 ms |
| | 개선 | +1% | **2.20×** | **2.17×** | 2.82× | −22% |
| 2× (ts=0.5) | w/o global | 0.932 | 0.212 | 0.49 | 92 | 65.9 ms |
| | **w/ global** | 0.960 | **0.387** | **0.93** | **229** | 48.4 ms |
| | 개선 | +3% | **1.82×** | **1.91×** | 2.50× | −27% |

논문과 일치하는 점: 부하가 오르면 양쪽 다 attainment가 떨어지지만 **격차는 유지**된다.
그리고 이득이 **TPOT에 집중**된다 — 논문 §7.2의 "TPOT는 KVPR 균형이 GPU당 메모리 경합을
잡으면서 부수적으로 좋아진다"와 같은 방향이다. TTFT는 양쪽 다 이미 0.93+이라 여지가 없다.

모델별로 보면 재배치의 승자와 패자가 뚜렷하다 (1× 부하, TPOT attain):

| | model_1 | model_4 | model_5 | ALL |
| --- | ---: | ---: | ---: | ---: |
| w/o global | 0.260 | **0.008** | 0.825 | 0.259 |
| w/ global | **0.807** | **0.454** | 0.392 | **0.569** |

무거운 model_1·model_4가 크게 이득을 보고 model_5가 손해를 본다. 순효과는 크게 플러스다.

### 4.4 결과 (Figure 7b 대응) — GPU 부하 균형

nvidia-smi 2초 샘플링 (하네스가 논문의 "요청당 가용 KV 메모리"를 내보내지 않아 대용):

| 부하 | arm | GPU0 mem / util | GPU1 mem / util | 메모리 불균형 |
| --- | --- | --- | --- | ---: |
| 1× | w/o global | 48.2 GiB / **75%** | 30.2 GiB / **34%** | 1.59× |
| 1× | w/ global | 27.9 GiB / 62% | 34.3 GiB / 67% | **1.23×** |
| 2× | w/o global | 48.8 GiB / **78%** | 30.3 GiB / **49%** | 1.61× |
| 2× | w/ global | 30.2 GiB / 76% | 35.4 GiB / 77% | **1.17×** |

논문 Figure 7b가 말하는 그림이다. controller 없이는 GPU1이 놀고 GPU0가 과부하다
(75% vs 34%). 켜면 **활용률이 62/67%, 76/77%로 거의 같아진다.**

### 4.5 이득의 출처 — 논문과 다른 지점

controller 로그를 세어보면 (1×, 2× 부하에서 **동일**):

```
activations   : 3
deactivations : 6   (전부 idle instance eviction)
migrations    : 0
migration 패스가 아무것도 못 찾은 횟수: 122 (1×) / 62 (2×)
```

시간순 액션:

```
deactivate model_3 on GPU 0   (idle)
deactivate model_4 on GPU 0   (idle)
deactivate model_8/7/6 on GPU 1 (idle)
activate   model_4 on GPU 1   <-- 부하 재분배가 일어난 지점
deactivate model_2 on GPU 0   (idle)
activate   model_3 on GPU 0
activate   model_8 on GPU 0
```

즉 **두 번째로 무거운 model_4(262 요청)가 GPU0에서 축출된 뒤 GPU1에 재활성화된 것**이
전부다. GPU0 메모리가 48.2 → 27.9 GiB로 떨어지고 GPU1이 30.2 → 34.3 GiB로 오른 이유가
이것이다.

**명시적 migration 경로는 한 번도 타지 않았다.** 로그가 이유를 그대로 말한다:
`Current unstable pairs: 1` → `No migrations found that reduce unstable pairs`.
`simple_global.py:183` 의 `MEMORY_PER_REQUEST_RATIO_THRESHOLD = 15` 가 원인이다 —
기본 migrate 정책(`memory_per_request`)은 두 GPU의 요청당 메모리 비율이 **15배** 넘게
벌어져야 움직인다. 실측 불균형은 1.6배였다. 이 상수는 플래그가 아니라 하드코딩이다.

부작용도 기록해둔다. `model_8`은 요청이 2건뿐이라 idle 축출 대상이 되었고, 재활성화
콜드스타트 때문에 TTFT p99가 **4,478 ms**까지 튀어 attainment 0.000이 됐다(controller를
끄면 1.000). 논문 §A.4의 idle threshold 트레이드오프가 그대로 관찰된 것이고, 코드의
`MODEL_IDLE_THRESHOLD = 50 s`는 논문이 최적이라고 한 ~45 s와 가깝다.

---

## 5. 재현 방법

```bash
source exp/scripts/env.sh

# 1-GPU 검증 (CUDA_VISIBLE_DEVICES 를 반드시 줄 것 — §6 참고)
CUDA_VISIBLE_DEVICES=0 TAG=verify ./exp/scripts/run_sanity.sh A   # 이어서 B, C

# 2-GPU Figure 7
./exp/scripts/run_multigpu.sh glob_on  1     # 이어서 glob_off, 그리고 0.5
python exp/scripts/compare_fig7.py --tag fig7 --ts 1
```

산출물: `results/3-placement/*_slo.json`, `results/3-placement/summary.tsv`,
`results/3-placement/*_actions.txt`, `server-logs/fig7_*/gpu_timeline.txt`.

---

## 6. 멀티 GPU로 옮기며 밟은 함정 (전부 `run_multigpu.sh`에 반영됨)

1. **tmux 서버가 `CUDA_VISIBLE_DEVICES`를 물고 있다.** tmux는 사용자당 서버 프로세스
   하나를 유지하고, 모든 `new-session`이 **그 서버가 처음 뜰 때의 환경**을 상속한다.
   1-GPU sanity를 `CUDA_VISIBLE_DEVICES=0`으로 돌린 뒤 같은 tmux 서버에서 2-GPU 서버를
   띄우면 GPU 1 워커가 전부
   `RuntimeError: CUDA error: invalid device ordinal` (`init_torch_distributed`의
   `torch.cuda.set_device(1)`)로 죽는다. 스크립트가 상속에 기대지 않고 명시적으로
   export 하고, 기동 전에 `torch.cuda.device_count()`를 검사한다.
2. **tmux는 세션 이름의 `.`을 `_`로 조용히 바꾼다.** `time_scale=0.5` → 세션 이름
   `...ts0.5`로 만들면 `tmux has-session`이 영원히 실패해서, 서버는 멀쩡히 뜨는 중인데
   런처가 즉시 "DIED"로 판정하고 **서버를 고아로 남긴다.** 이름을 미리 정규화한다.
3. **결과 파일 이름에 GPU 수가 박힌다.** `benchmark.py`가
   `f"{exp}_e2e_{num_gpus}gpu_..."`로 저장하는데 `run_sanity.sh`는 `*_e2e_1gpu_*`를
   하드코딩 glob 한다. 2 GPU에서 그대로 쓰면 분석 단계가 조용히 실패한다.
4. **모든 GPU에 `on: true` 모델이 최소 하나 있어야 한다.**
   `launch_multi_model_server`가 초기 배치에 등장하는 gpu_id에 대해서만 GPU 스케줄러를
   띄우므로, 빈 GPU에는 이후에도 모델을 올릴 수 없다.
5. **`launch_model_service()`는 `--num-gpus`가 아니라 `torch.cuda.device_count()`를
   본다** (`multi_model_server.py:576`). 2-GPU 박스에서 1-GPU 실험을 하려면
   `CUDA_VISIBLE_DEVICES=0`이 필수다. 안 그러면 model service는 GPU 2개용 broker를
   띄우는데 worker pool 엔진은 GPU 0에만 있다. 같은 함수의
   `num_model_service_workers = 1`은 플래그를 덮어쓴다.
6. `get_model_names` 엔드포인트가 마지막 GPU의 모델만 반환한다(2-GPU에서
   `["model_5".."model_8"]`). 표시상 문제일 뿐 라우팅은 8개 전부 정상이다 —
   controller 로그와 요청 완료 수(754/754)로 확인했다.

---

## 7. 논문 대비 이 실행이 검증한 것 / 못 한 것

**검증됨**
- §5 kvcached 밸루닝 — 로그에 `Elastic memory: kv_cache_manager_v0 initialized`
- §6.1 global placement의 **효과** — 위 §4
- §A.4 idle threshold 트레이드오프 — model_8 콜드스타트로 관찰

**재현 안 됨 (공개 코드에 없음)**
- **Algorithm 1 (KVPR) — 부분 구현.** 두 공개 레포의 전 커밋을 감사한 결과 `KVPR` /
  `w_token_rate` / `shared_kv` 라는 이름은 어디에도 없다. 구성요소로 보면 분모
  (`memory_available_for_requests = gpu_mem − weights`, `simple_global.py:93`)는 같은
  개념이 있으나, 분자가 `token_rate*token_size/SLO`가 아니라 **평활 요청 수**이고,
  그 비율로 모델을 정렬하는 코드도 없다. τ에 해당하는 임계값은 있으나 다른 지표에
  적용된다. 목적(GPU별 메모리 압력 균형)은 구현되어 있고 지표가 다르다.
- **Algorithm 2 (Moore-Hodgson) — 재료는 있고 메커니즘이 없음.** 데드라인 `a+s`와
  실행시간 추정 `p/c`가 모두 계산되고(`request_queue.py:27,30`) 데드라인 순으로
  처리되지만, 최적성 증명의 근거인 **완료 불가 시 최장 작업 제거** 단계가 없어 단순
  EDF가 된다. 이를 적용할 admission control도 비활성이다.
- **시기 문제가 아니다.** "코드가 논문보다 먼저라 없는 것"이라는 설명은 성립하지
  않는다. arXiv 2505.04021 **v1(2025-05-06)에 이미 두 알고리즘이 다 있고**, 프로토타입
  공개는 그 3개월 뒤인 2025-08-09이다. OSDI 개정(v3, 2026-06-10)이 바꾼 건 KVPR의
  분자뿐이다: `req_rate/SLO` → `token_rate*token_size/SLO`. 공개 코드는 둘 중 어느
  것과도 다르게 SLO 가중 없는 요청 수를 쓴다. 논문이 공식 링크하는 아티팩트는
  kvcached뿐이며 prism-research는 논문에 등장하지 않는다. 위는 *공개 코드에 대한*
  진술이다.
- **§6.2 admission control.** `request_queue.py:137`이 `net_available = float("inf")`.
  런타임 로그에서도 매 초 `net_available: inf`로 확인된다.
- **§6.1 overlapped migration.** 코드는 source를 먼저 deactivate 하고 target을
  activate 한다(정렬 순서상 deactivate 우선). 논문의 "target 준비될 때까지 source가
  계속 서빙 + NVLink로 weight/KV 전송"이 아니다. 게다가 이번 실행에서 migration 자체가
  0회다.
- MuxServe++/QLM/ServerlessLLM 베이스라인 미설치, 프로덕션 트레이스 비공개.

---

## 8. 주의사항

- `real_trace.pkl`의 프롬프트는 `"Hello "*n` 합성이다. 이번 실행은 레포 관례대로
  `--disable-radix-cache`로 돌렸으므로 문제없지만, **radix cache를 켜는 순간 prefill의
  99.3%가 캐시 히트가 되어 결과가 무의미해진다**(`SETUP.md` 참고). 실제 텍스트가
  필요하면 `build_sharegpt_trace.py`의 `content` 변형을 쓸 것.
- TPOT attainment는 `analyze_slo.py`가 재계산한 값이다. `benchmark.py`의
  `average_attainment_tpot`은 ms/s 단위 불일치로 항상 1.0이므로 쓰면 안 된다.
- 두 arm은 각 부하 지점당 1회씩만 돌렸다. TTFT p99 같은 꼬리 지표는 반복 없이 비교하면
  안 된다(예: `glob_on` ts=0.5의 TTFT p99 2,326 ms는 model_8 콜드스타트 1건이 만든 값).
  §4.3의 attainment·goodput 같은 집계 지표는 754건 위에서 계산되어 훨씬 안정적이다.
