# Prism sanity check — 1× A100-80G 위 Llama 실험

작성일: 2026-08-03 · 목적: 이후 작업에서 **Prism (OSDI'26) 자체를 baseline으로 쓰기 위해**
환경이 end-to-end로 동작하는지 검증. 논문에 나오는 다른 baseline들(MuxServe/MuxServe++,
QLM, ServerlessLLM)은 **의도적으로 설치하지 않음** — `../../README.md` §7 참고.

---

## 1. 환경 (가정이 아니라 실제 확인함)

| 구성요소 | 버전 / 상태 |
| --- | --- |
| GPU | 1× NVIDIA A100-SXM4-80GB, driver 595.71.05 (`driver_max_cuda` 13.2), cc 8.0 |
| venv | `/workspace/prism-exp/prism-venv` (Python 3.10) |
| torch | 2.4.0+cu121, `torch.cuda.is_available() == True` |
| sglang | 0.3.4.post2 (Prism fork, `prism-research/`) |
| vllm | 0.6.3.post1 — 정상 import |
| transformers | 4.45.2 — 핀 유지됨 (5.x는 vLLM을 `ImportError: DTensor`로 깨뜨림) |
| kvcached | branch `prism/shm`, `vmm_ops` 확장 빌드 완료, import 정상 |
| redis | supervisor 서비스, `RUNNING`, `PING → PONG` |

주의: 셋업 로그 `setup_prism_env.log`에는 `transformers 5.14.1`과 vLLM import 실패가
찍혀 있는데, 이는 빌드 시점 기록이고 그 뒤에 수정되었다. 위 표는 이번 실행을 위해
**현재 상태를 다시 확인한** 값이다.

### 모델

`meta-llama/Llama-3.1-8B` (15 GB), `meta-llama/Llama-3.2-3B` (6 GB)를
`$HF_HOME=/workspace/.hf_home`에 다운로드. 두 모델 모두 Prism의 프로파일 파일
`model_info.json`에 이미 등록되어 있어 `profile_models.py` 단계는 불필요했다.

> **Llama-3.1에는 3B가 없다.** 3B는 Llama 3.2에만 존재하므로 케이스 C는
> `meta-llama/Llama-3.2-3B`를 사용했다.

---

## 2. 무엇을 돌렸는가

**full Prism 모드** (ballooning §5 + global placement §6.1 + slack-aware local
arbitration §6.2), A100 한 장, 세 가지 colocation 케이스:

| 케이스 | config | GPU 0에 올린 모델 | 모델별 KV pool 상한 |
| --- | --- | --- | --- |
| A | `exp/configs/llama_1x8b.json` | `model_1` = Llama-3.1-8B | 45 GiB |
| B | `exp/configs/llama_2x8b.json` | `model_1`, `model_4` = Llama-3.1-8B ×2 | 22 GiB |
| C | `exp/configs/llama_8b_3b.json` | `model_1` = Llama-3.1-8B, `model_2` = Llama-3.2-3B | 25 GiB |

재현: `exp/scripts/run_sanity.sh <A|B|C>` 실행 후 `exp/scripts/summarize_sanity.py`.

**슬롯 이름은 임의로 정한 게 아니다.** `trace.py::generate_e2e_benchmark_reqs`가 슬롯별
SLO baseline을 하드코딩해 두었고 그 값은 특정 모델을 대상으로 측정된 것이다. 즉
`model_1/model_4/model_5`가 Llama-3.1-8B 슬롯이고 `model_2`가 Llama-3.2-3B 슬롯이다.
따라서 케이스 B는 `model_1 + model_2`가 아니라 **`model_1 + model_4`**를 써서 두 모델
모두 8B 기준 SLO를 받도록 했다.

### 데이터셋

`prism-research/benchmark/multi-model/real_trace.pkl` — 저장소에 함께 배포되는 파일.
구조는 `[어댑터 이름 27개, 요청 1500개]`이며, `model_dir`은 전부 `huggyllama/llama-7b`,
어댑터 이름은 `dummy-lora-7b-rank-8-{0..26}` — S-LoRA 계열 멀티어댑터 트레이스 포맷이다.
전체 span 600초. Prism은 선택된 어댑터 rank를 `model_N` 슬롯에 매핑한다.

**프롬프트 내용은 합성이다.** 모든 프롬프트가 `"Hello " * prompt_len`이다 (1500건 전부
검증함). 이 트레이스가 제공하는 것은 **도착 시각, 입출력 길이, 모델 라우팅**뿐이고
내용은 없다. 서빙 성능 지표에는 문제없지만 출력 품질이나 prefix cache 효과를 보려면
쓸 수 없다. (`--disable-radix-cache`를 켰으므로 어차피 prefix 재사용도 없었다.)

전체 트레이스: 입력 길이 p50 40 / p99 902 토큰, 출력 길이 p50 211 / p99 785 토큰.

| 슬롯 | adapter rank | 요청 수 | 입력 p50/p99 | 출력 p50/p99 |
| --- | --- | --- | --- | --- |
| model_1 (8B) | rank-2 | 296 | 72 / 764 | 252 / 824 |
| model_2 (3B) | rank-14 | 22 | 192 / 880 | 147 / 555 |
| model_4 (8B) | rank-10 | 262 | 30 / 946 | 317 / 513 |

논문 §3의 프로덕션 트레이스(Hyperbolic / Novita / Arena)는 **비공개**이고 릴리스된
코드에서 로더도 제거되어 있다 (`--csv-trace`는 파싱만 되고 실제로 쓰이지 않음). 따라서
사용하지 않았고, 사용할 수도 없다.

### SLO 정의

`SLO = 논문의 슬롯별 p95 baseline × scale`, **TTFT ×5, TPOT ×2** 적용 (README의 검증된
예제에서 쓰는 배수). 결과 절대값은 아래 표에 있다.

- attainment(TTFT) = `ttft ≤ slo_ttft`인 요청 비율
- attainment(TPOT) = `tpot ≤ slo_tpot`인 요청 비율
- **goodput** = 두 SLO를 **모두** 만족한 요청의 초당 처리량 (및 그 요청들의 출력 토큰/초)
- **위반율** = `1 − attainment(both)`
- 실패 요청은 위반으로 계산되나, 이번엔 실패가 하나도 없었다

---

## 3. 결과

`real_trace.pkl`, time_scale 1, replication 1, 케이스당 약 600초.
**세 케이스 모두 실패 요청 0건.**

| 케이스 / 모델 | 요청 | 완료 | SLO TTFT | SLO TPOT | tput req/s | tput tok/s | att TTFT | att TPOT | att 둘다 | 위반율 | goodput r/s | goodput t/s | TTFT p50/p99 | TPOT p50/p99 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **A** model_1 (8B) | 296 | 296 | 0.214 s | 22.9 ms | 0.491 | 143.4 | 1.000 | 1.000 | **1.000** | 0.000 | 0.491 | 143.4 | 33.5 / 77.7 ms | 13.6 / 14.7 ms |
| **B** model_1 (8B) | 296 | 296 | 0.214 s | 22.9 ms | 0.484 | 141.3 | 1.000 | 0.348 | 0.348 | 0.652 | 0.168 | 43.2 | 54.7 / 141.1 ms | 28.0 / 31.1 ms |
| **B** model_4 (8B) | 262 | 262 | 0.232 s | 22.5 ms | 0.428 | 131.7 | 0.981 | 0.084 | 0.084 | 0.916 | 0.036 | 10.9 | 60.0 / 2976.6 ms | 28.3 / 31.4 ms |
| **B** 전체 | 558 | 558 | – | – | 0.912 | 273.0 | 0.991 | 0.224 | **0.224** | 0.776 | 0.204 | 54.0 | 58.3 / 195.4 ms | 28.2 / 31.2 ms |
| **C** model_1 (8B) | 296 | 296 | 0.214 s | 22.9 ms | 0.491 | 143.4 | 1.000 | 1.000 | 1.000 | 0.000 | 0.491 | 143.4 | 36.3 / 91.0 ms | 14.3 / 22.1 ms |
| **C** model_2 (3B) | 22 | 22 | 0.204 s | 19.2 ms | 0.036 | 7.6 | 1.000 | 0.409 | 0.409 | 0.591 | 0.015 | 3.8 | 44.7 / 80.1 ms | 20.7 / 21.7 ms |
| **C** 전체 | 318 | 318 | – | – | 0.527 | 150.9 | 1.000 | 0.959 | **0.959** | 0.041 | 0.506 | 147.1 | 36.6 / 89.8 ms | 14.3 / 22.0 ms |

실행 시간: A 602.8초, B 611.7초, C 603.0초.

### 해석

- **A** — 80 GB A100을 8B 모델 하나가 독점하므로 부하가 없다: attainment 100%. 이것이
  기준점이며, 스택이 end-to-end로 정상임을 확인해 준다.
- **B** — 8B 두 개를 colocate하면 처리량은 거의 두 배(0.491 → 0.912 req/s)가 되지만
  TPOT p50이 13.6 → 28.2 ms로 밀려 TPOT SLO(약 22.5 ms)를 넘긴다. 그 결과 TPOT
  attainment가 0.224로 무너지는 반면 TTFT는 0.991로 유지된다. 논문이 말하는 decode
  측 contention이 그대로 나타난 것이고, 처리량/TPOT 트레이드오프의 형태가 예상과 맞는다.
- **C** — 3B 슬롯은 실제로 희소한 long-tail 모델(요청 22건)이라 8B를 거의 방해하지
  않는다. 전체 attainment는 0.959로 유지된다.
- **B의 model_4 TTFT p99 = 2977 ms는 cold start이지 장애가 아니다.** 1초를 넘는 요청이
  정확히 4건이고 인덱스 101–105로 연속되어 있다 — model_4의 최초 활성화 구간, 즉
  Prism의 §5.3 모델 로딩 경로다. 나머지 요청은 모두 261 ms 미만이다.

절대 처리량이 낮아 보이는 이유는 트레이스가 설계상 희소하고 버스티하기 때문이다(가장
바쁜 슬롯도 600초에 296건). 이건 프로덕션 형태의 replay이지 saturation sweep이 아니다.

---

## 4. 발견한 하네스 결함 2건 — 우회 처리함

모든 attainment/goodput 수치가 여기에 의존하므로 중요하다.

1. **TPOT SLO 단위 불일치.** `trace.py`는 `model_tpot_slo_baseline_p95`를
   **밀리초** 단위로 저장하는데(model_1 = 11.46), `benchmark.py`는 이를 **초** 단위인
   `output.tpot`과 비교한다 (`tpot = (finish_time − prefill_finish_time)/n`, 출력할 때만
   `*1000`). 따라서 `outputs[i].tpot < outputs[i].slo_tpot`는 항상 참이고
   **릴리스된 코드가 보고하는 TPOT attainment는 전부 1.0이다.** TTFT SLO는 초 단위라
   정상이다. `analyze_slo.py`는 TPOT baseline을 1000으로 나눈다.

2. **이 코드 경로에서는 SLO 통계를 아예 계산하지 않는다.** `--model-paths`와
   `--real-trace`를 함께 주면 클라이언트가 `run_tp_mode`로 빠지는데, 여기서는 per-request
   원본만 덤프하고 `get_benchmark_metrics`를 완전히 건너뛴다 — 릴리스된 e2e 결과 JSON에
   attainment 필드가 없는 이유다. `analyze_slo.py`가 원본 덤프에서 전부 재계산한다.

**Prism과 비교할 때 `benchmark.py`의 `average_attainment_tpot`은 쓰면 안 된다.**

---

## 5. 산출물

| 경로 | 내용 |
| --- | --- |
| `exp/configs/llama_{1x8b,2x8b,8b_3b}.json` | 세 가지 모델 config |
| `exp/scripts/run_sanity.sh` | launch + bench + analyze, 호출당 케이스 하나 |
| `exp/scripts/analyze_slo.py` | 원본 덤프에서 attainment / goodput / 위반율 재계산 |
| `exp/scripts/summarize_sanity.py` | 비교 표 출력 |
| `exp/results/sanity/sanity_{A,B,C}_slo.json` | 계산된 모델별 지표 |
| `exp/results/sanity/sanity_{A,B,C}_e2e_1gpu_1.0x_1rep.json` | 하네스 원본 지표 |
| `exp/results/sanity/requests/*_output_requests.json` | per-request 원본 기록 |
| `exp/results/sanity/summary.tsv` | 8행 전체, machine-readable |
| `exp/server-logs/sanity_{A,B,C}/` | server stdout, gpu_scheduler, model_service, bench 로그 |

---

## 6. 한계 / 이 실험이 증명하지 못하는 것

- GPU 한 장뿐이다. §7.2 (8모델 / 2 GPU), global placement의 부하 분산 효과(≥2 GPU 필요),
  §7.4 (58모델 / 32 GPU), TP 실험은 이 장비에서 불가능하다.
- **`prism` 모드만 돌렸다.** 이 Llama 구성으로 `static`(S-Partition)과 `elastic`은
  재실행하지 않았으므로 **baseline 대비 delta가 없다.** 이 보고서는 시스템이 동작한다는
  확인이지, 이긴다는 확인이 아니다.
- `trace.py`의 SLO baseline은 저자들이 **H100**에서 측정한 값인데 이 장비는 **A100**이다.
  따라서 절대 attainment를 논문 수치와 직접 비교하면 안 된다. TTFT ×5 / TPOT ×2 스케일링이
  차이를 일부 흡수하지만, 본격적인 주장에 쓰려면 §7.1대로 전용 A100에서 baseline을
  재측정해야 한다.
- 케이스당 seed 1개, replication 1회. error bar 없음.

> ⚠️ 이 인스턴스에서 `/workspace`는 **volume이 아니다** (`workspace_is_volume: false`).
> stop/start는 유지되지만 **recycle 또는 destroy 시 전부 삭제된다.** 현재 git으로
> 관리되고 있지도 않다. 인스턴스 수명주기를 건드리기 전에 `exp/`를 외부로 옮겨야 한다.
