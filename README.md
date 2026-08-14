# Prism (OSDI'26) 기준선 — 실험 환경

> **새 GPU 서버에 세팅하려면 → [`SETUP.md`](SETUP.md) 를 읽고 `./bootstrap.sh` 를 실행하세요.**
> 한 줄 요약: `git clone <this repo> && cd prism-exp && ./bootstrap.sh`
> 검증된 sanity 결과 + 전체 보고서: [`exp/results/1-env-verification/REPORT.md`](exp/results/1-env-verification/REPORT.md)
>
> `prism-research/`, `kvcached*/`, `prism-venv/`, 모델 가중치는 저장소에 없습니다 —
> `bootstrap.sh` 가 고정 SHA/lockfile로 재생성합니다.

**Prism: Cost-Efficient Multi-LLM Serving via GPU Memory Ballooning** 을 재현하고
기준선을 잡기 위한 모든 것이 `/workspace/prism-exp` 아래에 있습니다.

> ⚠️ **이 인스턴스의 `/workspace` 는 볼륨이 아닙니다** (`workspace_is_volume: false`).
> stop/start 는 보존되지만 **recycle 이나 destroy 는 전부 지웁니다.** 잃으면 안 되는 것
> (설정, 결과, 패치)은 git 이나 외부 저장소로 밀어 두세요.

---

## 1. 무엇이 어디에 있는가

| 경로 | 내용 |
| --- | --- |
| `prism-research/` | **Prism 본체** — SGLang `v0.3.4.post2` 의 포크 (`github.com/Multi-LLM/prism-research`). 멀티모델 서버, 전역 배치, GPU-로컬 스케줄러, 워커 풀, 모델 서비스, 벤치마크 클라이언트가 들어 있습니다. |
| `kvcached-prism/` | kvcached **`prism/shm` 브랜치** — Prism 이 링크하는 벌룬 드라이버 버전 (`kvcached.ops`, `kvcached.slab_allocator`). kvcached main 과는 API 가 다릅니다. |
| `prism-venv/` | 위 두 가지를 위한 Python **3.10** venv. torch 2.4.0+cu121 / vllm 0.6.3.post1 / flashinfer 0.1.6 / transformers 4.45.2. |
| `kvcached/` | kvcached **main 브랜치** (`v0.1.5`) — 독립 배포판 벌룬 드라이버와 자체 컨트롤러(라우터 + 슬립 매니저 + 트래픽 모니터), 마이크로 벤치마크. |
| `kvcached/engine_integration/sglang-pip-venv/` | SGLang **0.5.10** + kvcached 0.1.5 이 든 Python **3.11** venv. `prism-venv` 와 완전히 독립입니다. |
| `exp/` | 이 세팅을 위해 작성한 모든 것: 설정, 실행/벤치 스크립트, 결과. |
| `/workspace/.hf_home` | HF 캐시 (`HF_HOME`). |

**두 스택은 독립이며 섞으면 안 됩니다.**
`prism-venv` 가 논문의 실제 시스템(구버전 SGLang)이고, `sglang-pip-venv` 는 최신
kvcached 전용 스택으로 현행 SGLang 과 비교하거나 kvcached 자체 벤치마크를 돌릴 때
씁니다.

서비스: **redis** 가 supervisor 서비스로 `127.0.0.1:6379` 에서 돕니다
(`supervisorctl status redis`) — Prism 의 컨트롤러와 GPU 스케줄러가 이를 필요로 합니다.

---

## 2. 빠른 시작 (Prism)

```bash
source /workspace/prism-exp/exp/scripts/env.sh          # prism-venv 활성화, HF_HOME 설정

# 1) 서버 기동 (tmux 세션 prism-<mode>)
exp/scripts/launch_server.sh <static|elastic|prism> <config.json> <port>

# 2) 그 서버를 향해 벤치마크 클라이언트 실행
exp/scripts/run_bench.sh <exp-name> <num-models> <port> [benchmark.py 인자...]
```

### 세 가지 모드

| 모드 | 추가되는 플래그 | 대응 |
| --- | --- | --- |
| `static` | – | **S-Partition** 기준선 (§7.1): 모델별 정적 KV 풀 |
| `elastic` | `--enable-elastic-memory --use-kvcached-v0` | Prism 의 **메모리 벌루닝만** (§5) |
| `prism` | + `--enable-cpu-share-memory --enable-gpu-scheduler --enable-controller --policy simple-global --enable-model-service --enable-worker-pool --max-mem-usage --num-gpus` | **완전한 Prism** (§5 + §6): 벌루닝 + 전역 배치 + slack 인지 로컬 조정 |

`prism` 모드는 `launch_server.sh` 가 읽는 환경변수로 조정합니다:
`NUM_GPUS`, `MAX_MEM`(GPU 당 GiB 예산), `WORKERS_PER_GPU`, `MODEL_SERVICE_WORKERS`.

### 동작이 확인된 예시

```bash
source exp/scripts/env.sh

# --- static 기준선, GPU 1개에 모델 2개 병치
exp/scripts/launch_server.sh static exp/configs/smoke_2model.json 30000
exp/scripts/run_bench.sh smoke_static 2 30000 --req-rate 4 --micro-benchmark

# --- elastic (kvcached 벌루닝)
exp/scripts/launch_server.sh elastic exp/configs/smoke_2model.json 30001
exp/scripts/run_bench.sh smoke_elastic 2 30001 --req-rate 4 --micro-benchmark --enable-elastic-memory

# --- 완전한 Prism, GPU 1개에 모델 8개, 실제 트레이스
NUM_GPUS=1 MAX_MEM=67.28 WORKERS_PER_GPU=8 MODEL_SERVICE_WORKERS=4 \
  exp/scripts/launch_server.sh prism exp/configs/qwen_1gpu_8model_prism.json 30002
exp/scripts/run_bench.sh prism_8m_e2e 8 30002 \
  --e2e-benchmark --real-trace ./real_trace.pkl --time-scale 1 --replication 1 \
  --num-gpus 1 --enable-elastic-memory --ttft-slo-scale 5 --tpot-slo-scale 2
```

결과는 `exp/results/` (`*_key_metrics.tsv`, `*_all.jsonl`) 와
`exp/results/requests/` 에 떨어집니다. 서버 로그는 `exp/server-logs/`
(`<mode>_stdout.log`, 그리고 `<mode>.log.gpu_scheduler.log` /
`.model_service.log`) — **완전한 Prism 모드가 실패하면 뒤의 두 개를 보세요.**
최상위 로그는 에러를 삼킵니다.

---

## 3. 트레이스

* `prism-research/benchmark/multi-model/real_trace.pkl` 은 **레포에 함께 들어 있습니다** —
  어댑터 27개 / 요청 1500개로, §7.2(8모델)와 §7.4(18모델) e2e 실험의 바탕이 된
  트레이스입니다. `--e2e-benchmark --real-trace ./real_trace.pkl`.
* 합성 생성기(`--micro-benchmark`, `--uniform-trace`, `--two-phase-trace`)는 내장이라
  데이터 파일이 필요 없습니다.
* §3 의 **Hyperbolic / Novita / Arena** 프로덕션 트레이스는 **공개되어 있지 않습니다.**
  `benchmark.py` 에 `--hyper-trace` / `--csv-trace` 플래그는 남아 있지만 로더가 공개
  코드에서 제거되었습니다 — `--csv-trace` 는 파싱만 되고 쓰이지 않습니다. 이 트레이스를
  쓰려면 저자에게 요청하거나, 대체물(예: LMSYS `lmsys-chat-1m`)을 자체 생성기로
  재생해야 합니다.
* 모델별 **SLO 기준선이 `trace.py` 에 하드코딩**되어 있고
  (`model_ttft_slo_baseline_p95` / `model_tpot_slo_baseline_p95`) 논문의 Llama 혼합에
  대해 측정된 값입니다. 모델을 바꾸면 전용 GPU 에서 다시 측정해야 하며(논문 §7.1),
  그러지 않으면 달성률 수치를 비교할 수 없습니다.

---

## 4. 모델

`exp/configs/*` 는 현재 토큰 없이도 돌아가도록 **Qwen2.5**(비게이팅)를 씁니다:

| Prism 설정 슬롯 | 논문 모델 | 사용 중인 대체 모델 |
| --- | --- | --- |
| large | `meta-llama/Llama-3.1-8B` | `Qwen/Qwen2.5-7B-Instruct` |
| mid | `meta-llama/Llama-3.2-3B` | `Qwen/Qwen2.5-3B-Instruct` |
| small | `meta-llama/Llama-3.2-1B` | `Qwen/Qwen2.5-1.5B-Instruct`, `Qwen/Qwen2.5-0.5B-Instruct` |

이미 내려받음: Qwen2.5-0.5B/1.5B/3B/7B-Instruct.

**논문의 실제 Llama/Mistral 모델을 쓰려면** (HF 게이팅 대상):

```bash
echo 'HF_TOKEN=hf_xxx' >> /workspace/.env       # exp/scripts/env.sh 가 읽습니다
source exp/scripts/env.sh
hf download meta-llama/Llama-3.2-1B
hf download meta-llama/Llama-3.2-3B
hf download meta-llama/Llama-3.1-8B
# 그 다음 prism-research/benchmark/multi-model/model_configs/*.json 을 그대로 사용
```

이 Llama/Mistral 모델들은 Prism 의 프로파일 파일 `model_info.json` 에 **이미 등록되어**
있으므로 별도 프로파일링 단계가 필요 없습니다.

### 그 밖의 모델 추가하기

Prism 의 GPU 스케줄러는
`prism-research/python/sglang/multi_model/utils/model_info.json` 에 없는 모델에 대해
기동을 거부합니다
(`ValueError: Model path ... not found in the profiled model info file`).
다음으로 등록하세요:

```bash
source exp/scripts/env.sh
python exp/scripts/profile_models.py <hf/model/path> [...]
```

(위의 Qwen2.5 모델들은 이미 추가되어 총 28개 항목입니다.)

---

## 5. 구축하면서 겪은 함정 (여기서는 전부 해결됨)

1. **`prism-research/install.md` 는 docker 를 전제합니다**
   (`lmsysorg/sglang:v0.3.4.post2-cu121`). 이 컨테이너는 docker-in-docker 를 돌릴 수
   없어서 네이티브로 설치했습니다 — 정확한 레시피는 `setup_prism_env.sh` 참조.
2. **transformers 는 4.45.2 로 고정해야 합니다.** 레포의 `python[all]` extra 에 상한이
   없어서 새로 해석하면 transformers 5.x 가 딸려 오고, vLLM 0.6.3.post1 이
   `ImportError: cannot import name 'DTensor'` 로 깨집니다.
3. **`pyairports` 2.1.1 이 PyPI 에서 제거되었습니다**(빈 0.0.1 자리표시자만 남음).
   그런데 vLLM 0.6.3.post1 이 `outlines<0.1` 을 핀 고정하고 그것이 이를 import 합니다.
   `git+https://github.com/NICTA/pyairports.git` 에서 설치하고, `pkg_resources` 를 아직
   쓰므로 `setuptools<81` 도 함께 설치했습니다.
4. **`profile_model_info.py` 가 `cache_config=None` 을 넘깁니다.** vLLM 에서 Qwen2 계열
   모델이 여기서 죽습니다 (`'NoneType' object has no attribute 'sliding_window'`).
   `exp/scripts/profile_models.py` 는 실제 `CacheConfig` 를 넘깁니다.
5. **`--workers-per-gpu` 는 그 GPU 의 `on: true` 모델 수 이상이어야 합니다.** 아니면
   모델이 영원히 `activating` 에 걸려 기동이 데드락에 빠집니다(최상위 로그에는 아무것도
   안 나오고 `*.log.gpu_scheduler.log` 에 대기 루프가 보입니다).
6. 설정 JSON 의 `max_memory_pool_size` 는 **모델당 KV 풀의 GiB** 입니다.
   `elastic`/`prism` 모드에서는 가상 상한이며, 초과 할당하는 것이 바로 요점입니다.

---

## 6. 단일 GPU 와 논문의 차이

이 인스턴스에는 **A100-80GB 1장**이 있습니다. 그대로 재현 가능한 것:

* §7.3 모델 간 유연한 메모리 공유 (2모델 static 대 elastic)
* §7.3 요청 조정 (GPU-로컬 스케줄러 on/off)
* §7.5 모델 활성화 지연 (`--enable-cpu-share-memory` 웜 스타트)
* §A.3 elastic 메모리 오버헤드 (정속 최악 사례 대 정적 분할)
* GPU 1장에 모델 8개 병치 (§7.2 의 축소판)

GPU 가 더 필요한 것: §7.2(모델 8개 / GPU 2장), §7.3 전역 배치(부하 분산을 보이려면
GPU 2장 이상 필요), §7.4(모델 58개 / GPU 32장), TP 실험.
스크립트는 이미 `--num-gpus` / `NUM_GPUS` 를 받으므로, 더 큰 장비로 옮길 때는 설정
JSON 의 `gpu_ids` 만 고치면 됩니다.

---

## 7. 논문의 기준선 중 설치하지 않은 것

| 기준선 | 상태 |
| --- | --- |
| S-Partition | ✅ = `static` 모드 |
| Prism | ✅ = `prism` 모드 |
| MuxServe / MuxServe++ | ❌ 미설치 (`github.com/hao-ai-lab/MuxServe`. MuxServe++ 는 저자들의 SGLang 포팅 + kvcached 이며 공개되지 않음) |
| QLM | ❌ 미설치 (`github.com/QLM-project/QLM`) |
| ServerlessLLM | ❌ 미설치 (`github.com/ServerlessLLM/ServerlessLLM`) |

각각 별도 venv 가 필요합니다 — 서로 충돌하는 torch/vllm 버전을 핀 고정합니다.

---

## 8. Sanity 체크: A100-80G 1장에서 Llama (완전한 Prism 모드)

`exp/scripts/run_sanity.sh <A|B|C>` 가 함께 제공되는 `real_trace.pkl` 로 세 가지 병치
사례를 돌리고(time_scale 1, replication 1, 약 600초), 이어서
`exp/scripts/analyze_slo.py` 가 SLO 통계를 다시 계산합니다.
`exp/scripts/summarize_sanity.py` 가 결과를 한 표로 출력합니다.

| 사례 | 설정 | 모델 |
| --- | --- | --- |
| A | `exp/configs/llama_1x8b.json` | `model_1` = Llama-3.1-8B |
| B | `exp/configs/llama_2x8b.json` | `model_1`, `model_4` = Llama-3.1-8B ×2 |
| C | `exp/configs/llama_8b_3b.json` | `model_1` = Llama-3.1-8B, `model_2` = Llama-3.2-3B |

슬롯 이름은 임의가 **아닙니다.** `trace.py::generate_e2e_benchmark_reqs` 가 특정 모델에
대해 측정된 슬롯별 SLO 기준선을 하드코딩하므로, `model_1/model_4/model_5` 가
Llama-3.1-8B 슬롯이고 `model_2` 가 Llama-3.2-3B 슬롯입니다. 사례 B 가
`model_1 + model_2` 가 아니라 `model_1 + model_4` 를 쓰는 이유는 두 모델 모두 8B 기준의
SLO 를 받게 하기 위함입니다.

### 이 스크립트가 우회하는 하네스 결함 두 가지

1. **TPOT SLO 단위 불일치.** `trace.py` 는 `model_tpot_slo_baseline_p95` 를
   **밀리초**로 저장하는데, `benchmark.py` 는 이를 **초** 단위인 `output.tpot` 과
   비교합니다 (`tpot = (finish_time - prefill_finish_time)/n`). 따라서
   `outputs[i].tpot < outputs[i].slo_tpot` 비교가 항상 참이 되고 **보고되는 TPOT
   달성률이 전부 1.0** 입니다. TTFT SLO 는 초 단위라 올바릅니다.
   `analyze_slo.py` 는 TPOT 기준선을 1000 으로 나눕니다.
2. **이 코드 경로에서는 SLO 통계가 아예 나오지 않습니다.** `--model-paths` 를
   `--real-trace` 와 함께 넘기면 클라이언트가 `run_tp_mode` 로 들어가는데, 여기서는
   요청 단위 원시 레코드만 덤프하고 `get_benchmark_metrics` 를 통째로 건너뜁니다 —
   그래서 공개된 e2e 결과 JSON 에 달성률 필드가 없습니다. `analyze_slo.py` 가 원시
   덤프에서 다시 계산합니다.

Prism 과 비교할 때 `benchmark.py` 의 `average_attainment_tpot` 을 쓰지 마세요.

해당 스윕의 전체 기록 — 환경 버전, 데이터셋 출처, 결과, 위 두 하네스 결함, 주의사항 —
은 **`exp/results/1-env-verification/REPORT.md`** 에 있습니다.
기계 판독용 표: `exp/results/1-env-verification/summary.tsv`.

---

## 9. Paper-Faithful Prism (논문 알고리즘 구현)

브랜치 `exp/paper-faithful-prism` 에서, 공개 프로토타입과 **스케줄러 정책만** 다른
논문 충실 구현을 만들어 비교했습니다.

| 문서 | 내용 |
| --- | --- |
| [`docs/paper_faithful/design_analysis.md`](docs/paper_faithful/design_analysis.md) | 공개 코드 감사, 논문 Algorithm 1·2 와의 차이, 논문이 확정하지 않은 항목과 우리 선택의 근거 |
| [`exp/results/paper-faithful-comparison/REPORT.md`](exp/results/paper-faithful-comparison/REPORT.md) | 측정 결과, 원인 분석, 논문 중 구현하지 못한 부분 |

```bash
source exp/scripts/env.sh
./exp/run_paper_faithful_comparison.sh --dry-run   # 실행 계획 출력
./exp/run_paper_faithful_comparison.sh --resume    # 스윕 실행 / 재개
python exp/tests/test_moore_hodgson.py             # Algorithm 2 단위 테스트
python exp/tests/test_kvpr_placement.py            # Algorithm 1 단위 테스트
```
