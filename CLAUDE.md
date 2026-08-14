# 에이전트 런북 — Prism (OSDI'26) 실험 장비

새로 빌린 GPU 서버에서 Prism 실험을 세팅하거나 돌리는 중일 것입니다. 이 파일이
가장 빠른 길입니다. 무엇이든 실행하기 전에 끝까지 읽으세요. 아래 함정 4~5개를 피할 수
있고, 각각을 직접 발견하려면 20분 이상이 듭니다.

`SETUP.md` 와 `README.md` 가 사람이 읽는 문서이며 더 깊이 다룹니다. 이 파일은 순서대로
따라갈 절차와, 한 번 당해 봐야 알게 되는 것들을 모은 것입니다.

---

## 0. 이 레포는 무엇인가

멀티 LLM 서빙 시스템 **Prism**(OSDI'26) 을 둘러싼 재현 가능한 하네스입니다. 레포 자체는
환경을 담지 않습니다. `bootstrap.sh` 가 고정 SHA 와 lockfile 로 재구축합니다. Prism 은
SGLang v0.3.4.post2 의 포크에 `kvcached` 를 더한 것이고, 둘 다 bootstrap 이 클론합니다.

논문에 대응하는 서버 모드 세 가지:

| 모드 | 플래그 | 논문 |
| --- | --- | --- |
| `static` | – | S-Partition 기준선 |
| `elastic` | `--enable-elastic-memory --use-kvcached-v0` | §5 벌루닝만 |
| `prism` | + controller / gpu-scheduler / model-service / worker-pool | §5 + §6 전체 |

---

## 1. 장비부터 확인 (60초, 건너뛰지 말 것)

```bash
nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv
nvidia-smi topo -m | head -5     # GPU 간 NVLink 여부
free -g; df -h /workspace; nproc
```

**필수 조건: compute capability 가 10.0 미만이어야 합니다.** 스택이 torch
2.4.0+cu121 로 고정되어 있고 여기에는 Blackwell 커널이 없습니다. B200 / RTX 50xx 에서는
설치는 깔끔히 되고 첫 GPU 연산에서 `no kernel image is available` 로 죽습니다.
A100(8.0)에서 검증했고 H100(9.0), L40S(8.9)도 동작해야 합니다. `bootstrap.sh` 가 경고와
함께 확인을 요구하는데, 그 프롬프트를 보면 `y` 를 누르지 말고 멈춘 뒤 사용자에게 다른
장비를 고르라고 알리세요.

그리고 **`/workspace` 는 대개 볼륨이 아닙니다**
(`vast-capabilities | jq '.instance.workspace_is_volume'`). recycle/destroy 가 전부
지웁니다. 장비가 사라지기 전에 결과를 git 으로 밀어 두세요.

---

## 2. 부트스트랩 (약 20분, 대부분 다운로드)

```bash
apt-get install -y redis-server          # 베이스 이미지에 없음
mkdir -p /etc/supervisor/conf.d          # 재부팅 후에도 redis 유지
cat > /etc/supervisor/conf.d/redis.conf <<'EOF'
[program:redis]
environment=PROC_NAME="%(program_name)s"
command=/usr/bin/redis-server --bind 127.0.0.1 --port 6379 --save ""
autostart=true
autorestart=true
stdout_logfile=/dev/stdout
redirect_stderr=true
stdout_logfile_maxbytes=0
EOF
supervisorctl reread && supervisorctl update && redis-cli ping   # -> PONG

git clone https://github.com/meojun/Prism-singleGPU-A100- /workspace/prism-exp
cd /workspace/prism-exp
echo 'HF_TOKEN=hf_xxx' >> /workspace/.env && chmod 600 /workspace/.env
./bootstrap.sh                            # 멱등적. 실패하면 그냥 다시 실행
```

경로가 `exp/scripts/env.sh` 안에 `/workspace/prism-exp` 로 하드코딩되어 있습니다.
그 경로에 클론하거나 `PRISM_ROOT` 를 고치세요.

HF 토큰은 **meta-llama** 라이선스에 동의한 계정의 것이어야 합니다 — Llama 는 게이팅
대상이라 그렇지 않으면 모든 다운로드가 401 을 냅니다. 사용자에게 요청하고, 절대
지어내지 마세요. bootstrap 은 백그라운드로 돌리고 폴링하세요. 약 20분이 걸려서 포그라운드
도구 타임아웃을 넘깁니다.

검증 블록이 torch 2.4.0, sglang 0.3.4.post2, vllm 0.6.3.post1, transformers 4.45.2,
flashinfer 0.1.6, kvcached+vmm_ops, cuda available 에 대해 `OK` 를 찍으면 완료입니다.
하나라도 `BAD` 면 멈추고 고치세요.

---

## 3. 이 장비가 커밋된 기준선을 재현하는지 확인

새 수치를 믿기 전에 항상 이것부터 하세요. 사례당 약 12분입니다.

```bash
source exp/scripts/env.sh
CUDA_VISIBLE_DEVICES=0 TAG=verify ./exp/scripts/run_sanity.sh A    # 이어서 B, C
```

`TAG` 는 필수입니다 — 없으면 커밋된 `results/1-env-verification/` 을 덮어씁니다.
멀티 GPU 장비에서는 `CUDA_VISIBLE_DEVICES=0` 도 필수입니다(아래 함정 5).

`exp/results/1-env-verification/` 및 `exp/results/2-colocation/REPORT.md` 와
비교하세요. 반드시 성립해야 하는 것: **사례 A 달성률 ≈ 1.0**, **사례 B 의 TPOT 붕괴**
(model_1 ≈ 0.35, model_4 ≈ 0.08), **사례 C 높은 달성률**. 절대 지연값은 GPU 에 따라
달라지지만 이 패턴들은 변하면 안 됩니다. `exp/results/3-placement/REPORT.md` §3 에
정상 런의 기준값 대 재실행 비교표가 있습니다 — 디코드 지표가 1 % 이내로 들어왔습니다.

고립된 위반 하나(296개 요청 중 1~2개)는 잡음이지 장비 고장이 아닙니다. *패턴*이 바뀌면
장비 고장입니다.

---

## 4. 실험 실행 — GPU 개수는 파라미터다

**배치 설정을 손으로 쓰지 마세요.** `trace.py::generate_e2e_benchmark_reqs` 가 특정
모델에 대해 측정된 슬롯별 SLO 기준선을 하드코딩하므로, 슬롯에 엉뚱한 모델을 넣으면
조용히 잘못된 기준선과 비교됩니다. 대신 생성하세요:

```bash
python exp/scripts/make_config.py --num-gpus <N> -o exp/configs/mine.json
```

슬롯 8개는 고정이며 논문 §7.2/§7.3 의 혼합입니다:

| 슬롯 | 모델 | 가중치 | `real_trace.pkl` 요청 수 |
| --- | --- | ---: | ---: |
| model_1 | Llama-3.1-8B | 15.08 GiB | **296** |
| model_2 | Llama-3.2-3B | 6.00 | 22 |
| model_3 | Llama-3.2-1B | 2.28 | 22 |
| model_4 | Llama-3.1-8B | 15.08 | **262** |
| model_5 | Llama-3.1-8B | 15.08 | **120** |
| model_6 / 7 / 8 | Llama-3.2-1B | 각 2.28 | 19 / 11 / 2 |

부하가 극도로 치우쳐 있습니다 — 상위 세 슬롯이 요청의 90 % 입니다. 배치 모드:
`blocks`(기본값. N=2 에서 80/20 인 소박한 분할. 배치 *자체*가 변수일 때 쓰세요),
`roundrobin`(60/40), `balanced`(49/51. 배치를 변수에서 빼고 싶을 때).

그다음 실행합니다. **`run_multigpu.sh` 는 `NGPU` 를 보이는 GPU 전부로 기본 설정하고
그에 맞는 config 를 생성하므로, 같은 명령이 1 GPU · 2 GPU · N GPU 장비에 그대로
적응합니다:**

```bash
./exp/scripts/run_multigpu.sh glob_on  1      # 완전한 Prism,  time_scale 1
./exp/scripts/run_multigpu.sh glob_off 1      # 전역 컨트롤러 없음
python exp/scripts/compare_fig7.py --tag fig7 --ts 1
```

`time_scale` 은 도착 시각에 곱해집니다. 따라서 **작을수록 부하가 크고 런이 짧습니다**:
1.0 → 1배 부하 / 600초, 0.5 → 2배 / 300초, 0.25 → 4배 / 150초.

조절 항목(전부 환경변수): `NGPU`, `NMODELS`(≤8), `PLACEMENT`, `CFG`, `WORKERS`,
`MAXMEM`, `TAG`, `TRACE`, `TTFT_SCALE`, `TPOT_SCALE`. `WORKERS` 는 config 에서 자동
유도되며, 마이그레이션 여유를 넓히려는 경우에만 덮어쓰세요.

### GPU 개수별로 무엇이 가능한가

| | 1 GPU | 2 GPU | 4 GPU 이상 |
| --- | --- | --- | --- |
| §7.3 메모리 공유(static 대 elastic), §A.3 오버헤드, 병치 | ✅ | ✅ | ✅ |
| §7.3 Fig 7 전역 배치, 모델 마이그레이션 경로 | ❌ 불가능 | ✅ | ✅ |
| §7.2 Fig 5 (GPU 2장에 모델 8개) — 논문의 주 e2e | ❌ | ✅ | ✅ |
| §5.3 병렬 가중치 로딩 (Fig 10) | ❌ **조용한 no-op** | ✅ 부분적 | ✅ |
| TP | ❌ | TP=2 | 논문처럼 TP=4/8 |
| §7.4 (모델 58개 / GPU 32장) | ❌ | ❌ | ❌ |

§5.3 이 GPU 1장에서 no-op 인 이유는 `model_sevice.py` 가
`broker_gpu_id = (broker_id + target_gpu_id + 1) % num_gpus` 를 계산하기 때문입니다 —
GPU 가 하나면 브로커가 항상 대상이라 병렬성이 사라집니다.

---

## 5. 함정 — 전부 실제로 시간을 잡아먹은 것들

1. **tmux 가 낡은 `CUDA_VISIBLE_DEVICES` 를 물려줍니다.** tmux 는 *사용자당 서버
   하나*를 유지하고, 모든 `new-session` 은 그 서버가 처음 시작될 때의 환경을 상속합니다.
   `CUDA_VISIBLE_DEVICES=0` 로 1-GPU sanity 스윕을 돌린 뒤 같은 tmux 서버에서 2-GPU
   서버를 띄우면, GPU>0 워커가 전부
   `RuntimeError: CUDA error: invalid device ordinal` 로 죽습니다.
   `run_multigpu.sh` 는 이를 명시적으로 고정하고 실행 전에
   `torch.cuda.device_count()` 를 단언합니다. 새 런처를 쓴다면 똑같이 하세요.
2. **tmux 는 세션 이름의 `.` 을 `_` 로 바꿉니다.** `...ts0.5` 라는 세션은 `ts0_5` 가
   되므로 `tmux has-session -t ...ts0.5` 는 절대 매치되지 않습니다. 그래서 준비 대기
   루프가 멀쩡한 서버를 DIED 로 선언하고 **고아로 만들어**, GPU 를 조용히 점유한 채
   남깁니다. 이름을 먼저 정규화하세요.
3. **`num_gpus` 가 결과 파일 이름에 박힙니다.** `benchmark.py` 는
   `{exp}_e2e_{num_gpus}gpu_...` 로 쓰는데 `run_sanity.sh` 는 리터럴 `1gpu` 로
   glob 합니다. 그 glob 을 GPU 2장에서 재사용하면 분석 단계가 조용히 실패합니다.
4. **모든 GPU 에 `on: true` 모델이 최소 하나 있어야 합니다.**
   `launch_multi_model_server` 는 초기 배치에 존재하는 gpu_id 에 대해서만 GPU 스케줄러를
   시작하므로, 비어서 시작한 GPU 는 런 내내 죽어 있습니다.
5. **`launch_model_service()` 는 `--num-gpus` 가 아니라
   `torch.cuda.device_count()` 를 읽습니다** (`multi_model_server.py:576`). 2-GPU
   장비에서 `CUDA_VISIBLE_DEVICES=0` 없이 1-GPU 실험을 돌리면 브로커는 GPU 2개분,
   엔진은 1개분이 생깁니다. 같은 함수가 `num_model_service_workers = 1` 도 하드코딩해
   플래그를 무시합니다.
6. **`--workers-per-gpu` 는 그 GPU 의 `on: true` 모델 수 이상이어야 합니다.** 아니면
   모델이 `activating` 에 갇혀 기동이 데드락에 빠집니다. 최상위 로그에는 아무것도
   안 나옵니다.
7. **함께 제공되는 트레이스에서 `--disable-radix-cache` 를 절대 빼지 마세요.**
   `real_trace.pkl` 의 프롬프트는 `"Hello "*n` 이라 짧은 프롬프트가 긴 프롬프트의 정확한
   접두사이고, radix 캐시가 prefill 의 **99.3 %** 를 캐시에서 제공합니다. 결과가
   무의미해집니다. 실제 텍스트가 필요하면 `build_sharegpt_trace.py` 의 `content`
   변형을 쓰세요.
8. **`benchmark.py` 의 `average_attainment_tpot` 은 항상 1.0 입니다** — `trace.py` 가
   TPOT 기준선을 ms 로 저장하는데 비교는 초 단위와 이루어집니다. 단위를 바로잡는
   `analyze_slo.py` 를 쓰세요. 원본 필드를 인용하지 마세요.

### `prism` 모드 서버가 죽었을 때

최상위 로그는 에러를 삼킵니다. 다음 순서로 읽으세요:

```
exp/server-logs/<exp>/server.log                        # 워커 트레이스백
exp/server-logs/<exp>/server.log.gpu_scheduler.log
exp/server-logs/<exp>/server.log.global_controller.log  # --enable-controller 일 때만
exp/server-logs/<exp>/server.log.model_service.log
exp/server-logs/<exp>/stdout.log                        # 가장 쓸모없음
```

실패한 런 뒤의 정리 — **세션을 개별적으로 죽이세요.** `tmux kill-server` 는 절대
금물입니다(사용자 본인의 셸까지 죽입니다):

```bash
tmux kill-session -t <session>; pkill -f launch_multi_model_server
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader   # 0 MiB 여야 함
```

---

## 6. 논문 대 공개 코드 — 돌린 것 이상을 주장하지 말 것

이 레포는 상류가 공개한 것을 충실히 배선하지만, **상류 코드는 논문과 다릅니다.**
고정 SHA 의 `prism-research` 를 직접 읽어 확인했고 런타임에서도 확인했습니다:

- **Algorithm 1 (KVPR) 은 부분적으로만 구현되어 있습니다.** 두 공개 레포의 모든 커밋을
  훑어본 결과 KVPR / `w_token_rate` / `shared_kv` 라는 이름이 어디에도 없습니다.
  구성요소별로 보면 *분모*는 있습니다
  (`memory_available_for_requests = gpu_mem - weights`, `simple_global.py:93`).
  그러나 *분자*는 `token_rate*token_size/SLO` 가 아니라 평활된 요청 개수이고, 그 비율로
  모델을 정렬하는 코드도 없으며, τ 임계는 존재하되 다른 지표에 적용됩니다. 목표 — GPU
  별 메모리 압력의 균형 — 은 구현되어 있지만 구체적 지표는 아닙니다. 따라서 여기서의
  ablation 은 전역 배치가 도움이 된다는 것을 보일 뿐 Algorithm 1 을 검증하지 못합니다.
- **Algorithm 2 (Moore-Hodgson): 재료는 있고 메커니즘이 없습니다.** 마감 `a + s` 와
  실행 추정 `p/c` 는 둘 다 계산되고(`request_queue.py:27,30`) 요청도 마감 순서로
  pop 되지만, 최적성 증명이 기대는 실행가능성 검사와 가장 긴 작업 제거 단계가
  없습니다 — 결국 평범한 EDF 로 환원됩니다. 그것을 적용할 admission control 자체가
  어차피 비활성화되어 있습니다(`net_available = inf`).
- **시기 문제가 아닙니다.** 코드가 논문보다 앞선다는 뻔한 변명은 성립하지 않습니다.
  arXiv 2505.04021 **v1 (2025-05-06) 에 이미 Algorithm 1(KVPR)과
  Algorithm 2(Moore-Hodgson)가 둘 다 들어 있고**, 이는 2025-08-09 프로토타입 공개보다
  석 달 앞섭니다. OSDI 개정판(v3, 2026-06-10)이 바꾼 것은 KVPR 의 *분자*입니다 —
  v1 의 `req_rate/SLO` 가 `token_rate*token_size/SLO` 가 되었습니다. 공개 코드는 둘 중
  어느 쪽과도 맞지 않습니다 — SLO 가중 없는 단순 요청 개수를 씁니다. prism-research 를
  논문의 아티팩트가 아니라 단순화된 프로토타입으로 다루세요. OSDI 논문은 kvcached 만
  링크하고, prism-research 의 README 는 아직 v1 제목("Unleashing GPU Sharing")과
  13인 저자 목록을 인용하는 반면 kvcached 의 README 는 21인 OSDI 판을 인용합니다.
  Algorithm 1 을 구현한다면 OSDI 의 토큰 기반 분자를 쓰세요.
- **§6.2 admission control 이 꺼져 있습니다.** `request_queue.py:137` 이
  `net_available = float("inf")` 로 두고, GPU 스케줄러가 런타임에 매초
  `net_available: inf` 를 찍습니다.
- **§6.1 오버랩 마이그레이션이 구현되어 있지 않습니다.** 코드는 원본을 비활성화한
  *다음* 대상을 활성화합니다(동작이 비활성화 우선으로 정렬됨). NVLink 가중치/KV 전송도
  없고, 준비될 때까지 계속 서비스하는 동작도 없습니다. 실제로는 마이그레이션 자체가
  거의 발동하지 않습니다. `MEMORY_PER_REQUEST_RATIO_THRESHOLD = 15`
  (`simple_global.py:183`, 플래그가 아니라 하드코딩)가 GPU 간 15배 불균형을 요구하는데
  현실적인 값은 약 1.6배입니다.
- **기준선 MuxServe++/QLM/ServerlessLLM 은 설치되어 있지 않습니다**(torch/vllm 핀이
  충돌해 각각 자기 venv 가 필요합니다). S-Partition 과 Prism 만 있습니다.
- **프로덕션 트레이스(Hyperbolic/Novita/Arena)는 공개되어 있지 않습니다.**
  `--csv-trace` 는 파싱만 되고 쓰이지 않습니다.

`MODEL_IDLE_THRESHOLD = 50 s` 는 논문의 최적값 약 45초(§A.4)와 부합합니다.

정리하면, ablation 은 전역 컨트롤러가 *도움이 된다*는 것을 보일 수 있지만 Algorithm 1 을
검증할 수는 없습니다. 둘 중 무엇을 측정했는지 분명히 밝히세요.

> 논문 알고리즘을 직접 구현한 브랜치가 있습니다: `exp/paper-faithful-prism`.
> 설계 분석은 `docs/paper_faithful/design_analysis.md`, 측정 결과와 원인 분석은
> `exp/results/paper-faithful-comparison/REPORT.md` 를 보세요.

---

## 7. 결과가 쌓이는 곳

| 경로 | 내용 |
| --- | --- |
| `exp/results/1-env-verification/` | 커밋된 1-GPU 기준 스윕(합성 프롬프트) |
| `exp/results/2-colocation/` | 커밋된 ShareGPT + slowdown-SLO 병치 연구 |
| `exp/results/3-placement/` | 2-GPU §7.3 전역 배치 ablation + `REPORT.md` |
| `exp/results/4-rate-sweep/` | 2-GPU 유입률 스윕 + `REPORT_rate_sweep.md` |
| `exp/results/paper-faithful-comparison/` | 논문 충실 구현 대 공개 프로토타입 비교 |

각 실험은 `<exp>_slo.json`(`analyze_slo.py` 경유), `<exp>_actions.txt`(컨트롤러 활동),
`requests/`(요청 단위 원시 덤프), `server-logs/<exp>/gpu_timeline.txt`(nvidia-smi 샘플)
을 남깁니다.

새 연구의 관례: 결과 옆에 `REPORT.md` 를 두고 환경, 방법, 수치, 그리고 *결론 내리지
못한 것*을 적으세요. `exp/results/3-placement/REPORT.md` 가 템플릿입니다. 꼬리 지표
(p99)를 인용하려면 각 arm 을 최소 두 번 돌리세요 — 단일 런은 수백 개 요청에 대한
집계에는 괜찮지만 꼬리에는 부적합합니다.

---

## 8. v2 함정 — 전부 실제로 실행을 멈춰 세운 것들

`exp/paper-faithful-v2` 브랜치에서 하루치 실행 시간을 잡아먹은 것들입니다.
**장시간 스윕을 시작하기 전에 이 절을 끝까지 읽으세요.** 각각은 증상이 원인과
멀리 떨어져 있어서, 직접 겪으면 한 시간씩 날아갑니다.

1. **`kill 0` 은 프로세스 그룹 전체를 죽입니다.** `run_pf_case.sh` 의 정리 트랩에
   `kill "${SAMPLER:-0}"` 이 있습니다. 서버가 준비되기 전에 스크립트가 빠져나가면
   `SAMPLER` 가 아직 없어서 `kill 0` 이 되고, POSIX 에서 이것은 **자기 프로세스
   그룹 전체에 SIGTERM** 입니다 — 스윕 드라이버도, watchdog 도, 부모 셸도 같이
   죽습니다. 세 번의 멀티시간 런이 이것 때문에 사라졌고, supervisord 로그의
   `terminated by SIGTERM; not expected` 를 보고서야 잡혔습니다. 시작하지도 않은
   PID 에 절대 시그널을 보내지 마세요. `run_v2_case.sh` 는 고쳐져 있습니다.

2. **장시간 작업은 supervisor 아래에 두세요.** tmux 세션도, `setsid` 로 PPID 1 ·
   독립 세션 · tty 없음까지 만든 프로세스도 이 장비에서 죽었습니다(위 1번이
   주원인이지만 그것만은 아닙니다). supervisor 아래 redis 는 10시간 무중단입니다.
   `/etc/supervisor/conf.d/prism_pipeline.conf` 가 예시이고, `autorestart=unexpected`
   덕분에 무엇이 죽이든 수 초 안에 되살아납니다. 모든 단계가 `--resume` 안전이면
   재시작이 무해해집니다.

3. **포트를 고정하지 마세요.** `launch_multi_model_server` 는 `--port` 에서 모델
   엔진별 포트를 파생시키고(port+1, port+2, ...), `is_port_available` 은
   `127.0.0.1` 이 아니라 **`0.0.0.0`** 에 바인드해서 확인합니다. 게다가 30000–40000
   대는 에디터/툴링 서버가 동적 포트를 잡는 구간입니다. 고정 베이스는 언젠가 반드시
   충돌하고, 충돌은 **모델을 전부 로드한 뒤에** 드러납니다. `find_free_port.py` 로
   연속 블록을 런타임에 확보하세요.

4. **패치 적용기의 "이미 적용됨" 검사는 고유 문자열이어야 합니다.**
   `apply_patches.py` 가 addition 의 첫 줄로 중복을 판단했는데, argparse choices
   편집의 첫 줄이 `choices=[` 였습니다. 그 문자열은 파일에 수십 번 나오므로 편집이
   **조용히 건너뛰어졌고**, 나머지는 전부 정상이라 몇 시간 뒤 서버가
   `--policy: invalid choice: 'kvpr-global'` 로 죽고서야 드러났습니다. 지금은 13개
   랜딩 포인트를 명시적으로 검증합니다.

5. **요청 덤프에는 도착 시각도 프롬프트 길이도 없습니다.** `--model-paths` +
   `--real-trace` 경로의 덤프는 `{success, latency, ttft, tpot, output_len, model,
   error}` 뿐이고, JSONL 이 아니라 pretty-print 된 **단일 배열**입니다. 트레이스
   순서 그대로 기록되므로 도착 시각 정렬한 트레이스와 **인덱스로 조인**해야 합니다
   (`collect_v2_metrics.py::join_trace`). 조인이 어긋나면 조용히 엉뚱한 값이 붙으므로
   모델 시퀀스 불일치를 반드시 세세요.

6. **실행 중인 bash 스크립트를 편집하지 마세요.** bash 는 스크립트를 바이트 오프셋
   기준으로 이어 읽으므로, 실행 중 편집하면 다음 명령이 엉뚱하게 잘립니다.
   calibration 이 도는 중에 러너를 고쳤더니 이후 런들이 옛 인자로 실행됐습니다.
   멈추고 → 고치고 → 재시작하세요.

7. **ShareGPT 프롬프트가 담긴 pickle 은 GitHub 이 거부합니다.** 데이터셋 대화 중
   일부에 제3자의 실제 자격증명(우리 경우 Slack incoming webhook)이 들어 있어서
   push protection 에 걸립니다. 워크로드 pickle 은 gitignore 하고, seed 로
   재생성하세요. `paired_requests_*.json` 과 `phases_*.json` 이 프롬프트 본문 없이
   페어링 검증에 필요한 것을 전부 담습니다.

8. **프로파일이 하나라도 빠지면 config 를 쓰지 마세요.** 슬롯이 빠지면 Algorithm 2 가
   조용히 프로토타입의 2048 tok/s 상수로 되돌아가고 SLO 조회는 스윕 중간에
   KeyError 를 냅니다. `run_profiling_v2.sh` 는 folding 전에 여섯 슬롯을 확인합니다.

9. **이 6모델 구성은 5~10 req/s 에서 포화합니다.** v1 의 3×Llama-8B 무릎(≈26 req/s)을
   근거로 유입률을 잡으면 전 구간이 붕괴 영역이 됩니다. 그리고 병목은 TTFT 가 아니라
   **TPOT** 입니다(5 req/s 에서 TTFT 달성률 0.995 대 TPOT 0.690). 유입률은 반드시
   `run_calibration_v2.sh` 로 먼저 재세요.

### 새 장비 세팅 — 0 에서 실험까지

이번에 이 절이 없어서 세팅에만 몇 시간이 들었습니다. 순서대로 하면 됩니다.

```bash
# 1. 레포 (경로 고정 — exp/scripts/env.sh 가 하드코딩)
git clone https://github.com/meojun/Prism-singleGPU-A100- /workspace/prism-exp
cd /workspace/prism-exp

# 2. HF 토큰. meta-llama 라이선스에 동의한 계정의 것이어야 합니다.
#    없으면 Llama 다운로드가 전부 401 이고, 사용자에게 받으세요. 지어내지 마세요.
echo 'HF_TOKEN=hf_xxx' > /workspace/.env && chmod 600 /workspace/.env

# 3. 나머지 전부 (약 35분, 무인)
./setup/quickstart.sh
```

`quickstart.sh` 가 하는 일: redis 를 supervisor 아래 올리고 → 모델 가중치
(6개, 약 47 GB)와 ShareGPT(약 670 MB)를 백그라운드로 받고 → `bootstrap.sh` 로
고정 SHA 스택을 세우고 → Algorithm 1/2 패치를 적용해 13개 랜딩 포인트를 검증하고
→ 단위 테스트를 돌리고 → 파이프라인을 supervisor 에 등록(중지 상태로)합니다.

그다음 두 단계만 사람이 시작합니다.

```bash
source exp/scripts/env.sh
./exp/scripts/run_profiling_v2.sh      # c_i / SLO 기준선, 약 40분
supervisorctl start prism_pipeline     # 나머지 전부 무인
```

**`run_profiling_v2.sh` 결과를 다른 장비에서 재사용하지 마세요.** c_i 와 SLO
기준선은 이 GPU 에서 측정한 값이고, 그것이 Algorithm 2 의 실행가능성 판정과
Algorithm 1 의 KVPR 가중을 직접 결정합니다. 커밋된
`exp/results/paper-faithful-v2/sanity/profiling/` 은 A100-SXM4-80GB 기준입니다.

세팅에서 실제로 걸렸던 것들:
* `bootstrap.sh` 3단계의 torch 인덱스 문제는 이미 고쳐져 있습니다(README §7).
* ShareGPT 는 번들되어 있지 않고 `env.sh` 가 경로를 하드코딩합니다. 안 받으면
  워크로드 생성이 FileNotFoundError 로 죽습니다 — bootstrap 을 다 기다린 뒤에.
* Algorithm 1/2 구현은 `patches/paper_faithful/` 에 있고 bootstrap 이
  gitignore 된 `prism-research/` 클론 위에 복사합니다. v1 때는 이것이 커밋되어
  있지 않아 장비와 함께 사라졌습니다.

### v2 를 이어받는 가장 빠른 길

```bash
./setup/quickstart.sh                       # redis→bootstrap→패치→모델→단위테스트
supervisorctl status prism_pipeline         # 이미 돌고 있는지
tail -f /workspace/logs/pipeline.log        # 어디까지 왔는지
grep -E "^=====" /workspace/logs/pipeline.log   # 단계 전이만
cat /workspace/logs/chosen_rates.txt        # 확정된 부하 사다리
```

파이프라인은 `exp/run_pipeline_v2.sh` 이고
calibration → 부하 구간 선택 → sanity 게이트 → 본 스윕 → ablation → REPORT 순입니다.
**모든 단계가 `--resume` 안전**하므로 언제 죽어도 이어서 돌리면 됩니다. 단계마다
REPORT.md 를 다시 만들고 push 하므로, 원격에는 항상 그 시점까지의 보고서가 있습니다.
설계 근거와 논문 대조는 `exp/results/paper-faithful-v2/fragments/` 에 있습니다.
