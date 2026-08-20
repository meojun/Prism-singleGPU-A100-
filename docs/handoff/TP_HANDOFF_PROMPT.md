# 4×A100 서버 인수인계 프롬프트 — TP 지원 + anti-affinity

아래 전체를 새 서버의 Claude Code 세션에 그대로 붙여넣으면 된다.

---

## 붙여넣을 프롬프트

```
https://github.com/meojun/Prism-singleGPU-A100- 이 레포로 작업한다.

이건 Prism 논문(OSDI'26)의 released prototype 을 paper-faithful 하게 재구현하는
프로젝트고, 너의 임무는 그중 **아직 아무도 손대지 못한 TP(Tensor Parallelism)
지원과 TP anti-affinity 제약** 하나다. 다른 항목은 건드리지 마라.

## 0. 먼저 읽어라 (구현 전에 반드시)

브랜치 exp/paper-faithful-v6 를 기준으로 한다.

  git clone https://github.com/meojun/Prism-singleGPU-A100- /workspace/prism-exp
  cd /workspace/prism-exp && git checkout exp/paper-faithful-v6
  git checkout -b exp/paper-faithful-tp

읽을 순서:
  1. HANDOVER.md                                            ← 현재 ground truth
  2. exp/results/paper-faithful-v4/tp-validation/FINDING.md  ← TP=2 가 왜 FAIL 했는지
  3. exp/results/paper-faithful-v4/IMPLEMENTATION_AUDIT.md   ← 무엇이 FULL/PARTIAL/미구현인지
  4. docs/paper_faithful/design_analysis.md  §5b             ← anti-affinity 절, V6 에서 정정됨
  5. CLAUDE.md                                               ← 장비 세팅 런북

**HANDOVER.md 를 ground truth 로 쓰되, 코드/raw data 와 모순되면 코드와 raw data 를
우선해서 보고해라.** 실제로 V6 에서 그런 모순이 세 건 나왔다.

## 1. 이미 확립된 것 — 다시 파지 마라

* released prototype 에서 **TP>1 과 Prism 스케줄링은 출고 상태로는 상호 배타적**이다.
  TP=2 검증은 FAIL 이고, 원인은 하드웨어가 아니라 코드 구조다. 다만 **어느 층이
  문제인지 정확히 알아라** — 엔진은 TP 를 이미 한다(§2(1)). 막는 것은 그 위의
  worker-pool 계층이다. "TP 는 불가능하다" 로 읽고 포기하지 마라.
* `--enable-worker-pool` 은 선택이 아니다. GPU 스케줄러와 마이그레이션 전부가 그
  경로에 산다. 따라서 "worker-pool 을 끄고 TP 를 쓰면 되지 않나" 는 답이 아니다.
* anti-affinity 제약은 **구현되어 있지 않다.** design_analysis.md 가 한때
  "구현되어 있으나 TP=1 이라 발동하지 않는다" 고 적고 있었지만 그건 오기였고
  V6 에서 정정했다. kvpr_global*.py 어디에도 그런 분기가 없고 patches/ 전체에
  tp_size / tp_rank 참조가 0 건이다.
* `tp_size > 1` 이면 upstream 이 model-service 경로를 끄므로
  (`model_runner.py:133-136`) **TP 모델은 V4 의 병렬 가중치 로딩도, P2P
  마이그레이션도 통째로 못 쓴다.** 이건 사소한 한계가 아니다 — 이 프로젝트가
  지금까지 만든 fast activation / fast migration 이 TP arm 에서는 전부 무효라는
  뜻이고, TP arm 의 성능 수치를 non-TP arm 과 나란히 놓고 읽으면 안 된다는 뜻이다.
  고치려면 model service 가 per-rank 샤드를 다뤄야 한다. **고치든 한계로 남기든,
  리포트에 이 사실을 명시하고 수치 해석에 반영해라.**

## 2. 막혀 있는 지점 세 개 (전부 file:line 확인됨)

pinned SHA 595ec1f 의 prism-research 기준이다.

(1) **worker-pool 이 엔진을 단일 GPU 에 묶는다** — `multi_model_server.py:817`
    `launch_worker_pool_engines` 가 (GPU, worker slot) 당 엔진 하나를 만들면서
    GPU 목록으로 `[gpu_id]` (길이 1) 를 넘긴다.

    **여기서 오해하지 마라 — 엔진 자체는 TP 를 이미 한다.**
    `launch_engine` (`multi_model_server.py:313-344`) 는 SGLang 상류의 TP 런처
    그대로이고 tp_rank 마다 scheduler 프로세스를 띄운다:
        tp_rank_range = range(tp_size_per_node * node_rank, ...)
        assert len(tp_rank_range) == len(gpu_ids)      # <- TP=2 가 깨지는 지점
        for tp_rank in tp_rank_range:
            gpu_id = gpu_ids[tp_rank % tp_size_per_node]
            proc = mp.Process(target=run_scheduler_process, args=(..., gpu_id, tp_rank, ...))
    `tp_size=2` 인데 `gpu_ids` 가 길이 1 이면 `assert 2 == 1` 에서 죽는다.
    즉 이건 엔진 재작성이 아니라 **배선 문제**다. 올바른 GPU 리스트와 모델별
    tp_size 를 넘기면 된다.

(1b) **worker-pool 의 슬롯 회계가 "1 인스턴스 = 1 GPU" 를 전제한다** — 여기가
    진짜 구조 작업이다. `scheduling/gpu/worker_pool.py:101-134` 의
    `handle_activate_model` 은 **한 GPU 안에서** 유휴 워커 하나를 고른다
    (`get_idle_worker` / `assign_worker`). TP 그룹은 k 개 GPU 의 슬롯을 **동시에,
    묶어서** 잡았다가 함께 놓아야 하는데 그 개념이 없다. 마이그레이션과 자원
    회계(`resource_manager.py`)도 같은 전제 위에 있다.
    **작업량의 대부분이 여기다.** (1) 은 배선이고 (1b) 가 구조다.

(2) **컨트롤러가 TP 그룹을 rank0 로 축약한다** — `controller_global.py:97-99`
        # NOTE(ke): For TP case, only consider rank0 state
        gpu_ids = set([mod.gpu_ids[0] for mod in models])
    배치 코드가 TP 그룹을 멀티-GPU 객체로 볼 수 없다. anti-affinity 를 "표현"
    하려면 여기가 먼저 열려야 한다.

(3) **가중치 조회 키가 어긋난다** — `model_runner.py:274-282`
        model_key = (model_path, self.tp_size)
        return self.shared_cpu_models[model_key][self.tp_rank]
    TP=2 가 죽는 실제 지점이다 (`Model model_5 not found in shared cpu models`).

    **여기에 검증되지 않은 가설이 하나 있으니 네가 확인해라.**
    `load_shared_cpu_models` (`multi_model_server.py:541-563`) 는 이미
    `(model_path, tp_size)` 로 키를 잡고 tp_size 별 샤드를 로드한다. 즉 **CPU 쪽
    샤딩 인프라는 이미 존재한다.** 그렇다면 (3) 은 독립적인 결함이 아니라 (1) 의
    결과일 가능성이 높다 — 엔진의 server_args 가 모델별 tp_size 가 아니라 서버
    수준 `--tensor-parallel-size` 를 들고 가서 키가 안 맞는 것. 이 가설이 맞으면
    (1) 을 고치는 것만으로 (3) 이 따라 풀린다. **가장 먼저 이걸 확인해라.**
    맞든 틀리든 결과를 기록해라.

## 3. 작업 순서 — 이 순서를 지켜라

각 단계가 끝날 때마다 커밋하고, 다음으로 넘어가기 전에 그 단계가 실제로 동작하는지
런타임 로그로 확인해라. 뒤 단계는 앞 단계 없이는 의미가 없다.

**1단계. TP>1 이 worker-pool 경로에서 일단 돌게 만든다.** (배선. 작은 편)
  - §2(3) 의 가설부터 확인
  - `launch_worker_pool_engines` 가 `[gpu_id]` 가 아니라 그 모델의 TP 그룹 GPU
    리스트를 넘기도록 (`multi_model_server.py:817`)
  - 엔진 server_args 의 tp_size 가 서버 수준이 아니라 모델 config 의 tp_size 를
    따르도록
  - 1b 단계(슬롯 회계)와 얽힌다 — TP 그룹에 k 개 슬롯을 묶어서 배정하는 것이
    선행되어야 GPU 리스트를 만들 수 있다. 순서를 어떻게 풀지는 네가 판단해라
  - 성공 기준: TP=2 모델이 활성화되고 추론이 성공하며 rank 가 서로 다른 GPU 에
    관측된다 (`exp/scripts/collect_tp2_evidence.py` 가 이미 이 체크를 한다 —
    v4 의 `tp2_validation.json` 이 무엇을 검사하는지 보고 그 형식을 재사용해라)

**1b단계. worker-pool 이 TP 그룹을 k 개 슬롯의 묶음으로 다루게 한다.** (구조. 큰 편)
  - `worker_pool.py` 의 워커 배정/해제가 그룹 단위로 원자적이어야 한다
  - 부분 배정 실패 시 롤백이 있어야 한다 (k-1 개만 잡고 멈추면 데드락)
  - 성공 기준: TP=2 모델의 두 rank 가 서로 다른 GPU 의 슬롯을 동시에 점유하고,
    비활성화 시 둘 다 반환된다

**2단계. 컨트롤러가 TP 그룹을 멀티-GPU 로 유지한다.**
  - `controller_global.py:97-99` 의 rank0 축약 제거
  - 배치/마이그레이션 코드가 TP 그룹 전체를 하나의 배치 단위로 다루게
  - 성공 기준: alg1 로그에 TP 모델의 gpu_ids 가 단일값이 아니라 집합으로 찍힌다

**3단계. Algorithm 1 에 anti-affinity 필터를 넣는다.**
  - 논문 제약: tp_size=k 인 모델의 k 개 샤드는 서로 다른 k 개 GPU 에 놓인다
  - 이건 새 메커니즘이 아니라 argmin 후보 필터다. 1·2단계가 끝나면 10 줄 남짓
  - **프로젝트 관례를 지켜라**: 새 코드 경로는 전부 플래그 뒤에 두고 기본값은
    released prototype 을 그대로 재현한다 (design_analysis.md §4). 이 항목도
    `--enable-tp-anti-affinity` 같은 opt-in 으로 만들어라
  - `exp/tests/test_kvpr_placement.py` 형식으로 단위테스트를 추가해라

**4단계. 제약이 실제로 구속력을 갖는지 측정한다.**
  - **GPU 2 장으로는 검증이 불가능하다.** TP=2 면 배치 선택지가 {0,1} 하나뿐이라
    제약이 자동 만족되어 검증할 것이 없다. 4 장이 최소선이고 8 장이면 논문의
    TP=4/8 을 재현할 수 있다
  - 검증 설계의 핵심: **제약이 없었다면 argmin 이 샤드를 겹쳐 놓았을 시나리오를
    구성**하고, 제약이 켜졌을 때 다른 배치가 선택되는 것을 보여야 한다.
    "제약을 켜도 아무 일도 안 일어났다" 는 검증이 아니다
  - anti-affinity ON / OFF 두 arm 을 같은 워크로드·seed 로 돌리고, 제약이 발동한
    횟수와 그때 배치가 어떻게 달라졌는지를 raw data 로 남겨라

## 4. 이 장비에서 반드시 다시 재야 하는 것

**건너뛰면 모든 SLO 판정이 조용히 틀린다.** 이전 장비 값을 그대로 쓰지 마라.
HANDOVER.md §4.2 에 근거가 있고, 실제로 장비를 옮길 때마다 이만큼 달랐다:

  | TTFT p95 기준선 | 이전 장비 대비 +10 ~ +35 % |
  | c_i (Algorithm 2) | -6 ~ -11 % |
  | tau (Algorithm 1) | 이전 값이 두 자릿수 어긋나 결정의 29 % 를 통과시킴 |

  ./setup/quickstart.sh                                  # redis, 스택, 모델, 패치
  ./exp/scripts/run_profiling_v2.sh <outdir>             # SLO 기준선 + c_i, 약 40분
  ./exp/scripts/calibrate_tau_v4.sh <outdir>             # tau 재유도

그리고 TP 는 all-reduce 가 매 디코드 스텝에 있으므로 **토폴로지를 반드시 먼저
확인하고 기록해라**:

  nvidia-smi topo -m           # NV# 여야 한다. PCIe-only 면 TP 수치가 무의미하다
  nvidia-smi nvlink --status
  python -c "import torch; print({f'{i}->{j}': torch.cuda.can_device_access_peer(i,j)
             for i in range(torch.cuda.device_count())
             for j in range(torch.cuda.device_count()) if i!=j})"

**GPU 4 장 전부에 대해** peer access 행렬을 남겨라. 2 장짜리 결과를 4 장 토폴로지로
확대 해석하면 안 된다.

## 5. 하지 말 것

* **기각된 가설을 다시 파지 마라.** HANDOVER.md §2 에 다섯 개가 근거와 함께
  정리돼 있다 (마이그레이션 억제, Moore-Hodgson 과소수용, admission 실행비용 등).
* **V5_2 의 deactivation 원인 규명(12~15초 블로킹)은 별개의 분석 작업이다.**
  TP 구현에 섞지 마라. 참고로 V6 에서 유력한 단서가 나왔다 — 비활성화 경로가
  `preempt=False` 라서 `_run_to_completion_normal()` 이 진행 중 배치를 **끝까지
  드레인**한다 (`scheduler.py:559, 1728`). 긴 생성이 걸리면 비활성화가 길어진다.
  이건 그 분석 트랙에 넘길 단서이고 네 작업이 아니다.
* **성능 목표와 충실도 목표를 한 브랜치에 섞지 마라.** TP 는 충실도 목표다.
* 논문에 없는 항을 paper-faithful arm 에 넣지 마라. 필요하면 라벨링된 별도 arm.

## 6. 장비 함정 (전부 실제로 시간을 잡아먹은 것들)

CLAUDE.md §5/§8 에 전체 목록이 있고, 최근에 다시 겪은 것들만:

* supervisor 는 프로세스 그룹째 멈춰야 한다 — `stopasgroup=true` / `killasgroup=true`
* supervisor 자식은 fd 1024 로 시작한다 — 파이프라인과 watchdog 양쪽에 `ulimit -n 65535`
* **실행 중인 bash 스크립트를 편집하지 마라.** bash 가 바이트 오프셋으로 이어 읽어서
  `syntax error near unexpected token` 이 난다. 후속 작업은 별도 supervisor 프로그램으로
* `env.sh` 가 `/workspace/.env` 를 `set -a` 로 다시 읽어 호출자의 `PRISM_V4_*` 를 덮는다
* `exp/scripts/env.sh` 가 `PRISM_ROOT=/workspace/prism-exp` 를 하드코딩한다.
  레포를 반드시 그 경로에 두어라
* HF_TOKEN 은 meta-llama 라이선스를 승인한 계정의 것이어야 한다. 없으면 Llama
  다운로드가 전부 401 이다. **지어내지 말고 사용자에게 받아라**
* Qwen2.5-7B 는 flashinfer prefill workspace 가 기본 384 MiB 로 모자란다.
  `FLASHINFER_WORKSPACE_SIZE` 를 int 로 캐스트하는 수정이 v4 패치에 있다

## 7. 산출물

* `exp/results/paper-faithful-tp/REPORT.md` — 측정 결과
* `exp/results/paper-faithful-tp/IMPLEMENTATION_AUDIT.md` 갱신분 — TP 관련 행의
  판정이 무엇으로 바뀌었고 근거가 무엇인지
* raw data 를 반드시 보존해라. 집계가 raw 를 대체하지 않는다
* 단계별로 **안 된 것은 안 됐다고** 적어라. 이 프로젝트의 기존 리포트들이 전부
  그렇게 쓰여 있다

먼저 위 문서들을 읽고, **구현 전에** 무엇을 어떤 순서로 할 것인지 나에게 요약해라.
```

---

## 지금 이 서버 작업과의 관계

**병렬로 시작 가능하다.** 파일이 겹치지 않는다.

| | 이 서버 (v6) | 4×A100 (tp) |
| --- | --- | --- |
| 건드리는 파일 | `scheduler.py`, `kvcached`, 마이그레이션 회계 | `multi_model_server.py`, `controller_global.py`, `model_runner.py` |
| 목표 | KV migration (충실도) | TP + anti-affinity (충실도) |
| 필요 GPU | 2 장 | **4 장 이상** |

유일한 접점은 `kvpr_global_v4.py` 다 — anti-affinity 필터가 거기 들어가고,
KV migration 은 거기를 건드리지 않는다. 그래도 나중에 합칠 때 충돌 지점이므로
양쪽 다 그 파일에서는 작업을 최소화하는 편이 좋다.
