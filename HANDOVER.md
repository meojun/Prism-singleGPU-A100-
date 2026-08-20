# 인수인계 — Paper-Faithful Prism V4 / V5 / V5_2

`CLAUDE.md` 가 장비 세팅 런북이고, 이 문서는 **그 위에서 수행된 연구의 인수인계**다.
무엇이 밝혀졌고, 무엇이 기각되었고, 무엇이 아직 열려 있는지를 적는다.

---

## 0. 브랜치 지도

| 브랜치 | 내용 | 상태 |
| --- | --- | --- |
| `exp/paper-faithful-v3-validation` | 이전 장비의 V3 연구 | 참조용 |
| `exp/paper-faithful-v4` | V4 구현 + 27런 본 스윕 + τ 연구 | 완료 |
| `exp/paper-faithful-v5` | ablation(어느 알고리즘이 비용인가) + P2P 재검증 15런 | 완료 |
| `exp/paper-faithful-v5_2` | 계측 전용 — 스케줄러 루프 / deactivation 구간 9런 | 완료 (현재 브랜치) |

각 브랜치의 `exp/results/<name>/REPORT.md` 가 그 연구의 결과다. raw data 는
`raw/{requests,migrations,scheduler,gpu_metrics}/` 에 요청·마이그레이션·사이클·GPU샘플
단위로 보존되어 있고, `summary.csv` 는 그것으로부터 파생된 것이지 그 반대가 아니다.

---

## 1. 확립된 것

### 1.1 메커니즘은 주장대로 작동한다

| 항목 | 결과 | 근거 |
| --- | --- | --- |
| 병렬 가중치 로딩 | sequential 9.01 → V3 11.39 → **V4 25.51 GB/s** | `v4/microbench/loading.json` |
| NVLink P2P 마이그레이션 (8B) | V3 0.773s → **V4 0.220s**, 72.9 GB/s | `v4/microbench/migration.json` |
| P2P 경로 실서빙 검증 | 3런, P2P 전송 6~8건, **요청 실패 0** | `v5/REPORT.md` |

NVLink 사용은 추정이 아니라 드라이버 링크 카운터로 계측했다. broker 경로는 모델의
정확히 절반이, P2P 경로는 전체(Llama-3.1-8B = 14.96 GiB)가 링크를 건넌다.

**page-lock 이 로딩 이득의 전부다.** 세 arm 의 바이트 분할이 동일하므로 이득은 옮긴
양이 아니라 방식에서 온다. helper leg 파이프라이닝은 구현했으나 이 장비에서 더
느렸고(12.67 GB/s), ablation 으로 남겼다 — **PARTIAL 을 무조건 FULL 로 만들 이유는 없다.**

### 1.2 그런데 e2e goodput 으로 이어지지 않는다

| 조건 | prototype | V3 | V4 |
| --- | ---: | ---: | ---: |
| bursty 8 | 2.53 ± 2.27 | 1.46 ± 0.07 | 1.37 ± 0.20 |
| bursty 20 | 0.56 ± 0.35 | **4.46 ± 3.25** | **4.21 ± 3.40** |
| steady 8 | 0.84 ± 0.58 | 0.65 ± 0.11 | 0.53 ± 0.07 |

* 부하에 따라 **부호가 바뀐다.** 중부하에서는 paper-faithful arm 이 지고, 포화 구간
  (20 req/s)에서는 평균 8배 이긴다.
* **V3 와 V4 는 어느 조건에서도 구분되지 않는다.** 전송을 2배 빠르게 해도(e2e 에서도
  16.8 대 8.3 GB/s 로 재현됨) goodput 이 움직이지 않는다.
* 20 req/s 는 이 구성의 포화점(5~10 req/s)의 2~4배 지점이고 seed 간 분산이 평균의
  70~80% 다. 평균비만 인용하지 말고 seed 별 값을 함께 볼 것.

---

## 2. 기각된 가설 — 여기를 다시 파지 말 것

이번 연구에서 다섯 개의 그럴듯한 원인이 데이터에 의해 기각됐다. 각각 재현 가능한
근거가 있으므로 같은 길을 다시 가지 않기 바란다.

| 가설 | 기각 근거 | 위치 |
| --- | --- | --- |
| 마이그레이션이 비용이라 줄이면 낫다 | τ 를 이 장비에서 재유도(0.00035 → 0.146)해 억제하니 bursty 20 goodput 이 **4.21 → 0.30** 으로 붕괴 | `v4/tau-study/` |
| 마이그레이션이 임계 경로 밖이다 | 위약 대조 TTFT 1.30배 — 단 **arm 별로 갈린다**, 아래 2.1 | `v4/v5_2-analysis/q1_*` |
| KV 손실이 비싸다 | 위약 대조 TPOT 1.04 / E2E 0.93 — **약한 증거**, 아래 2.1 | `v4/v5_2-analysis/q4_*` |
| Moore-Hodgson 의 과소 수용 | steady 8 3 seed 에서 **3482/3483 수용, deferred 합계 1** 인데 goodput 은 낮음 | `v5/summary.csv` |
| Moore-Hodgson 의 실행 비용 | admission 이 iteration 당 **+0.001 ms**. 설명해야 할 TTFT 차이는 2.7 ms | `v5_2/REPORT.md` |

### 2.1 위약 대조 두 건은 표가 읽히는 것보다 약하다 (V6 재계산)

raw CSV 를 arm 별로 다시 계산한 결과다. 표의 한 줄로 압축하면 실제보다 단단하게
읽히므로 여기에 남긴다.

**q1 (임계 경로).** 24 런 중 **11 런만** 위약 대조가 존재한다 — 마이그레이션이 잦으면
조용한 구간이 남지 않아 `far_n = 0` 이 된다. 그 11 런을 arm 별로 나누면:

| arm | n | TTFT near/far median | 범위 |
| --- | ---: | ---: | --- |
| released-prototype | 6 | **1.55** | 1.22 ~ 7.80 |
| paper-faithful (v3+v4) | 5 | **1.08** | 0.97 ~ 1.30 |

pooled median 1.30 은 **프로토타입 arm 이 끌어올린 값**이다. 프로토타입은 source-first
라 전송 구간 전체가 서비스 공백이고(§1.1 의 downtime 0.482s 대 0), v3/v4 는
target-first 라 downtime 이 0 이다. **서로 다른 메커니즘을 pool 한 것**이다.
읽어야 할 결론은 이렇다 — 마이그레이션 비용은 **프로토타입에서는 임계 경로에 있고,
우리 target-first arm 에서는 거의 1.0 이라 근거가 약하다.** 따라서 3.1 의 중부하
지연 세금을 마이그레이션 임계경로로 설명하려는 시도는 이 데이터가 지지하지 않는다.

덧붙여 `analysis.txt` 는 자기 숫자(1.30)와 반대되는 결론
("the migration cost is NOT on the request critical path")을 무조건 출력하는
legend 문구를 달고 있었다. V6 에서 `analyze_v5_2.py` 가 판정을 숫자에서 유도하고
arm 별 분해를 함께 찍도록 고쳤다.

**q4 (KV 손실).** median 1.04 / 0.93 은 재계산해도 정확히 일치한다. 다만 **n=8** 이고
개별 비율이 TPOT `[0.28, 1.53]`, E2E `[0.27, 1.39]` 로 1 을 양쪽으로 크게 넘나든다.
"기각" 이 아니라 **"이득이 크지 않다는 약한 증거"** 가 정확한 표현이다. KV migration
을 짓지 않을 근거로 이것만 인용하지 말 것 (3.3 의 판단은 그대로 유효하다 — 성능이
아니라 충실도가 이유이므로).

**Algorithm 2 = goodput 39% 라는 전제.** V5_2 가 출발한 이 수치는
released-prototype steady 8 = **0.844 ± 0.578** (sd 가 평균의 69%) 대 alg2only 0.513
비교다. **차이가 잡음 안에 있다.** V5_2 의 계측 자체는 유효하지만, 그것이 설명하려던
격차는 통계적으로 확립된 적이 없다. 이 방향을 더 파려면 seed 를 먼저 늘려야 한다.

**방법론 교훈이 하나 있다.** 위 두 번째와 세 번째는 처음에 "대조군 = 나머지 전부" 로
쟀다가 틀렸다. 마이그레이션을 걸친 요청을 고르면 자동으로 *긴 요청* 이 뽑히고, 300초에
13회 마이그레이션이면 ±10초 창이 시간의 87% 를 덮어 대조군이 조용한 구간만 남는다.
**위약(sham) 시점 대조**로 바꾸자 결론이 뒤집혔다. 같은 종류의 비교를 할 때 반드시
같은 편향을 갖는 대조군을 쓸 것.

---

## 3. 열려 있는 질문

### 3.1 중부하에서 paper-faithful arm 이 지는 원인 (최우선)

증상은 명확하다. **처리율은 세 arm 이 동일**하고(8.05 / 19.89 / 8.07), steady 에서는
deferral 도 0 인데, 요청마다 **5~6% 지연이 균일하게 얹힌다** (TTFT p50 106.5 → 112.7 →
117.3 ms). goodput 이 SLO 임계 지표라 5% 지연이 23% goodput 손실로 증폭된다.

원인은 미상이다. 남은 단서 둘:

1. **`--overlap-migration` 의 readiness barrier.** ablation arm 에 이것이 섞여 있었다
   (`v3-alg2only` 는 Moore-Hodgson 외에 `--parallel-model-loading` 과
   `--overlap-migration` 도 켠다). 제어 액션이 엔진 완료까지 블로킹되고 그동안
   컨트롤러가 다른 결정을 못 한다. **다음에 할 ablation 은 이 플래그를 뺀 arm 이다.**
2. **deactivate 를 12초 막는 것.** 아래 3.2.

### 3.2 일부 deactivation 이 12초 블로킹되는 이유

제어 경로 자체는 0.16 ms 로 무시할 수준이고 전부 엔진 대기다. 그런데 분포가 극단적이다 —
3 seed 를 합친 62 건 기준 **중앙값 0.90초(≈ 엔진 자체 teardown 0.96초), 평균 3.45초,
최대 15.38초.** 대부분은
순수 teardown 이고 소수의 긴 것이 총합을 지배한다. 엔진이 배치 경계까지 기다리는 것으로
추정되지만 현재 계측은 거기까지 닿지 않는다. 다음 계측은 엔진 내부(스케줄러가 deactivate
요청을 집어드는 시점 대 배치 상태)여야 한다.

> **정정 (V6).** 이 문단은 원래 "중앙값 0.84초, 최대 12.52초" 였다. 그 값은
> **seed 1 만** 의 것이다. `raw/scheduler/paper-faithful-v3_bursty_r8_s*_actions.jsonl`
> 을 합치면 seed 별 max 가 12.53 / 15.38 / 13.16 초이고, 평균 3.45 초만 pooled 값이었다.
> seed 를 섞어 인용하지 말 것.

| seed | n | median (s) | mean (s) | max (s) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 20 | 0.851 | 2.943 | 12.530 |
| 2 | 21 | 1.104 | 4.075 | **15.378** |
| 3 | 21 | 0.764 | 3.314 | 13.161 |
| **합** | **62** | **0.899** | **3.452** | **15.378** |

### 3.3 미구현 (논문 충실도)

| | 상태 | 비고 |
| --- | --- | --- |
| KV migration | 미구현 | 프로세스 간 스케줄러 상태 이관이 본체. 위약 대조상 성능 이득은 낮게 예상되나, **논문 메커니즘이므로 충실도 목표에는 구현이 맞다** |
| TP anti-affinity | 불가 | worker-pool 엔진이 `[gpu_id]` 단일 GPU 에 묶여 TP 샤드가 GPU 를 가로지를 구조가 없다. `v4/tp-validation/FINDING.md` |
| TP=2 검증 | GPU 2장에서 불가 | TP=2 면 선택지가 하나뿐이라 anti-affinity 가 자동 만족돼 검증할 것이 없다. **GPU 4장 이상 필요하고, 그러면 별도 연구가 된다** |

---

## 4. 새 장비에서 재현하기

### 4.1 절차

```bash
git clone https://github.com/meojun/Prism-singleGPU-A100- /workspace/prism-exp
cd /workspace/prism-exp && git checkout exp/paper-faithful-v5_2
echo 'HF_TOKEN=hf_xxx' > /workspace/.env && chmod 600 /workspace/.env   # meta-llama 승인 계정
./setup/quickstart.sh                       # redis → 모델 47GB → bootstrap → 패치 → 단위테스트

# 이 장비에서 반드시 다시 재야 하는 것 (약 40분)
./exp/scripts/run_profiling_v2.sh exp/results/<study>/profiling

# 패치 체인 (v2 → v3 → v4 → v5_2 순서, 멱등적)
for f in patches/paper_faithful/apply_patches.py patches/paper_faithful_v3/apply_v3.py \
         patches/paper_faithful_v4/apply_v4.py patches/paper_faithful_v5_2/apply_v5_2.py; do
    python3 $f --repo /workspace/prism-exp/prism-research; done

supervisorctl start prism_v4      # 또는 prism_v5 / prism_v5_2
```

### 4.2 장비마다 반드시 다시 재야 하는 값

**이걸 건너뛰면 모든 SLO 판정이 조용히 틀린다.** 이번에 이전 장비 값을 그대로 쓸 뻔했고,
재보니 이만큼 달랐다:

| | 차이 |
| --- | --- |
| TTFT p95 기준선 | 이전 장비 대비 **+10 ~ +35 %** |
| `c_i` (Algorithm 2 실행가능성) | **-6 ~ -11 %** |
| `τ` (Algorithm 1 line 8) | 이전 값 0.00035 가 이 장비 delta 분포보다 두 자릿수 작아 결정의 29 % 를 통과시킴 |

`exp/scripts/calibrate_tau_v4.sh` + `derive_tau_v4.py` 가 τ 를 마이그레이션 억제 상태에서
재유도한다(자기가 통제할 대상을 통해 자기를 재는 순환을 피하기 위함).
이전 장비 값은 `v4/provenance/*_committed_other_box.json` 에 보존돼 있다.

### 4.3 2×A100 장비에서 그대로 도는가 — 그렇다

이번 연구는 4×A100 노드에서 **GPU 2장만** 써서 수행했고, 실제 2-GPU 장비에서도 동일하게
동작한다. 확인한 것:

* 모든 파이프라인이 `NGPU=2`, `CUDA_VISIBLE_DEVICES=0,1` 고정. GPU 가 2장뿐이면 그대로 0,1 이다.
* 유휴 GPU 검사(`env_check`)는 "할당 쌍 밖의 GPU 중 사용 중인 것"을 찾는데, 2장뿐이면
  검사 대상이 없어 빈 결과 → 통과한다.
* microbenchmark 는 `--gpu-ids 0,1` 로 두 장만 쓴다.

**다만 확인할 것이 하나 있다 — GPU 간 P2P 지원 여부.** 이 노드는 NVSwitch 를 거쳐 모든
쌍이 NV12 였다. PCIe 로만 연결된 2-GPU 장비라면 `torch.cuda.can_device_access_peer` 가
False 가 되고, V4 의 P2P 마이그레이션은 자동으로 host 경로로 폴백한다(코드가 확인하고
분기한다). 죽지는 않지만 **P2P 수치는 나오지 않으므로**, 리포트에
`peer access: {'0->1': ...}` 값을 반드시 확인하고 기록할 것.

```bash
nvidia-smi topo -m            # NV# 이면 NVLink
nvidia-smi nvlink --status
```

### 4.4 이 박스에서 쓴 환경 설정

`/workspace/.env` (저장소 밖, 권한 600):

```
HF_TOKEN=...                          # meta-llama 라이선스 승인 계정
CUDA_VISIBLE_DEVICES=0,1              # 할당 강제
FLASHINFER_WORKSPACE_SIZE=1073741824  # 아래 5.1 참조
PRISM_V4_P2P_MIGRATION=0              # 아래 5.2 참조
```

---

## 5. 이번에 시간을 잡아먹은 함정

`CLAUDE.md` §5/§8 의 함정 목록에 더해, 이번 연구에서 새로 겪은 것들이다.

### 5.1 flashinfer workspace 가 기본값으로는 모자라고, 환경변수는 깨져 있었다

Qwen2.5-7B 프로파일링이 세 번 죽었다. `batch_prefill_tmp_v` 에 ~450 MiB 가 필요한데
기본이 384 MiB 다(이 모델만 GQA 비가 28:4 여서 그렇다). `FLASHINFER_WORKSPACE_SIZE` 가
그걸 위해 있는데 `global_config.py` 가 **문자열로 읽어** `torch.empty("...")` 가 죽는다.
`int()` 캐스트가 v4 패치에 들어 있다. 동시성을 낮추는 것으로는 해결되지 않는다.

### 5.2 P2P 마이그레이션은 CUDA IPC 매핑을 반환해야 한다

마이그레이션 소스로 쓰려고 model service 가 엔진 GPU 가중치를 IPC 로 매핑하는데,
Python 참조를 버리는 것만으로는 메모리가 반환되지 않는다. `empty_cache()` 는 자기
프로세스 할당자 캐시만 비운다 — **다른 프로세스 메모리의 IPC 매핑은
`torch.cuda.ipc_collect()` 가 필요하다.** 비활성화 24회가 누수로 쌓여 OOM 났고 요청
3387건 중 1300건이 실패했다. 수정 후 재검증에서 실패 0건.

`/workspace/.env` 의 `PRISM_V4_P2P_MIGRATION=0` 은 그 사건 뒤 27런의 설정을 하나로
고정하려고 꺼둔 것이다. **V5 에서 재활성화해 검증했고 통과했으므로, 새 연구에서는 켜도 된다**
(`V5_P2P=1` 로 넘기면 `env.sh` 가 덮어쓰지 못한다 — 아래 5.3).

### 5.3 `env.sh` 가 호출자의 설정을 덮어쓴다

`run_v4_case.sh` 가 `env.sh` 를 source 하고, `env.sh` 는 `/workspace/.env` 를 `set -a` 로
다시 읽는다. 그래서 호출자가 `PRISM_V4_*` 를 지정해도 조용히 덮인다. `V5_P2P` /
`V5_PAGELOCK` 은 그 파일에 없으므로 살아남아 이긴다.

### 5.4 supervisor 는 프로세스 그룹째 멈춰야 한다

`supervisorctl stop` 이 watchdog 만 죽이고 그것이 띄운 파이프라인은 살아남아 watchdog 의
flock 을 쥔 채로 남았다. 다음 start 는 lock 을 못 잡아 즉시 exit 0 했고 supervisor 가
네 번 만에 FATAL 로 포기했다. `stopasgroup=true` / `killasgroup=true` 가 필요하다.

### 5.5 supervisor 자식은 fd 1024 로 시작한다

microbenchmark 가 6개 모델의 모든 가중치 텐서에 `share_memory_()` 를 부르는데 텐서당
fd 하나다(약 1500개). `ulimit -n 65535` 를 watchdog 과 파이프라인 양쪽에 넣었다.

### 5.6 `/dev/shm` 정리 패턴이 실제 이름과 달랐다

정리 코드가 `ipc_0_model_*_root` 를 지웠지만 엔진이 만드는 이름은
`ipc_<gpu>_<worker>_root` 다. 죽은 런이 세그먼트를 전부 남겼다.

### 5.7 실행 중인 bash 스크립트를 편집하지 말 것 (재확인)

bash 는 스크립트를 바이트 오프셋으로 이어 읽는다. 파이프라인이 도는 중에 고쳤더니
`syntax error near unexpected token 'fi'` 가 났다. **후속 작업은 별도 supervisor
프로그램으로 붙일 것** — 이번에 τ 연구와 리포트 발행이 그 방식으로 안전하게 동작했다.

### 5.8 마이그레이션 개수는 컨트롤러 결정 로그로 셀 것

Activate 를 가장 가까운 Deactivate 와 짝지으면 유휴 축출과 재활성화가 한 번의
마이그레이션으로 읽힌다. 한 프로토타입 런에서 컨트롤러 결정 2회 대 페어링 4회였고,
그중 둘은 25초·30초짜리 "마이그레이션" 이었다. `collect_v4_metrics.py` 는
`"Reason: migrate model"` / `migration_decision=MIGRATE` 를 권위로 삼고 액션 기록은
시간을 재는 데만 쓴다.

### 5.9 readiness barrier 가 없으면 액션 시간은 제출 시간이다

`--overlap-migration` 이 없으면 제어 핸들러가 요청을 **제출하는 순간** 반환한다. 그래서
released prototype 의 액션은 가중치가 얼마나 걸리든 ~15 ms 로 찍힌다. V3/V4 의 실제
전송 시간 옆에 놓으면 프로토타입이 50배 빠른 것처럼 읽힌다. raw data 의
`latency_is_submission_only` 열이 그 표시다.

---

## 6. 다음에 할 것 (권고 순서)

1. **`--overlap-migration` 을 뺀 ablation.** 3.1 의 첫 번째 단서. 지금까지의 모든
   paper-faithful arm 에 이 플래그가 섞여 있었고, 제어 액션을 블로킹시킨다. 이것이
   중부하 지연 세금의 원인인지가 가장 값싸게 확인 가능한 가설이다.
2. **엔진 내부 계측.** deactivate 요청이 스케줄러에 도달한 시점 대 엔진이 실제 teardown 을
   시작한 시점. 3.2 의 12초를 특정한다.
3. **KV migration 구현** (충실도 목표라면). 성능 이득은 낮게 예상되나 그 자체가 결과다.
4. **TP=2** — GPU 4장 이상 확보 후, 별도 연구로.

**성능이 목표라면 1과 2가, 논문 충실도가 목표라면 3과 4가 맞다. 두 목표를 한 버전에
섞으면 어느 쪽도 깨끗하게 끝나지 않는다.**
