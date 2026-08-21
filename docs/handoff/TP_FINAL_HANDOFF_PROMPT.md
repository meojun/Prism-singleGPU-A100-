# 인수인계 프롬프트 — 최종 비교 실험과 70B 검증

https://github.com/meojun/Prism-singleGPU-A100- 이 레포로 작업한다.

Prism 논문(OSDI'26)의 released prototype 을 paper-faithful 하게 재구현하는
프로젝트다. **구현은 끝났다. 네가 이어받을 것은 측정이다.**

## 0. 먼저 읽어라 (실행 전에 반드시)

    git clone https://github.com/meojun/Prism-singleGPU-A100- /workspace/prism-exp
    cd /workspace/prism-exp && git checkout exp/tp-v6-merge

읽을 순서:
  1. RESUME_HERE_TP.md                                    ← 현재 상태. 여기부터.
  2. exp/results/paper-faithful-tp/REPORT.md              ← TP 측정 결과
  3. exp/results/paper-faithful-tp/DESIGN_DECISIONS.md    ← 논문 §A.2.2 대조
  4. exp/results/paper-faithful-tp/findings/              ← 열린 결함과 철회 기록
  5. HANDOVER.md                                          ← 연구 전반 ground truth
  6. CLAUDE.md                                            ← 장비 런북

**RESUME_HERE_TP.md 를 기준으로 쓰되, 코드나 raw data 와 모순되면 코드와 raw
data 를 우선해서 보고해라.** 이번 연구에서 그런 모순이 여러 건 나왔고, 문서
쪽이 틀린 경우가 많았다. 논문 원문이 저장소에 없어서 저장소의 한 줄 요약을
논문의 전부로 읽었다가 두 건을 정정한 적도 있다 — 논문 PDF 가 있으면 §A.2.2 와
§5.3 을 직접 확인해라.

## 1. 브랜치

| 브랜치 | 내용 |
| --- | --- |
| `exp/tp-v6-merge` | **작업 대상.** TP + KV migration 병합본 |
| `exp/paper-faithful-tp` | TP 구현 원본 |
| `exp/paper-faithful-v6` | KV migration 원본 |

병합은 이미 검증했다. 패치 체인이 **양방향**으로 적용된다
(v6→tp, tp→v6). 유일한 실질 위험이었던 `server_args.py` 의 `keys_to_remove`
앵커 충돌은 해소돼 있다.

## 2. 이미 확립된 것 — 다시 파지 마라

* **TP=2/TP=4 는 worker-pool 경로에서 돈다.** V4 판정 FAIL 이었던 것이 PASS 다.
  원인은 기록된 가설과 달랐다 — 가중치 조회 결함이 아니라 `tp_size` 가
  `server_args.py:265` 의 `keys_to_remove` 에서 버려진 것이다.
* **Llama-3.1-70B 이 TP=4 와 TP=2 로 서빙된다.** 스모크 수준은 확인됐다.
* **논문 규칙(second-lowest KVPR)과 strict 규칙은 4 GPU 에서 원리상 구분
  불가능하다.** `k>=3` 이고 `n>k` 여야 갈리는데 어느 모델도 TP=3 을 못 한다
  (`num_key_value_heads` 가 8/4/2). 8 GPU 가 필요하다. 테스트에 박혀 있다.
* **anti-affinity 는 배치를 바꾸지 못하는 경우가 많다.** engine pool 에서
  배치 단위가 미리 구성된 그룹이라 충돌 배치가 애초에 emit 될 수 없다.
  이것이 이 연구의 주된 결과다. `DESIGN_DECISIONS.md` §2.

## 3. 네가 할 일 — 순서를 지켜라

### (0) 새 박스 재보정 — 건너뛰면 모든 SLO 판정이 조용히 틀린다

`RESUME_HERE_TP.md` §4 그대로. 요점만:

* `quickstart.sh` 는 v2 패치만 적용한다. **체인을 v6/tp 까지 마저 돌려라.**
  안 하면 flashinfer workspace 가 문자열로 읽혀 프로파일링이 6모델 전부에서
  죽는다.
* `run_profiling_v2.sh` → 트레이스 빌드 → `calibrate_tau_v4.sh` 순서.
  tau 스크립트는 gitignore 된 `.pkl` 을 전제한다.
* 이전 박스 값(τ=0.171086 등)을 **이어받지 마라.** 이 프로젝트에서 반복적으로
  사고를 낸 항목이다.
* 토폴로지를 확인하고 기록해라. PCIe-only 면 TP 수치가 무의미하다.

### (1) worktree venv 링크 — 이걸 안 하면 최종 비교가 0런으로 끝난다

지난 세션이 여기서 12시간을 날렸다. `env.sh` 가
`$PRISM_ROOT/prism-venv/bin/activate` 를 소스하는데 worktree 에는 venv 가 없어
시스템 python 이 쓰였고, `transformers` 가 없어 워크로드가 0개 생성됐다.
49런이 전부 "트레이스 없음" FAILED 로 찍히고 25초 만에 끝났다.

    ln -sfn /workspace/prism-exp/prism-venv     /workspace/prism-merge/prism-venv
    ln -sfn /workspace/prism-exp/kvcached       /workspace/prism-merge/kvcached
    ln -sfn /workspace/prism-exp/kvcached-prism /workspace/prism-merge/kvcached-prism

**확인까지 해라:**

    ( cd /workspace/prism-merge && PRISM_ROOT=$PWD source exp/scripts/env.sh && \
      python3 -c "import transformers,torch;print('OK',torch.__version__)" )

### (2) TP 스윕 실패 2런 재실행 (30분)

`strict/s1`, `paper/s3` 가 shm 잔재(`Error: '/ipc_2_3_root'`)로 기동 실패했다.
TP 와 무관하다. 재실행하면 9/9 가 된다.

### (3) 최종 비교 — 이번 세션의 본체

**arm 두 개.** 사용자 지시대로 비교 그래프는 프로토타입 대 최종 둘뿐이다.

    A  released-prototype  @ TP 패치 없는 트리   (exp/paper-faithful-v6 기준)
    C  paper-faithful-v6   @ 병합 트리, 전부 켜짐

**회귀 게이트만 세 번째 arm 을 쓴다** (bursty 8, seed 1 한 쌍):

    B  released-prototype  @ 병합 트리, TP 플래그 꺼짐

게이트가 통과하지 못하면 **본 스윕을 시작하지 마라.** A-vs-C 수치가 "논문
메커니즘의 효과"가 아니라 "다른 코드베이스와의 비교"가 되기 때문이고, 12시간을
쓰고 나서 알아내면 늦다.

조건:
* TP=1, **GPU 2장 고정** (박스에 4장이 있어도. 전 arm 이 같은 하드웨어를 봐야 한다)
* bursty **2 / 4 / 8 / 14 / 20** req/s × seed 1,2,3
* steady **4 / 8 / 20** req/s × seed 1,2,3
* 20 req/s 는 일반 운영점이 아니라 **Extreme Load / Stress** 로 라벨해라.
  이 구성의 포화점은 5~10 req/s 로 측정돼 있다.

**실측 기준 런당 14.4분(중앙값), 50런 ≈ 12시간.** 줄이려면 steady 를 8 req/s
만 남겨라 (~9시간). steady 의 목적은 "이점이 시간적 이질성에 몰려 있는가"
확인이고 무릎 하나면 그 질문에 답한다. **seed 를 2개로 줄이지는 마라** —
이 프로젝트의 seed 분산이 sd/mean 17% 라 n=2 면 판정기가 대부분
"insufficient seeds" 로 떨어뜨린다.

인프라는 다 있다:
* `exp/scripts/orchestrate_final_compare.sh` — 오케스트레이터 (supervisor 로 등록)
* `exp/scripts/compare_arms.py` — 집계, 게이트, 개선율
* 결과: `exp/results/final-prototype-vs-paper-faithful/`

**판정 규칙은 이미 코드에 고정돼 있다. 바꾸지 마라.** 개선은 pooled sd 를 넘을
때만 주장하고, 못 넘으면 `within noise` 로 적는다. 성공 seed 2 미만이면 집계
하지 않는다. 부호는 통일돼 있다 — 양수면 언제나 Final 이 낫다.

### (4) 70B 안정성 (~1.3시간)

3단계 게이트다. 앞이 실패하면 뒤를 돌리지 마라.

    Stage 1 startup    rank 전부 생성, NCCL, rank->GPU 서로 다름, OOM 없음
    Stage 2 basic      짧은 요청 여러 개 성공, rank crash 없음
    Stage 3 sustained  최소 30분 지속 부하. 안정성이 peak 보다 우선

`exp/scripts/run_70b_sustained.sh` + `build_70b_report.py`. 실패하면 단순 FAIL
로 끝내지 말고 **어느 단계인지** 적어라 — "70B 가 안 된다" 는 조치가 불가능하고
"가중치 로딩에서 죽는다" 는 가능하다.

### (5) V6 KV 주입 — 아직 막혀 있다

4단계 중 3단계까지 됐고 주입이 안 된다. **추측하지 마라. 이 항목에서 세 번
틀렸다.** 계측이 이미 들어가 있다(`patches/paper_faithful_tp/apply_v6_probe.py`,
동작 변경 없음). 돌리면 이 두 줄이 찍힌다:

    [KV-PROBE service] key=... have=[...] id=...
    [KV-PROBE engine]  key=... have=[...] id=...

**`run_v6_validation.sh` 를 쓰지 마라** — `R=/workspace/prism-exp` 를 하드코딩해서
v6 패치가 없는 트리를 가리킨다. `run_v6_validation_merge.sh` 를 써라.

## 4. 하지 말 것

* **기각된 가설을 다시 파지 마라.** `HANDOVER.md` §2 에 다섯 개,
  V6 쪽에 세 개(전달 채널)가 근거와 함께 정리돼 있다.
* **철회된 실험을 되살리지 마라.**
  `findings/WITHDRAWN_regression_step.md` — 같은 코드베이스에서 다른 시스템을
  돌리는 것은 회귀 검증이 아니다.
* **TP>1 을 프로토타입과 겨루지 마라.** 프로토타입은 worker-pool 에서 TP>1 을
  아예 못 돌린다. **Unsupported** 로 적지 성능 arm 을 만들지 마라.
* **커밋된 `slo_base.json` / `prefill_speed.json` 을 믿지 마라.** 다른 박스 값이다.
* **논문에 없는 항을 paper-faithful arm 에 넣지 마라.**

## 5. 장비 함정 (전부 실제로 시간을 잡아먹은 것)

`CLAUDE.md` §5/§8 에 더해 이번에 새로 겪은 것:

* worktree venv 링크 (§3(1)) — 12시간을 날렸다
* 런 사이 `/dev/shm` 미정리 → 다음 서버가 기동에서 죽는다. spawn 자식이 init 으로
  재부모화되어 늦게 죽으므로 `reap` 대기를 넉넉히
* sglang 은 spawn 을 쓴다. 부모를 죽여도 워커가 GPU 를 쥔 채 남는다
* 패치에서 편집을 지워도 **이미 적용된 저장소는 되돌아가지 않는다.**
  `git checkout -- python/` 후 체인 재적용
* 패치 앵커와 probe 는 **파일에서 유일해야** 한다. `apply_tp.py` 의 `replace()`
  가 이제 강제한다 — 도입하자마자 실제 버그 두 개를 잡았다
* 실행 중인 bash 스크립트를 편집하지 마라
* 프로파일링은 무경합 단독 측정이다
* 상대 경로 `--model-config-file` 은 죽는다 (서버가 cd 한다)
* 요청 필드는 `model` 이지 `model_name` 이 아니다
* `ActivateReqInput.gpu_id` 는 기본값이 없다 (빠뜨리면 FastAPI 422)
* 장시간 작업은 supervisor 아래에. tmux 는 이 박스에서 죽는다

## 6. 산출물

* `exp/results/final-prototype-vs-paper-faithful/REPORT.md` — 다섯 질문에 데이터로
* `.../large-model-70b/STABILITY_REPORT.md`
* `IMPLEMENTATION_AUDIT` 갱신분 — **생성기(`build_report_tp.py`)를 고쳐라.**
  표만 손으로 고치면 다음 생성 때 되돌아간다
* **raw data 를 보존해라.** 집계가 raw 를 대체하지 않는다
* **안 된 것은 안 됐다고 적어라.** 이 프로젝트의 기존 리포트가 전부 그렇게
  쓰여 있고, 그게 이 연구의 가치다

먼저 위 문서들을 읽고, **실행 전에** 무엇을 어떤 순서로 할 것인지 나에게
요약해라.
