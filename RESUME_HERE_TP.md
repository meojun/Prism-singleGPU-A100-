# 돌아왔을 때 여기부터 — TP + anti-affinity, 그리고 최종 비교

이 파일이 **현재 상태**다. 연구 전반의 ground truth 는 `HANDOVER.md`, 장비
런북은 `CLAUDE.md`. 셋이 어긋나면 **코드와 raw data 를 우선**해라 — 이번에도
문서가 틀린 경우가 여러 건 나왔다.

브랜치 두 개, 둘 다 푸시돼 있다.

| 브랜치 | 내용 |
| --- | --- |
| `exp/paper-faithful-tp` | TP 지원 + anti-affinity 구현과 검증 |
| `exp/tp-v6-merge` | 위 + `exp/paper-faithful-v6`(KV migration) 병합 |

---

## 0. 30초 상황 파악

```bash
cd /workspace/prism-exp && git log --oneline -8
supervisorctl status | grep tp_
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
```

---

## 1. 끝난 것

| | 상태 | 근거 |
| --- | --- | --- |
| **1+1b** TP 슬롯 그룹 | ✅ | TP=2/TP=4 서빙, 슬롯 반환, 그룹 마이그레이션 관측 |
| **2** 컨트롤러가 TP 그룹 유지 | ✅ | `current_groups: [0,1]`, `any_tp_group_active: true` |
| **3** anti-affinity (논문 §A.2.2) | ✅ | `--enable-tp-anti-affinity` / `-strict` 두 경로, 단위테스트 |
| **4** ON/OFF 측정 | ✅ 7/9런 | `exp/results/paper-faithful-tp/step4-final/` |
| **5** 70B 서빙 | ✅ TP=4, TP=2 | `serve-70b-tp4/`, `serve-70b-tp2/` |
| 병합 (TP + v6) | ✅ 양방향 패치 체인 검증 | `exp/tp-v6-merge` |

### TP=2 는 이제 돈다 (V4 판정은 FAIL 이었다)

```
[PAPER-TP] engine rank: tp_rank=0 gpu_id=0 tp_size=2
[PAPER-TP] engine rank: tp_rank=1 gpu_id=1 tp_size=2      <- 서로 다른 GPU
[PAPER-TP] WorkerPool gpu=0 owned=[0,1] shadow=[]
[PAPER-TP] WorkerPool gpu=1 owned=[0]   shadow=[1]
```

**V4 의 실패 원인은 기록된 가설과 달랐다.** 가중치 조회의 독립 결함이 아니라
`tp_size` 가 `server_args.py:265` 의 `keys_to_remove` 에서 버려져 기본값 1 로
떨어진 것이다. 그래서 `--tensor-parallel-size 2` 가 worker-pool 경로에서 조용히
사라졌다. 결정적 증거는 uniform-TP 시도가 같은 에러로 죽었다는 것이다 —
tp_size 가 2 로 도달했다면 `launch_engine` 의
`assert len(tp_rank_range) == len(gpu_ids)` 가 먼저 터졌어야 한다.

### 4단계 측정 결과

| arm | seed별 위반(반사실) | 우회 | 요청 |
| --- | --- | --- | --- |
| off | 26, 11, 14 | **0, 0, 0** | 3387/3387 |
| paper | 28, 12 | **28, 12** | 3387/3387 |
| strict | 10, 21 | **10, 21** | 3372/3372 |

제약이 켜지면 위반이 100% 우회된다.

---

## 2. 이 연구의 주된 결과 — 논문과 구현의 간극

논문 §A.2.2 는 **샤드를 GPU 에 자유롭게 배치할 수 있다**고 전제한다. engine
pool 구현에서 배치 단위는 **미리 구성된 그룹**이다(논문 §4 의 "indivisible
resource unit"). 그룹이 서로 다른 GPU 로 구성되는 한 **충돌 배치는 애초에
emit 될 수 없다.**

```
OFF: argmin 이 (3,3) 을 원함 -> 실행할 엔진이 없음 -> 스냅 -> (1,3)
ON : argmin 이 (3,1) 을 원함 -> 그대로              -> (1,3)   ← 같은 그룹
```

즉 제약이 막는 것은 "불법 배치"가 아니라 **"planner 의 선택이 스냅에 덮이는
것"** 이고, 스냅이 차이를 흡수하는 경우가 많다.

그리고 **논문 규칙과 strict 규칙은 이 장비에서 원리상 구분 불가**다. 갈리려면
`k>=3` 이고 `n>k` 여야 하는데, 4 GPU 에서 남는 건 k=3 뿐이고 **이 실험의 어느
모델도 TP=3 을 못 한다**(`num_key_value_heads` 가 전부 8/4/2). 8 GPU 가 필요하다.
논증이 아니라 테스트로 박아뒀다 (`test_tp_anti_affinity.py` case 9).

---

## 3. 안 끝난 것

| | 상태 |
| --- | --- |
| **최종 프로토타입 비교** | ❌ **0/50런.** 아래 §5 가 실패 원인 |
| **70B 안정성 (sustained)** | ❌ 앞 단계 실패로 미실행 |
| **V6 KV 주입 (4단계)** | ❌ 여전히 막힘. 계측도 못 돌렸다 (§5) |
| TP 스윕 2런 | ❌ `strict/s1`, `paper/s3` — shm 잔재로 기동 실패 |
| Algorithm 1 마이그레이션 | ⚠️ 측정 런 전부 `migrations=0`. τ=0.02 arm 은 돌았으니 결과 확인할 것 |

---

## 4. 새 박스에서 반드시 다시 재라

**건너뛰면 모든 SLO 판정이 조용히 틀린다.** 이번 박스에서 실측한 차이:

| | 이전 박스 대비 |
| --- | --- |
| TTFT p95 기준선 | **-15 ~ -29 %** |
| `c_i` | **+11 ~ +23 %** |
| `τ` | 0.171086 (이전 박스 0.146, 커밋 기본값 0.35) |

절차:

```bash
git clone https://github.com/meojun/Prism-singleGPU-A100- /workspace/prism-exp
cd /workspace/prism-exp && git checkout exp/tp-v6-merge      # 병합본이 최신
echo 'HF_TOKEN=hf_...' > /workspace/.env && chmod 600 /workspace/.env
./setup/quickstart.sh

# quickstart 는 v2 패치만 적용한다. 체인을 마저 돌려라 -- 안 하면 flashinfer
# workspace 가 문자열로 읽혀 프로파일링이 6모델 전부에서 죽는다 (v4 패치에
# int() 캐스트가 있다).
for f in patches/paper_faithful/apply_patches.py patches/paper_faithful_v3/apply_v3.py \
         patches/paper_faithful_v4/apply_v4.py patches/paper_faithful_v5_2/apply_v5_2.py \
         patches/paper_faithful_v6/apply_v6.py patches/paper_faithful_tp/apply_tp.py; do
    python3 $f --repo /workspace/prism-exp/prism-research; done

./exp/scripts/run_profiling_v2.sh <outdir>          # 약 15분, 무경합 단독
# tau 전에 트레이스부터. calibrate_tau_v4.sh 는 gitignore 된 .pkl 을 전제한다.
python3 exp/scripts/build_paired_workload.py --rate 8 --duration 420 --seed 1 \
    --slo-base exp/configs/v2/slo_base.json --outdir exp/workloads/paper-faithful-v4
./exp/scripts/calibrate_tau_v4.sh <outdir>
```

토폴로지도 반드시 확인하고 기록해라. 이 박스는 4×A100-80GB 전 쌍 NV12
(NVSwitch), peer access 12쌍 전부 true 였다. PCIe-only 면 TP 수치가 무의미하다.

---

## 5. 최종 비교가 왜 0런으로 끝났나 — 반드시 먼저 고쳐라

**worktree 에 `prism-venv` 가 없었다.**

3-arm 비교는 저장소를 두 개 쓴다. `exp/scripts/env.sh` 가
`$PRISM_ROOT/prism-venv/bin/activate` 를 소스하는데 worktree 에는 venv 가 없어
시스템 python 이 쓰였고, `transformers` 가 없어 **워크로드가 0개** 생성됐다.
그래서 49런이 전부 "트레이스 없음" FAILED 로 찍히고 25초 만에 끝났다.

고친 방법(새 박스에서 다시 해야 한다):

```bash
ln -sfn /workspace/prism-exp/prism-venv     /workspace/prism-merge/prism-venv
ln -sfn /workspace/prism-exp/kvcached       /workspace/prism-merge/kvcached
ln -sfn /workspace/prism-exp/kvcached-prism /workspace/prism-merge/kvcached-prism
# prism-base 에도 동일하게
```

확인:

```bash
( cd /workspace/prism-merge && PRISM_ROOT=$PWD source exp/scripts/env.sh && \
  python3 -c "import transformers,torch;print('OK',torch.__version__)" )
```

**V6 검증도 별개 이유로 실패했다.** `run_v6_validation.sh` 가
`R=/workspace/prism-exp` 를 하드코딩하는데 이 박스에서 그 경로는 v6 패치가 없는
TP 브랜치라 `unknown system: paper-faithful-v6` 로 죽는다. 병합 트리용
`exp/scripts/run_v6_validation_merge.sh` 를 새로 뒀으니 그걸 써라.

---

## 6. 실험 인프라 (전부 커밋돼 있다)

| 스크립트 | 무엇 |
| --- | --- |
| `exp/scripts/run_tp_boot.sh` | TP=k 부팅/서빙 검증 |
| `exp/scripts/run_tp_serve.sh` | 위 + 정책 선택 (`TP_POLICY`) |
| `exp/scripts/run_tp_cycle.sh` | 활성화/비활성화 순환, 슬롯 반환 |
| `exp/scripts/run_tp_case.sh` / `2.sh` | 4단계 arm 하나 (실제 트레이스) |
| `exp/scripts/collect_tp_aa.py` | 사이클 raw + 요약 |
| `exp/scripts/build_report_tp.py` | TP REPORT + AUDIT 델타 생성 |
| `exp/scripts/compare_arms.py` | 3-arm 집계, 회귀 게이트, 개선율 |
| `exp/scripts/sustained_load.py` | open-loop 지속 부하 (70B) |
| `exp/scripts/run_70b_sustained.sh` | 70B Stage 3 |
| `exp/scripts/build_70b_report.py` | 70B 안정성 판정 |
| `exp/scripts/run_v6_validation_merge.sh` | v6 KV 검증 (병합 트리) |

supervisor 프로그램(`/opt/supervisor-scripts/`)은 **새 박스에서 다시 만들어야
한다.** 저장소 밖이다. `tp_final_compare.sh` 만
`exp/scripts/orchestrate_final_compare.sh` 로 복사해 뒀다.

---

## 7. 장비 함정 (이번에 새로 겪은 것)

`CLAUDE.md` §5/§8 목록에 더해:

* **worktree 에 venv 링크가 없으면 조용히 시스템 python 이 쓰인다.** 위 §5.
* **런 사이에 `/dev/shm` 이 안 치워지면 다음 서버가 기동에서 죽는다.**
  `Error: '/ipc_2_3_root'`. TP 스윕 2런이 이것으로 날아갔다. `reap()` 의
  대기 시간을 넉넉히 잡아라 (spawn 자식이 init 으로 재부모화되어 늦게 죽는다).
* **sglang 은 spawn 을 쓴다.** 부모를 죽여도 워커가 GPU 메모리를 쥔 채 남는다.
  `pgrep -f "prism-venv/bin/python3"` 로 잡아야 한다.
* **패치에서 편집을 지워도 이미 적용된 저장소는 되돌아가지 않는다.** probe 는
  멱등을 보장할 뿐 되돌리지 않는다. `git checkout -- python/` 후 체인 재적용.
* **상대 경로 `--model-config-file` 은 죽는다.** 서버가
  `benchmark/multi-model` 로 cd 한다.
* **요청 필드는 `model` 이지 `model_name` 이 아니다.** 잘못된 키는 200 OK 후
  디스패치되지 않아 TP 데드락처럼 보인다.
* **`ActivateReqInput.gpu_id` 는 기본값이 없다.** 빠뜨리면 FastAPI 422.

---

## 8. 다음 세션이 할 일 (권고 순서)

1. §4 재보정 + §5 venv 링크
2. TP 스윕 실패 2런 재실행 (30분) → 9/9
3. **최종 프로토타입 비교** — 실측 기준 런당 14.4분(중앙값), 50런 ≈ **12시간**
   * 줄이려면 steady 를 8 req/s 만 → ~9시간 (권장)
4. **70B 안정성** 3단계 게이트 (~1.3시간)
5. **V6 KV 주입** — 계측부터. 추측하지 마라, 이 항목에서 세 번 틀렸다.

세부 프롬프트는 `docs/handoff/TP_FINAL_HANDOFF_PROMPT.md`.
