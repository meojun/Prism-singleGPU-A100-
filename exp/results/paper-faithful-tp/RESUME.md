# 이어받는 지점 (2026-08-20 07:20 UTC / 16:20 KST 기준)

브랜치 `exp/paper-faithful-tp`, 기준 `exp/paper-faithful-v6 @ 66f5b39`. rebase 하지 않았다.

## 확인 먼저

```bash
supervisorctl status | grep tp_        # tp_commute2 가 RUNNING 이거나 EXITED
tail -40 /workspace/logs/tp_commute2.log
ls /workspace/logs/.tp*done            # 단계별 완료 스탬프
```

`tp_commute2` 는 스탬프 기반이라 재시작해도 끝난 단계를 다시 돌지 않는다.
특정 단계를 다시 돌리려면 해당 스탬프를 지우고 `supervisorctl start tp_commute2`.

| 스탬프 | 단계 |
| --- | --- |
| `.tp2_regression` | A) 회귀 테스트 |
| `.tp2_70b_tp2` | B) 70B TP=2 |
| `.tp2_cycle` | C) TP 활성화/비활성화 순환 |
| `.tp2_aa_off` / `.tp2_aa_on` | D) anti-affinity 라이브 스모크 |

## 끝난 것

| | 판정 | 근거 |
| --- | --- | --- |
| 0단계 환경·프로파일링·τ | 완료 | τ=0.171086 (이 박스). `calibration/tau.json` |
| 1+1b TP 슬롯 그룹 | **PASS** | `boot-tp2/`, `boot-tp2-g4/`, `boot-tp4-g4/` |
| 2 컨트롤러 멀티-GPU | 코드 완료 | 라이브 검증은 D) 에서 처음 이뤄진다 |
| 3 anti-affinity | 코드 + 단위테스트 | `exp/tests/test_tp_anti_affinity.py` |
| 5 70B TP=4 | **서빙 성공** | `serve-70b-tp4/` |

## 안 끝난 것

* **4단계 ON/OFF 측정 — 이것이 실제 산출물이고 미착수다.**
* `REPORT.md`, `IMPLEMENTATION_AUDIT.md` 갱신분 미작성.
* TP 모델의 마이그레이션(그룹 A → 그룹 B) 미검증. C) 는 같은 그룹 재활성화까지만 본다.
* TP=1 과 TP>1 혼재 + 실제 워크로드 미검증. 지금까지는 스모크 4요청뿐이다.

## 4단계를 설계할 때 반드시 반영할 것

측정으로 확인했다 (`test_tp_anti_affinity.py` case 2 대 case 4a):

* **균형 잡힌 클러스터에서 anti-affinity 는 구속하지 않는다.** 샤드 단위 argmin 이
  배치할 때마다 해당 GPU 에 부하를 물리므로 두 번째 샤드는 저절로 다른 GPU 로 간다.
  ON 과 OFF 가 같은 `(0,1)` 을 낸다.
* 구속하는 조건은 **한 GPU 가 나머지보다 압도적으로 한가할 때** 다. 샤드 하나를
  물려도 여전히 argmin 이면 두 번째 샤드도 같은 GPU 로 간다. 단위테스트의
  불균형 시나리오에서 `OFF -> (3,3)` (불법), `ON -> (3,1)`.
* 따라서 워크로드는 **GPU 하나를 드레인시키는 국면**을 포함해야 한다. 균등 부하로
  돌리면 `aa_violations = 0` 이 나오고, 그것은 "제약이 동작했다" 가 아니라
  "제약이 한 번도 구속하지 않았다" 로 보고해야 하는 결과다.
* 반사실은 두 arm 모두에서 기록된다: `[PAPER-ALG1-TP]` 의 `aa_violations`,
  `aa_diverted`, `aa_infeasible`. OFF arm 도 자기가 저지른 위반을 센다.

## 수치 해석 시 주의

`model_runner.py:133-136` 이 `tp_size > 1` 에서 model service 경로를 끈다
(런타임 로그의 `model_service=False` 로 확인). TP arm 은 V4 의 병렬 가중치
로딩도 P2P 마이그레이션도 쓰지 않는다. **TP arm 의 시간 수치를 non-TP arm 과
나란히 놓지 말 것.**

## 이 박스에서 재측정한 값 (커밋된 값은 다른 박스 것이다)

| | 이전 박스 대비 |
| --- | --- |
| TTFT p95 기준선 | **-15 ~ -29 %** |
| `c_i` | **+11 ~ +23 %** |
| `τ` | 0.171086 (이전 박스 0.146, 커밋 기본값 0.35) |

이 박스가 더 빠르다. 커밋된 `slo_base.json` 을 그대로 썼다면 SLO 문턱이
15~29% 느슨해져 goodput 이 과대평가됐을 것이다.

## 다른 브랜치와 합칠 때 보면 되는 곳

패치 앵커가 겹칠 수 있는 파일은 두 개뿐이다.

* `srt/server_args.py` — `keys_to_remove` 에 `enable_tp_worker_pool`,
  `tp_max_groups`, `enable_tp_anti_affinity` 추가
* `multi_model_server_args.py` — 위 세 플래그 선언 + `--policy` choices 에
  `kvpr-global-tp` 추가

`run_v4_case.sh` 는 건드리지 않았다 (별도 `run_tp_boot.sh` / `run_tp_serve.sh` /
`run_tp_cycle.sh` 를 신설했다). `simple_global.py`, `scheduler.py`,
`worker_pool_model_runner.py`, `model_sevice.py` 도 건드리지 않았다.

`patches/paper_faithful_tp/apply_tp.py` 의 `replace()` 는 앵커와 probe 의
유일성을 강제한다. 원본 `595ec1f` 에서 `v2 -> v3 -> v4 -> v5_2 -> tp` 전체를
엄격 모드로 재적용해 검증했다.
