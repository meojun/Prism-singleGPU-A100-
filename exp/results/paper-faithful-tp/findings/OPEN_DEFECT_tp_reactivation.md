# 해결된 결함 — TP 모델이 재활성화되지 못하고 요청이 쌓였다

발견: 4단계 첫 런 (`step4/tp-aa-off_r8_s1`), 2026-08-20.
상태: **해결 및 재검증 완료.** 아래 증상과 원인은 수정 전 기록으로 보존한다.

## 해결

컨트롤러가 샤드별 목적지를 고른 다음 `_snap_to_group` 으로 실행 가능한 유휴 TP
그룹에 스냅하고, 그 그룹을 실제로 소유한 rank 0 워커에게 activate 를 보낸다.
초기 그룹 소유권도 GPU 0에 몰리지 않도록 rank 0을 그룹 전체에 분산했다. 후자가
없으면 GPU 3은 어떤 그룹도 소유하지 않아 스냅 후에도 재활성화가 반복 거절됐다.

수정 후 최종 4단계는 9/9런 모두 끝났다. 특히 이전에 적체됐던 arm은 다음과 같다.

* `strict/s1`: 3372/3372 완료, 87 사이클, 반사실 위반/우회 25/25
* `paper/s3`: 3372/3372 완료, 88 사이클, 반사실 위반/우회 23/23

근거는 `step4-final/`, `REPORT.md`, `raw/placements/`에 있다.

## 증상

벤치마크가 `BENCHMARK_TIMEOUT` 1800초에 잘린다 (`benchmark_rc=124`). 대기 중인
작업 **1672건이 전부 `model_1`** 이고, model_1 은 이 구성의 **유일한 TP=2 모델**
이다. 나머지 5개 TP=1 모델은 정상 완료된다.

```
$ grep -oE "Waiting for task .*_(model_[0-9])" bench.log | grep -oE "model_[0-9]$" | sort | uniq -c
   1672 model_1
```

런 종료 시점에 전 GPU 스케줄러가 `model_1: 'deactivated'` 를 들고 있다.

## 원인

컨트롤러는 재활성화를 시도한다:

```
ACTION: activate inactive model model_1 on GPU 1. Reason: inactive models but with requests
```

그런데 워커풀이 거절한다:

```
No idle tp_size=2 worker for GPU 1; free=[...] {1: 1}
```

`_greedy_placement_tp` 는 KVPR 만 보고 샤드의 GPU 를 고르고, **그 GPU 가 비어
있는 TP 그룹을 rank0 로 소유하는지 확인하지 않는다.** 소유하지 않거나 그룹이 이미
쓰이고 있으면 activate 가 실패하고, 실패한 활성화는 다시 시도되지 않은 채 모델이
내려가 있는다. TP=1 모델에는 이 문제가 없다 — 모든 GPU 가 자기 TP=1 슬롯을 갖기
때문이다.

관련해서 배치와 emit 사이에 그룹 정보가 사라지는 것도 같은 뿌리다.
`_find_optimal_migrations` 는 부모 계약상 `(name, src_gpu, dst_gpu)` 단일 GPU 쌍을
반환하므로, 컨트롤러가 emit 하는 것은 "GPU 1 로 옮겨라" 이지 "그룹 (1,3) 으로
옮겨라" 가 아니다.

## 이 런에서도 유효한 것

이 결함에도 불구하고 아래는 실측으로 남는다.

* **TP 그룹 마이그레이션은 동작한다.** 스케줄러가 자율적으로 그룹을 바꿨다:
  `[2,3] slot=7` → `[1,2] slot=5` → `[1,3] slot=6` → `[2,3] slot=7`.
  이전까지 미검증으로 적어둔 항목이다.
* **anti-affinity 반사실 데이터는 온전하다.** 348 사이클, 위반 23회, 충돌 배치
  282건. 계획 단계의 기록이므로 요청 타임아웃과 독립이다.

## 당시 기록한 수정 방향 (현재 적용됨)

샤드의 GPU 후보를 **비어 있는 그룹을 소유한 GPU** 로 제한하거나, 배치가 끝난 뒤
선택된 GPU 집합을 실제 그룹으로 매핑하고 그 그룹의 owner 로 activate 를 보내야
한다. 후자가 부모 계약을 덜 건드린다.

`findings/pre-fix/`의 수정 전 요청/SLO 수치는 쓰면 안 된다. TP 모델이 대부분의
시간 내려가 있었으므로 그 goodput은 anti-affinity가 아니라 이 결함을 잰다. 최종
수치는 `step4-final/`만 사용한다.
