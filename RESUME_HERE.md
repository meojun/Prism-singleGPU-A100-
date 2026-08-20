# 돌아왔을 때 여기부터 — 2×A100 박스, V6

이 파일은 세션이 끊긴 뒤 다시 시작할 때 **가장 먼저 읽는 것**이다.
연구 전반의 ground truth 는 `HANDOVER.md`, 이 파일은 *지금 이 순간* 무엇이
어디까지 갔는지만 적는다.

## 1. 30초 안에 상황 파악하기

```bash
cd /workspace/prism-exp
git log --oneline -8
tmux ls                                   # 살아있는 작업
tail -40 /workspace/logs/stage_b_sweep.log
cat exp/results/paper-faithful-v6/sweep/SUMMARY.txt   # 있으면 스윕이 끝난 것
```

스윕은 끝나면 **스스로 집계하고 커밋하고 푸시한다.** 그러므로:

* `git log` 에 `v6 sweep:` 커밋이 있으면 → 끝났다. SUMMARY.txt 를 읽어라.
* `tmux ls` 에 `stage_b` 가 있고 커밋이 없으면 → 아직 돌고 있다. 기다려라.
* 둘 다 없으면 → 중간에 죽었다. 아래 3번.

## 2. 여기까지 끝난 것

| | 상태 |
| --- | --- |
| A — 이 박스 재보정 (SLO 기준선, c_i, tau) | ✅ 완료, 커밋됨 |
| C — 감사 정정 3건 + P2P FULL 승격 | ✅ 완료, 커밋됨 |
| B — KV migration 모듈 + 25개 단위테스트 | ✅ 통과 |
| B — 배선 패치 (5개 지점, opt-in) | ✅ 적용, 체인 멱등 확인 |
| B — e2e 검증: v4 대조군 | ✅ 통과 — 요청 3387, 가중치 전송 24 (P2P 7), KV 마커 0 |
| B — e2e 검증: v6 처리군 | ⚠️ **부분 성공** — 캡처 O, 전달 X. `e2e/FINDING.md` |
| B — 스윕 (효과가 있는가) | ⛔ **던지지 않았다.** 아래 2.1 |

**환경 값은 절대 다른 박스 것을 쓰지 마라.** 이 박스 값은
`exp/results/paper-faithful-v6/profiling/*_this_box.json` 이고
`SLO_BASE_FILE` / `PREFILL_SPEED_FILE` 로 가리킨다.
`exp/configs/v2/` 의 추적본은 **다른 박스 값이므로 건드리지 않았다.**
이유는 `exp/results/paper-faithful-v6/provenance/ENVIRONMENT.md`.

대조군의 `stash=0 inject=0` 은 그 자체가 결과다 — **플래그가 꺼져 있으면 v6 코드
경로에 진입조차 하지 않는다**는 증거이고, 기본 arm 이 released prototype 을 그대로
재현한다는 불변식이 데이터로 확인된 것이다. 그리고 그 런의 가중치 전송 24건 중
7건이 gpu-to-gpu 였다 — V4 의 P2P 가 이 박스에서 마이크로벤치가 아니라 **실부하
e2e** 로 동작한다.

### 2.1 다음에 할 일은 이것 하나다 — 전달 채널 수정

**스윕을 먼저 돌리지 마라.** 지금 상태로 돌리면 오염된 숫자가 나온다.

무엇이 되고 무엇이 안 되는지는 `exp/results/paper-faithful-v6/e2e/FINDING.md`
에 근거와 함께 있다. 요약:

* **된다** — preempt → retract → KV 캡처 → 바이트 계량. stash 6회, capture
  실패 0회, 최대 89요청 / 40,066토큰 / 4.6 GB. `kv_bytes` 가 하드코딩 0 이
  아니라 실측값이 됐다.
* **안 된다** — 엔진이 캡슐을 돌려받지 못한다. 서비스는 정상적으로 돌려주는데
  (`fetch ... -> 4 requests`) 엔진은 30초 뒤 타임아웃한다.
* **원인** — 내가 응답을 엔진의 `output_queue` 에 얹었는데, 그건 **가중치 로딩
  핸드셰이크가 이미 쓰는 채널**이다 (`worker_pool_model_runner.py:277-279,
  359-367` 이 활성화마다 3번 블로킹 `get()`). `activate_async` 에서는 그 로딩이
  별도 스레드라 두 리더가 경합하고, 로더가 캡슐을 가져가 버린다.
  v4 의 `__release__` 는 응답이 없는 단방향이라 채널 공유가 안전했다.
  요청/응답은 다르다.

**수정 방향** — 전용 응답 큐를 `input_queue` / `output_queue` 와 같은 방식으로
관통시켜라: `launch_worker_pool_engines` → `launch_engine` →
`run_scheduler_process` → `ModelRunner`, 그리고 서비스 쪽에 대응 dict.
그러면 기존 프로토콜을 건드리지 않는다.
같이 할 것: 타임아웃 30초는 너무 길다. 정상 응답이 그렇게 걸릴 이유가 없고,
실패 시 활성화가 30초 멈추는 것이 측정을 오염시킨다.

**하지 말 것** — fetch 를 non-blocking 으로 바꾸고 두 채널을 폴링하는 것은
경합을 줄일 뿐 없애지 못한다. 가중치 응답에 캡슐을 얹는 것은 독립적으로
실패하는 두 메커니즘을 묶는다.

**수정 후 절차**: 단위테스트 → v6 단독 검증(`exp/scripts/run_v6_validation.sh`,
약 10분) → `inject > 0` 과 `kv_transfers.jsonl` 존재 확인 → 그 다음에 스윕.

## 3. 스윕이 중간에 죽었다면

**resumable 이다.** 끝난 런은 `DONE` 마커가 있어 건너뛴다. 그대로 다시 던져라:

```bash
tmux new-session -d -s stage_b \
  "bash -lc 'ulimit -n 65535; /workspace/prism-exp/exp/scripts/run_v6_sweep.sh 2>&1 | tee -a /workspace/logs/stage_b_sweep.log; sleep infinity'"
```

(`/workspace/stage_b_sweep.sh` 도 같은 내용이지만 레포 안의 것을 써라.)

죽은 런의 잔해부터 치워야 다음 서버가 뜬다 (실제로 겪은 함정):

```bash
ps -eo pid,cmd | grep -E "launch_multi_model_server|benchmark.py" | grep -v grep
rm -f /dev/shm/ipc_*          # 엔진이 만드는 이름은 ipc_<gpu>_<worker>_root
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
```

## 4. 결과를 읽는 규칙

`SUMMARY.txt` 는 두 질문을 **분리해서** 답한다. 순서를 지켜 읽어라.

1. **발동했는가** — `stash` / `inject` / KV MiB / 실패 요청.
   KV 가 0 이면 두 arm 은 사실상 같은 시스템이고, 그 아래 goodput 표는
   아무 의미가 없다. 집계기가 그 경우 비교를 거부하도록 되어 있다.
2. **효과가 있는가** — seed 별 값과 spread 를 함께 본다.
   v4 연구에서 이 유입률의 seed 간 분산이 평균의 70~80% 였다.
   **평균 차이가 spread 안에 들어오면 결과가 아니다.**

## 5. 다음에 할 일 (권고 순서)

1. **전달 채널 수정** (§2.1). 이것 말고 다른 것을 먼저 하지 마라.
2. 수정 후 검증이 통과하면 — `IMPLEMENTATION_AUDIT.md` 의 KV-cache migration 행을
   `NOT IMPLEMENTED` 에서 갱신하고 근거를 raw data 로 가리켜라.
   `build_report_v4.py` 가 그 표를 생성하므로 **생성기를 고쳐야 한다.**
   손으로 표만 고치면 다음 생성 때 되돌아간다.
3. **HANDOVER.md 갱신** — §3.3 의 "KV migration 미구현" 항목.
4. TP 브랜치와의 머지는 사용자가 따로 물어보기로 했다. 먼저 하지 마라.

## 6. 병행 중인 다른 작업

4×A100 박스에서 **TP + anti-affinity** 를 별도 세션이 구현 중이다
(`exp/paper-faithful-tp`, `exp/paper-faithful-v6` @ 66f5b39 에서 분기).
프롬프트는 `docs/handoff/TP_HANDOFF_PROMPT.md`.

**그쪽 구역을 건드리지 마라:** `multi_model_server.py`, `worker_pool.py`,
`resource_manager.py`, `controller_global.py`, `model_runner.py`,
`kvpr_global*.py`.

접점은 **패치 앵커**다. 두 브랜치가 같은 파일의 같은 근처를 치면 나중에
적용되는 쪽이 `anchor not found` 로 죽는다. 합칠 때 체인을 **핀 고정 원본의
깨끗한 사본에 처음부터** 돌려서 확인해라 — 이미 패치된 트리에 재실행하면
멱등성 probe 때문에 바뀐 블록이 조용히 건너뛰어진다 (이번에 실제로 당했다).

## 7. 이 세션에서 새로 겪은 함정

* `--enable-kv-migration` 같은 필드를 `MultiModelServerArgs` 에 추가하면
  `srt/server_args.py` 의 `keys_to_remove` 에도 **반드시** 넣어야 한다.
  안 넣으면 `ServerArgs.__init__() got an unexpected keyword argument` 로
  **플래그를 쓰지 않는 arm 까지 전부** 기동에서 죽는다.
* `calibrate_tau_v4.sh` 는 단독 실행이 안 된다. gitignore 된 `.pkl` 트레이스를
  전제하며 그건 `run_pipeline_v4.sh` STAGE 5 가 만든다. `build_paired_workload.py`
  로 먼저 만들어라 — **이 박스의 slo_base 로**.
* kvcached 는 프로세스당 전역 allocator 하나다. 한 프로세스에서 두 풀을 만들면
  먼저 것이 무효화되고, 살아남은 쪽은 용량을 보고하면서 `alloc()` 이 None 을
  반환한다. 조용해서 찾기 어렵다.
* kvcached 는 종료 시 `cuMemRelease` 실패로 프로세스를 abort 시킨다. 판정이 다
  끝난 뒤라 결과에는 영향이 없지만 종료 코드를 덮는다.
* **런 사이에 `/dev/shm` 을 반드시 치워라.** 끝난 서버가 `ipc_<gpu>_<worker>_root`
  와 multiprocessing 의 `sem.mp-*` / `mp-*` 를 남기고, 다음 서버가 그것과 충돌해
  기동 중에 죽는다 (`resource_tracker: process died unexpectedly` + 세그먼트
  이름 KeyError). v6 검증이 실제로 이렇게 죽었고, 바로 옆에서 v4 대조군은
  깨끗하게 끝났다 — **v6 코드와 무관하고 순서만 반대였으면 v4 가 죽었다.**
  `exp/scripts/shm_clean.sh` 가 이걸 한다. 스윕은 매 런 사이에 호출한다.
