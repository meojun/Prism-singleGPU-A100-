# 돌아왔을 때 여기부터 — Paper-Faithful Prism V6

이 파일이 **현재 상태**다. 연구 전반의 ground truth 는 `HANDOVER.md`.
둘이 어긋나면 **코드와 raw data 를 우선**해라 — 실제로 이번에 그런 모순이 여러 건 나왔다.

---

## 0. 30초 상황 파악

```bash
cd /workspace/prism-exp && git log --oneline -8
tmux ls                                    # 남아있는 세션 (전부 완료된 것들)
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
```

브랜치는 `exp/paper-faithful-v6`, 최신 커밋 `9faf7b8`. 전부 푸시돼 있다.

---

## 1. 끝난 것

| | 상태 |
| --- | --- |
| **A** 환경 부트스트랩 + 이 박스 재보정 | ✅ SLO 기준선, `c_i`, τ=0.15992 |
| **C** 감사 정정 3건 + P2P FULL 승격 | ✅ |
| 대조군 스윕 (v4, 6런) | ✅ 6/6 실패 0, raw 99파일 보존 |
| **B** KV migration 1~3단계 | ✅ preempt → 캡처 → 서비스까지 전달 |
| **B** KV migration 4단계 (주입) | ❌ **여기서 막혀 있다** |

### 이 프로젝트가 처음으로 얻은 수치

`kv_bytes` 가 모든 arm 에서 하드코딩 `0` 이던 것이 **실측값**이 됐다.
한 번의 마이그레이션에서 최대 **89요청 / 40,066토큰 / 4.6 GB**.
그게 released prototype 이 매번 버리고 프롬프트부터 재계산하는 양이다.

### 대조군 스윕 결과 (v6 와 비교할 문턱)

| workload | n | goodput | seed 별 |
| --- | ---: | ---: | --- |
| bursty 8 | 3 | 4.940 ± 0.846 | 5.53, 5.54, **3.74** |
| steady 8 | 3 | 3.579 ± 0.221 | 3.80, 3.28, 3.66 |

sd 가 평균의 17% 다. **n=3 으로는 20% 미만 차이를 주장할 수 없다.**
집계기가 그 판정을 자동으로 한다 (spread 안이면 `INDISTINGUISHABLE`).

이 숫자를 **v4 리포트의 1.30 과 나란히 놓지 마라** — 박스도, P2P 설정도,
트레이스도 다르다. 근거는 `exp/results/paper-faithful-v6/sweep/README.md`.

---

## 2. 다음에 할 일 — 정확히 하나

### 2.1 먼저: engine_id 계측 (2줄, 약 15분)

**추측하지 마라. 오늘 세 번 틀렸다.**

증상은 이렇게 좁혀져 있다:

```
09:31:13.832  서비스: fetch Qwen2.5-7B -> 2 requests    ← 큐에 넣음
09:31:18.836  엔진:   fetch timed out (5초)             ← 못 받음
```

엔진은 **진짜 큐를 기다리고 있었다** — `kv_replies` 가 없었으면 로그 없이
조기 반환했을 것이다. 즉 양쪽 다 큐를 쥐고 있는데 **같은 큐가 아니다.**

코드만 읽어서는 안 갈린다. 양쪽 다 `mr.engine_id` 를 쓴다고 되어 있다.
그러니 찍어봐라:

* 서비스 `model_sevice.py` 의 `__kv_fetch__` 핸들러 —
  `self.v6_kv_queues[gpu_model].put(caps)` 직전에
  `gpu_model` 값과 `list(self.v6_kv_queues.keys())` 를 로그
* 엔진 `scheduler.py:1867` 근처 —
  `mr.engine_id` 와 `list(getattr(q, "kv_replies", {}).keys())` 를 로그

한 번 돌리면(약 10분) 어느 키가 어긋나는지 확정된다.

### 2.2 그 다음: 원인에 맞춰 수정 + 검증

키 불일치면 몇 줄이다. 구조 문제면 그때 범위를 다시 잡아라.
검증은 `./exp/scripts/run_v6_validation.sh` (약 10분).

**통과 조건**: `inject > 0`, `kv_transfers.jsonl` 존재하고 `kv_bytes > 0`,
요청 실패 0, `fetch timed out` 0건, 그리고 끝까지 완주.

### 2.3 별도로 설명해야 할 것

완료 요청이 **3387 → 1296** 으로 떨어졌다. 같은 설정인데 타임아웃은 오히려
줄었다(30초×4 → 5초×2). **타임아웃으로 설명되지 않는 방향이다.**
주입 문제와 무관할 수 있으니 따로 봐라.

### 2.4 주입이 되면

1. v6 스윕: `SWEEP_ARMS=paper-faithful-v6 ./exp/scripts/run_v6_sweep.sh`
   (6런 약 1시간, 무인, 끝나면 스스로 집계·커밋·푸시).
   대조군은 이미 있으니 바로 비교된다.
2. `IMPLEMENTATION_AUDIT.md` 의 KV-cache migration 행 갱신 —
   **`exp/scripts/build_report_v4.py` 생성기를 고쳐라.** 표만 손으로 고치면
   다음 생성 때 되돌아간다.
3. `HANDOVER.md` §3.3 의 "KV migration 미구현" 갱신.

---

## 3. 이미 실패한 설계 — 다시 하지 마라

전달 채널을 **세 번** 시도했다.

| 설계 | 결과 | 근거 |
| --- | --- | --- |
| 엔진 `output_queue` 에 답신 | ❌ 경합 — 가중치 로더가 캡슐을 가져감, 매번 30초 타임아웃 | `e2e/FINDING.md` |
| 핸드셰이크에 4번째 메시지 | ❌ **런이 행** — 원인 끝내 미확정 | `e2e/attempt-4th-message/FINDING.md` |
| **전용 per-engine 큐** (현재) | 행은 해결, 주입은 여전히 안 됨 | 위 2.1 |

**확정된 제약**: 같은 큐에 두 리더 ❌, 가중치 핸드셰이크 공유 ❌.

**확정된 사실**: 캡슐은 프로세스 간 전송이 문제없다. 자식이 부모의 캡슐
바이트를 정확히 읽고, 부모가 참조를 버리고 풀을 해제한 뒤에도 그렇다.
`e2e/attempt-4th-message/ipc_capsule_probe.py` 가 그걸 증명한다.
**전송은 용의선상에서 빠졌다.**

기각된 가설도 적어둔다 (다시 파지 마라):
* 행 = 메시지 개수 불일치 → put/get 전수 감사, 양쪽 경로 균형
* 행 = IPC use-after-free → torch 가 프로세스 간 참조카운트를 검 + probe 로 반증

---

## 4. 새 박스로 옮긴다면 — 반드시 다시 재라

**건너뛰면 모든 SLO 판정이 조용히 틀린다.** 이 프로젝트에서 반복적으로 사고를
낸 항목이다.

```bash
git clone https://github.com/meojun/Prism-singleGPU-A100- /workspace/prism-exp
cd /workspace/prism-exp && git checkout exp/paper-faithful-v6
echo 'HF_TOKEN=hf_xxx' > /workspace/.env && chmod 600 /workspace/.env
./setup/quickstart.sh                                       # 약 35분

# 패치 체인 (순서 지켜라, 멱등적)
for f in patches/paper_faithful/apply_patches.py patches/paper_faithful_v3/apply_v3.py \
         patches/paper_faithful_v4/apply_v4.py patches/paper_faithful_v5_2/apply_v5_2.py \
         patches/paper_faithful_v6/apply_v6.py; do
    python3 $f --repo /workspace/prism-exp/prism-research; done

./exp/scripts/run_profiling_v2.sh exp/results/<study>/profiling   # 약 40분
# tau 전에 트레이스부터! (아래 5번)
./exp/scripts/calibrate_tau_v4.sh exp/results/<study>/calibration
```

**`exp/configs/v2/slo_base.json` 과 `prefill_speed.json` 을 커밋하지 마라.**
추적되는 공유 경로라, 두 박스가 각자 값을 커밋하면 **어느 쪽도 맞지 않는
충돌**이 난다. 이 박스는 값을 `exp/results/paper-faithful-v6/profiling/*_this_box.json`
에 두고 `SLO_BASE_FILE` / `PREFILL_SPEED_FILE` 로 가리킨다. 같은 방식을 써라.

P2P 여부는 박스마다 다르다. 반드시 확인하고 기록해라:

```bash
nvidia-smi topo -m          # NV# 이면 NVLink
python3 -c "import torch; print({f'{i}->{j}': torch.cuda.can_device_access_peer(i,j)
            for i in range(torch.cuda.device_count())
            for j in range(torch.cuda.device_count()) if i!=j})"
```

---

## 5. 이 세션에서 새로 겪은 함정

CLAUDE.md §5/§8 목록에 더해서:

* **`calibrate_tau_v4.sh` 는 단독 실행이 안 된다.** gitignore 된 `.pkl` 트레이스를
  전제하고 그건 `run_pipeline_v4.sh` STAGE 5 가 만든다. 먼저:
  ```bash
  python3 exp/scripts/build_paired_workload.py --rate 8 --duration 420 --seed 1 \
      --slo-base <이 박스의 slo_base.json> --outdir exp/workloads/paper-faithful-v4
  ```
  **반드시 그 박스의 slo_base 로.** 트레이스에 슬롯별 SLO 가 박힌다.

* **플래그를 `MultiModelServerArgs` 에만 추가하면 안 된다.**
  `srt/server_args.py` 의 `keys_to_remove` 집합에도 넣어야 한다. 안 넣으면
  `ServerArgs.__init__() got an unexpected keyword argument` 로
  **그 플래그를 쓰지 않는 arm 까지 전부** 기동에서 죽는다.

* **런 사이에 `/dev/shm` 을 치워라.** `exp/scripts/shm_clean.sh` 가 한다.
  안 치우면 다음 서버가 기동 중에 죽는다 (`resource_tracker: process died
  unexpectedly` + 세그먼트 이름 KeyError). 스윕은 매 런 사이에 호출한다.

* **패치 앵커는 파일에서 유일해야 한다.** 오늘 세 번 당했다 —
  `self.tp_worker.deactivate_model_runner()` 의 첫 매치가
  `handle_deactivate_request` 가 아니라 `__init__` 이었고,
  `put(t1-t0)` + `put(service_id)` 쌍이 다른 브랜치에도 있었다.
  `probe` 문자열도 그 삽입에만 고유해야 한다 — 같은 패치의 다른 부분이 쓴
  문자열을 probe 로 쓰면 조용히 건너뛴다.

* **패치 검증은 "다시 돌려서 성공 메시지 보기" 가 아니다.** 멱등성 probe 때문에
  바뀐 블록이 건너뛰어진다. **핀 고정 원본을 리셋하고 체인 전체를 재생한 뒤
  줄 번호로 확인해라:**
  ```bash
  (cd prism-research && git checkout -- python/)
  for f in patches/*/apply*.py; do python3 $f --repo $PWD/prism-research; done
  grep -n "<네가 넣은 표시>" prism-research/python/.../파일.py
  ```

* kvcached 는 **프로세스당 전역 allocator 하나**다. 한 프로세스에서 두 풀을
  만들면 먼저 것이 무효화되고, 살아남은 쪽은 용량을 보고하면서 `alloc()` 이
  None 을 반환한다. 조용해서 찾기 어렵다.

* kvcached 는 종료 시 `cuMemRelease` 실패로 프로세스를 abort 시킨다. 판정이
  끝난 뒤라 결과에는 영향이 없지만 종료 코드를 덮는다.

---

## 6. 병행 중인 다른 작업 — TP

4×A100 박스에서 **TP + anti-affinity** 를 별도 세션이 구현 중이다
(`exp/paper-faithful-tp`). 프롬프트는 `docs/handoff/TP_HANDOFF_PROMPT.md`.

**겹치는 파일 세 개** — 합칠 때 여기만 보면 된다:
* `srt/server_args.py` 의 `keys_to_remove`
* `multi_model_server_args.py` (플래그 선언)
* `exp/scripts/run_v4_case.sh` (arm 추가)

합칠 때는 **깨끗한 핀 고정 원본에 체인을 처음부터** 돌려서 앵커를 확인해라.
