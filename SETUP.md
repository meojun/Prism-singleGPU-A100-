# 새 GPU 서버에 그대로 세팅하기

이 저장소는 **환경을 통째로 담고 있지 않다.** venv(약 15 GB)와 모델 가중치(약 21 GB)는
git에 넣을 물건이 아니다. 대신 **그것들을 똑같이 다시 만드는 데 필요한 것 전부**를 담는다:
upstream 커밋 SHA 고정, 전체 의존성 lockfile, 프로파일 테이블, 실험 스크립트, 그리고
검증된 기준 결과.

## 한 줄 요약

```bash
git clone https://github.com/meojun/Prism-singleGPU-A100- prism-exp
cd prism-exp
echo 'HF_TOKEN=hf_xxx' >> /workspace/.env     # Llama는 gated 모델
./bootstrap.sh
```

A100-80G 기준 약 15~25분 (대부분 다운로드 시간). 끝나면 바로:

```bash
source exp/scripts/env.sh
./exp/scripts/run_sanity.sh A     # 이어서 B, C
python exp/scripts/summarize_sanity.py
```

나온 숫자를 `exp/results/sanity/REPORT.md` §3의 표와 비교하면 새 장비가 제대로
돌아가는지 확인된다.

---

## bootstrap.sh가 하는 일

| 단계 | 내용 |
| --- | --- |
| 0 | preflight — GPU 확인, compute capability 검사, `uv` 없으면 설치 |
| 1 | `prism-research`, `kvcached-prism`, `kvcached`를 **고정 SHA**로 clone (`setup/pins.env`) |
| 2 | Python 3.10 venv 생성 |
| 3 | torch 2.4.0+cu121 (PyTorch 전용 index) |
| 4 | flashinfer 0.1.6 (flashinfer 전용 index) |
| 5 | 나머지 전부를 `setup/requirements.lock.txt`에서 **정확한 핀**으로 설치 |
| 6 | `prism-research/python`, `kvcached-prism`를 editable로 설치 + C++ 확장 빌드 |
| 7 | 프로파일된 `model_info.json` 복사 (Llama/Mistral + 우리가 추가한 Qwen2.5) |
| 8 | redis 기동 (127.0.0.1:6379) |
| — | 모델 가중치 다운로드, 그리고 import/버전/CUDA 검증 |

**멱등하다.** 중간에 실패해도 그냥 다시 실행하면 된다. 이미 된 단계는 건너뛴다.

옵션:

```bash
SKIP_MODELS=1 ./bootstrap.sh          # 가중치 다운로드 생략
SKIP_KVCACHED_MAIN=1 ./bootstrap.sh   # 독립 kvcached(main) clone 생략
ROOT=/data/prism ./bootstrap.sh       # 다른 경로에 설치
```

---

## 왜 `setup_prism_env.sh`를 그대로 안 쓰는가

원래 셋업 스크립트(참고용으로 저장소에 남겨둠)는 **재현되지 않는다.** 매번 의존성을
새로 resolve하는데, 그러면 다음 함정들을 매번 다시 밟는다. `setup_prism_env.log`를 보면
그 스크립트가 실제로 `transformers 5.14.1`과 vLLM import 실패로 끝났다는 걸 알 수 있다 —
그 뒤에 수동으로 고친 것이다. `bootstrap.sh`는 그 수정들이 이미 반영된 상태를 설치한다.

### 함정 (전부 lockfile로 해결됨)

1. **transformers.** `python[all]` extra에 상한이 없어서 새로 resolve하면 5.x가 딸려오고,
   vLLM 0.6.3.post1이 `ImportError: cannot import name 'DTensor'`로 죽는다. → 4.45.2 고정.
2. **pyairports.** 2.1.1이 PyPI에서 내려갔다(빈 0.0.1 placeholder만 남음). 그런데 vLLM
   0.6.3.post1이 `outlines<0.1`을 핀하고 outlines가 이걸 import한다. → lockfile이 git URL로
   직접 지정.
3. **setuptools.** `pkg_resources`를 아직 쓰므로 <81 필요. → 80.10.2 고정.
4. **lockfile은 내부적으로 일관되지 않다.** litellm 1.95.0은 `tokenizers>=0.21`을 요구하는데
   transformers 4.45.2는 `tokenizers 0.20.3`을 핀한다. 그래서 **어떤 resolver로도 풀리지
   않는다** (`No solution found`). 실제로는 무해하다 — litellm은
   `sglang/lang/backend/litellm.py`에서만 닿고 multi-model 서버는 그 경로를 절대 import하지
   않는다. `bootstrap.sh`가 `uv pip install --no-deps`를 쓰는 이유가 이것이다. 최적화가
   아니라 **필수**다. `--no-deps`를 빼면 5단계에서 실패한다.
5. **docker 불필요.** upstream의 `install.md`는 `lmsysorg/sglang:v0.3.4.post2-cu121` 이미지를
   전제하는데 Vast 컨테이너는 docker-in-docker가 안 된다. 전부 네이티브로 설치한다.

---

## GPU 호환성 — 읽고 넘어갈 것

이 스택은 **torch 2.4.0+cu121**에 고정되어 있다. 이건 GPU 아키텍처에 대한 제약이지
드라이버 버전에 대한 제약이 아니다.

| GPU | compute cap | 상태 |
| --- | --- | --- |
| A100 | 8.0 | ✅ 검증됨 (이 저장소의 결과가 나온 환경) |
| H100 | 9.0 | ✅ 동작해야 함 (cu121 커널 존재) |
| L40S / A6000 | 8.9 | ✅ 동작해야 함 |
| **B200 / RTX 50xx** | **10.0+** | ❌ **안 됨** |

Blackwell(cc ≥ 10.0)은 CUDA ≥ 12.8 빌드가 필요하다. cu121 휠은 **설치는 깨끗하게 되고**
첫 GPU 연산에서 `no kernel image is available for execution on the device`로 죽는다.
lockfile은 여기서 아무 도움이 안 된다 — torch/vllm/flashinfer 조합을 통째로 다시 풀어야
하고, 그러면 위의 함정들을 전부 다시 상대해야 한다. `bootstrap.sh`가 preflight에서 이걸
감지하고 경고 후 확인을 받는다.

드라이버 버전은 걱정할 필요 없다. CUDA minor-version compatibility 덕분에 CUDA 12.0+를
지원하는 드라이버면 12.x 휠이 전부 돈다. **호스트 드라이버를 절대 apt로 업그레이드하지 말 것.**

---

## 저장소에 들어있는 것 / 없는 것

**있음**

| 경로 | 내용 |
| --- | --- |
| `bootstrap.sh` | 원샷 재구축 |
| `setup/pins.env` | upstream 3개 저장소의 고정 커밋 SHA |
| `setup/requirements.lock.txt` | 154개 패키지 전체 freeze (검증된 조합) |
| `setup/model_info.json` | 프로파일 테이블 28개 엔트리 (Llama/Mistral + 우리가 추가한 Qwen2.5) |
| `setup/download_models.sh` | 가중치 다운로드 |
| `exp/configs/` | 모델 배치 config (Llama 3종 + Qwen 계열) |
| `exp/scripts/` | launch / bench / analyze / summarize |
| `exp/results/sanity/` | 기준 결과 + `REPORT.md` (한국어 전문 보고서) |
| `setup_prism_env.sh` | 원래 셋업 스크립트 (역사적 참고용, 재현 불가) |

**없음** (`.gitignore`)

- `prism-venv/`, `sglang-pip-venv/` — bootstrap이 재생성
- `prism-research/`, `kvcached/`, `kvcached-prism/` — bootstrap이 고정 SHA로 clone.
  별도 git 저장소라서 커밋하면 깨진 gitlink가 된다
- 모델 가중치 — `$HF_HOME`에 있고 `setup/download_models.sh`가 받는다
- `exp/server-logs/` — 실행할 때마다 재생성
- 탐색용 대용량 결과 (`exp/results/requests/`, `*_all.jsonl`) — sanity 결과는 커밋됨

---

## 새 장비에서 확인할 것

`bootstrap.sh`의 검증 단계가 버전과 CUDA 가용성을 찍는다. 그 다음:

1. `./exp/scripts/run_sanity.sh A` — 케이스 A는 **attainment 1.000**이 나와야 한다.
   안 나오면 환경 문제이지 부하 문제가 아니다 (8B 하나가 80 GB를 독점하는 상황).
2. `exp/results/sanity/REPORT.md` §3과 비교. 절대 latency는 GPU가 다르면 달라지지만,
   B의 TPOT 붕괴(colocation contention)와 C의 높은 attainment라는 **패턴**은 유지되어야 한다.
3. 서버가 `activating`에서 멈추면 → redis 확인, 그리고 `--workers-per-gpu`가 그 GPU에
   올라간 `on: true` 모델 수 이상인지 확인 (`exp/server-logs/*/server.log.gpu_scheduler.log`를
   볼 것. 최상위 로그는 이 에러를 삼킨다).

---

## 이 스택을 바꾸게 되면

`setup/pins.env`의 SHA를 올리거나 패키지를 바꾸면 lockfile이 무효가 된다. 다시 만들려면:

```bash
source prism-venv/bin/activate
uv pip freeze | grep -vE '^-e file://' | grep -vE '^(torch|torchvision|flashinfer)==' \
  > setup/requirements.lock.txt
```

(editable 2개와 전용 index가 필요한 3개는 bootstrap이 따로 설치하므로 제외한다.)
