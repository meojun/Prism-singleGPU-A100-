# 실험 결과 — 인덱스

폴더는 **연구 단위**로 4개다. 예전에는 실행마다 폴더가 생겨 10개까지 늘어났는데,
그 원인은 `TAG` 하나가 *결과 폴더*와 *파일 접두사* 두 역할을 겸했기 때문이다.
지금은 분리되어 있다:

| 변수 | 뜻 | 예 |
| --- | --- | --- |
| `STUDY` | 결과가 들어갈 **폴더** | `4-rate-sweep` |
| `TAG` | run 이름 = **파일 접두사** | `exp`, `probe`, `burst` |

```bash
STUDY=4-rate-sweep TAG=myrun ./exp/scripts/run_multigpu.sh glob_on 1
python exp/scripts/collect_metrics.py --exp myrun_glob_on_ts1 --study 4-rate-sweep
```

---

| 폴더 | 보고서 | 내용 |
| --- | --- | --- |
| [`1-env-verification/`](1-env-verification/) | [REPORT.md](1-env-verification/REPORT.md) | 1-GPU sanity 스윕(A/B/C), 재실행 분산, 2×A100 박스에서의 재검증(`verify_*`), slowdown SLO의 무경합 기준(`base_*`) |
| [`2-colocation/`](2-colocation/) | [REPORT.md](2-colocation/REPORT.md) | ShareGPT + slowdown SLO colocation 연구 (8B 1개 / 8B 2개 / 3B+8B), 1 GPU |
| [`3-placement/`](3-placement/) | [REPORT.md](3-placement/REPORT.md) | §7.3 Figure 7 — global placement on/off, 2 GPU 8 모델 |
| [`4-rate-sweep/`](4-rate-sweep/) | [REPORT_rate_sweep.md](4-rate-sweep/REPORT_rate_sweep.md) | 3× Llama-3.1-8B rate sweep + burst, capacity 프로파일링(`probe_*`, `rampLO_*`), 무경합 기준(`ref_*`) |
| — | [STATUS_REPORT.md](STATUS_REPORT.md) | **전체 상태 자동 생성 보고서** (환경·전 실험·논문 대비 코드 검증) |

## 파일 규칙

폴더 안에서 run은 파일 접두사로 구분된다.

| 패턴 | 내용 |
| --- | --- |
| `<run>_slo.json` | `analyze_slo.py`가 재계산한 모델별 attainment·지연 백분위 |
| `<run>_e2e_*gpu_*.json` | 하네스가 낸 원본 지표 |
| `<run>_summary.csv` | rate-vs-X 표 한 줄 |
| `<run>_timeseries.csv` | 1초 bin 시계열 (모델별 KV·running·queue, GPU별 util·메모리) |
| `<run>_windows.csv` | 도착 구간별 요청 통계 (capacity 곡선용) |
| `<run>_actions.txt` | 컨트롤러 activation / deactivation / migration |
| `requests/<run>_*_output_requests.json` | per-request 원본 덤프 |

`server-logs/<run>/`(gitignore)에는 원시 서버 로그와 `gpu_timeline.txt`가 남는다.
