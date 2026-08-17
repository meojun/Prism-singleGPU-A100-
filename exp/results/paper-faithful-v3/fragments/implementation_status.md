## 3. v3 구현 및 실행 설정

| 항목 | v3 설정 | 런타임 증거 |
| --- | --- | --- |
| Algorithm 1 line 8 | `current_r - best_r > tau`, absolute tau; KV bytes를 GiB로 정규화 | `server.log.global_controller.log`의 `[PAPER-ALG1-V3]` JSON |
| 초기 로딩 | GPU별 모델을 병렬 활성화 | `--parallel-model-loading` |
| 마이그레이션 | target 활성화 완료 후 source 비활성화 | action 로그의 `ActivateAction` → `DeactivateAction` |
| 제어 경합 방지 | 응답 waiter 선등록, 활성화 직렬화, worker-slot 예약 검사, 600초 readiness timeout, 부분 성공 시 안전한 source 정리 | timeout/assert/worker 부족이 없는 최종 run의 controller/stdout 로그 |
| 로컬 admission | Moore-Hodgson | GPU scheduler 로그의 `[PAPER-ALG2] enabled=True` |
| 비교 페어링 | 같은 trace, seed, duration, SLO scale | `META.txt`, paired request/phase JSON, canonical run 경로 |

요청 실패가 발생한 시도는 한 번 자동 재실행한다. 재시도에서 회복되면
`RECOVERED_AFTER_RETRY`, 다시 발생하면 `REPRODUCED_REQUEST_FAILURE`로 표시하며,
재현된 실패는 시스템 결과로 보존하고 달성률/goodput의 분모에 포함한다.
응답 waiter 등록 경합이 있던 구현으로 수행된 진단 시도는 `*_invalid_response_race_*`로
보존하되 정규 seed 집계에서 제외하고, 수정 코드로 다시 측정했다.
또한 메모리만 보고 이미 worker 4개가 찬 GPU를 재활성화 대상으로 고르던 진단 시도는
정규 결과에서 제외했다. 최종 구현은 현재 active instance와 같은 scheduling cycle에서
예약된 target slot을 함께 계산하며, 수정 코드로 14/20 req/s를 다시 측정했다.
