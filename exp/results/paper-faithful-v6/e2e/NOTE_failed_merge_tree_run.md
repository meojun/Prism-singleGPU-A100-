# 주의 — 이 박스에서 시도한 v6 검증은 실패했다 (결과 아님)

병합 트리에서 `run_v6_validation.sh` 를 돌렸더니 한 줄로 죽었다:

```
unknown system: paper-faithful-v6
```

원인은 v6 코드가 아니라 스크립트의 경로다. `run_v6_validation.sh` 는
`R=/workspace/prism-exp` 를 하드코딩하는데, 이 박스에서 그 경로는 **v6 패치가
없는 TP 브랜치**였다. `/workspace/shm_clean.sh` 도 존재하지 않는 경로다.

`paper-faithful-v6/run.log` 에 남아 있는 12줄짜리 정상 로그는 **원래 V6 박스의
것**이고 유효하다. 병합 과정에서 이 실패 로그가 그것을 덮어쓸 뻔했고, 병합
충돌에서 원본을 지켰다.

이 박스에서 v6 를 검증하려면 `exp/scripts/run_v6_validation_merge.sh` 를 써라 —
병합 트리와 이 박스의 보정값에 배선돼 있고, KV 핸드오프 프로브 출력도 함께
찍는다.
