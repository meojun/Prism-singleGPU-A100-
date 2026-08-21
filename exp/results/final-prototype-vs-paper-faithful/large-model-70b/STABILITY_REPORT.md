# Llama 70B stability: FAIL

실패 단계: **startup 또는 basic inference**.
sustained 는 시작하지 않았다 -- 기본 서빙이 확인되기 전에 stress 를
돌리면 어느 단계가 문제인지 알 수 없게 된다.

근거: `logs/stage12.log`, `raw/stage12/server-logs/`.
