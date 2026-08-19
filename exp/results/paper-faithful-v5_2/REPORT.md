# V5_2 — where the two unexplained costs actually go

계측만 넣었고 동작은 바꾸지 않았다. 두 질문에 답한다.

## (가) Algorithm 2 를 켜면 왜 goodput 이 사라지는가

released-prototype 과 v3-alg2only 는 GPU 스케줄러 루프 안의 함수 하나만 다르다.
그 루프를 구간별로 잰다. 값은 iteration 당 밀리초.

| Arm | seed | iters | 루프 주기 | 메모리읽기 | Redis | admission | 디스패치 | 최대 iter |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| paper-faithful-v3-alg2only | 1 | 79268 | 1.078 | 0.026 | 1.011 | 0.014 | 0.026 | 4.5 |
| paper-faithful-v3-alg2only | 2 | 79833 | 1.062 | 0.026 | 0.997 | 0.014 | 0.025 | 5.0 |
| paper-faithful-v3-alg2only | 3 | 77778 | 1.098 | 0.026 | 1.030 | 0.015 | 0.026 | 7.3 |
| released-prototype | 1 | 80139 | 0.959 | 0.024 | 0.899 | 0.012 | 0.024 | 5.7 |
| released-prototype | 2 | 78844 | 1.137 | 0.028 | 1.068 | 0.014 | 0.028 | 7.6 |
| released-prototype | 3 | 78478 | 1.063 | 0.026 | 0.999 | 0.013 | 0.026 | 8.5 |

| Arm | 루프 주기 (ms) | admission (ms) | Redis (ms) |
| --- | ---: | ---: | ---: |
| paper-faithful-v3-alg2only | 1.079 ± 0.018 | 0.014 ± 0.000 | 1.013 ± 0.017 |
| released-prototype | 1.053 ± 0.090 | 0.013 ± 0.001 | 0.988 ± 0.085 |

Moore-Hodgson 을 켜면 루프 주기가 **+0.026 ms**, 그 중 admission 이 **+0.001 ms** 다.
루프가 길어지면 새로 도착한 요청이 디스패치까지 더 기다린다. 8 req/s 에서
평균 추가 대기는 루프 주기 증가의 절반, 약 **+0.013 ms** 이다.
관측된 TTFT 차이와 비교하면 이 경로가 설명하는 몫을 알 수 있다.

## (나) deactivation 의 82% 는 어디에 있는가

[V5-HOP] 은 제어 요청을 보낸 시점과 엔진의 응답이 돌아온 시점을 잰다.
엔진 자체 teardown 은 v4 측정에서 평균 0.96 초였다.

| Arm | seed | action | n | 등록·전송 (s) | 엔진 대기 (s) | 총 (s) |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| paper-faithful-v3 | 1 | ActivateReqInput | 19 | 0.000 ± 0.000 | 1.253 ± 0.456 | 1.253 ± 0.456 |
| paper-faithful-v3 | 1 | DeactivateReqInput | 20 | 0.000 ± 0.000 | 2.936 ± 3.661 | 2.936 ± 3.661 |
| paper-faithful-v3 | 2 | ActivateReqInput | 21 | 0.000 ± 0.000 | 1.620 ± 0.720 | 1.620 ± 0.720 |
| paper-faithful-v3 | 2 | DeactivateReqInput | 21 | 0.000 ± 0.000 | 4.069 ± 4.550 | 4.069 ± 4.550 |
| paper-faithful-v3 | 3 | ActivateReqInput | 20 | 0.000 ± 0.000 | 1.472 ± 0.613 | 1.472 ± 0.613 |
| paper-faithful-v3 | 3 | DeactivateReqInput | 21 | 0.000 ± 0.000 | 3.308 ± 4.261 | 3.308 ± 4.261 |

Deactivate 의 엔진 대기 평균 **3.45 초** 대 엔진 자체 teardown 0.96 초.
차이 **2.49 초**가 요청이 스케줄러에 도달해 엔진이 실제로 teardown 을
시작하기까지의 구간이다 — 스케줄러 루프가 그 요청을 집어들 때까지의 대기가
여기에 포함된다. (가)의 루프 주기와 같은 원인일 수 있다.

