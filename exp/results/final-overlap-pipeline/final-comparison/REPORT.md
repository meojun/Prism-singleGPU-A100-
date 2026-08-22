# Released Prototype vs Final Paper-Faithful Prism

생성: `exp/scripts/compare_arms.py`. 수치는 전부 `raw/*/summary.csv` 에서 파생된 것이고 그 반대가 아니다. 판정 규칙은 숫자를 보기 전에 코드에 고정돼 있다.


| arm | 무엇 |
| --- | --- |
| A | released-prototype (TP patch absent) |
| B | released-prototype (TP patch present, off) |
| C | paper-faithful-v6 (all mechanisms on) |

증가가 좋은 지표는 `(Final-Proto)/Proto`, 지연은 `(Proto-Final)/Proto` 로 계산해 **양수면 언제나 Final 이 낫다**는 뜻이 되게 했다.

판정은 보수적이다. 이 프로젝트의 대조군 스윕이 seed 분산 sd/mean=17% 를 기록했고, 과거에 seed 가 평균을 사이에 두고 갈라지는데도 평균비를 인용해 한 번 틀렸다. 그래서 차이가 pooled sd 를 넘지 못하면 **within noise** 로 적는다.

## Q1. Prototype 과 Final-TP-OFF 가 E2E 에서 같게 동작하는가

`aggregated/regression_check.csv` 참조. 게이트 판정: **PASS**

게이트가 STOP 이면 A-vs-C 수치는 '논문 메커니즘의 효과' 가 아니라 '다른 코드베이스와의 비교' 로 읽어야 한다.

## Q2, Q3. 부하 구간별 차이, 그리고 bursty 에서 더 강한가

| workload | rate | metric | Proto | Final | 개선(%) | 판정 |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| bursty | 2 | e2e_p99 | 1.876e+04 ± 1.9e+03 | 2.091e+04 ± 3.4e+03 | -11.46 | within noise |
| bursty | 2 | goodput | 0.9144 ± 0.22 | 0.7256 ± 0.3 | -20.66 | within noise |
| bursty | 2 | joint_slo | 0.4554 ± 0.11 | 0.3611 ± 0.15 | -20.72 | within noise |
| bursty | 2 | ttft_p99 | 4237 ± 1.8e+03 | 4937 ± 1.7e+03 | -16.51 | within noise |
| bursty | 4 | e2e_p99 | 2.154e+04 ± 3.8e+02 | 3.028e+04 ± 2.6e+03 | -40.6 | exceeds seed spread |
| bursty | 4 | goodput | 1.239 ± 0.16 | 0.8556 ± 0.081 | -30.94 | exceeds seed spread |
| bursty | 4 | joint_slo | 0.308 ± 0.039 | 0.2125 ± 0.017 | -31.0 | exceeds seed spread |
| bursty | 4 | ttft_p99 | 3680 ± 1.8e+03 | 5078 ± 4e+03 | -37.96 | within noise |
| bursty | 8 | e2e_p99 | 3.279e+04 ± 6.5e+03 | 5.424e+04 ± 1.3e+04 | -65.44 | exceeds seed spread |
| bursty | 8 | goodput | 2.487 ± 2.5 | 0.7067 ± 0.29 | -71.58 | exceeds seed spread |
| bursty | 8 | joint_slo | 0.3066 ± 0.3 | 0.08796 ± 0.037 | -71.31 | exceeds seed spread |
| bursty | 8 | ttft_p99 | 3112 ± 2.2e+03 | 4192 ± 2e+03 | -34.71 | within noise |
| bursty | 14 | e2e_p99 | 6.742e+04 ± 7.5e+03 | 7.399e+04 ± 5.2e+04 | -9.74 | within noise |
| bursty | 14 | goodput | 0.8289 ± 0.54 | 0.3722 ± 0.11 | -55.09 | exceeds seed spread |
| bursty | 14 | joint_slo | 0.05926 ± 0.039 | 0.02653 ± 0.0079 | -55.24 | exceeds seed spread |
| bursty | 14 | ttft_p99 | 4639 ± 1.3e+03 | 2526 ± 2.3e+03 | 45.54 | exceeds seed spread |
| bursty | 20 | e2e_p99 | 8.633e+04 ± 1e+04 | 1.11e+05 ± 6.7e+03 | -28.53 | exceeds seed spread |
| bursty | 20 | goodput | 0.6467 ± 0.55 | 0.3878 ± 0.27 | -40.03 | within noise |
| bursty | 20 | joint_slo | 0.03283 ± 0.029 | 0.01966 ± 0.014 | -40.12 | within noise |
| bursty | 20 | ttft_p99 | 5858 ± 2.7e+03 | 3.61e+04 ± 5.3e+04 | -516.33 | within noise |
| steady | 4 | e2e_p99 | 2.164e+04 ± 4.7e+02 | 2.25e+04 ± 1.5e+03 | -3.97 | within noise |
| steady | 4 | goodput | 0.375 ± 0.064 | 0.34 ± 0.12 | -9.33 | within noise |
| steady | 4 | joint_slo | 0.09126 ± 0.017 | 0.08325 ± 0.03 | -8.78 | within noise |
| steady | 4 | ttft_p99 | 325.8 ± 24 | 304 ± 13 | 6.7 | exceeds seed spread |
| steady | 8 | e2e_p99 | 2.627e+04 ± 1.5e+03 | 2.686e+04 ± 1.2e+03 | -2.23 | within noise |
| steady | 8 | goodput | 0.3211 ± 0.0084 | 0.2656 ± 0.042 | -17.3 | exceeds seed spread |
| steady | 8 | joint_slo | 0.03978 ± 0.001 | 0.0329 ± 0.0052 | -17.3 | exceeds seed spread |
| steady | 8 | ttft_p99 | 398.9 ± 39 | 403.9 ± 41 | -1.26 | within noise |
| steady | 20 | e2e_p99 | 6.473e+04 ± 2e+04 | 7.19e+04 ± 1.6e+04 | -11.09 | within noise |
| steady | 20 | goodput | 0.1456 ± 0.13 | 0.1889 ± 0.067 | 29.77 | within noise |
| steady | 20 | joint_slo | 0.007309 ± 0.0064 | 0.00945 ± 0.0033 | 29.29 | within noise |
| steady | 20 | ttft_p99 | 1528 ± 1.4e+03 | 814.4 ± 1.6e+02 | 46.71 | within noise |

20 req/s 는 일반 운영점이 아니라 **Extreme Load / Stress** 구간이다. 이 구성의 포화점은 5~10 req/s 로 측정돼 있다.

## Q4. Final 의 migration/KV 메커니즘이 실제로 얼마나 쓰이는가

**Final arm 에서 마이그레이션이 한 번도 일어나지 않았다.** 0 을 '메커니즘이 동작한다' 로 읽지 말 것 — 이 워크로드와 이 박스의 tau 에서 Algorithm 1 이 이동을 낼 조건이 오지 않았다는 사실이다.

## Q5. Prototype 이 못 하는 TP>1 / 대형 모델을 Final 은 하는가

| | Released Prototype | Final Paper-Faithful |
| --- | --- | --- |
| worker-pool 하 TP>1 | **Unsupported** | Supported |
| TP=2 / TP=4 | Unsupported | 검증됨 |
| anti-affinity | 미구현 | 논문 §A.2.2 구현 |
| Llama-3.1-70B | Unsupported | 서빙 확인 (TP=4, TP=2) |

프로토타입은 worker-pool 경로에서 TP>1 을 아예 못 돌린다. 성능 arm 을 억지로 만들지 않고 **Unsupported** 로 적는다. 근거는 `exp/results/paper-faithful-v4/tp-validation/FINDING.md` 와 이 브랜치의 `paper-faithful-tp/REPORT.md`.
