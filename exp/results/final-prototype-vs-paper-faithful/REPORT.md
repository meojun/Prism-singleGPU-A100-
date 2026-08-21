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

**해당 수치 없음.**

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
