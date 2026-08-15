## 9. 핵심 발견

- **v1 의 under-admission 은 사라졌고, 원인은 `c_i` 였다.** 24개 런 전체에서
  Moore-Hodgson 은 eligible 요청의 95.5~100% 를 선택했고, pathological 라운드는
  **0건**, `eligible>0 이면서 selected=0` 인 최대 연속 라운드는 0이다. v1 에서는
  `eligible=123, selected=0` 이 일상적으로 나왔다. Algorithm 2 는 한 줄도 바꾸지
  않았다. 바뀐 것은 `c_i` 값뿐으로, Llama-3.1-8B 에 대해 직접 측정한 13,702 tok/s
  대 v1 의 4,214 tok/s 다. 논문의 단일 기계 실행가능성 검사는 `c_i` 를 엔진의
  총 chunked-prefill 처리량으로 읽으면 정합적이며, 논문 자신의 최적성 논증이
  전제하는 것도 그것이다.

- **이 영역은 TPOT 바운드라 Algorithm 2 가 대표 지표를 움직일 수 없다.**
  TTFT 달성률은 모든 런에서 0.89~0.99 이므로 joint 달성률은 사실상 TPOT 달성률이다.
  Algorithm 2 는 설계대로 동작한다 — 10 req/s bursty 에서 TTFT p99 를 8,370 ms 에서
  2,374 ms 로 **3.5배** 줄였다. 다만 여기서는 TTFT 가 애초에 병목이 아니었으므로
  그 이득이 joint 달성률에 나타나지 않는다.

- **도착 타이밍만 바꿨을 뿐인데 Prism 의 상대적 우열이 뒤집히고, 그 뒤집힘은
  부하에 대해 단조롭다.** 기준선 대비 Prism 의 이득을 bursty 에서 steady 를 뺀 값:
  1 req/s **+18.4pp**, 2 req/s **+14.9pp**, 8 req/s **−22.4pp**, 10 req/s **−24.9pp**.
  request set, 모델별 요청 수, 프롬프트, 평균 offered load 가 전부 같고 요청이
  *언제* 도착하는지만 다르다.

- **Algorithm 1 의 `tau` 는 부하가 오를수록 오히려 더 보수적이 된다.** 8 과
  10 req/s bursty 에서 Prism 은 마이그레이션을 **0회** 한 반면 프로토타입은 각각
  2회와 5회 했다. 상대 기준 `(peak_now − peak_after)/peak_now > tau` 는 모든 GPU 가
  부하를 받을 때 작아진다 — 절대 불균형은 더 커졌는데도 모델 하나를 옮겨서 바뀌는
  peak 의 *비율*은 줄기 때문이다. 결국 이 규칙은 자신이 고치려던 불균형이 가장 클 때
  동작을 억제한다. 이는 KVPR 의 성질이 아니라 **상대 임계값의 성질**이다.

- **steady 에서는 Algorithm 1 이 하지 말아야 할 마이그레이션을 한다.** 1 req/s
  steady 에서 프로토타입이 0회인데 Prism 은 8회 옮겼고 4.8% 나쁜 결과로 끝났다.
  steady 에는 고칠 모델별 불균형이 없으므로 stop-the-world 마이그레이션은 순수 비용이다.
  반대로 축출과 활성화는 **bursty 에서만** 발생했다(런당 7~9회, 6~8회. steady 런에서는
  전부 0). 페어링된 워크로드가 의도한 메커니즘을 정확히 분리해 냈다는 증거다.
