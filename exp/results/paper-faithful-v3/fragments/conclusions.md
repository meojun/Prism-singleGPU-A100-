## 8. 결론

한 문장으로 요약하면, v3는 **shifting-bursty의 중·고부하에서 released prototype보다
유리했지만 모든 부하와 steady에서 일관되게 우세하지는 않았다.** 모든 정규 런은 요청
실패 0건으로 끝났고, 비교한 8개 workload-rate 점 중 Joint SLO 달성률은 5개 점에서
높았다.

- Bursty에서는 8/14/20 req/s에서 Joint 달성률이 각각 +12.5/+0.8/+8.3%p였고,
  goodput은 +35.8%/+2.8%/+60.2%였다. 특히 20 req/s TTFT p99는 24.88초에서
  1.52초로 93.9% 감소했다.
- Bursty 4 req/s에서는 반대로 Joint 달성률과 goodput이 23.5% 낮았다. 네 부하의
  goodput 합은 v3 15.11 req/s, prototype 13.20 req/s로 v3가 14.5% 높지만, 낮은
  부하의 회귀를 감추지 않고 함께 봐야 한다.
- Steady에서는 4/8 req/s가 +5.4/+4.9%p, 14/20 req/s가 -2.3/-0.7%p였다.
  네 부하의 goodput 합은 7.33 대 7.18 req/s(+2.1%)로 차이가 작다. v3가 steady에서도
  매 런 13~14회 migration을 수행한 반면 prototype은 0회였다는 점은 고부하 steady의
  손실이 동적 제어 비용과 양립하는 결과임을 보여 주지만, 본 실험만으로 인과를 확정할
  수는 없다.
- Bursty 14/20 req/s에서 TTFT tail은 크게 개선됐지만 TPOT p99는 악화됐다. 그럼에도
  TPOT 달성률과 Joint 달성률은 높아졌으므로, v3의 이득은 최악 1% TPOT을 줄이는 것보다
  더 많은 요청을 SLO 경계 안으로 넣는 형태다.

따라서 이 결과가 지지하는 범위는 “v3의 paper-faithful 제어가 모델별 hot set이 이동하는
중·고부하에서 유효하다”까지다. “모든 부하 패턴에서 prototype을 지배한다”는 결론은
지지하지 않는다.

## 9. 해석상의 한계

- 각 점은 seed 1 한 번의 측정이다. paired request set은 시스템 간 비교 잡음을 줄이지만,
  run-to-run 분산이나 신뢰구간을 대신하지 않는다.
- 모델 프로파일과 sanity/calibration은 동일 장비에서 직전에 수행한 v2 결과를 재사용했다.
- 결과는 A100 80GB 2장, 6개 모델, 300초 측정 구간과 본 SLO scale에 한정된다.
- 최종 16개 정규 런만 표와 그림에 포함했다. 응답 waiter 경합 및 worker-slot 초과 예약을
  찾는 데 사용한 진단 실행은 수정 전 구현의 결과이므로 별도 보관하고 집계에서 제외했다.
- Algorithm 1, overlap migration, Moore-Hodgson을 함께 켠 시스템 비교이므로 각 구성 요소의
  독립 효과를 분리하려면 추가 ablation이 필요하다.
