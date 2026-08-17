## 0. 비교 범위

이 보고서는 동일한 6개 모델, A100 80GB 2장, 동일한 paired request set과 평균 offered load에서
`released-prototype`과 `paper-faithful-v3`를 비교한다. 워크로드는 steady와 shifting-bursty,
aggregate 요청률은 4/8/14/20 req/s이며 각 점은 seed 1의 300초 측정 구간이다.

v3는 Algorithm 1 line 8의 절대 `tau` 판정, 초기 모델 병렬 로딩, target-first overlap
migration, GPU 로컬 Moore-Hodgson admission을 함께 활성화한다.

모델 프로파일과 sanity/calibration은 같은 장비·모델·SLO 설정으로 직전에 수행한 v2
결과를 재사용하고, 본 비교의 16개 측정 런은 v3 결과 디렉터리에 별도로 기록한다.
