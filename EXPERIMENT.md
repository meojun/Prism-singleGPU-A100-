# Llama-3.1-8B 3개(A100-80G 2장) 유입률 스윕 실험

`exp/results/{ref,probe,exp,burst}/` 아래 모든 런을 재현하는 명령입니다.
발견과 해석: [`exp/results/4-rate-sweep/REPORT_rate_sweep.md`](exp/results/4-rate-sweep/REPORT_rate_sweep.md).

아래 내용은 전부 **손대지 않은 Prism** 을 구동합니다. `prism-research/` 나
`kvcached-prism/` 아래 어떤 파일도 패치하지 않았습니다. 워크로드 생성기는 하네스가
이미 재생할 수 있는 형식의 pickle 을 내보내고, 모든 지표는 원래 엔진/스케줄러/
컨트롤러가 이미 남기는 로그에서 파싱합니다.

---

## 0. 준비

```bash
source exp/scripts/env.sh
export SLO_BASE_FILE=/workspace/prism-exp/exp/configs/slo_base_3x8b_sharegpt.json
hf download anon8231489123/ShareGPT_Vicuna_unfiltered \
    ShareGPT_V3_unfiltered_cleaned_split.json --repo-type dataset \
    --local-dir $DATASETS/sharegpt          # 673 MB, 비게이팅
```

## 1. 모델 구성과 그 이유

슬롯 `model_1`, `model_4`, `model_5` 에 **`meta-llama/Llama-3.1-8B`** 3개.

* **슬롯은 자유롭게 고를 수 없습니다.** `benchmark.py` 는 `--model-paths` 가 붙은 런을
  전부 `trace.py::generate_e2e_benchmark_reqs` 로 보내는데, 여기에 특정 모델에 대해
  측정된 슬롯별 SLO 기준선이 하드코딩되어 있습니다. 슬롯 1/4/5 가 Llama-3.1-8B 슬롯
  셋이고, 2는 3B, 3/6/7/8 은 1B 입니다. 8B 를 슬롯 2로 돌리면 3B 기준선으로
  채점됩니다.
* **`-Instruct` 가 아니라 베이스 모델.** `Llama-3.1-8B-Instruct` 는
  `model_info.json` 에 없어서, 내려받아 `profile_models.py` 를 거치기 전까지는 GPU
  스케줄러가 기동을 거부합니다
  (`ValueError: Model path ... not found in the profiled model info file`).
  그리고 얻을 것도 없습니다. `benchmark.py:64-65` 가 `max_new_tokens=output_len` 과
  함께 `ignore_eos=True` 를 보내므로 디코드 길이가 **강제**되고, 인스트럭션 튜닝이
  측정되는 값을 단 하나도 바꿀 수 없습니다. 아키텍처가 같으므로 `cell_size` 도
  동일(131072 B/token)하고 가중치도 15.08 GiB 로 같습니다.
* **배치는 산술적으로 1 + 2 입니다.** GPU 2장에 모델 3개면 한 GPU 는 반드시 2개를
  갖습니다. 모델별 유입률이 같으면 두 개가 올라간 GPU 가 부하의 2/3 를 지고, 어떤 배치
  정책도 이를 고칠 수 없습니다 — 모델을 옮겨봐야 짝이 다른 곳에 다시 생길 뿐입니다.
  이 비대칭이 이 실험의 주된 레버이고, 마이그레이션 경로를 발동시키는 요인입니다.

```bash
python exp/scripts/make_config.py --num-gpus 2 --slots 1,4,5 \
    --placement balanced -o exp/configs/llama_2gpu_3x8b.json
```

## 2. 무경합 SLO 기준선 (논문 7.1)

내장된 SLO 표는 저자들이 다른 하드웨어에서 측정한 값입니다. 이 장비에서는 같은 슬롯이
**TTFT 1.77배, TPOT 1.57배 느립니다.** 그래서 내장 임계값은 부하가 없어도 도달할 수
없고 "달성률" 이 신호로서 의미를 잃습니다. 레포에 이미 있는 도구로 다시 유도합니다.

```bash
python exp/scripts/build_sharegpt_trace.py --variant rate --slots 1 \
    --phase-rates "4" --phase-len 180 --cv 1.0 --seed 42 \
    --out $DATASETS/sharegpt/ref_solo.pkl

SLOTS=1 NGPU=1 CFG=$PWD/exp/configs/llama_1gpu_solo8b.json \
  TAG=ref TRACE=$DATASETS/sharegpt/ref_solo.pkl \
  ./exp/scripts/run_multigpu.sh glob_on 1

python exp/scripts/derive_slo_baseline.py --run ref:glob_on_ts1:model_1 \
    --out exp/configs/slo_base_3x8b_sharegpt.json
# 그 다음 model_1 행을 model_4 / model_5 에 복사 -- 같은 모델, 같은 워크로드
```

측정값: **TTFT p95 76.1 ms, TPOT p95 18.04 ms** (내장값: 42.9 / 11.46).
보고에 쓰는 SLO 스케일은 TTFT ×5(380 ms), TPOT ×3(54.1 ms)입니다. 스케일은 후처리
인자이므로, 저장된 요청 덤프에서 다른 조합도 재실행 없이 다시 계산할 수 있습니다.

## 3. 용량 프로파일링 — 람다를 추측하지 말 것

유입률을 **한 번의 런 안에서** 램프시키고 결과를 도착 구간으로 버킷팅합니다. 그래서
유입률마다 런을 하나씩 돌리지 않고 한 번에 용량 곡선을 얻습니다.

```bash
# 낮은 램프: 90초 6단계로 1 -> 8 req/s
python exp/scripts/build_sharegpt_trace.py --variant rate --slots 1,4,5 \
  --phase-rates "0.333,0.333,0.333;0.667,0.667,0.667;1,1,1;1.333,1.333,1.333;2,2,2;2.667,2.667,2.667" \
  --phase-len 90 --cv 1.0 --seed 42 --out $DATASETS/sharegpt/probe_ramp.pkl

# 높은 램프: 60초 5단계로 8 -> 30 req/s
python exp/scripts/build_sharegpt_trace.py --variant rate --slots 1,4,5 \
  --phase-rates "2.667,2.667,2.667;4,4,4;5.333,5.333,5.333;7.333,7.333,7.333;10,10,10" \
  --phase-len 60 --cv 1.0 --seed 42 --out $DATASETS/sharegpt/probe_ramp_hi.pkl

SLOTS=1,4,5 NGPU=2 CFG=$PWD/exp/configs/llama_2gpu_3x8b.json \
  TAG=probe TRACE=$DATASETS/sharegpt/probe_ramp.pkl \
  ./exp/scripts/run_multigpu.sh glob_on 1
python exp/scripts/collect_metrics.py --exp probe_glob_on_ts1 --tag probe \
    --trace $DATASETS/sharegpt/probe_ramp.pkl --window 90
# probe_ramp_hi.pkl 과 --window 60 으로 반복
```

두 램프는 7.8 req/s 에서 겹치고 서로 일치합니다(출력 1612 대 1680 tok/s, TTFT p95
151 대 157 ms) — 공짜로 얻는 재현성 검사입니다. TTFT 무릎이 23~31 req/s 사이이므로
**lambda_base = 12 req/s, 용량의 약 46 %** 로 잡았고, 이는 목표 구간인 40~60 % 안에
들어갑니다.

> 두 probe 런은 `TAG=probe` 를 공유해서 서로의 로그를 덮어씁니다. 낮은 램프의 CSV 는
> `results/4-rate-sweep/rampLO_*` 로 보존했습니다. 다시 돌린다면 램프마다 다른 TAG 를
> 쓰세요.

## 4. 실험 1(기준선)과 2(경합)

트레이스 하나를 `--time-scale` 로 스윕합니다. 이렇게 하면 **요청 집합, 프롬프트와 출력
길이, 모델별 혼합, seed 가 모든 유입률 지점에서 바이트 단위로 동일**하게 유지되고 도착
시계만 압축됩니다. 유입률마다 새로 생성하면 표본이 바뀌어 비교가 교란됩니다.

```bash
python exp/scripts/build_sharegpt_trace.py --variant rate --slots 1,4,5 \
  --phase-rates "4,4,4" --phase-len 420 --cv 1.0 --seed 42 \
  --out $DATASETS/sharegpt/exp_base12.pkl        # 요청 5090개, 12.1 req/s

for ts in 1 0.8 0.6667 0.5 0.4; do               # 1.0x 1.25x 1.5x 2.0x 2.5x
  SLOTS=1,4,5 NGPU=2 CFG=$PWD/exp/configs/llama_2gpu_3x8b.json \
    TAG=exp TRACE=$DATASETS/sharegpt/exp_base12.pkl \
    TPOT_SCALE=3 TTFT_SCALE=5 \
    ./exp/scripts/run_multigpu.sh glob_on $ts
  python exp/scripts/collect_metrics.py --exp exp_glob_on_ts$ts --tag exp
done
```

`--time-scale t` 는 `lambda_base / t` 유입률을 만듭니다: 1.0 → 12, 0.8 → 15,
0.6667 → 18, 0.5 → 24, 0.4 → 30 req/s. 실험 1이 `ts=1` 지점이고 나머지가 실험 2입니다.

## 5. 버스트 시나리오 — 뜨거운 모델 수를 늘려가기

같은 모델 3개에 대해, 동시에 뜨거운 모델의 수가 1 → 2 → 3 으로 올라가도록 모델별
유입률을 단계적으로 바꾸되 `model_1` 은 내내 8 req/s 에 고정합니다. 한 모델의 유입률을
고정해 두는 것이 모델 간 간섭을 측정 가능하게 만드는 장치입니다.

```bash
python exp/scripts/build_sharegpt_trace.py --variant rate --slots 1,4,5 \
  --phase-rates "8,0.5,0.5;8,8,0.5;8,8,8" --phase-len 150 --cv 1.0 --seed 42 \
  --out $DATASETS/sharegpt/exp_burst.pkl

SLOTS=1,4,5 NGPU=2 CFG=$PWD/exp/configs/llama_2gpu_3x8b.json \
  TAG=burst TRACE=$DATASETS/sharegpt/exp_burst.pkl TPOT_SCALE=3 TTFT_SCALE=5 \
  ./exp/scripts/run_multigpu.sh glob_on 1
python exp/scripts/collect_metrics.py --exp burst_glob_on_ts1 --tag burst \
    --trace $DATASETS/sharegpt/exp_burst.pkl --window 150 --tpot-slo-scale 3
```

"낮음" 을 0 이 아니라 0.5 req/s 로 둔 이유가 있습니다. 0 이면 모델이
`MODEL_IDLE_THRESHOLD = 50 s` 를 넘겨 축출되고, 그러면 이 시나리오가 겨냥하는 KV 캐시
재분배가 아니라 축출과 콜드 스타트를 측정하게 됩니다.

## 6. 산출물

| 파일 | 내용 |
| --- | --- |
| `<exp>_slo.json` | 모델별 달성률, TTFT·TPOT 의 평균/p50/p95/p99, e2e, goodput |
| `<exp>_summary.csv` | 한 줄 — 유입률 대 X 표의 열들 |
| `<exp>_timeseries.csv` | 1초 구간 — 모델별 도착/실행/큐/KV 토큰/KV 풀 비율/디코드 처리율, GPU 별 스케줄러 큐/디바이스 메모리/사용률, 컨트롤러 동작 |
| `<exp>_windows.csv` | 도착 구간별 요청 통계(용량 곡선) |
| `<exp>_actions.txt` | 활성화 / 비활성화 / 마이그레이션 |
| `server-logs/<exp>/gpu_timeline.txt` | 2초 주기 nvidia-smi 원시 샘플 |

시계열 열들은 버스트 → 경합 → 스케줄링/메모리 변화 → 지연을 하나의 시계 위에 그리기
위해 필요한 것들입니다.

## 7. 만들어낼 수 없는 두 지표와 그 이유

* **`rejected` 는 항상 0 입니다.** 관측되지 않은 것이 아니라 불가능합니다.
  `request_queue.py:137` 이 `net_available = float("inf")` 로 설정하며 주석에
  *"the actual implementation doesn't seem to limit resources"* 라고 적혀 있습니다.
  따라서 논문 6.2 에 서술된 메모리 기반 admission control 은 아무것도 거부하지
  않습니다. 런타임 로그가 매초 `net_available: inf` 를 찍고, `collect_metrics.py` 가
  그 문자열을 기록해 두어 이 주장을 계속 검증할 수 있게 했습니다.
* **포화 이전에는 큐 길이가 거의 0 입니다.** 전부 즉시 admit 되므로 백프레셔가 큐
  깊이가 아니라 `#running-req` 와 TTFT 로 나타납니다. 큐는 포화를 지나야 생깁니다
  (30 req/s 에서 모델 큐 38, 스케줄러 큐 184). 부하 신호로는 큐 길이가 아니라
  `*_running` 과 TTFT p95 를 보세요.
