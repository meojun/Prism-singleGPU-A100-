# Prism experiment box — status report

Generated 2026-08-12 09:54 UTC by `exp/scripts/build_status_report.py`. Every figure is probed or read from committed results at generation time; nothing in this file is typed by hand.

## 1. Environment

**GPU** — driver 595.71.05, GPU0↔GPU1 link `NV12`

| idx | name | memory | compute cap |
| --- | --- | --- | --- |
| 0 | NVIDIA A100-SXM4-80GB | 81920 MiB | 8.0 |
| 1 | NVIDIA A100-SXM4-80GB | 81920 MiB | 8.0 |

**Host** — 128 threads, RAM 2003 GiB total, 1891 GiB available, disk at repo 512G total, 478G free, /dev/shm 250G

**Stack** (from `prism-venv`, i.e. what the runs actually loaded)

| package | version |
| --- | --- |
| torch | 2.4.0+cu121 |
| sglang | 0.3.4.post2 |
| vllm | 0.6.3.post1 |
| transformers | 4.45.2 |
| flashinfer | 0.1.6+cu121torch2.4 |
| kvcached | n/a |
| cuda_available | True |

**Pinned upstream** (`setup/pins.env`) — checked out HEADs

| repo | HEAD |
| --- | --- |
| prism-research | `595ec1f` |
| kvcached (prism/shm) | `d78649d` |
| kvcached (main) | `ce76a12` |

**redis** — `PONG`, supervisor: `redis                            RUNNING   pid 3178, uptime 4:38:14`

**Model weights**

| size | model |
| --- | --- |
| 15G | `meta-llama/Llama-3.1-8B` |
| 2.4G | `meta-llama/Llama-3.2-1B` |
| 6.0G | `meta-llama/Llama-3.2-3B` |

**Right now** — serving processes: 0, GPU memory in use: 0 MiB 0 MiB

## 2. Repository state

branch `main`, working tree **dirty**, in sync with origin

| commit | date | subject |
| --- | --- | --- |
| ddfb5d7 | 2026-08-12 | Rate-sweep and burst experiments on 3x Llama-3.1-8B, 2 GPUs |
| a5f676b | 2026-08-12 | Generalise the runner to N GPUs and add an agent runbook |
| 12462c5 | 2026-08-12 | Two-GPU support: environment re-verification and §7.3 placement ablation |
| ddff762 | 2026-08-04 | ShareGPT colocation experiment with slowdown-based SLO on 1x A100 |
| eabf72e | 2026-08-03 | Reproducible single-GPU Prism (OSDI'26) baseline environment + A100 sanity sweep |

Uncommitted:
```
?? exp/results/STATUS_REPORT.md
?? exp/scripts/build_status_report.py
```

## 3. Experiment inventory

28 analysed runs across 10 result namespaces. Every row is read from the committed `*_slo.json` / `*_summary.csv`.

| namespace | run | reqs | dur | att TTFT | att TPOT | TTFT p95 ms | TPOT p50 ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `base` | `base_M1` | 296 | 603s | 0.983 | 0.986 | — | 14 |
| `base` | `base_M2` | 22 | 272s | 1.000 | 1.000 | — | 12 |
| `base` | `base_M4` | 262 | 607s | 1.000 | 1.000 | — | 14 |
| `burst` | `burst_glob_on_ts1` | 7569 | 497s | 0.990 | 0.483 | 231 | 63 |
| `exp` | `exp_A` | 296 | 603s | 1.000 | 1.000 | — | 14 |
| `exp` | `exp_B` | 558 | 611s | 0.991 | 0.229 | — | 28 |
| `exp` | `exp_C` | 318 | 603s | 0.997 | 0.959 | — | 14 |
| `exp` | `exp_glob_on_ts0.4` | 5090 | 210s | 0.707 | 0.291 | 21,325 | 111 |
| `exp` | `exp_glob_on_ts0.5` | 5090 | 243s | 0.987 | 0.307 | 239 | 89 |
| `exp` | `exp_glob_on_ts0.6667` | 5090 | 311s | 0.999 | 0.379 | 191 | 68 |
| `exp` | `exp_glob_on_ts0.8` | 5090 | 362s | 1.000 | 0.447 | 177 | 57 |
| `exp` | `exp_glob_on_ts1` | 5090 | 443s | 1.000 | 0.872 | 165 | 44 |
| `fig7` | `fig7_glob_off_ts0.5` | 754 | 316s | 0.932 | 0.212 | — | 34 |
| `fig7` | `fig7_glob_off_ts1` | 754 | 616s | 0.954 | 0.259 | — | 34 |
| `fig7` | `fig7_glob_on_ts0.5` | 754 | 310s | 0.960 | 0.387 | — | 24 |
| `fig7` | `fig7_glob_on_ts1` | 754 | 608s | 0.966 | 0.569 | — | 20 |
| `probe` | `probe_glob_on_ts1` | 5375 | 352s | 0.894 | 0.206 | 3,868 | 56 |
| `ref` | `ref_glob_on_ts1` | 702 | 193s | 1.000 | 0.994 | 76 | 15 |
| `sanity` | `sanity_A` | 296 | 603s | 1.000 | 1.000 | — | 14 |
| `sanity` | `sanity_B` | 558 | 612s | 0.991 | 0.224 | — | 28 |
| `sanity` | `sanity_C` | 318 | 603s | 1.000 | 0.959 | — | 14 |
| `sanity_repeat` | `sanity_repeat_C` | 318 | 603s | 1.000 | 0.912 | — | 15 |
| `sharegpt_content` | `sharegpt_content_A` | 296 | 603s | 1.000 | 1.000 | — | 14 |
| `sharegpt_content` | `sharegpt_content_B` | 558 | 613s | 0.991 | 0.224 | — | 29 |
| `sharegpt_content` | `sharegpt_content_C` | 318 | 603s | 0.997 | 0.918 | — | 16 |
| `verify` | `verify_A` | 296 | 603s | 0.997 | 0.997 | — | 14 |
| `verify` | `verify_B` | 558 | 612s | 0.987 | 0.219 | — | 29 |
| `verify` | `verify_C` | 318 | 604s | 0.987 | 0.934 | — | 14 |

> attainment is computed by `analyze_slo.py`, never by `benchmark.py` — the latter's `average_attainment_tpot` compares millisecond baselines against second-valued measurements and is always 1.0.

## 4. Rate sweep — 3× Llama-3.1-8B on 2 GPUs

λ_base = **12 req/s**, chosen as ~46% of the profiled TTFT knee (~26 req/s). One request sequence replayed at every rate via `--time-scale`, so lengths, model mix and seed are identical across rows.

| offered λ | ×base | achieved | out tok/s | TTFT p50 ms | TTFT p95 ms | TPOT p50 ms | att TPOT | KV pool m1/m4/m5 | max queue mdl/sch |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | :---: |
| 12 | 1.00× | 11.5 | 2,360 | 70 | **165** | 44 | 0.872 | 0.08 / 0.16 / 0.16 | 0 / 0 |
| 15 | 1.25× | 14.1 | 2,885 | 75 | **177** | 57 | 0.447 | 0.09 / 0.21 / 0.20 | 0 / 0 |
| 18 | 1.50× | 16.4 | 3,365 | 79 | **191** | 68 | 0.379 | 0.11 / 0.28 / 0.26 | 0 / 0 |
| 24 | 2.00× | 21.0 | 4,304 | 92 | **239** | 89 | 0.307 | 0.23 / 0.69 / 0.58 | 0 / 0 |
| 30 | 2.50× | 24.3 | 4,983 | 132 | **21,325** | 111 | 0.291 | 0.29 / 0.98 / 0.98 | 38 / 184 |

Between 2.0× and 2.5× λ_base, TTFT **p95 grows 89×** while **p50 grows only 1.4×** — mean and median hide the cliff. The cliff coincides exactly with the two-model GPU's KV pool reaching ~0.98 and with the first non-empty queue.

### 4.1 Colocation dominates arrival rate

| offered λ | model_1 (GPU0, alone) | model_4 (GPU1, shared) | model_5 (GPU1, shared) |
| ---: | :---: | :---: | :---: |
| 12 | 1.000 / 89 / 17 | 0.835 / 170 / 47 | 0.787 / 183 / 49 |
| 15 | 1.000 / 89 / 19 | 0.189 / 185 / 62 | 0.177 / 199 / 62 |
| 18 | 1.000 / 95 / 20 | 0.093 / 200 / 76 | 0.072 / 216 / 74 |
| 24 | 0.854 / 153 / 24 | 0.056 / 247 / 115 | 0.037 / 277 / 101 |
| 30 | 0.826 / 177 / 30 | 0.062 / 23,054 / 133 | 0.009 / 12,411 / 127 |

`att_tpot / TTFT p95 ms / TPOT p50 ms`. At λ_base the model alone on GPU0 runs at the uncontended baseline while the two sharing a GPU are ~2.8× slower — before any rate increase.

### 4.2 Burst — hot models 1 → 2 → 3

| phase | hot models | total λ | att both | TPOT p50 ms | model_1 TTFT p95 | model_4 TTFT p95 | model_5 TTFT p95 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1 hot (8 / 0.5 / 0.5) | 8.8 | **0.986** | 22 | **100** | 155 | 141 |
| 1 | 2 hot (8 / 8 / 0.5) | 17.3 | **0.489** | 41 | **111** | 217 | 155 |
| 2 | 3 hot (8 / 8 / 8) | 24.3 | **0.207** | 97 | **271** | 250 | 334 |

model_1 is pinned at 8 req/s in all three phases, so its column is pure cross-model interference.

Controller actions during the burst run:
```
ACTION: deactivate model_5 on GPU 1 and activate model_5 on GPU 0. Reason: migrate model
ACTION: deactivate model_5 on GPU 0 and activate model_5 on GPU 1. Reason: migrate model
```

### 4.3 Capacity profiling (how λ_base was chosen, not guessed)

| ramp | offered req/s | out tok/s | TTFT p95 ms | TPOT p50 ms |
| --- | ---: | ---: | ---: | ---: |
| low ramp (1 → 8 req/s) | 1.1 | 262 | 154 | 29 |
| low ramp (1 → 8 req/s) | 1.9 | 373 | 128 | 27 |
| low ramp (1 → 8 req/s) | 2.9 | 576 | 132 | 29 |
| low ramp (1 → 8 req/s) | 3.9 | 794 | 146 | 31 |
| low ramp (1 → 8 req/s) | 5.9 | 1,191 | 137 | 33 |
| low ramp (1 → 8 req/s) | 7.8 | 1,680 | 157 | 38 |
| high ramp (8 → 31 req/s) | 7.8 | 1,612 | 151 | 36 |
| high ramp (8 → 31 req/s) | 11.5 | 2,336 | 151 | 41 |
| high ramp (8 → 31 req/s) | 16.5 | 3,525 | 186 | 58 |
| high ramp (8 → 31 req/s) | 23.0 | 4,811 | 228 | 97 |
| high ramp (8 → 31 req/s) | 30.8 | 6,085 | 5,552 | 109 |

The two ramps overlap at ~7.8 req/s and agree, which doubles as a reproducibility check.

## 5. Paper vs released code — re-verified against the pinned source

Each row below is checked against the source **at report time**. A claim that no longer matches prints `NOT CONFIRMED` instead of being repeated.

| claim | status | location | evidence |
| --- | --- | --- | --- |
| §6.2 admission control is disabled | confirmed | `request_queue.py:137` | `net_available = float("inf")` |
| §6.1 migration threshold is hard-coded and very loose | confirmed | `simple_global.py:183` | `self.MEMORY_PER_REQUEST_RATIO_THRESHOLD = 15` |
| idle eviction threshold | confirmed | `simple_global.py:181` | `self.MODEL_IDLE_THRESHOLD = 50  # seconds` |
| GPU-local scheduling is slack-ordered EDF, not Moore-Hodgson | confirmed | `request_queue.py:30` | `return req.arrival_time + req.slo - profiled_prefill_time` |
| model service sizes itself from the device count, not --num-gpus | confirmed | `multi_model_server.py:579` | `num_devices = torch.cuda.device_count()` |

Confirmed **absent** (the grep finds nothing, which is the finding):

| paper mechanism | status | searched in | meaning |
| --- | --- | --- | --- |
| Algorithm 1 (KVPR) | absent | `simple_global.py` | no KV Pressure Ratio anywhere in the placement policy; the released code uses `violation` and `memory_per_request` heuristics instead |
| Algorithm 2 (Moore-Hodgson) | absent | `request_queue.py` | no reference to the algorithm the paper proves optimality with |

Not reproducible for other reasons: MuxServe++/QLM/ServerlessLLM baselines are not installed (conflicting torch/vllm pins), and the Hyperbolic / Novita / Chatbot Arena production traces are not public.

## 6. Reproducing any of this

Full command list with rationale: [`EXPERIMENT.md`](../../EXPERIMENT.md). Setting up a fresh box: [`CLAUDE.md`](../../CLAUDE.md).

```bash
source exp/scripts/env.sh
export SLO_BASE_FILE=$PWD/exp/configs/slo_base_3x8b_sharegpt.json

# environment check against the committed 1-GPU baseline
CUDA_VISIBLE_DEVICES=0 TAG=verify ./exp/scripts/run_sanity.sh A   # then B, C

# N-GPU placement config, then a run; NGPU defaults to every visible GPU
python exp/scripts/make_config.py --num-gpus 2 --slots 1,4,5 \
    --placement balanced -o exp/configs/llama_2gpu_3x8b.json
SLOTS=1,4,5 CFG=$PWD/exp/configs/llama_2gpu_3x8b.json TAG=exp \
  TRACE=$DATASETS/sharegpt/exp_base12.pkl TPOT_SCALE=3 \
  ./exp/scripts/run_multigpu.sh glob_on 1
python exp/scripts/collect_metrics.py --exp exp_glob_on_ts1 --tag exp

# regenerate this report
python exp/scripts/build_status_report.py
```

## 7. Where the detail lives

| document | contents | size |
| --- | --- | --- |
| [`exp/results/exp/REPORT_rate_sweep.md`](exp/REPORT_rate_sweep.md) | 3× Llama-3.1-8B rate sweep + burst (this study) | 8 KB |
| [`exp/results/fig7/REPORT.md`](fig7/REPORT.md) | environment verification + §7.3 global-placement ablation | 13 KB |
| [`exp/results/exp/REPORT.md`](exp/REPORT.md) | ShareGPT colocation study, 1 GPU (pre-existing) | 18 KB |
| [`exp/results/sanity/REPORT.md`](sanity/REPORT.md) | original 1-GPU sanity sweep (pre-existing) | 10 KB |
| [`EXPERIMENT.md`](../../EXPERIMENT.md) | every command, with the reasoning behind each choice | 8 KB |
| [`CLAUDE.md`](../../CLAUDE.md) | runbook for setting this up on a fresh rented GPU box | 12 KB |

## 8. Caveats that apply to every number above

- **Single run per data point.** Aggregates over thousands of requests (attainment, throughput, p50) are stable; **tail values such as the 21 s TTFT p95 at 30 req/s must not be quoted as precise** without repeats. The existence and order of magnitude of the cliff are solid.
- **`--disable-cuda-graph` throughout** (repo convention). Absolute TPOT is ~1.57× slower than the paper's baseline hardware, which is why SLO baselines were re-derived here. Do not compare absolute latency with the paper.
- **The shipped `real_trace.pkl` has synthetic `"Hello "*n` prompts** with ~99% prefix overlap. All rate-sweep work uses ShareGPT text (~2-4% overlap). Never enable the radix cache with the synthetic trace.
- **Queue length is ~0 below saturation** and rejections are always 0 — both structural, see §5. Use `#running-req` and TTFT p95 as the load signal.
- **`/workspace` is not a persistent volume** on this instance: recycle or destroy wipes the venv, the 24 GB of weights and `exp/server-logs/`. Committed results survive because they are pushed to git.

