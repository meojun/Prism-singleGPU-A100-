# Prism (OSDI'26) baseline — experiment environment

> **새 GPU 서버에 세팅하려면 → [`SETUP.md`](SETUP.md) 를 읽고 `./bootstrap.sh` 를 실행하세요.**
> 한 줄 요약: `git clone <this repo> && cd prism-exp && ./bootstrap.sh`
> 검증된 sanity 결과 + 전체 보고서(한국어): [`exp/results/1-env-verification/REPORT.md`](exp/results/1-env-verification/REPORT.md)
>
> `prism-research/`, `kvcached*/`, `prism-venv/`, 모델 가중치는 저장소에 없습니다 —
> `bootstrap.sh` 가 고정 SHA/lockfile로 재생성합니다.

Everything for reproducing / baselining **Prism: Cost-Efficient Multi-LLM Serving via
GPU Memory Ballooning** lives under `/workspace/prism-exp`.

> ⚠️ **`/workspace` on this instance is NOT a volume** (`workspace_is_volume: false`).
> Stop/start preserves it; **recycle or destroy wipes it**. Push anything you care
> about (configs, results, patches) to git or off-box storage.

---

## 1. What is where

| Path | What it is |
| --- | --- |
| `prism-research/` | **Prism itself** — a fork of SGLang `v0.3.4.post2` (`github.com/Multi-LLM/prism-research`). Contains the multi-model server, global placement, GPU-local scheduler, worker pool, model service, and the benchmark client. |
| `kvcached-prism/` | kvcached **branch `prism/shm`** — the balloon-driver version Prism links against (`kvcached.ops`, `kvcached.slab_allocator`). Not the same API as kvcached main. |
| `prism-venv/` | Python **3.10** venv for the two above. torch 2.4.0+cu121 / vllm 0.6.3.post1 / flashinfer 0.1.6 / transformers 4.45.2. |
| `kvcached/` | kvcached **main branch** (`v0.1.5`) — the standalone/published balloon driver, plus its own controller (router + sleep manager + traffic monitor) and micro-benchmarks. |
| `kvcached/engine_integration/sglang-pip-venv/` | Python **3.11** venv with SGLang **0.5.10** + kvcached 0.1.5. Completely independent of `prism-venv`. |
| `exp/` | Everything written for this setup: configs, launch/bench scripts, results. |
| `/workspace/.hf_home` | HF cache (`HF_HOME`). |

**Two independent stacks — do not mix them.**
`prism-venv` is the paper's actual system (old SGLang). `sglang-pip-venv` is the
modern kvcached-only stack, useful if you want to compare against current SGLang
or use kvcached's own benchmarks.

Services: **redis** runs as a supervisor service on `127.0.0.1:6379`
(`supervisorctl status redis`) — Prism's controller and GPU scheduler need it.

---

## 2. Quick start (Prism)

```bash
source /workspace/prism-exp/exp/scripts/env.sh          # activates prism-venv, sets HF_HOME

# 1) start a server (tmux session prism-<mode>)
exp/scripts/launch_server.sh <static|elastic|prism> <config.json> <port>

# 2) run the benchmark client against it
exp/scripts/run_bench.sh <exp-name> <num-models> <port> [benchmark.py args...]
```

### The three modes

| Mode | Flags added | Corresponds to |
| --- | --- | --- |
| `static` | – | **S-Partition** baseline (§7.1): per-model static KV pool |
| `elastic` | `--enable-elastic-memory --use-kvcached-v0` | Prism's **memory ballooning only** (§5) |
| `prism` | + `--enable-cpu-share-memory --enable-gpu-scheduler --enable-controller --policy simple-global --enable-model-service --enable-worker-pool --max-mem-usage --num-gpus` | **Full Prism** (§5 + §6): ballooning + global placement + slack-aware local arbitration |

`prism` mode is tuned via env vars read by `launch_server.sh`:
`NUM_GPUS`, `MAX_MEM` (per-GPU GiB budget), `WORKERS_PER_GPU`, `MODEL_SERVICE_WORKERS`.

### Verified working examples

```bash
source exp/scripts/env.sh

# --- static baseline, 2 models colocated on 1 GPU
exp/scripts/launch_server.sh static exp/configs/smoke_2model.json 30000
exp/scripts/run_bench.sh smoke_static 2 30000 --req-rate 4 --micro-benchmark

# --- elastic (kvcached ballooning)
exp/scripts/launch_server.sh elastic exp/configs/smoke_2model.json 30001
exp/scripts/run_bench.sh smoke_elastic 2 30001 --req-rate 4 --micro-benchmark --enable-elastic-memory

# --- full Prism, 8 models on 1 GPU, real trace
NUM_GPUS=1 MAX_MEM=67.28 WORKERS_PER_GPU=8 MODEL_SERVICE_WORKERS=4 \
  exp/scripts/launch_server.sh prism exp/configs/qwen_1gpu_8model_prism.json 30002
exp/scripts/run_bench.sh prism_8m_e2e 8 30002 \
  --e2e-benchmark --real-trace ./real_trace.pkl --time-scale 1 --replication 1 \
  --num-gpus 1 --enable-elastic-memory --ttft-slo-scale 5 --tpot-slo-scale 2
```

Results land in `exp/results/` (`*_key_metrics.tsv`, `*_all.jsonl`) and
`exp/results/requests/`. Server logs in `exp/server-logs/`
(`<mode>_stdout.log`, plus `<mode>.log.gpu_scheduler.log` / `.model_service.log`
— **read those two when full Prism mode fails**, the top-level log swallows the error).

---

## 3. Traces

* `prism-research/benchmark/multi-model/real_trace.pkl` **ships with the repo** —
  27 adapters / 1500 requests, the trace behind the §7.2 (8-model) and §7.4
  (18-model) e2e experiments. `--e2e-benchmark --real-trace ./real_trace.pkl`.
* Synthetic generators (`--micro-benchmark`, `--uniform-trace`, `--two-phase-trace`)
  are built in — no data files needed.
* The **Hyperbolic / Novita / Arena** production traces of §3 are **not public**.
  `benchmark.py` still has `--hyper-trace` / `--csv-trace` flags but the loader
  for them was stripped from the released code — `--csv-trace` is parsed and never
  used. To use those traces you must request them from the authors, or replay a
  substitute (e.g. LMSYS `lmsys-chat-1m`) through your own generator.
* Per-model **SLO baselines are hard-coded** in `trace.py`
  (`model_ttft_slo_baseline_p95` / `model_tpot_slo_baseline_p95`) and were measured
  for the paper's Llama mix. If you swap models, re-measure them on dedicated GPUs
  (paper §7.1) or your attainment numbers are not comparable.

---

## 4. Models

`exp/configs/*` currently use **Qwen2.5** (ungated) so everything runs without a token:

| Prism config slot | paper model | substitute in use |
| --- | --- | --- |
| large | `meta-llama/Llama-3.1-8B` | `Qwen/Qwen2.5-7B-Instruct` |
| mid | `meta-llama/Llama-3.2-3B` | `Qwen/Qwen2.5-3B-Instruct` |
| small | `meta-llama/Llama-3.2-1B` | `Qwen/Qwen2.5-1.5B-Instruct`, `Qwen/Qwen2.5-0.5B-Instruct` |

Downloaded already: Qwen2.5-0.5B/1.5B/3B/7B-Instruct.

**To use the paper's actual Llama/Mistral models** (gated on HF):

```bash
echo 'HF_TOKEN=hf_xxx' >> /workspace/.env       # picked up by exp/scripts/env.sh
source exp/scripts/env.sh
hf download meta-llama/Llama-3.2-1B
hf download meta-llama/Llama-3.2-3B
hf download meta-llama/Llama-3.1-8B
# then use prism-research/benchmark/multi-model/model_configs/*.json directly
```

Those Llama/Mistral models are **already present** in Prism's profiled
`model_info.json`, so no profiling step is needed for them.

### Adding any other model

Prism's GPU scheduler refuses to start for a model missing from
`prism-research/python/sglang/multi_model/utils/model_info.json`
(`ValueError: Model path ... not found in the profiled model info file`).
Register it with:

```bash
source exp/scripts/env.sh
python exp/scripts/profile_models.py <hf/model/path> [...]
```

(The Qwen2.5 models above have already been added — 28 entries total.)

---

## 5. Gotchas hit while building this (all already fixed here)

1. **`prism-research/install.md` assumes docker** (`lmsysorg/sglang:v0.3.4.post2-cu121`).
   This container cannot run docker-in-docker, so the stack was installed natively —
   see `setup_prism_env.sh` for the exact recipe.
2. **transformers must be pinned to 4.45.2.** The repo's `python[all]` extra has no
   upper bound, so a fresh resolve pulls transformers 5.x, which breaks vLLM
   0.6.3.post1 with `ImportError: cannot import name 'DTensor'`.
3. **`pyairports` 2.1.1 was removed from PyPI** (only an empty 0.0.1 placeholder
   remains), but vLLM 0.6.3.post1 pins `outlines<0.1`, which imports it. Installed
   from `git+https://github.com/NICTA/pyairports.git`, plus `setuptools<81`
   (it still uses `pkg_resources`).
4. **`profile_model_info.py` passes `cache_config=None`**, which crashes for Qwen2
   models in vLLM (`'NoneType' object has no attribute 'sliding_window'`).
   `exp/scripts/profile_models.py` supplies a real `CacheConfig`.
5. **`--workers-per-gpu` must be ≥ the number of `on: true` models on that GPU**,
   otherwise startup deadlocks with models stuck in `activating` forever (the
   top-level log shows nothing; `*.log.gpu_scheduler.log` shows the wait loop).
6. `max_memory_pool_size` in the config JSON is **GiB of KV pool per model**. In
   `elastic`/`prism` mode it is a virtual ceiling — over-subscribing it is the point.

---

## 6. Single-GPU vs the paper

This instance has **1× A100-80GB**. Reproducible as-is:

* §7.3 flexible cross-model memory sharing (2-model static vs elastic)
* §7.3 request arbitration (GPU-local scheduler on/off)
* §7.5 model activation latency (`--enable-cpu-share-memory` warm start)
* §A.3 elastic memory overhead (constant-rate worst case vs static partition)
* 8-model colocation on one GPU (a scaled-down §7.2)

Needs more GPUs: §7.2 (8 models / 2 GPUs), §7.3 global placement (needs ≥2 GPUs to
show load balancing), §7.4 (58 models / 32 GPUs), TP experiments.
The scripts already take `--num-gpus` / `NUM_GPUS`; only the config JSONs
(`gpu_ids`) need editing when you move to a bigger box.

---

## 7. Baselines from the paper not installed here

| Baseline | Status |
| --- | --- |
| S-Partition | ✅ = `static` mode |
| Prism | ✅ = `prism` mode |
| MuxServe / MuxServe++ | ❌ not installed (`github.com/hao-ai-lab/MuxServe`; MuxServe++ is the authors' SGLang port + kvcached, not released) |
| QLM | ❌ not installed (`github.com/QLM-project/QLM`) |
| ServerlessLLM | ❌ not installed (`github.com/ServerlessLLM/ServerlessLLM`) |

Each needs its own venv — they pin conflicting torch/vllm versions.

---

## 8. Sanity check: Llama on 1× A100-80G (full Prism mode)

`exp/scripts/run_sanity.sh <A|B|C>` runs three colocation cases against the shipped
`real_trace.pkl` (time_scale 1, replication 1, ~600 s), then
`exp/scripts/analyze_slo.py` recomputes the SLO stats.
`exp/scripts/summarize_sanity.py` prints them as one table.

| case | config | models |
| --- | --- | --- |
| A | `exp/configs/llama_1x8b.json` | `model_1` = Llama-3.1-8B |
| B | `exp/configs/llama_2x8b.json` | `model_1`, `model_4` = Llama-3.1-8B ×2 |
| C | `exp/configs/llama_8b_3b.json` | `model_1` = Llama-3.1-8B, `model_2` = Llama-3.2-3B |

Slot names are **not** arbitrary: `trace.py::generate_e2e_benchmark_reqs` hard-codes a
per-slot SLO baseline measured for a specific model, so `model_1/model_4/model_5` are the
Llama-3.1-8B slots and `model_2` is the Llama-3.2-3B slot. Case B uses `model_1 + model_4`
(not `model_1 + model_2`) so that both models get an 8B-derived SLO.

### Two harness defects this works around

1. **TPOT SLO unit mismatch.** `trace.py` stores `model_tpot_slo_baseline_p95` in
   **milliseconds**, but `benchmark.py` compares it against `output.tpot`, which is in
   **seconds** (`tpot = (finish_time - prefill_finish_time)/n`). The comparison
   `outputs[i].tpot < outputs[i].slo_tpot` is therefore always true and **every reported
   TPOT attainment is 1.0**. TTFT SLOs are in seconds and are correct.
   `analyze_slo.py` divides the TPOT baseline by 1000.
2. **No SLO stats at all on this code path.** Passing `--model-paths` together with
   `--real-trace` routes the client into `run_tp_mode`, which dumps raw per-request
   records and skips `get_benchmark_metrics` entirely — hence no attainment field in the
   released e2e result JSONs. `analyze_slo.py` recomputes from the raw dump.

Do not use `average_attainment_tpot` from `benchmark.py` when comparing against Prism.

Full write-up of that sweep — environment versions, dataset provenance, results,
the two harness defects, and caveats — is in **`exp/results/1-env-verification/REPORT.md`**.
Machine-readable table: `exp/results/1-env-verification/summary.tsv`.
