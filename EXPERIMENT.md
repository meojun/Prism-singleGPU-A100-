# Rate-sweep experiments on 3x Llama-3.1-8B (2x A100-80G)

Commands to reproduce every run under `exp/results/{ref,probe,exp,burst}/`.
Findings and interpretation: [`exp/results/exp/REPORT_rate_sweep.md`](exp/results/exp/REPORT_rate_sweep.md).

Everything below drives **stock Prism**. No file under `prism-research/` or
`kvcached-prism/` is patched. The workload generator emits a pickle in the
format the harness already replays, and every metric is parsed out of logs the
stock engine/scheduler/controller already write.

---

## 0. Setup

```bash
source exp/scripts/env.sh
export SLO_BASE_FILE=/workspace/prism-exp/exp/configs/slo_base_3x8b_sharegpt.json
hf download anon8231489123/ShareGPT_Vicuna_unfiltered \
    ShareGPT_V3_unfiltered_cleaned_split.json --repo-type dataset \
    --local-dir $DATASETS/sharegpt          # 673 MB, ungated
```

## 1. Model configuration and why

Three **`meta-llama/Llama-3.1-8B`** instances in slots `model_1`, `model_4`,
`model_5`.

* **Slots are not free.** `benchmark.py` routes every `--model-paths` run into
  `trace.py::generate_e2e_benchmark_reqs`, which hard-codes a per-slot SLO
  baseline measured for one specific model. Slots 1/4/5 are the three
  Llama-3.1-8B slots; 2 is 3B and 3/6/7/8 are 1B. Driving 8B through slot 2
  would score it against a 3B-derived baseline.
* **Base, not `-Instruct`.** `Llama-3.1-8B-Instruct` is absent from
  `model_info.json`, so the GPU scheduler refuses to start
  (`ValueError: Model path ... not found in the profiled model info file`) until
  it is downloaded and run through `profile_models.py`. It would buy nothing:
  `benchmark.py:64-65` sends `ignore_eos=True` with
  `max_new_tokens=output_len`, so decode length is **forced** and
  instruction-tuning cannot move a single measured quantity. Same architecture
  -> same `cell_size` (131072 B/token) and the same 15.08 GiB of weights.
* **Placement is 1 + 2 by arithmetic.** Three models on two GPUs always leaves
  one GPU with two. With equal per-model rates the doubled GPU carries 2/3 of
  the load, and no placement policy can fix it -- moving a model just recreates
  the pair elsewhere. This asymmetry is the experiment's main lever, and it is
  what makes the migration path fire.

```bash
python exp/scripts/make_config.py --num-gpus 2 --slots 1,4,5 \
    --placement balanced -o exp/configs/llama_2gpu_3x8b.json
```

## 2. No-contention SLO baseline (paper 7.1)

The built-in SLO table was measured by the authors on different hardware. On
this box the same slot is **1.77x slower on TTFT and 1.57x on TPOT**, so the
built-in thresholds are unreachable even unloaded and "attainment" stops being
a signal. Re-derive with the repo's existing tool.

```bash
python exp/scripts/build_sharegpt_trace.py --variant rate --slots 1 \
    --phase-rates "4" --phase-len 180 --cv 1.0 --seed 42 \
    --out $DATASETS/sharegpt/ref_solo.pkl

SLOTS=1 NGPU=1 CFG=$PWD/exp/configs/llama_1gpu_solo8b.json \
  TAG=ref TRACE=$DATASETS/sharegpt/ref_solo.pkl \
  ./exp/scripts/run_multigpu.sh glob_on 1

python exp/scripts/derive_slo_baseline.py --run ref:glob_on_ts1:model_1 \
    --out exp/configs/slo_base_3x8b_sharegpt.json
# then copy model_1's row onto model_4 / model_5 -- same model, same workload
```

Measured: **TTFT p95 76.1 ms, TPOT p95 18.04 ms** (built-in: 42.9 / 11.46).
Reported SLO scales are x5 TTFT (380 ms) and x3 TPOT (54.1 ms). Scales are a
post-processing argument, so any other pair can be recomputed from the stored
request dumps without re-running.

## 3. Capacity profiling -- do not guess lambda

Rate is ramped **inside one run** and the results bucketed by arrival window, so
a capacity curve costs one run instead of one run per rate.

```bash
# low ramp: 1 -> 8 req/s in six 90 s steps
python exp/scripts/build_sharegpt_trace.py --variant rate --slots 1,4,5 \
  --phase-rates "0.333,0.333,0.333;0.667,0.667,0.667;1,1,1;1.333,1.333,1.333;2,2,2;2.667,2.667,2.667" \
  --phase-len 90 --cv 1.0 --seed 42 --out $DATASETS/sharegpt/probe_ramp.pkl

# high ramp: 8 -> 30 req/s in five 60 s steps
python exp/scripts/build_sharegpt_trace.py --variant rate --slots 1,4,5 \
  --phase-rates "2.667,2.667,2.667;4,4,4;5.333,5.333,5.333;7.333,7.333,7.333;10,10,10" \
  --phase-len 60 --cv 1.0 --seed 42 --out $DATASETS/sharegpt/probe_ramp_hi.pkl

SLOTS=1,4,5 NGPU=2 CFG=$PWD/exp/configs/llama_2gpu_3x8b.json \
  TAG=probe TRACE=$DATASETS/sharegpt/probe_ramp.pkl \
  ./exp/scripts/run_multigpu.sh glob_on 1
python exp/scripts/collect_metrics.py --exp probe_glob_on_ts1 --tag probe \
    --trace $DATASETS/sharegpt/probe_ramp.pkl --window 90
# repeat with probe_ramp_hi.pkl and --window 60
```

Both ramps overlap at 7.8 req/s and agree (1612 vs 1680 out tok/s, TTFT p95 151
vs 157 ms) -- a free reproducibility check. The TTFT knee is between 23 and 31
req/s, so **lambda_base = 12 req/s, about 46% of capacity**, inside the 40-60%
target.

> The two probe runs share `TAG=probe` and therefore overwrite each other's
> logs. The low ramp's CSVs were preserved as `results/probe/rampLO_*`. Use a
> distinct TAG per ramp if you rerun.

## 4. Experiment 1 (baseline) and 2 (contention)

One trace, swept with `--time-scale`. This keeps the **request set, prompt and
output lengths, per-model mix and seed byte-identical** across every rate point;
only the arrival clock is compressed. Regenerating per rate would change the
sample and confound the comparison.

```bash
python exp/scripts/build_sharegpt_trace.py --variant rate --slots 1,4,5 \
  --phase-rates "4,4,4" --phase-len 420 --cv 1.0 --seed 42 \
  --out $DATASETS/sharegpt/exp_base12.pkl        # 5090 reqs, 12.1 req/s

for ts in 1 0.8 0.6667 0.5 0.4; do               # 1.0x 1.25x 1.5x 2.0x 2.5x
  SLOTS=1,4,5 NGPU=2 CFG=$PWD/exp/configs/llama_2gpu_3x8b.json \
    TAG=exp TRACE=$DATASETS/sharegpt/exp_base12.pkl \
    TPOT_SCALE=3 TTFT_SCALE=5 \
    ./exp/scripts/run_multigpu.sh glob_on $ts
  python exp/scripts/collect_metrics.py --exp exp_glob_on_ts$ts --tag exp
done
```

`--time-scale t` yields rate `lambda_base / t`: 1.0 -> 12, 0.8 -> 15,
0.6667 -> 18, 0.5 -> 24, 0.4 -> 30 req/s. Experiment 1 is the `ts=1` point;
Experiment 2 is the rest.

## 5. Burst scenario -- growing number of hot models

Same three models, per-model rates stepped so the count of simultaneously hot
models goes 1 -> 2 -> 3 while model_1 stays pinned at 8 req/s throughout.
Holding one model's rate constant is what makes cross-model interference
measurable.

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

"low" is 0.5 req/s rather than 0: at 0 the models would cross
`MODEL_IDLE_THRESHOLD = 50 s` and be evicted, which measures eviction and
cold-start instead of the KV-cache redistribution this scenario is aimed at.

## 6. Outputs

| file | contents |
| --- | --- |
| `<exp>_slo.json` | per-model attainment, mean/p50/p95/p99 TTFT & TPOT, e2e, goodput |
| `<exp>_summary.csv` | one row: the rate-vs-X table columns |
| `<exp>_timeseries.csv` | 1 s bins -- per-model arrivals / running / queue / KV tokens / KV pool fraction / decode throughput, per-GPU scheduler queue / device memory / utilisation, controller actions |
| `<exp>_windows.csv` | per-arrival-window request stats (the capacity curve) |
| `<exp>_actions.txt` | activations / deactivations / migrations |
| `server-logs/<exp>/gpu_timeline.txt` | raw nvidia-smi samples @2 s |

The time-series columns are the ones needed to plot
burst -> contention -> scheduling/memory change -> latency on a common clock.

## 7. Two metrics that cannot be produced, and why

* **`rejected` is always 0.** Not unobserved -- impossible.
  `request_queue.py:137` sets `net_available = float("inf")` with the comment
  *"the actual implementation doesn't seem to limit resources"*, so the
  memory-based admission control described in 6.2 of the paper never rejects.
  The runtime log prints `net_available: inf` every second, and
  `collect_metrics.py` records that string so the claim stays checkable.
* **Queue length is ~0 below saturation.** Everything is admitted immediately,
  so back-pressure appears as `#running-req` and TTFT rather than queue depth.
  Queues only form past saturation (at 30 req/s: model queue 38, scheduler
  queue 184). Read `*_running` and TTFT p95, not queue length, as the load
  signal.
