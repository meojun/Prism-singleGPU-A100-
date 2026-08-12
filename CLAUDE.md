# Agent runbook — Prism (OSDI'26) experiment box

You are setting up or running Prism experiments on a freshly rented GPU server.
This file is the fast path. Read it fully before running anything; it will save
you the 4–5 traps below, each of which costs 20+ minutes to rediscover.

`SETUP.md` and `README.md` are the human-facing docs and go deeper. This file is
the ordered procedure plus the things that are only obvious after you have been
bitten.

---

## 0. What this repo is

A reproducible harness around **Prism**, a multi-LLM serving system (OSDI'26).
The repo holds no environment: `bootstrap.sh` rebuilds it from pinned SHAs and a
lockfile. Prism itself is a fork of SGLang v0.3.4.post2 plus `kvcached`, both
cloned by bootstrap.

Three server modes, mapping to the paper:

| mode | flags | paper |
| --- | --- | --- |
| `static` | – | S-Partition baseline |
| `elastic` | `--enable-elastic-memory --use-kvcached-v0` | §5 ballooning only |
| `prism` | + controller / gpu-scheduler / model-service / worker-pool | §5 + §6 full |

---

## 1. Check the box first (60 seconds, do not skip)

```bash
nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv
nvidia-smi topo -m | head -5     # NVLink between GPUs?
free -g; df -h /workspace; nproc
```

**Hard requirement: compute capability must be < 10.0.** The stack is pinned to
torch 2.4.0+cu121, which has no Blackwell kernels. On a B200 / RTX 50xx it
installs cleanly and then dies at the first GPU op with `no kernel image is
available`. Verified on A100 (8.0); H100 (9.0) and L40S (8.9) should work.
`bootstrap.sh` warns and prompts — if you see that prompt, stop and tell the
user to pick another machine rather than answering `y`.

Also note **`/workspace` is usually not a volume** (`vast-capabilities | jq
'.instance.workspace_is_volume'`). Recycle/destroy wipes it. Push results to git
before the box goes away.

---

## 2. Bootstrap (~20 min, mostly downloads)

```bash
apt-get install -y redis-server          # not in the base image
mkdir -p /etc/supervisor/conf.d          # keep redis alive across reboots
cat > /etc/supervisor/conf.d/redis.conf <<'EOF'
[program:redis]
environment=PROC_NAME="%(program_name)s"
command=/usr/bin/redis-server --bind 127.0.0.1 --port 6379 --save ""
autostart=true
autorestart=true
stdout_logfile=/dev/stdout
redirect_stderr=true
stdout_logfile_maxbytes=0
EOF
supervisorctl reread && supervisorctl update && redis-cli ping   # -> PONG

git clone https://github.com/meojun/Prism-singleGPU-A100- /workspace/prism-exp
cd /workspace/prism-exp
echo 'HF_TOKEN=hf_xxx' >> /workspace/.env && chmod 600 /workspace/.env
./bootstrap.sh                            # idempotent; re-run after any failure
```

Paths are hard-coded to `/workspace/prism-exp` in `exp/scripts/env.sh`. Clone
there or edit `PRISM_ROOT`.

The HF token must come from an account that accepted the **meta-llama** license
— Llama is gated and every download 401s otherwise. Ask the user for it; never
invent one. Run bootstrap in the background and poll: it takes ~20 min and will
blow past a foreground tool timeout.

Bootstrap is done when the verify block prints `OK` for torch 2.4.0, sglang
0.3.4.post2, vllm 0.6.3.post1, transformers 4.45.2, flashinfer 0.1.6,
kvcached+vmm_ops, and cuda available. Anything `BAD` means stop and fix.

---

## 3. Confirm the box reproduces the committed baseline

Always do this before trusting a new number. ~12 min per case.

```bash
source exp/scripts/env.sh
CUDA_VISIBLE_DEVICES=0 TAG=verify ./exp/scripts/run_sanity.sh A    # then B, C
```

`TAG` is mandatory — without it you overwrite the committed `results/1-env-verification/`.
`CUDA_VISIBLE_DEVICES=0` is mandatory on a multi-GPU box (trap 5 below).

Compare against `exp/results/1-env-verification/` and `exp/results/2-colocation/REPORT.md`. What must
hold: **case A attainment ≈ 1.0**, **case B's TPOT collapse** (model_1 ≈ 0.35,
model_4 ≈ 0.08), **case C high attainment**. Absolute latency shifts with the
GPU; those patterns must not. `exp/results/3-placement/REPORT.md` §3 has a
reference-vs-rerun table from a known-good run — decode metrics landed within 1%.

An isolated stray violation (1–2 requests out of 296) is noise, not a broken
box. A changed *pattern* is a broken box.

---

## 4. Run an experiment — the GPU count is a parameter

**Never hand-write a placement config.** `trace.py::generate_e2e_benchmark_reqs`
hard-codes a per-slot SLO baseline measured for one specific model, so putting
the wrong model in a slot silently compares against the wrong baseline. Generate
instead:

```bash
python exp/scripts/make_config.py --num-gpus <N> -o exp/configs/mine.json
```

The eight slots are fixed and are the paper's §7.2/§7.3 mix:

| slot | model | weights | requests in `real_trace.pkl` |
| --- | --- | ---: | ---: |
| model_1 | Llama-3.1-8B | 15.08 GiB | **296** |
| model_2 | Llama-3.2-3B | 6.00 | 22 |
| model_3 | Llama-3.2-1B | 2.28 | 22 |
| model_4 | Llama-3.1-8B | 15.08 | **262** |
| model_5 | Llama-3.1-8B | 15.08 | **120** |
| model_6 / 7 / 8 | Llama-3.2-1B | 2.28 ea | 19 / 11 / 2 |

Load is extremely skewed — the top three slots are 90% of requests. Placement
modes: `blocks` (default; 80/20 at N=2 — the naive split, use it when placement
*is* the variable), `roundrobin` (60/40), `balanced` (49/51 — use when you want
placement out of the way).

Then run. **`run_multigpu.sh` defaults `NGPU` to every visible GPU and generates
the matching config, so the same command adapts to a 1-, 2-, or N-GPU box:**

```bash
./exp/scripts/run_multigpu.sh glob_on  1      # full Prism,      time_scale 1
./exp/scripts/run_multigpu.sh glob_off 1      # no global controller
python exp/scripts/compare_fig7.py --tag fig7 --ts 1
```

`time_scale` multiplies arrival times, so **smaller = more load and a shorter
run**: 1.0 → 1× load / 600 s, 0.5 → 2× / 300 s, 0.25 → 4× / 150 s.

Knobs (all env vars): `NGPU`, `NMODELS` (≤8), `PLACEMENT`, `CFG`, `WORKERS`,
`MAXMEM`, `TAG`, `TRACE`, `TTFT_SCALE`, `TPOT_SCALE`. `WORKERS` auto-derives
from the config; override only to widen migration headroom.

### What each GPU count buys you

| | 1 GPU | 2 GPUs | ≥4 GPUs |
| --- | --- | --- | --- |
| §7.3 memory sharing (static vs elastic), §A.3 overhead, colocation | ✅ | ✅ | ✅ |
| §7.3 Fig 7 global placement, model migration path | ❌ impossible | ✅ | ✅ |
| §7.2 Fig 5 (8 models on 2 GPUs) — the paper's main e2e | ❌ | ✅ | ✅ |
| §5.3 parallel weight loading (Fig 10) | ❌ **silent no-op** | ✅ partial | ✅ |
| TP | ❌ | TP=2 | TP=4/8 as in the paper |
| §7.4 (58 models / 32 GPUs) | ❌ | ❌ | ❌ |

§5.3 is a no-op on one GPU because `model_sevice.py` computes
`broker_gpu_id = (broker_id + target_gpu_id + 1) % num_gpus` — with one GPU the
broker is always the target and the parallelism vanishes.

---

## 5. Traps — every one of these has cost real time

1. **tmux inherits a stale `CUDA_VISIBLE_DEVICES`.** tmux keeps *one server per
   user*, and every `new-session` inherits the environment that server first
   started with. Run a 1-GPU sanity sweep under `CUDA_VISIBLE_DEVICES=0`, then
   launch a 2-GPU server from the same tmux server, and every GPU>0 worker dies
   with `RuntimeError: CUDA error: invalid device ordinal`.
   `run_multigpu.sh` pins it explicitly and asserts `torch.cuda.device_count()`
   before launching. If you write a new launcher, do the same.
2. **tmux rewrites `.` to `_` in session names.** A session named `...ts0.5`
   becomes `ts0_5`, so `tmux has-session -t ...ts0.5` never matches: your
   readiness loop declares a perfectly healthy server DIED and **orphans it**,
   silently holding the GPUs. Normalise the name first.
3. **`num_gpus` is stamped into result filenames.** `benchmark.py` writes
   `{exp}_e2e_{num_gpus}gpu_...`, but `run_sanity.sh` globs a literal `1gpu`.
   Reuse that glob on 2 GPUs and the analysis step fails silently.
4. **Every GPU needs ≥1 `on: true` model.** `launch_multi_model_server` starts a
   GPU scheduler only for gpu_ids present in the initial placement, so a GPU
   that starts empty stays dead for the whole run.
5. **`launch_model_service()` reads `torch.cuda.device_count()`, not
   `--num-gpus`** (`multi_model_server.py:576`). Running a 1-GPU experiment on a
   2-GPU box without `CUDA_VISIBLE_DEVICES=0` gives you brokers for two GPUs and
   engines on one. Same function hard-codes `num_model_service_workers = 1`,
   overriding the flag.
6. **`--workers-per-gpu` must be ≥ on:true models on that GPU**, else startup
   deadlocks with models stuck in `activating`. The top-level log shows nothing.
7. **Never drop `--disable-radix-cache` with the shipped trace.**
   `real_trace.pkl` prompts are `"Hello "*n`, so short prompts are exact
   prefixes of long ones and radix cache serves **99.3%** of prefill from cache.
   Results become meaningless. Use `build_sharegpt_trace.py`'s `content` variant
   if you need real text.
8. **`benchmark.py`'s `average_attainment_tpot` is always 1.0** — `trace.py`
   stores TPOT baselines in ms and the comparison is against seconds. Use
   `analyze_slo.py`, which corrects the unit. Never quote the raw field.

### When a `prism`-mode server dies

The top-level log swallows the error. Read, in this order:

```
exp/server-logs/<exp>/server.log                        # worker tracebacks
exp/server-logs/<exp>/server.log.gpu_scheduler.log
exp/server-logs/<exp>/server.log.global_controller.log  # only if --enable-controller
exp/server-logs/<exp>/server.log.model_service.log
exp/server-logs/<exp>/stdout.log                        # least useful
```

Cleanup after a failed run — **kill sessions individually**, never
`tmux kill-server` (it kills the user's own shell):

```bash
tmux kill-session -t <session>; pkill -f launch_multi_model_server
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader   # expect 0 MiB
```

---

## 6. Paper vs released code — do not claim more than you ran

The repo faithfully wires up what upstream released, but **upstream's code
diverges from the paper**. Verified by reading `prism-research` at the pinned
SHA and confirmed at runtime:

- **Algorithm 1 (KVPR) is partially implemented.** Audited across all commits
  of both public repos: the names KVPR / `w_token_rate` / `shared_kv` appear
  nowhere. Component by component, the *denominator* is there
  (`memory_available_for_requests = gpu_mem - weights`, `simple_global.py:93`),
  but the *numerator* is a smoothed request count rather than
  `token_rate*token_size/SLO`, nothing sorts models by that ratio, and the τ
  threshold exists but is applied to a different metric. The goal — balance
  per-GPU memory pressure — is implemented; the specific metric is not, so an
  ablation here shows global placement helps without validating Algorithm 1.
- **Algorithm 2 (Moore-Hodgson): ingredients present, mechanism absent.** The
  deadline `a + s` and the execution estimate `p/c` are both computed
  (`request_queue.py:27,30`) and requests are popped in deadline order, but the
  feasibility check and the drop-the-longest-job step that the optimality proof
  rests on do not exist — it reduces to plain EDF. The admission control that
  would apply it is disabled anyway (`net_available = inf`).
- **Caveat on both.** `prism-research` is a 4-commit curated release from
  2025-08-09; the paper appeared at OSDI in July 2026. What is public may be an
  earlier or reduced snapshot. These are statements about the released code,
  not about what the authors built.
- **§6.2 admission control is disabled.** `request_queue.py:137` sets
  `net_available = float("inf")`; the GPU scheduler logs `net_available: inf`
  every second at runtime.
- **§6.1 overlapped migration is not implemented.** The code deactivates the
  source *then* activates the target (actions are sorted deactivate-first);
  there is no NVLink weight/KV transfer and no keep-serving-until-ready. In
  practice migration also never fires: `MEMORY_PER_REQUEST_RATIO_THRESHOLD = 15`
  (`simple_global.py:183`, hard-coded, not a flag) demands a 15× per-GPU
  imbalance, and a realistic one is ~1.6×.
- **Baselines MuxServe++/QLM/ServerlessLLM are not installed** (conflicting
  torch/vllm pins — each needs its own venv). Only S-Partition and Prism exist.
- **Production traces (Hyperbolic/Novita/Arena) are not public.** `--csv-trace`
  is parsed and never used.

`MODEL_IDLE_THRESHOLD = 50 s` does match the paper's ~45 s optimum (§A.4).

So: an ablation can show the global controller *helps*, but it cannot validate
Algorithm 1. Say which one you measured.

---

## 7. Where results go

| path | contents |
| --- | --- |
| `exp/results/1-env-verification/` | committed 1-GPU reference sweep (synthetic prompts) |
| `exp/results/2-colocation/` | committed ShareGPT + slowdown-SLO colocation study |
| `exp/results/1-env-verification/` | 1-GPU re-verification on a 2× A100 box |
| `exp/results/3-placement/` | 2-GPU §7.3 global placement ablation + `REPORT.md` |

Each experiment writes `<exp>_slo.json` (via `analyze_slo.py`),
`<exp>_actions.txt` (controller activity), `requests/` (raw per-request dump),
and `server-logs/<exp>/gpu_timeline.txt` (nvidia-smi samples).

House style for a new study: put a `REPORT.md` next to the results with
environment, method, numbers, and what you could *not* conclude.
`exp/results/3-placement/REPORT.md` is the template. Run each arm at least twice before
quoting tail metrics (p99) — single runs are fine for aggregates over hundreds
of requests, not for tails.
