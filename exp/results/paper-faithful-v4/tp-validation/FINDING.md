# TP=2 does not run in the worker-pool path

Two independent configurations were tried and both failed identically, at model
activation, with:

    ValueError: Model model_5 not found in shared cpu models

1. **Mixed TP** — one model at `tp_size: 2` spanning GPUs [0,1], one at
   `tp_size: 1`, server `--tensor-parallel-size 1`.
2. **Uniform TP** — both models at `tp_size: 2` spanning [0,1], server
   `--tensor-parallel-size 2`.

## Why

Worker-pool engines are not per-model. `launch_worker_pool_engines`
(`multi_model_server.py`) creates one engine per (GPU, worker slot):

```python
for gpu_id in range(num_gpus):
    for worker_id in range(workers_per_gpu):
        engine_launch_args.append((server_args, port_args, [gpu_id], ...))
```

Every engine is bound to exactly **one** GPU — `[gpu_id]`. There is no
construct in this path for a model's tensor-parallel shards to span two GPUs.
The weights are looked up as `(model_path, tp_size)`
(`model_runner.py::_get_cpu_model_ref`), so an engine and a model that disagree
about `tp_size` can never meet, which is the error above.

The global controller is built the same way. It collapses a TP group to its
rank-0 GPU, with the reason stated in the source:

```python
# NOTE(ke): For TP case, only consider rank0 state
gpu_ids = set([mod.gpu_ids[0] for mod in models])
```

So a TP group is invisible to placement as a multi-GPU object even in
principle.

## What this means for the study

`--enable-worker-pool` is not optional here: the GPU scheduler, the migration
machinery and everything Prism's §6 describes live in that path. TP > 1 and
Prism's scheduling are therefore mutually exclusive in this prototype, and the
anti-affinity constraint the paper places on TP shards has nothing to constrain.

Two further consequences, both recorded rather than worked around:

* `model_runner.py` disables the model-service weight path when `tp_size > 1`
  ("Tensor parallelism is enabled, model service will not be used"), so a TP
  model would not use parallel weight loading even if it ran.
* The study's TP=2 verdict is **FAIL**, and it is a property of the prototype,
  not of this harness. No TP=2 latency numbers are reported, because none were
  produced.

Raw evidence: `server-logs/` from both attempts, and `tp2_validation.json`.
