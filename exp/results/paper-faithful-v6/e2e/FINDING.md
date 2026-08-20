# KV migration end to end: capture works, the handover channel does not

One pair of runs, bursty 8 req/s, seed 1, tau 0.00035, on this box.
`paper-faithful-v4` and `paper-faithful-v6` differ by exactly one flag.

**This is a validation, not a performance result.** One seed per arm cannot
support a goodput claim, and no goodput comparison is made below.

## What passed

| | v4 (control) | v6 (KV migration) |
| --- | ---: | ---: |
| requests completed | 3387 | 3387 |
| failed requests | 0 | 0 |
| weight transfers | 24 (7 gpu-to-gpu-p2p) | — |
| KV stash events | 0 | 6 |
| capture failures | 0 | 0 |

The control's `stash=0, inject=0` is itself evidence: with the flag off the v6
code paths are not entered at all, so the default arm still reproduces the
released prototype. And its 7 gpu-to-gpu weight transfers confirm v4's P2P path
working end to end on this box under load, not only in the microbenchmark.

**Capture works, and the volumes are real:**

```
 4 requests /    885 tokens /  101.5 MB   Llama-3.2-3B   gpu1
 7 requests /  1,523 tokens /   56.1 MB   Qwen2.5-3B     gpu0
89 requests / 40,066 tokens / 4,595.1 MB  Llama-3.2-3B   gpu0
 3 requests /    980 tokens /   36.1 MB   Qwen2.5-3B     gpu1
 3 requests /  1,100 tokens /   31.5 MB   Qwen2.5-1.5B   gpu1
 5 requests /  1,026 tokens /   37.8 MB   Qwen2.5-3B     gpu0
```

So the chain up to the hand-off is real: the migration preempts, the retract
path is taken, each in-flight request's KV is gathered off the source before
`cache_finished_req` frees it, and the bytes are measured. `kv_bytes` is no
longer a hardcoded 0.

The 89-request, 4.6 GB capture is worth noting on its own. That is the scale the
paper's mechanism exists for -- 40,066 tokens that the prototype would re-prefill
from scratch.

## What failed, and why

The service side of the hand-off also works:

```
06:51:44  stash meta-llama/Llama-3.2-3B from gpu 1 (4 requests)
06:53:13  fetch meta-llama/Llama-3.2-3B -> 4 requests      <- returned them
06:52:14  stash Qwen/Qwen2.5-3B-Instruct from gpu 0 (7 requests)
06:53:46  fetch Qwen/Qwen2.5-3B-Instruct -> 7 requests     <- returned them
```

But the engine never receives them:

```
06:53:43  GPU=1 Worker 2 (model_3) [PAPER-KV-V6] fetch timed out; falling back to recompute
06:54:16  GPU=0 Worker 2 (model_4) [PAPER-KV-V6] fetch timed out; falling back to recompute
06:57:26  GPU=1 Worker 0 (model_4) [PAPER-KV-V6] fetch timed out; falling back to recompute
06:58:28  GPU=1 Worker 2 (model_2) [PAPER-KV-V6] fetch timed out; falling back to recompute
```

**Cause: the reply channel already has an owner.** The wiring returns capsules
on the engine's `output_queue`, which is the weight-loading handshake channel --
`worker_pool_model_runner.py:277-279` and `:359-367` each take three blocking
`get()`s from it per activation. Under `activate_async` that loading runs on its
own thread, concurrently with the activation path where the KV fetch waits. Two
readers, one queue: whichever arrives first takes the message, so the capsules
are consumed by the weight loader and the fetch waits out its timeout.

This is a design mistake in the wiring, not a subtlety of the mechanism. The
`__release__` sentinel that v4 added is fire-and-forget and needs no reply, so
reusing the service's existing channels was safe there; a request/response
exchange is not the same thing and needs a channel of its own.

## The cost of the fallback

The fallback is safe -- a fetch that fails costs a recompute, exactly the
prototype's behaviour, and both arms completed 3387 requests with zero failures.
But it is **not free to measure against**: four timeouts at 30 s each put ~120 s
of blocked activation into a 420 s run. Any goodput comparison made in this
state would show v6 losing for a reason that has nothing to do with KV
migration.

**Do not run the sweep until the hand-off is fixed.** It would produce clean-
looking numbers that mean the opposite of what they appear to.

## The fix

A dedicated reply channel, threaded the way `input_queue` / `output_queue`
already are: `launch_worker_pool_engines` -> `launch_engine` ->
`run_scheduler_process` -> `ModelRunner`, and a matching dict on the service.
Then `__kv_fetch__` replies there and no existing protocol is disturbed.

Cheaper alternatives exist and are worse. Making the fetch non-blocking and
polling both channels leaves the same race, only less often. Pushing capsules
with the weight-load reply couples two mechanisms that fail independently.

While fixing it, drop the timeout well below 30 s. It exists to keep a failed
hand-off from hanging activation forever, and 30 s is far longer than any
legitimate reply needs.
