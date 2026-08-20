# The fourth-message hand-off hangs. Reverted.

`NEXT_FIX.md` proposed carrying the capsules as a fourth message on the
activation handshake. It was implemented (`49c360e`), run, and **reverted**
(`git revert 49c360e`). This records what happened so the next attempt does not
repeat it.

## What it fixed

It did remove the race, and cleanly:

* `fetch timed out`: **4 -> 0**
* the service delivered: `handover meta-llama/Llama-3.2-3B -> 1_2: 4 reqs`
* capture kept working: 2 stashes, 4 requests / 875 tokens and
  7 requests / 3,704 tokens, zero capture failures

So the diagnosis in `../FINDING.md` was right and the direction was right.

## What it broke

The run **hung**. Twelve minutes after the last hand-off: both GPUs at 0%
utilisation, the benchmark parked on `Waiting for task req_model_1#39_model_1`,
no further service activity. The previous design at least completed -- 3387
requests, zero failures, paying 30 s per missed hand-off. This one does not
finish at all, which is strictly worse.

## Why -- the first hypothesis is REFUTED, and the real one is worse

**Refuted.** The guess below was that message counts got out of step. They do
not. Auditing every put and get on that channel:

| path | service puts | engine gets | balanced |
| --- | --- | --- | --- |
| success, v4 branch | `"success"` (410), elapsed (433), service_id (434), **caps** | 359, 366, 367, **new get** | yes |
| error | `None` (441), error string (442) | 359, then 364, then `continue` | yes |

Both paths balance, with and without the fourth message. Whatever hung the run,
it was not a desynchronised handshake.

**The likely real cause is tensor lifetime across CUDA IPC, and it invalidates
the planned fix as well.**

A capsule holds CUDA tensors on the *source* GPU. Putting it on a
`torch.multiprocessing.Queue` does not copy them -- it sends CUDA IPC handles.
So the model service ends up holding a mapping of the source engine's memory.
Then, immediately:

```
scheduler.py:2016   q.put(("__kv_stash__", key, gpu_id, self._v6_captured))
scheduler.py:2025   finally: self._v6_captured = []      # source drops its refs
scheduler.py:1759   self._v6_stash_captured()
scheduler.py:1760   self.tp_worker.deactivate_model_runner()   # frees GPU memory,
                                                              # empty_cache()
```

The source releases its only references and then tears the engine down, while
another process is still mapped to that memory. That is a use-after-free across
IPC, and a hang is a plausible way for it to present.

This is the same class of bug v4 hit with weights, and v4's fix is the tell: the
model service deliberately *keeps* the source copy alive
(`if not self.v4_p2p: del gpu_model`) and calls `torch.cuda.ipc_collect()` only
once the migration has displaced it. Capsules need the same discipline and do
not have it.

**What this changes for the next attempt.** A dedicated per-engine reply queue
fixes the channel, which was a real bug, but it does not touch this one -- the
lifetime problem is independent of which queue carries the handles. Any fix has
to keep the source-side tensors alive until the target has copied out of them,
or avoid crossing IPC with them at all (stage through host memory at the cost of
a bounce). Decide that before writing more plumbing.

### ...and that one does not survive either

Checked before building on it, and it is not supported.

`torch.multiprocessing.reductions.reduce_tensor` passes a `ref_counter_handle`
alongside the IPC handle -- torch maintains a **cross-process reference count**
for shared CUDA storages, so a producer that drops its Python reference does not
hand the memory back to the allocator while a consumer still holds it. That is
the exact race the hypothesis above described, and torch guards it.

The v4 experience points the same way, in the opposite direction: its bug was
that IPC-mapped memory was *retained* rather than freed, which is why it needed
`torch.cuda.ipc_collect()`. Retention, not premature release, is the failure
mode this machinery produces.

So the pattern is: dropping `self._v6_captured` after the put should be safe, and
the worst it should cost is memory held longer than expected.

## The honest state: the cause is unknown

Two hypotheses, both proposed with reasons and both undermined by checking. I am
not going to write a third; a confident-sounding guess is what cost this attempt
in the first place.

What is actually established:

* the fourth message **removed the race it targeted** -- timeouts 4 -> 0, and the
  service delivered capsules for the first time
* the run **hung** shortly after the first hand-off carrying a non-empty list
* it is **not** a message-count desync (audited, both paths balance)
* it is **not** obviously a producer-frees-early race (torch refcounts across
  processes)

What would settle it, in rough order of cost: log every `get()` on the engine's
`output_queue` with the type of what it received, so a message arriving where
another was expected becomes visible; stamp entry and exit of the service's
`while True` iteration to see whether the service itself stalled rather than the
engine; and watch source-GPU memory across the stash to see whether anything is
retained.

Until one of those runs, treat the fourth-message design as **known-broken for
an unknown reason** -- which is enough to not use it, and not enough to design
the replacement around.

## The original guess, kept for the record

The fourth message assumes the engine always performs a matching fourth `get()`.
That holds only on the one load path the patch edits
(`worker_pool_model_runner.py:359-372`). Any activation that returns weights
through another path -- the retry loop after `msg is None`, an activation the
service answers from a branch that is not the v4 one, a path where
`use_model_service` is false -- leaves the capsule list sitting unread in the
queue. The next activation's `msg = output_queue.get()` then receives a list of
capsules where it expected `"success"`, and the handshake is desynchronised from
that point on. Activation never completes, requests wait forever, GPUs idle.

That is consistent with the evidence: the hang begins right after the first
hand-off that actually carried capsules (`-> 1_2: 4 reqs`), and every earlier
hand-off carried `0 reqs` -- an empty list is still a message, but the engine had
been reading them fine, so what changed at that point was the content, not the
count. **This is a hypothesis, not a confirmed cause; nobody has yet instrumented
the read side to see which get() received what.** Confirm before designing
against it.

## What this means for the next attempt

The channel has to be one where an unread message cannot corrupt anything else.
That rules out sharing the weight handshake in any form, whether by an extra
message or by piggybacking on an existing one.

So the remaining option is the one `NEXT_FIX.md` set aside for cost rather than
correctness: **a dedicated per-engine reply queue**, threaded through
`launch_model_service` -> `launch_worker_pool_engines` -> `launch_engine` ->
`run_scheduler_process` -> `ModelRunner`. Five signatures, all of them in the TP
branch's territory, so coordinate before starting.

Give it a short timeout on the read, and have a missed reply fall back to
recompute -- the property the reverted design had and this one must keep.

The server logs from the hung run are beside this file.

---

# Update: capsules cross processes fine. The channel is where the problem is.

Before instrumenting the server, the cheaper question: is the hang about
*sharing the weight handshake*, or about *moving capsules between processes at
all*? Only the second would also sink a dedicated queue, so it had to be
settled first.

`ipc_capsule_probe.py` (beside this file) settles it without a server. A child
process receives capsules over a `torch.multiprocessing.Queue` and reads every
byte, while the parent does exactly what the source engine does next -- drops
its references, stops the allocator thread, and releases the kvcached pool.

```
parent: expected k/v sums {'r0': (2252800.0, 92364800.0), ...}
parent: 4 capsules, 13.8 MiB
parent: put returned in 0.000s
parent: references dropped and pool released
child read back:  [('r0', 220, 2252800.0, 92364800.0), ...]
RESULT: crossing succeeded, bytes MATCH
```

(The first version of this probe passed vacuously: it wrote `+X` to k and `-X`
to v and compared their sum, so both sides came to zero. Values are now distinct
per capsule *and* per layer, with k and v checked separately, and the script
asserts the expected values are non-zero and distinguishable before trusting a
match.)

**So the IPC hypothesis is dead for a second reason** -- not just guarded in
principle by torch's refcounting, but observed working in practice, including the
teardown that was supposed to break it.

## What this leaves

The hang is a property of **sharing the weight-loading handshake**, not of
capsule transport. That is good news for the plan: a dedicated per-engine reply
queue is not undermined by anything known, and can be built.

What the probe does *not* cover, and what an instrumented run still would:
it used a fresh `spawn` queue rather than the service's long-lived router
queues; one hand-off rather than many across repeated activations; 13.8 MiB
rather than the 4.6 GB a real capture reached; and the child read immediately.

## Revised recommendation

Build the dedicated per-engine queue. Instrumenting the abandoned
fourth-message design would explain a design nobody intends to use, and the one
thing that explanation could have changed -- whether transport itself is viable
-- is now answered.

Keep a short read timeout with a recompute fallback, and prove it on the real
path with a `run_v6_validation.sh` run before any sweep.
