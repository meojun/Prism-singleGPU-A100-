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

## Why, most likely

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
