# The hand-off fix, worked out but not applied

The diagnosis is in `FINDING.md`. This is the design that follows from it,
written down while it was fresh. **It has not been applied or tested** -- it was
worked out with too little time left to validate, and a half-applied patch chain
is worse than none (this session lost two runs to exactly that).

## The shape of the fix

Do not add a channel. Use the one that already exists, and add one message to it.

The engine's activation handshake with the model service is exactly three
ordered messages, one reader:

```
engine  (worker_pool_model_runner.py:357)  input_queue.put((model_key, engine_id, gpu_id, model))
service (model_sevice.py:410)              output_queue[engine_id].put("success")
service (model_sevice.py:433)              output_queue[engine_id].put(t1 - t0)
service (model_sevice.py:434)              output_queue[engine_id].put(self.service_id)
engine  (worker_pool_model_runner.py:359)  msg          = output_queue.get(timeout=300)
engine  (worker_pool_model_runner.py:366)  loading_time = output_queue.get()
engine  (worker_pool_model_runner.py:367)  service_id   = output_queue.get()
```

Append a fourth message carrying the capsules. One reader, one ordered channel,
so the race the current design has cannot occur.

**Service**, right after `put(self.service_id)`:

```python
if os.environ.get("PRISM_V6_KV_MIGRATION") == "1":
    _caps = self.v6_kv_stash.pop(model_key, [])
    self.output_queue[engine_id].put(_caps)
    logging.info(f"[PAPER-KV-V6] handover {model_key} -> {engine_id}: {len(_caps)} reqs")
```

**Engine**, right after `service_id = self.output_queue.get()`:

```python
if os.environ.get("PRISM_V6_KV_MIGRATION") == "1":
    self.v6_pending_kv = self.output_queue.get()
```

**Scheduler**, in `_v6_inject_migrated_kv`, replace the fetch round-trip with:

```python
caps = getattr(mr, "v6_pending_kv", None)
mr.v6_pending_kv = None
```

Then `__kv_fetch__` and its router entry are dead and should be deleted; keep
`__kv_stash__`, which is fire-and-forget and works.

## Why this shape and not the others

*A dedicated queue per engine* is the textbook answer and would also work, but it
means changing five signatures -- `launch_model_service`,
`launch_worker_pool_engines`, `launch_engine`, `run_scheduler_process`,
`ModelRunner` -- and every one of those is shared with the TP branch's territory.
The fourth-message version touches three call sites and none of them.

*A reply queue sent inside the request* does not work: `multiprocessing.Queue`
cannot be pickled through another queue ("Queue objects should only be shared
between processes through inheritance"). A `Manager().Queue()` proxy would, at
the cost of a manager process.

*Polling both channels* leaves the same race, only rarer.

## The one thing to be careful about

Both sides gate on the same env var, so the number of messages in flight can
never disagree -- but only if the flag reaches both processes. It does today:
`run_v4_case.sh` exports `PRISM_V6_KV_MIGRATION` into the tmux command line
(tmux does not inherit the launching shell's environment), and the model service
is forked from that env.

If the service's weight copy raises, its three puts do not all happen and the
engine blocks on a `get()` that never comes. That hazard already exists upstream
for the current three; the fourth adds no new exposure, but it is worth knowing
that this path has no timeout after the first message.

## How to validate

```bash
cd /workspace/prism-exp
# reset and replay -- never test a patch by re-running it on an already-patched tree
(cd prism-research && git checkout -- python/)
for f in patches/paper_faithful/apply_patches.py patches/paper_faithful_v3/apply_v3.py \
         patches/paper_faithful_v4/apply_v4.py patches/paper_faithful_v5_2/apply_v5_2.py \
         patches/paper_faithful_v6/apply_v6.py; do python3 $f --repo $PWD/prism-research; done
python3 exp/tests/test_kv_migration.py          # expect 25 passed
./exp/scripts/run_v6_validation.sh              # ~10 min
```

Pass looks like: `inject > 0`, `kv_transfers.jsonl` present with non-zero
`kv_bytes`, zero failed requests, and no `fetch timed out` anywhere. The
service should log `handover ... -> N reqs` with N matching a preceding stash.

Only then run the sweep (`exp/scripts/run_v6_sweep.sh`, ~2 h, self-publishing).
