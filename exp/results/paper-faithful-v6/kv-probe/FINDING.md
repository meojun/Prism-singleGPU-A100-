# KV migration: what the code actually permits

Two things had to be settled before designing a KV migration, and reading the
source gave the wrong answer to one of them. Both are recorded here with the
probe that settled them (`kv_peer_probe.py`, run on this box).

## 1. There is no live KV at teardown time in ANY arm

Every migration path emits `DeactivateAction(preempt=False, ...)`:

* prototype — `simple_global.py:822-828` (idle eviction) and `:918-928` (migration)
* paper-faithful — `KVPRGlobalPolicy` subclasses `SimpleGlobalPolicy` and
  overrides only `_find_optimal_migrations`, which returns
  `List[(name, src, dst)]`. **Action emission stays in the parent**, so the
  paper-faithful arms inherit the same `preempt=False`.
* `patches/paper_faithful_v3/overlap_migration.py:39` also sets `preempt=False`
  (and is dead code besides).

`DeactivateAction`'s dataclass default is `preempt=True` (`action.py:82`), but
every construction site overrides it. With `preempt=False`,
`Scheduler.handle_deactivate_request` (`scheduler.py:1728`) takes
`_run_to_completion_normal()` (`:559-577`), which drains the running batch *and*
keeps pulling from the waiting queue until both are empty. Waiting requests are
separately returned to the frontend Redis queue and never had KV.

**So the KV is not dropped mid-flight — it is consumed to completion first.**
`kv_bytes == 0` everywhere is not merely missing accounting; there is genuinely
nothing to move.

This means a paper-faithful KV migration cannot be a pure addition. The paper's
mechanism presupposes preemption: retract in-flight requests, move their KV,
resume them on the target. The prototype instead drains, which *avoids needing*
KV migration at the cost of a long teardown. They are two different answers to
the same problem, and implementing the paper's requires switching the migration
path to `preempt=True, preempt_mode=RECOMPUTE` so that
`_retract_running_batch_normal` (`scheduler.py:1902-1954`) runs — which today
frees each retracted request's KV via `cache_finished_req` and then discards
the requeued `Req` objects, i.e. a full recompute from the prompt.

Per project convention (design_analysis.md §4) this goes behind an opt-in flag
so the default keeps reproducing the released prototype.

## 2. The transport is NOT blocked — a source-side gather makes it work

kvcached exposes six bindings and none of them copy
(`csrc/torch_bindings.cpp:37-51`); `csrc/page.cpp:26-41` calls `cuMemSetAccess`
with a single local `accessDesc` and never grants peer access. Reading that, a
peer read of a source `FTensor` should fault, which would force either a C++
change plus a rebuild, or a host bounce.

**Measured, it does not fault.** On this box, with `peer: True`:

```
--- same-GPU gather                                         ok
--- peer copy of the GATHERED tensor (normal allocator)     OK
--- DIRECT peer read of the VMM-backed FTensor view         OK
```

The third line is not the peer read it looks like. `k[0][idx]` is advanced
indexing with `idx` on the source device, so the gather executes **on GPU 0**
and materialises a temporary in the normal caching allocator; only that
temporary crosses the link. The VMM pages are never read from the peer.

That is not a workaround — it is the design. A request's KV slots are scattered
across pages shared with other requests (`slab_allocator.py:483-484`:
`page_id = idx * cell_size // PAGE_SIZE`), so a gather is required regardless of
transport. Gathering on the source and peer-copying the result is both necessary
and sufficient, and it reuses V4's existing machinery (`StreamPool`, `_emit`
JSONL accounting, the `v4_resident` registry).

**No kvcached C++ change is needed.** Recorded because the static reading said
otherwise, and rebuilding the extension would have been a day's detour.

## 3. What still has to move besides bytes

From `schedule_batch.py:172-270`, the per-request state that must travel:
`rid`, `origin_input_ids`, **`output_ids`** (the only record of decode progress),
`sampling_params`, `arrival_time`/`slo`, and the ordered slot vector
`req_to_token[req_pool_idx, :len(origin_input_ids)+len(output_ids)]`.

`req_pool_idx`, `prefix_indices` and `last_node` are source-local index space and
must be re-allocated on the target.

Note the existing serializer **loses `output_ids`**: `_convert_req_to_frontend_reqs`
(`scheduler.py:1772-1824`) rebuilds a `GenerateReqInput` from `origin_input_ids`
alone and hardcodes `output_len=512` (`:1822`). It cannot be reused as-is.

There is a pre-existing but dead hook for the arrival side:
`waiting_queue_stash` / `_restore_waiting_requests` (`scheduler.py:332, 1842-1844`)
is read on activation and never written. That is where migrated requests land.
