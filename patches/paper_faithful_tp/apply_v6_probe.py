#!/usr/bin/env python3
"""Instrumentation for the V6 KV hand-off, and nothing else.

The V6 handover is explicit that guessing the cause here failed three times, so
this adds the two prints its 2(1) asks for and changes no behaviour.  It is a
separate applier rather than an edit to ``apply_v6.py`` because that file
belongs to the other branch; keeping it separate also means it can be dropped
in one step once the cause is known.

The symptom, from the V6 run:

    09:31:13.832  service: fetch Qwen2.5-7B -> 2 requests     put on the queue
    09:31:18.836  engine:  fetch timed out (5 s)              never arrived

The engine really was waiting on a queue -- with no ``kv_replies`` it would have
returned early and silently -- so both sides hold a queue and they are not the
same one.  Reading the code does not settle it: the two sides *look* like they
agree.  ``scheduler.py`` sends

    q.put(("__kv_fetch__", mr.model_path, None, mr.engine_id))

against the service's unpack

    model_key, engine_id, target_gpu_id, gpu_model = self.input_queue.get(...)

so ``engine_id`` there is really the model path and ``gpu_model`` is really the
engine id -- confusing, but consistent, and the stash lookup does find the two
capsules.  What is not visible from the source is whether
``v6_kv_queues[<engine id>]`` in the service process and
``kv_replies[<engine id>]`` in the engine process are the same object after
both crossed a spawn boundary.  That is what these prints settle.
"""

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def replace(path, old, new, probe=None):
    text = path.read_text()
    if probe is not None:
        hits = text.count(probe)
        if hits > 1:
            raise RuntimeError(f"probe not unique in {path} ({hits}): {probe[:80]!r}")
        if hits == 1:
            return
    n = text.count(old)
    if n == 0:
        if new in text:
            return
        raise RuntimeError(f"anchor not found in {path}: {old[:140]!r}")
    if n > 1:
        raise RuntimeError(f"anchor not unique in {path} ({n}): {old[:140]!r}")
    path.write_text(text.replace(old, new, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=str(ROOT / "prism-research"))
    ns = ap.parse_args()
    repo = Path(ns.repo).resolve()
    service = repo / "python/sglang/multi_model/model_sevice.py"
    sched = repo / "python/sglang/srt/managers/scheduler.py"

    # Service side: which key is written, and what keys exist here.
    replace(service,
        "                self.v6_kv_queues[gpu_model].put(caps)\n",
        "                logging.info(\n"
        "                    \"[KV-PROBE service] key=%r have=%r id=%r\"\n"
        "                    % (gpu_model, sorted(self.v6_kv_queues.keys()),\n"
        "                       id(self.v6_kv_queues.get(gpu_model)))\n"
        "                )\n"
        "                self.v6_kv_queues[gpu_model].put(caps)\n",
        probe="[KV-PROBE service]")

    # Engine side: which key is read, and what keys exist there.
    replace(sched,
        "            oq = getattr(q, \"kv_replies\", {}).get(mr.engine_id)\n",
        "            _kvp = getattr(q, \"kv_replies\", {})\n"
        "            logger.info(\n"
        "                \"[KV-PROBE engine] key=%r have=%r id=%r\"\n"
        "                % (mr.engine_id, sorted(_kvp.keys()),\n"
        "                   id(_kvp.get(mr.engine_id)))\n"
        "            )\n"
        "            oq = getattr(q, \"kv_replies\", {}).get(mr.engine_id)\n",
        probe="[KV-PROBE engine]")

    missing = [f"{p}: {n}" for p, n in ((service, "[KV-PROBE service]"),
                                        (sched, "[KV-PROBE engine]"))
               if n not in p.read_text()]
    if missing:
        raise RuntimeError("v6 probe verification failed:\n" + "\n".join(missing))
    print("v6 KV hand-off probe applied (instrumentation only)")


if __name__ == "__main__":
    main()
