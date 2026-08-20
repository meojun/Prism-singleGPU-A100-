#!/usr/bin/env python3
"""Send a handful of requests to a running multi-model server and record what happened.

The payload shape is ``benchmark.py::send_generate_request``'s, deliberately.
The field is ``model``, not ``model_name``, and the SLO fields feed the
admission path.  A request with the wrong key is accepted with 200 OK and then
never dispatched, so the client hangs -- which reads exactly like a TP deadlock
and is not one.  That cost a cycle here; hence the note.

Usage: tp_probe_requests.py <port> <model_name> <out.json> [max_new_tokens]
"""

import json
import sys
import time
import urllib.request
import uuid

PROMPTS = [
    "The capital of France is",
    "Explain in one sentence why the sky is blue:",
    "List three prime numbers:",
    "Translate to French: good morning.",
]


def main():
    port, model, out = sys.argv[1], sys.argv[2], sys.argv[3]
    max_new = int(sys.argv[4]) if len(sys.argv) > 4 else 24

    ok, failed, errors, lat, samples = 0, 0, [], [], []
    for i, prompt in enumerate(PROMPTS):
        pload = {
            "text": prompt,
            "sampling_params": {"ignore_eos": True, "max_new_tokens": max_new},
            "rid": f"tpboot_{i}_{uuid.uuid4().hex[:8]}",
            "model": model,
            "slo": 60.0,
            "slo_ttft": 10.0,
            "slo_tpot": 1.0,
            "prompt_len": len(prompt.split()),
            "output_len": max_new,
        }
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/generate",
            data=json.dumps(pload).encode(),
            headers={"Content-Type": "application/json", "User-Agent": "tp-boot"},
        )
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                body = r.read().decode(errors="replace")
            dt = time.time() - t0
            lat.append(dt)
            ok += 1
            samples.append(body[:400])
            print(f"  ok  ({dt:5.2f}s) {prompt[:32]!r} -> {body[:130]}")
        except Exception as e:  # noqa: BLE001 -- the failure mode is the datum
            failed += 1
            errors.append(f"{type(e).__name__}: {e}")
            print(f"  FAIL {prompt[:32]!r}: {type(e).__name__}: {e}")

    json.dump(
        {
            "summary": {
                "requests": len(PROMPTS),
                "successful": ok,
                "failed": failed,
                "errors": errors,
                "latency_s": {
                    "mean": (sum(lat) / len(lat)) if lat else None,
                    "all": lat,
                },
            },
            "responses": samples,
        },
        open(out, "w"),
        indent=2,
    )
    print(f"=== requests: {ok} ok / {failed} failed")
    return 0 if failed == 0 and ok > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
