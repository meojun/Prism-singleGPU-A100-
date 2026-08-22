#!/usr/bin/env python3
"""Build the three-request mixed-model Algorithm-2 ordering trace.

The production placement puts model_1 and model_4 on the same GPU, so this is
the runtime equivalent of the abstract A(model1), B(model2), C(model1) unit
sequence: two independent backend consumers sharing one GPU arbitration set.
"""

import argparse
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(__file__))
from build_sharegpt_trace import Request, _Unpickler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.source, "rb") as f:
        _, source_requests = _Unpickler(f).load()
    prompts = {r.model: r for r in source_requests if r.model in {"model_1", "model_4"}}
    if set(prompts) != {"model_1", "model_4"}:
        raise SystemExit("source trace lacks model_1/model_4 prompts")

    spec = [
        # benchmark.py applies TTFT_SCALE=5.  Keep ample headroom for the
        # launcher's post-readiness warmups while retaining strict A<B<C EDF
        # deadlines in the real Algorithm-2 call.
        ("A", "model_1", 10.0),
        ("B", "model_4", 20.0),
        ("C", "model_1", 30.0),
    ]
    requests = []
    for rid, model, slo_ttft in spec:
        source = prompts[model]
        req = Request()
        req.req_id = rid
        req.prompt = source.prompt
        req.prompt_len = source.prompt_len
        req.output_len = 8
        req.arrival_time = 0.0
        req.model = model
        req.slo = slo_ttft
        req.slo_ttft = slo_ttft
        req.slo_tpot = max(float(source.slo_tpot), 0.02)
        requests.append(req)

    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "wb") as f:
        pickle.dump([["__PRISM_DIRECT__"], requests], f)
    print(f"wrote {output}: A(model_1), B(model_4), C(model_1), same GPU")


if __name__ == "__main__":
    main()
