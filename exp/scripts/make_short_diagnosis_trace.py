#!/usr/bin/env python3
import argparse
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(__file__))
from build_sharegpt_trace import _Unpickler

ap = argparse.ArgumentParser()
ap.add_argument("--input", required=True)
ap.add_argument("--output", required=True)
ap.add_argument("--duration", type=float, default=120.0)
a = ap.parse_args()
with open(a.input, "rb") as f:
    adapters, requests = _Unpickler(f).load()
kept = [r for r in requests if float(r.arrival_time) < a.duration]
os.makedirs(os.path.dirname(os.path.abspath(a.output)), exist_ok=True)
with open(a.output, "wb") as f:
    pickle.dump([adapters, kept], f)
print(f"wrote {a.output}: {len(kept)} requests, arrival < {a.duration}s")
