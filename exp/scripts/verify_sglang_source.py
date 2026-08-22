#!/usr/bin/env python3
"""Fail fast if a worktree run imports sglang from a different source tree."""

import sys
from pathlib import Path

import sglang


expected = (Path(sys.argv[1]).resolve() / "python")
actual = Path(sglang.__file__).resolve()
print(f"[PRISM-SOURCE] expected={expected} actual={actual}", flush=True)
if not actual.is_relative_to(expected):
    raise SystemExit("FATAL: sglang import escaped the selected PRISM_REPO")
