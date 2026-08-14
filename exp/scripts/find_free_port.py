#!/usr/bin/env python3
"""Print a base port with `--span` consecutive free ports, at or above `--from`.

Needed because launch_multi_model_server derives one port per model engine from
--port (port+1, port+2, ...) and refuses to start if --port itself is taken --
and it tests availability by binding 0.0.0.0, not 127.0.0.1. The 30000-40000
range is exactly where editor/tooling servers park their dynamic ports, so a
hard-coded base collides sooner or later and the run dies with
"Port N is not available" after the models have already been loaded.
"""
import argparse
import socket


def free(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("", port))
            s.listen(1)
            return True
        except OSError:
            return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", type=int, default=41000)
    ap.add_argument("--span", type=int, default=24)
    ap.add_argument("--step", type=int, default=50)
    ap.add_argument("--tries", type=int, default=200)
    a = ap.parse_args()
    base = a.start
    for _ in range(a.tries):
        if all(free(base + i) for i in range(a.span)):
            print(base)
            return
        base += a.step
    raise SystemExit("no free port block found")


if __name__ == "__main__":
    main()
