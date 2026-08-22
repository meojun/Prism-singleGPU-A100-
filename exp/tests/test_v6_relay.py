#!/usr/bin/env python3
"""Three-process CUDA IPC test for the V6 source -> service -> target relay."""

import queue
import sys
from pathlib import Path

import torch
import torch.multiprocessing as mp


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "patches/paper_faithful_v6"))
import kv_migration_v6 as kvm  # noqa: E402


def source_process(outgoing, release):
    torch.cuda.set_device(0)
    base = torch.arange(32, dtype=torch.float32, device="cuda:0")
    capsule = kvm.RequestKVCapsule(
        "relay-test", "model", [1], [2], {}, 0.0, 1.0,
        [base], [base + 100], 0,
    )
    outgoing.put([capsule])
    release.wait(30)


def relay_process(incoming, outgoing, release, status):
    try:
        capsules = incoming.get(timeout=20)
        kvm.clone_capsules_for_relay(capsules)
        outgoing.put(capsules)
        status.put(("relay", "ok"))
        release.wait(30)
    except Exception as exc:
        status.put(("relay", repr(exc)))


def target_process(incoming, status):
    try:
        capsules = incoming.get(timeout=20)
        capsule = capsules[0]
        expected_k = torch.arange(32, dtype=torch.float32)
        expected_v = expected_k + 100
        if not torch.equal(capsule.k[0].cpu(), expected_k):
            raise AssertionError("relay changed K bytes")
        if not torch.equal(capsule.v[0].cpu(), expected_v):
            raise AssertionError("relay changed V bytes")
        capsule, path = kvm.transfer_capsule(capsule, 1)
        if capsule.k[0].device.index != 1:
            raise AssertionError("target transfer did not reach GPU 1")
        if not torch.equal(capsule.k[0].cpu(), expected_k):
            raise AssertionError("target transfer changed K bytes")
        status.put(("target", path))
    except Exception as exc:
        status.put(("target", repr(exc)))


def main():
    if torch.cuda.device_count() < 2:
        print("SKIP: V6 relay test needs two GPUs")
        return 0

    ctx = mp.get_context("spawn")
    source_to_relay = ctx.Queue()
    relay_to_target = ctx.Queue()
    status = ctx.Queue()
    release = ctx.Event()
    processes = [
        ctx.Process(target=source_process, args=(source_to_relay, release)),
        ctx.Process(target=relay_process,
                    args=(source_to_relay, relay_to_target, release, status)),
        ctx.Process(target=target_process, args=(relay_to_target, status)),
    ]
    for process in processes:
        process.start()

    processes[2].join(40)
    release.set()
    for process in processes[:2]:
        process.join(15)

    messages = []
    while True:
        try:
            messages.append(status.get(timeout=1))
        except queue.Empty:
            break

    print("messages:", messages)
    print("exitcodes:", [process.exitcode for process in processes])
    relay_ok = ("relay", "ok") in messages
    target_ok = any(name == "target" and result in (
        "gpu-to-gpu-p2p", "via-host"
    ) for name, result in messages)
    clean_exit = all(process.exitcode == 0 for process in processes)
    if not (relay_ok and target_ok and clean_exit):
        return 1
    print("V6 THREE-PROCESS CUDA RELAY PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
