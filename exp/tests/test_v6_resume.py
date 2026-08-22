#!/usr/bin/env python3
"""CPU-only checks for joining migrated KV to a requeued target request."""

import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "patches/paper_faithful_v6/kv_migration_v6.py"
SPEC = importlib.util.spec_from_file_location("kv_migration_v6", MODULE_PATH)
KVM = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(KVM)


class FakeReq(SimpleNamespace):
    def __init__(self, rid, origin_input_text, origin_input_ids, sampling_params,
                 lora_path=None, arrival_time=None, slo=None):
        super().__init__(
            rid=rid,
            origin_input_text=origin_input_text,
            origin_input_ids=origin_input_ids,
            origin_input_ids_unpadded=origin_input_ids,
            output_ids=[],
            fill_ids=None,
            sampling_params=sampling_params,
            lora_path=lora_path,
            arrival_time=arrival_time,
            slo=slo,
            req_pool_idx=None,
            prefix_indices=[],
            extend_input_len=0,
            last_node=None,
            cached_tokens=0,
            tokenizer=None,
            stream=False,
        )


def make_pair(rid="request-1"):
    sampling = object()
    capsule = SimpleNamespace(
        rid=rid,
        origin_input_ids=[10, 11, 12],
        output_ids=[20, 21],
        sampling_params=sampling,
        arrival_time=12.5,
        slo=3.0,
        num_tokens=4,
        request_state={
            "origin_input_text": "the prompt",
            "lora_path": None,
            "stream": True,
            "decoded_text": "prior output",
        },
    )
    slots = torch.tensor([101, 37, 205, 9], dtype=torch.int32)
    return capsule, slots


class V6ResumeTests(unittest.TestCase):
    def test_prepare_resumed_request_rebuilds_target_scheduler_state(self):
        capsule, slots = make_pair()
        req = KVM.build_resumed_request(
            FakeReq, capsule, slots, "target-tokenizer")

        self.assertEqual(req.rid, "request-1")
        self.assertEqual(req.origin_input_text, "the prompt")
        self.assertEqual(req.origin_input_ids, [10, 11, 12])
        self.assertEqual(req.origin_input_ids_unpadded, [10, 11, 12])
        self.assertEqual(req.output_ids, [20, 21])
        self.assertEqual(req.fill_ids, [10, 11, 12, 20, 21])
        self.assertIs(req.sampling_params, capsule.sampling_params)
        self.assertEqual((req.arrival_time, req.slo), (12.5, 3.0))
        self.assertIsNone(req.req_pool_idx)
        self.assertIsNone(req.last_node)
        self.assertIs(req.prefix_indices, slots)
        self.assertEqual((req.extend_input_len, req.cached_tokens), (1, 0))
        self.assertEqual(req.tokenizer, "target-tokenizer")
        self.assertTrue(req.stream)
        self.assertEqual(req.decoded_text, "prior output")

    def test_prepare_resumed_request_rejects_slot_mismatch(self):
        capsule, slots = make_pair()
        with self.assertRaisesRegex(ValueError, "slot count"):
            KVM.build_resumed_request(
                FakeReq, capsule, slots[:-1], "target-tokenizer")

    def test_prepare_resumed_request_rejects_non_resumable_capsule(self):
        capsule, slots = make_pair()
        capsule.output_ids.append(22)
        with self.assertRaisesRegex(ValueError, "resumable prefix"):
            KVM.build_resumed_request(
                FakeReq, capsule, slots, "target-tokenizer")

    def test_recompute_fallback_preserves_progress_without_prefix_slots(self):
        capsule, _ = make_pair()
        req = KVM.build_recomputed_request(
            FakeReq, capsule, "target-tokenizer")

        self.assertEqual(req.output_ids, [20, 21])
        self.assertEqual(req.fill_ids, [10, 11, 12, 20, 21])
        self.assertEqual(req.prefix_indices, [])
        self.assertEqual(req.extend_input_len, 5)
        self.assertEqual(req.decoded_text, "prior output")


if __name__ == "__main__":
    unittest.main()
