#!/usr/bin/env python3
"""Regression tests for the V6 two-phase overlap contract."""

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HELPER = ROOT / "patches/paper_faithful_v6/action_order_v6.py"
SPEC = importlib.util.spec_from_file_location("action_order_v6", HELPER)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class Activate:
    def __init__(self, model_name="migrating"):
        self.model_name = model_name


class Deactivate:
    def __init__(self, model_name="migrating"):
        self.model_name = model_name


class Other:
    model_name = "other"


class V6ActionOrderTests(unittest.TestCase):
    def setUp(self):
        self.activate = Activate()
        self.deactivate = Deactivate()
        self.other = Other()
        self.actions = [self.activate, self.other, self.deactivate]

    def batches(self, overlap, kv):
        return MODULE.build_action_batches(
            self.actions, overlap, kv, Activate, Deactivate
        )

    def test_non_overlap_preserves_original_batch(self):
        self.assertEqual(self.batches(False, True), [(self.actions, 3)])

    def test_overlap_without_kv_remains_target_first(self):
        batches = self.batches(True, False)
        self.assertEqual([type(batch[0]) for batch, _ in batches],
                         [Activate, Other, Deactivate])
        self.assertEqual([workers for _, workers in batches], [1, 1, 1])

    def test_kv_migration_prepares_then_stashes_then_commits(self):
        batches = self.batches(True, True)
        self.assertEqual([type(batch[0]) for batch, _ in batches],
                         [Activate, Other, Deactivate, Activate])
        self.assertEqual(
            [getattr(batch[0], "phase", None) for batch, _ in batches],
            ["prepare", None, None, "commit"],
        )
        self.assertEqual([workers for _, workers in batches], [1, 1, 1, 1])

    def test_activation_without_source_is_not_split(self):
        activation = Activate("new-model")
        batches = MODULE.build_action_batches(
            [activation], True, True, Activate, Deactivate
        )
        self.assertEqual(batches, [([activation], 1)])
        self.assertFalse(hasattr(activation, "phase"))

    def test_policy_action_is_not_mutated_by_phase_split(self):
        self.batches(True, True)
        self.assertFalse(hasattr(self.activate, "phase"))


if __name__ == "__main__":
    unittest.main()
