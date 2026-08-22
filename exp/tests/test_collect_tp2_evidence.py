import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "collect_tp2_evidence.py"
SPEC = importlib.util.spec_from_file_location("collect_tp2_evidence", SCRIPT)
COLLECT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COLLECT)


class RankMapTests(unittest.TestCase):
    def test_tp1_helpers_are_not_merged_into_tp_rank_zero(self):
        text = "\n".join([
            "[PAPER-TP] engine rank: tp_rank=0 gpu_id=0 tp_size=2 model_service=False",
            "[PAPER-TP] engine rank: tp_rank=0 gpu_id=0 tp_size=1 model_service=True",
            "[PAPER-TP] engine rank: tp_rank=0 gpu_id=1 tp_size=1 model_service=True",
            "[PAPER-TP] engine rank: tp_rank=1 gpu_id=1 tp_size=2 model_service=False",
        ])

        self.assertEqual(COLLECT.rank_gpu_map([("server.log", text)]),
                         {0: [0], 1: [1]})


if __name__ == "__main__":
    unittest.main()
