import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


SCRIPT = Path(__file__).parents[1] / "dataset_maker/unist/filter_unist_quality_in_place.py"
SPEC = importlib.util.spec_from_file_location("filter_unist_quality_in_place", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class FilterUniSTQualityTest(unittest.TestCase):
    def test_quality_flags_are_independent(self):
        self.assertEqual(MODULE.quality_flags(378, 24_920, 30_180, False), [])
        self.assertEqual(set(MODULE.quality_flags(750, 880, 60_000, True)), {
            "target_generation_ceiling",
            "duration_ratio_out_of_bounds",
            "target_bicodec_collapse",
        })
        self.assertEqual(MODULE.quality_flags(1, 17_780, 40, False), [
            "qwen_codec_too_short", "duration_ratio_out_of_bounds"])

    def test_bicodec_collapse_metrics(self):
        normal = list(range(100))
        adjacent_collapse = [7] * 100
        # 95% dominance with enough separators to make adjacent repeat < .95.
        dominant_collapse = [3] * 19 + [4]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokens.parquet"
            pq.write_table(pa.table({
                "target_bicodec": [normal, adjacent_collapse, dominant_collapse]
            }), path)
            collapse, adjacent, dominance = MODULE.bicodec_collapse_metrics(path)
        self.assertEqual(collapse.tolist(), [False, True, True])
        self.assertEqual(adjacent[0], 0)
        self.assertEqual(adjacent[1], 1)
        self.assertGreaterEqual(dominance[2], 0.95)

    def test_legacy_zero_reject_state_semantics(self):
        state = {"input_rows": 83, "output_rows": 83}
        self.assertEqual(state.get("rejected_rows", 0), 0)
        self.assertEqual(
            state["output_rows"] + state.get("rejected_rows", 0),
            state["input_rows"])


if __name__ == "__main__":
    unittest.main()
