import sys
import unittest
from pathlib import Path


STAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STAGE))

from compare_checkpoint_gate import candidate_row
from merge_text_probe import percentile


class MergeAndGateTest(unittest.TestCase):
    def test_percentile_is_deterministic_for_small_probe(self) -> None:
        self.assertEqual(percentile([4.0, 1.0, 3.0, 2.0], 0.95), 4.0)
        self.assertEqual(percentile([4.0, 1.0, 3.0, 2.0], 0.50), 3.0)

    def test_candidate_row_uses_bidirectional_mean(self) -> None:
        row = candidate_row(
            {
                "summary": {
                    "checkpoint_iteration": 600,
                    "bleu": {
                        "eng->cmn": {"score": 20.0},
                        "cmn->eng": {"score": 10.0},
                    },
                    "compute_rtf_source_mean": 0.4,
                    "compute_rtf_source_p95": 0.8,
                    "first_text_token_seconds_mean": 0.2,
                }
            }
        )
        self.assertEqual(row["iteration"], 600)
        self.assertEqual(row["average_bleu"], 15.0)


if __name__ == "__main__":
    unittest.main()
