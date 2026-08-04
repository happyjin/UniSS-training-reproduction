import sys
import unittest
from pathlib import Path


STEP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STEP))

from compare_step1_gate import candidate_row


class Step1GateTest(unittest.TestCase):
    def test_candidate_uses_bidirectional_mean(self) -> None:
        row = candidate_row(
            {
                "summary": {
                    "checkpoint_iteration": 500,
                    "bleu": {
                        "eng->cmn": {"score": 24.0},
                        "cmn->eng": {"score": 20.0},
                    },
                    "compute_rtf_source_mean": 0.4,
                    "compute_rtf_source_p95": 0.8,
                    "first_text_token_seconds_mean": 0.2,
                }
            }
        )
        self.assertEqual(row["name"], "Step1 iter 500")
        self.assertEqual(row["average_bleu"], 22.0)


if __name__ == "__main__":
    unittest.main()
