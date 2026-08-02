from __future__ import annotations

import unittest
from pathlib import Path


class StageBV2ValidationTest(unittest.TestCase):
    def test_validation_source_uses_causal_target_gate(self) -> None:
        root = Path(__file__).resolve().parents[2]
        source = (
            root / "training/simul_uniss/subsecond_v2/validate_stage_b_v2.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"target_edit_agreement"', source)
        self.assertIn('"full_teacher_edit_agreement"', source)
        self.assertIn('"first_correct_stable_coverage"', source)
        self.assertIn("minimum_target_agreement", source)


if __name__ == "__main__":
    unittest.main()
