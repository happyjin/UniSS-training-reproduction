from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from training.simul_uniss.subsecond_v2.validate_stage_c import validate


class FormalStageCValidationTest(unittest.TestCase):
    def _calibration(self, directory: Path, recalls: tuple[float, float, float]) -> Path:
        path = directory / "calibration.json"
        path.write_text(
            json.dumps(
                {
                    "scope": "formal_target_microphrase_safe_commit_v2",
                    "records": 100,
                    "positive_rate": 0.5,
                    "calibrated_ece": 0.05,
                    "operating_points": {
                        name: {"recall": recall, "precision": 1.0, "threshold": 0.5}
                        for name, recall in zip(("fast", "balanced", "quality"), recalls)
                    },
                }
            ),
            encoding="utf-8",
        )
        return path

    def test_passes_only_with_nonzero_operating_point_recall(self) -> None:
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            result = validate(
                self._calibration(root, (0.5, 0.2, 0.01)),
                minimum_fast_recall=0.01,
                minimum_balanced_recall=0.01,
                minimum_quality_recall=0.001,
                maximum_ece=0.2,
            )
            self.assertEqual(result["status"], "passed")

    def test_rejects_zero_quality_recall(self) -> None:
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            result = validate(
                self._calibration(root, (0.5, 0.2, 0.0)),
                minimum_fast_recall=0.01,
                minimum_balanced_recall=0.01,
                minimum_quality_recall=0.001,
                maximum_ece=0.2,
            )
            self.assertEqual(result["status"], "failed")


if __name__ == "__main__":
    unittest.main()

