from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from experiments.evaluation.uniss_phase3_prefix_streaming_v3_inference_v1.checkpoint_selection import (
    SELECTION_METRICS,
    parse_validation_rows,
    rank_rows,
)


class CheckpointSelectionTest(unittest.TestCase):
    def test_parser_and_rank_only_saved_iterations(self) -> None:
        def line(iteration: int, values: list[float]) -> str:
            fields = " ".join(
                f"| {name} value: {value:.6E} "
                for name, value in zip(SELECTION_METRICS, values)
            )
            return f"validation loss at iteration {iteration} {fields}|\n"

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "train.log"
            path.write_text(
                line(500, [2, 2, 2, 2, 2, 2])
                + line(750, [0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
                + line(1000, [1, 1, 1, 1, 1, 1]),
                encoding="utf-8",
            )
            rows = parse_validation_rows(path)
            ranked = rank_rows(rows, {500, 1000})
        self.assertEqual([row["iteration"] for row in ranked], [1000, 500])


if __name__ == "__main__":
    unittest.main()

