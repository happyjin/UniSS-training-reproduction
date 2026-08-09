#!/usr/bin/env python3
"""Checks the CTC feasibility arithmetic behind the upsample-ratio recommendation.

Run directly:
``python experiments/simul_s2st_route_v1/step2_nar_ctc_head/tests/test_measure_upsample_ratio.py``
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from torch.nn import functional as F  # noqa: E402

from experiments.simul_s2st_route_v1.step2_nar_ctc_head.measure_upsample_ratio import (  # noqa: E402
    CURRENT_RATIO,
    Row,
    adjacent_repeats,
    feasibility,
    iter_lines,
    partition_degenerate,
    read_rows,
    smallest_ratio,
)


class RepeatTest(unittest.TestCase):
    def test_counts_only_adjacent_pairs(self) -> None:
        self.assertEqual(adjacent_repeats([]), 0)
        self.assertEqual(adjacent_repeats([5]), 0)
        self.assertEqual(adjacent_repeats([5, 5]), 1)
        self.assertEqual(adjacent_repeats([5, 5, 5]), 2)
        self.assertEqual(adjacent_repeats([5, 6, 5]), 0)
        self.assertEqual(adjacent_repeats([1, 1, 2, 2, 2, 3]), 3)


class FeasibilityTest(unittest.TestCase):
    def test_matches_pytorch_ctc_on_the_exact_boundary(self) -> None:
        """The rule must agree with what ``F.ctc_loss`` can actually align."""

        targets = torch.tensor([7, 7, 9], dtype=torch.long)
        required = len(targets) + adjacent_repeats(targets.tolist())
        self.assertEqual(required, 4)
        for frames, expect_finite in ((required - 1, False), (required, True)):
            logits = torch.zeros(1, frames, 11).log_softmax(-1).transpose(0, 1)
            loss = F.ctc_loss(
                logits,
                targets,
                torch.tensor([frames]),
                torch.tensor([len(targets)]),
                blank=10,
                reduction="sum",
                zero_infinity=False,
            )
            self.assertEqual(bool(torch.isfinite(loss)), expect_finite)

    def test_ratio_must_cover_repeats_not_just_length(self) -> None:
        # 4 text tokens, 8 units, all identical: needs 8 + 7 = 15 frames, so ratio 2 fails
        # even though 2 * 4 = 8 equals the unit count.
        rows = [Row(direction="eng->cmn", text_length=4, unit_length=8, adjacent_repeats=7)]
        self.assertEqual(feasibility(rows, 2)["feasible_fraction"], 0.0)
        self.assertEqual(feasibility(rows, 4)["feasible_fraction"], 1.0)

    def test_relative_cost_is_quadratic_and_anchored_at_the_current_ratio(self) -> None:
        rows = [Row(direction="eng->cmn", text_length=10, unit_length=20, adjacent_repeats=0)]
        self.assertAlmostEqual(feasibility(rows, CURRENT_RATIO)["relative_attention_cost"], 1.0)
        halved = feasibility(rows, CURRENT_RATIO // 2)["relative_attention_cost"]
        self.assertAlmostEqual(halved, 0.25, places=6)

    def test_occupancy_reports_how_much_of_the_lattice_is_real(self) -> None:
        rows = [Row(direction="eng->cmn", text_length=10, unit_length=100, adjacent_repeats=0)]
        self.assertAlmostEqual(feasibility(rows, 20)["mean_lattice_occupancy"], 0.5, places=6)

    def test_smallest_ratio_walks_the_grid_in_order(self) -> None:
        rows = [
            Row(direction="eng->cmn", text_length=10, unit_length=100, adjacent_repeats=0),
            Row(direction="cmn->eng", text_length=10, unit_length=300, adjacent_repeats=0),
        ]
        self.assertEqual(smallest_ratio(rows, 0.5, [8, 16, 32, 64]), 16)
        self.assertEqual(smallest_ratio(rows, 1.0, [8, 16, 32, 64]), 32)
        self.assertIsNone(smallest_ratio(rows, 1.0, [8, 16]))


class PartitionTest(unittest.TestCase):
    def test_splits_on_required_frames_and_keeps_every_row(self) -> None:
        rows = [
            Row(direction="eng->cmn", text_length=10, unit_length=200, adjacent_repeats=0),
            Row(direction="cmn->eng", text_length=1, unit_length=500, adjacent_repeats=0),
            Row(direction="cmn->eng", text_length=2, unit_length=200, adjacent_repeats=0),
        ]
        healthy, degenerate = partition_degenerate(rows, 100.0)
        self.assertEqual(len(healthy) + len(degenerate), len(rows))
        self.assertEqual([row.text_length for row in healthy], [10, 2])
        self.assertEqual([row.text_length for row in degenerate], [1])

    def test_boundary_row_counts_as_healthy(self) -> None:
        rows = [Row(direction="eng->cmn", text_length=2, unit_length=200, adjacent_repeats=0)]
        healthy, degenerate = partition_degenerate(rows, 100.0)
        self.assertEqual(len(healthy), 1)
        self.assertEqual(degenerate, [])


class SamplerTest(unittest.TestCase):
    def write(self, directory: str, count: int) -> Path:
        path = Path(directory) / "lines.jsonl"
        path.write_text("".join(f'{{"n": {index}}}\n' for index in range(count)), encoding="utf-8")
        return path

    def test_reads_everything_when_sampling_is_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self.write(directory, 50)
            self.assertEqual(len(list(iter_lines(path, None))), 50)
            self.assertEqual(len(list(iter_lines(path, 0))), 50)

    def test_sampled_lines_are_complete_distinct_and_spread(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self.write(directory, 1000)
            drawn = [json.loads(line)["n"] for line in iter_lines(path, 10)]
        self.assertEqual(len(drawn), 10)
        self.assertEqual(len(set(drawn)), 10)
        self.assertEqual(drawn, sorted(drawn))
        # The first line must not be skipped, and the sample must reach the far end.
        self.assertEqual(drawn[0], 0)
        self.assertGreater(drawn[-1], 850)

    def test_requesting_more_lines_than_exist_stops_cleanly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self.write(directory, 3)
            drawn = [json.loads(line)["n"] for line in iter_lines(path, 100)]
        self.assertTrue(drawn)
        self.assertTrue(all(0 <= value < 3 for value in drawn))


class ManifestTest(unittest.TestCase):
    def test_skips_unusable_and_out_of_scope_rows(self) -> None:
        records = [
            {
                "src_lang": "eng",
                "tgt_lang": "cmn",
                "target_qwen_ids": [1, 2, 3],
                "target_bicodec": [4, 4, 5],
            },
            {  # empty text
                "src_lang": "cmn",
                "tgt_lang": "eng",
                "target_qwen_ids": [],
                "target_bicodec": [1, 2],
            },
            {  # empty units
                "src_lang": "cmn",
                "tgt_lang": "eng",
                "target_qwen_ids": [1],
                "target_bicodec": [],
            },
            {  # direction outside the bilingual scope
                "src_lang": "eng",
                "tgt_lang": "eng",
                "target_qwen_ids": [1],
                "target_bicodec": [1],
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "joint.jsonl"
            manifest.write_text(
                "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
            )
            rows = list(read_rows([manifest]))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].direction, "eng->cmn")
        self.assertEqual(rows[0].text_length, 3)
        self.assertEqual(rows[0].unit_length, 3)
        self.assertEqual(rows[0].adjacent_repeats, 1)
        self.assertEqual(rows[0].required_frames, 4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
