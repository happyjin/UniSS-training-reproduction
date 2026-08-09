#!/usr/bin/env python3
"""Checks the CTC decode and edit distance behind the Step 2b unit error rate.

Run directly:
``python experiments/simul_s2st_route_v1/step2_nar_ctc_head/tests/test_evaluate_existing_head.py``
"""

from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from experiments.simul_s2st_route_v1.step2_nar_ctc_head.evaluate_existing_head import (  # noqa: E402
    blank_suppressed_decode,
    ctc_greedy_decode,
    edit_distance,
)


def naive_edit_distance(left, right):
    rows = [[0] * (len(right) + 1) for _ in range(len(left) + 1)]
    for i in range(len(left) + 1):
        rows[i][0] = i
    for j in range(len(right) + 1):
        rows[0][j] = j
    for i in range(1, len(left) + 1):
        for j in range(1, len(right) + 1):
            rows[i][j] = min(
                rows[i - 1][j] + 1,
                rows[i][j - 1] + 1,
                rows[i - 1][j - 1] + (left[i - 1] != right[j - 1]),
            )
    return rows[len(left)][len(right)]


class EditDistanceTest(unittest.TestCase):
    def test_known_cases(self) -> None:
        self.assertEqual(edit_distance([], []), 0)
        self.assertEqual(edit_distance([], [1, 2, 3]), 3)
        self.assertEqual(edit_distance([1, 2, 3], []), 3)
        self.assertEqual(edit_distance([1, 2, 3], [1, 2, 3]), 0)
        self.assertEqual(edit_distance([1, 2, 3], [1, 9, 3]), 1)
        self.assertEqual(edit_distance([1, 2, 3], [1, 3]), 1)
        self.assertEqual(edit_distance([1, 3], [1, 2, 3]), 1)
        self.assertEqual(edit_distance([1, 2, 3], [3, 2, 1]), 2)

    def test_matches_the_naive_implementation_on_random_streams(self) -> None:
        generator = random.Random(20260809)
        for _ in range(60):
            left = [generator.randrange(4) for _ in range(generator.randrange(0, 25))]
            right = [generator.randrange(4) for _ in range(generator.randrange(0, 25))]
            self.assertEqual(
                edit_distance(left, right),
                naive_edit_distance(left, right),
                msg=f"{left} vs {right}",
            )

    def test_is_symmetric(self) -> None:
        generator = random.Random(7)
        for _ in range(20):
            left = [generator.randrange(6) for _ in range(generator.randrange(1, 30))]
            right = [generator.randrange(6) for _ in range(generator.randrange(1, 30))]
            self.assertEqual(edit_distance(left, right), edit_distance(right, left))


class CTCDecodeTest(unittest.TestCase):
    def one_hot(self, ids, vocab):
        logits = torch.full((len(ids), vocab), -10.0)
        for position, value in enumerate(ids):
            logits[position, value] = 10.0
        return logits

    def test_collapses_repeats_and_drops_blanks(self) -> None:
        blank = 3
        frames = [0, 0, 3, 0, 1, 1, 3, 3, 2]
        self.assertEqual(ctc_greedy_decode(self.one_hot(frames, 4), blank), [0, 0, 1, 2])

    def test_all_blank_decodes_to_nothing(self) -> None:
        self.assertEqual(ctc_greedy_decode(self.one_hot([2, 2, 2], 3), 2), [])

    def test_single_run_yields_one_token(self) -> None:
        self.assertEqual(ctc_greedy_decode(self.one_hot([1, 1, 1, 1], 3), 2), [1])


class BlankSuppressedDecodeTest(unittest.TestCase):
    def test_recovers_the_runner_up_when_blank_wins_everywhere(self) -> None:
        # Blank (index 3) is the argmax on every frame, but the runner-up varies.
        logits = torch.tensor(
            [
                [1.0, 0.0, 0.0, 9.0],
                [1.0, 0.0, 0.0, 9.0],
                [0.0, 2.0, 0.0, 9.0],
                [0.0, 0.0, 3.0, 9.0],
            ]
        )
        self.assertEqual(ctc_greedy_decode(logits, 3), [])
        self.assertEqual(blank_suppressed_decode(logits, 3), [0, 1, 2])

    def test_never_emits_the_blank_symbol(self) -> None:
        logits = torch.zeros(5, 4)
        logits[:, 3] = 100.0
        self.assertNotIn(3, blank_suppressed_decode(logits, 3))

    def test_leaves_the_caller_logits_untouched(self) -> None:
        logits = torch.tensor([[1.0, 0.0, 9.0]])
        blank_suppressed_decode(logits, 2)
        self.assertEqual(float(logits[0, 2]), 9.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
