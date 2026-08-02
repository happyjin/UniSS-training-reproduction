from __future__ import annotations

import unittest

from training.simul_uniss.subsecond_v2.audit_teacher_prefix_ceiling import (
    build_immediate_causal_stream,
    score_prefix_sequences,
)


class TeacherPrefixCeilingTest(unittest.TestCase):
    def test_scores_tokens_once_at_their_commit_boundary(self) -> None:
        result = score_prefix_sequences(
            reference=[1, 2, 3, 4, 5, 6],
            token_end_ms=[80, 160, 240, 320, 400, 480],
            commit_end_ms=[160, 320, 480],
            visible_end_ms=[240, 400, 560],
            predictions=[
                [1, 9, 3],
                [1, 2, 3, 4, 5],
                [1, 2, 3, 4, 5, 6],
            ],
        )
        self.assertEqual(result["immediate_total"], 6)
        self.assertEqual(result["immediate_exact"], 5)
        self.assertEqual(result["revision_total"], 6)
        self.assertEqual(result["revision_160"], 1)
        self.assertEqual(result["first_correct_stable_visible_ms"], 240.0)

    def test_missing_prefix_tokens_count_as_errors(self) -> None:
        result = score_prefix_sequences(
            reference=[4, 5],
            token_end_ms=[80, 160],
            commit_end_ms=[160],
            visible_end_ms=[240],
            predictions=[[4]],
        )
        self.assertEqual(result["immediate_total"], 2)
        self.assertEqual(result["immediate_exact"], 1)

    def test_causal_stream_delays_missing_positions_without_using_reference(self) -> None:
        stream = build_immediate_causal_stream(
            token_end_ms=[80, 160, 240, 320],
            commit_end_ms=[160, 320],
            predictions=[[7], [7, 8, 9, 10]],
        )
        self.assertEqual(stream, [7, 8, 9, 10])


if __name__ == "__main__":
    unittest.main()
