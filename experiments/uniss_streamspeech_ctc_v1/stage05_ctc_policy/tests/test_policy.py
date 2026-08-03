import sys
import unittest
from pathlib import Path


STAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STAGE))

from policy import CTCReadWritePolicy, collapse_ctc


PIECES = {1: "▁hello", 2: "world", 3: "▁from", 4: "▁uniss"}


class CTCPolicyTest(unittest.TestCase):
    def test_ctc_collapse_removes_blank_and_repetition(self) -> None:
        self.assertEqual(collapse_ctc([0, 1, 1, 0, 2, 2, 3], 0), [1, 2, 3])

    def test_requires_cross_chunk_confirmation_and_source_event(self) -> None:
        policy = CTCReadWritePolicy(
            source_blank_id=0,
            target_blank_id=0,
            target_language="cmn",
            target_id_to_piece=lambda value: str(value),
            confirmations=2,
        )
        first = policy.update([1, 0], [7, 0])
        second = policy.update([1, 0], [7, 0])
        self.assertEqual(first.action, "WAIT")
        self.assertEqual(second.action, "WRITE")
        self.assertEqual(second.new_target_tokens, (7,))

    def test_english_withholds_unfinished_last_word(self) -> None:
        policy = CTCReadWritePolicy(
            source_blank_id=0,
            target_blank_id=0,
            target_language="eng",
            target_id_to_piece=lambda value: PIECES[value],
            confirmations=2,
        )
        policy.update([9, 0], [1, 2, 3, 0])
        decision = policy.update([9, 0], [1, 2, 3, 0])
        self.assertEqual(decision.new_target_tokens, (1, 2))
        final = policy.update([9, 0], [1, 2, 3, 0], final=True)
        self.assertEqual(final.new_target_tokens, (3,))

    def test_conflict_never_rewrites_committed_prefix(self) -> None:
        policy = CTCReadWritePolicy(
            source_blank_id=0,
            target_blank_id=0,
            target_language="cmn",
            target_id_to_piece=lambda value: str(value),
            confirmations=2,
        )
        policy.update([1, 0], [7, 0])
        policy.update([1, 0], [7, 0])
        conflict = policy.update([1, 2, 0], [8, 0])
        self.assertEqual(conflict.action, "WAIT")
        self.assertEqual(policy.committed_target, [7])
        self.assertGreaterEqual(conflict.target_conflicts, 1)


if __name__ == "__main__":
    unittest.main()

