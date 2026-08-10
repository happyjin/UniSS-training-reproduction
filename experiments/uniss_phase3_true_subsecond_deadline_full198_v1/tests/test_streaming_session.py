from __future__ import annotations

import unittest

from experiments.uniss_phase3_true_subsecond_deadline_full198_v1.inference import (
    StreamingSessionState,
)


class StreamingSessionTest(unittest.TestCase):
    def test_semantic_history_is_bounded_and_speaker_is_persistent(self) -> None:
        state = StreamingSessionState(tuple(range(32)), semantic_history_limit=5)
        state.append_source_time(640)
        state.commit_text([1, 2], forced=False)
        state.append_semantic(list(range(10)))
        self.assertEqual(state.semantic_history, [5, 6, 7, 8, 9])
        state.reset_utterance()
        self.assertEqual(state.speaker_global, tuple(range(32)))
        self.assertEqual(state.semantic_history, [])


if __name__ == "__main__":
    unittest.main()
