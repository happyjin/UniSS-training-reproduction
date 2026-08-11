from __future__ import annotations

import unittest

from training import constants_uniss as c
from web_demo.true_subsecond_pilot15_streaming_v1.qwen_runtime import (
    IncrementalQwenRuntime,
    repeated_text_reason,
    semantic_rejection_reason,
)


class QwenRuntimeTest(unittest.TestCase):
    def test_history_context_matches_training_packing_contract(self) -> None:
        runtime = IncrementalQwenRuntime.__new__(IncrementalQwenRuntime)
        runtime.target_lang = "cmn"
        runtime.semantic_history_tokens = 3
        runtime.committed_text_ids = [11, 12]
        runtime.committed_semantic_ids = [1, 2, 3, 4, 5]
        self.assertEqual(
            runtime._history_context_tokens(),
            [
                c.language_token_id("cmn"),
                c.speed_token_id(1.0),
                c.TOKEN_START_CONTENT,
                11,
                12,
                c.TOKEN_END_CONTENT,
                c.TOKEN_START_SEMANTIC,
                *c.encode_bicodec_semantic([3, 4, 5]),
                c.TOKEN_END_SEMANTIC,
            ],
        )

    def test_forced_write_never_emits_unsupervised_audio(self) -> None:
        runtime = IncrementalQwenRuntime.__new__(IncrementalQwenRuntime)
        runtime.allow_unsafe_forced_audio = False
        write = runtime.micro_write(
            object(),
            maximum_text_tokens=2,
            semantic_block_tokens=12,
            forced=True,
        )
        self.assertEqual(write.text_ids, ())
        self.assertEqual(write.semantic_ids, ())
        self.assertEqual(
            write.quality_rejected_reason,
            "forced_write_without_semantic_supervision",
        )

    def test_unsafe_forced_probe_is_explicitly_opt_in(self) -> None:
        class Tokenizer:
            @staticmethod
            def decode(values, **_kwargs):
                return "probe:" + ",".join(str(value) for value in values)

        runtime = IncrementalQwenRuntime.__new__(IncrementalQwenRuntime)
        runtime.allow_unsafe_forced_audio = True
        runtime.committed_text_ids = []
        runtime.committed_semantic_ids = []
        runtime.tokenizer = Tokenizer()
        runtime._candidate_text = lambda _observation, _maximum: [11]
        runtime._safe_prefix = lambda _observation, candidates: (list(candidates), (0.9,))
        runtime._semantic_block = lambda *_args, **_kwargs: list(range(12))
        write = runtime.micro_write(
            object(), maximum_text_tokens=1, semantic_block_tokens=12, forced=True
        )
        self.assertEqual(write.text_ids, (11,))
        self.assertEqual(write.semantic_ids, tuple(range(12)))

    def test_repetition_and_semantic_collapse_are_rejected(self) -> None:
        self.assertEqual(repeated_text_reason([7, 8], [8]), "repeated_text_delta")
        self.assertIsNone(repeated_text_reason([7, 8], [9]))
        self.assertEqual(
            semantic_rejection_reason([4] * 12), "semantic_identical_run:12"
        )
        self.assertIsNone(semantic_rejection_reason(list(range(12))))


if __name__ == "__main__":
    unittest.main()
