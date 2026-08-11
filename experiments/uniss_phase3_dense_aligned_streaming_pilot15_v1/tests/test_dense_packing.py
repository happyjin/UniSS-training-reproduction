from __future__ import annotations

import unittest
from pathlib import Path

from experiments.uniss_phase3_dense_aligned_streaming_pilot15_v1.data.build_dense_sessions import (
    build_dense_session,
)
from experiments.uniss_phase3_dense_aligned_streaming_pilot15_v1.data.packing import (
    ROLE_ACTION,
    build_session_token_sample,
    pack_session_samples,
)
from training import constants_uniss as c


def _encode(value: str) -> list[int]:
    return [1000 + ord(character) % 1000 for character in value]


class DensePackingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.formal = {
            "formal_a68_pass": True,
            "id": "pack-sample",
            "src_lang": "eng",
            "tgt_lang": "cmn",
            "source_duration_ms": 640,
            "target_duration_ms": 480,
            "source_glm": list(range(8)),
            "source_glm_end_ms": [80 * (index + 1) for index in range(8)],
            "target_bicodec": list(range(24)),
            "target_words": [{"text": "你"}, {"text": "好"}],
            "translation": "你好！",
            "micro_write_events": [
                {
                    "text": "你",
                    "target_word_start": 0,
                    "target_word_end": 1,
                    "semantic_start": 0,
                    "semantic_end": 12,
                    "earliest_safe_ms": 320,
                },
                {
                    "text": "好",
                    "target_word_start": 1,
                    "target_word_end": 2,
                    "semantic_start": 12,
                    "semantic_end": 24,
                    "earliest_safe_ms": 640,
                },
            ],
        }
        self.session = build_dense_session(
            self.formal,
            source_manifest=Path("/tmp/formal.jsonl"),
            source_index=0,
            split="train",
            speaker_global=range(32),
        )

    def test_session_is_one_causal_sequence(self) -> None:
        sample = build_session_token_sample(self.session, self.formal, _encode)
        self.assertEqual(
            sum(role == ROLE_ACTION for role in sample.token_roles),
            len(self.session.events),
        )
        encoded_source = c.encode_glm_semantic(self.formal["source_glm"])
        for token in encoded_source:
            self.assertEqual(sample.tokens.count(token), 1)
        self.assertEqual(len(sample.annotations), len(self.session.events))
        self.assertEqual(
            sample.annotations[-1]["stable_target_length"],
            len(_encode(self.formal["translation"])),
        )

    def test_packing_preserves_complete_session_boundaries(self) -> None:
        sample = build_session_token_sample(self.session, self.formal, _encode)
        packed = list(pack_session_samples([sample, sample], seq_length=4096))
        self.assertEqual(len(packed), 1)
        self.assertEqual(len(packed[0]["sample_boundaries"]), 2)
        first, second = packed[0]["sample_boundaries"]
        self.assertEqual(first, [0, sample.length])
        self.assertEqual(second, [sample.length, sample.length * 2])
        self.assertEqual(len(packed[0]["tokens"]), 4096)
        self.assertEqual(len(packed[0]["sessions"][0]["annotations"]), len(self.session.events))


if __name__ == "__main__":
    unittest.main()
