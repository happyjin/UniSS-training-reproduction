from __future__ import annotations

import unittest
from pathlib import Path

from experiments.uniss_phase3_dense_aligned_streaming_pilot15_v1.data.build_dense_sessions import (
    _exact_text_deltas,
    _target_word_spans,
    build_dense_session,
)
from experiments.uniss_phase3_dense_aligned_streaming_pilot15_v1.data.schema import (
    DenseSession,
    visible_prefix_length,
)


def _record(*, target_text: str, target_words: list[dict], target_lang: str) -> dict:
    return {
        "formal_a68_pass": True,
        "id": f"synthetic-{target_lang}",
        "src_lang": "cmn" if target_lang == "eng" else "eng",
        "tgt_lang": target_lang,
        "source_duration_ms": 960,
        "target_duration_ms": 800,
        "source_glm": list(range(12)),
        "source_glm_end_ms": [80 * (index + 1) for index in range(12)],
        "target_bicodec": list(range(40)),
        "target_words": target_words,
        "translation": target_text,
        "micro_write_events": [
            {
                "text": target_words[0]["text"],
                "target_word_start": 0,
                "target_word_end": 1,
                "semantic_start": 0,
                "semantic_end": 12,
                "earliest_safe_ms": 320,
            },
            {
                "text": "",
                "semantic_continuation": True,
                "target_word_start": 0,
                "target_word_end": 1,
                "semantic_start": 12,
                "semantic_end": 20,
                "earliest_safe_ms": 320,
            },
            {
                "text": target_words[1]["text"],
                "target_word_start": 1,
                "target_word_end": 2,
                "semantic_start": 20,
                "semantic_end": 30,
                "earliest_safe_ms": 640,
            },
            {
                "text": target_words[2]["text"],
                "target_word_start": 2,
                "target_word_end": 3,
                "semantic_start": 30,
                "semantic_end": 40,
                "earliest_safe_ms": 800,
            },
        ],
    }


class DenseDataTest(unittest.TestCase):
    def test_normalized_alignment_preserves_original_hyphens_and_periods(self) -> None:
        text = "The U.S. nonprofit Wi-Fi works."
        words = [
            {"text": "The"},
            {"text": "US"},
            {"text": "nonprofit"},
            {"text": "WiFi"},
            {"text": "works"},
        ]
        spans, kind, ratio = _target_word_spans(text, words)
        self.assertEqual(kind, "normalized_exact")
        self.assertGreater(ratio, 0.99)
        self.assertEqual([text[start:end] for start, end in spans], ["The", "U.S", "nonprofit", "Wi-Fi", "works"])

    def test_fuzzy_alignment_still_reconstructs_released_text_exactly(self) -> None:
        text = "She will depart soon."
        words = [
            {"text": "She"},
            {"text": "will"},
            {"text": "leaving"},
            {"text": "soon"},
        ]
        micro = [
            {
                "text": word["text"],
                "target_word_start": index,
                "target_word_end": index + 1,
                "semantic_start": index,
                "semantic_end": index + 1,
            }
            for index, word in enumerate(words)
        ]
        deltas, kind, ratio = _exact_text_deltas(text, words, micro)
        self.assertEqual(kind, "monotonic_fuzzy")
        self.assertGreater(ratio, 0.50)
        self.assertEqual("".join(deltas), text)

    def test_visible_prefix_binary_search(self) -> None:
        ends = [80, 160, 240, 400]
        self.assertEqual(visible_prefix_length(ends, 0), 0)
        self.assertEqual(visible_prefix_length(ends, 160), 2)
        self.assertEqual(visible_prefix_length(ends, 399), 3)
        self.assertEqual(visible_prefix_length(ends, 400), 4)

    def test_english_text_and_semantics_are_exact(self) -> None:
        record = _record(
            target_text="Hello, streaming world!",
            target_words=[
                {"text": "Hello"},
                {"text": "streaming"},
                {"text": "world"},
            ],
            target_lang="eng",
        )
        session = build_dense_session(
            record,
            source_manifest=Path("/tmp/formal.jsonl"),
            source_index=7,
            split="train",
            speaker_global=range(32),
        )
        restored = DenseSession.from_dict(session.to_dict())
        writes = [event for event in restored.events if event.action == "WRITE"]
        self.assertEqual("".join(event.text_delta for event in writes), record["translation"])
        self.assertEqual(writes[0].semantic_start, 0)
        self.assertEqual(writes[-1].semantic_end, 40)
        self.assertEqual(writes[-1].wall_time_ms, 960)
        self.assertTrue(writes[-1].source_finished)
        self.assertTrue(writes[-1].final_write)
        self.assertEqual(sum(event.final_write for event in restored.events), 1)

    def test_chinese_text_keeps_punctuation(self) -> None:
        record = _record(
            target_text="你好，流式世界！",
            target_words=[
                {"text": "你好"},
                {"text": "流式"},
                {"text": "世界"},
            ],
            target_lang="cmn",
        )
        session = build_dense_session(
            record,
            source_manifest=Path("/tmp/formal.jsonl"),
            source_index=0,
            split="valid",
            speaker_global=range(32),
        )
        writes = [event for event in session.events if event.action == "WRITE"]
        self.assertEqual(
            [event.text_delta for event in writes],
            ["你好，", "", "流式", "世界！"],
        )
        self.assertEqual(
            [(event.semantic_start, event.semantic_end) for event in writes],
            [(0, 12), (12, 20), (20, 30), (30, 40)],
        )

    def test_internal_event_order_is_checksum_protected(self) -> None:
        record = _record(
            target_text="a b c",
            target_words=[{"text": "a"}, {"text": "b"}, {"text": "c"}],
            target_lang="eng",
        )
        session = build_dense_session(
            record,
            source_manifest=Path("/tmp/formal.jsonl"),
            source_index=0,
            split="train",
            speaker_global=range(32),
        )
        value = session.to_dict()
        value["events"][0], value["events"][1] = value["events"][1], value["events"][0]
        with self.assertRaises(ValueError):
            DenseSession.from_dict(value)


if __name__ == "__main__":
    unittest.main()
