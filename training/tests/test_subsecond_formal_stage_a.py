from __future__ import annotations

import unittest

from training.simul_uniss.subsecond_v2.formal_supervision import (
    alignment_coverage,
    build_micro_write_supervision,
    build_support_alignment,
    normalize_words,
    safe_label,
)


class FormalStageASupervisionTest(unittest.TestCase):
    def test_support_is_monotonic_and_unaligned_words_are_uncertain(self) -> None:
        source = normalize_words(
            [
                {"text": "我", "start_ms": 40, "end_ms": 120},
                {"text": "明天", "start_ms": 180, "end_ms": 390},
                {"text": "北京", "start_ms": 500, "end_ms": 900},
            ],
            duration_ms=1000,
        )
        target = normalize_words(
            [
                {"text": "I", "start_ms": 0, "end_ms": 100},
                {"text": "will", "start_ms": 100, "end_ms": 220},
                {"text": "go", "start_ms": 220, "end_ms": 380},
                {"text": "to", "start_ms": 380, "end_ms": 450},
                {"text": "Beijing", "start_ms": 450, "end_ms": 800},
            ],
            duration_ms=900,
        )
        support = build_support_alignment(
            source,
            target,
            [
                {"source_index": 0, "target_index": 0, "confidence": 0.9},
                {"source_index": 2, "target_index": 2, "confidence": 0.8},
                {"source_index": 2, "target_index": 4, "confidence": 0.95},
            ],
        )
        self.assertEqual([value["support_end_ms"] for value in support], [120, 120, 900, 900, 900])
        self.assertTrue(support[1]["uncertain"])
        self.assertFalse(support[4]["uncertain"])

    def test_micro_write_covers_semantic_once_and_has_safe_time(self) -> None:
        words = normalize_words(
            [
                {"text": "Tomorrow", "start_ms": 0, "end_ms": 220},
                {"text": "morning", "start_ms": 220, "end_ms": 480},
                {"text": "at", "start_ms": 480, "end_ms": 560},
                {"text": "nine", "start_ms": 560, "end_ms": 800},
                {"text": ".", "start_ms": 800, "end_ms": 840},
            ],
            duration_ms=840,
        )
        support = [
            {
                "support_end_ms": value,
                "uncertain": False,
                "negation_or_entity_risk": False,
            }
            for value in (390, 520, 520, 760, 760)
        ]
        events = build_micro_write_supervision(
            words,
            support,
            language="en",
            target_duration_ms=840,
            target_semantic_count=42,
        )
        self.assertEqual(events[0]["semantic_start"], 0)
        self.assertEqual(events[-1]["semantic_end"], 42)
        self.assertEqual(
            sum(int(value["semantic_count"]) for value in events),
            42,
        )
        self.assertTrue(events[-1]["final_flush"])
        self.assertEqual(events[0]["earliest_safe_ms"] % 160, 0)
        self.assertEqual(safe_label(events[0], 0), 0)
        self.assertEqual(safe_label(events[0], int(events[0]["earliest_safe_ms"])), 1)

    def test_alignment_coverage_ignores_common_tokenization_differences(self) -> None:
        words = normalize_words(
            [
                {"text": "Hello", "start_ms": 0, "end_ms": 200},
                {"text": ",", "start_ms": 200, "end_ms": 220},
                {"text": "world", "start_ms": 220, "end_ms": 500},
                {"text": "!", "start_ms": 500, "end_ms": 520},
            ],
            duration_ms=600,
        )
        self.assertEqual(alignment_coverage(words, "Hello, world!", "en"), 1.0)


if __name__ == "__main__":
    unittest.main()
