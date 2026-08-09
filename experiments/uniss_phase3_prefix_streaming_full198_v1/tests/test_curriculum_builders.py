from __future__ import annotations

import unittest

from training import constants_uniss as c

from experiments.uniss_phase3_prefix_streaming_full198_v1 import builders
from experiments.uniss_phase3_prefix_streaming_full198_v1.curriculum import (
    choose_prefix_pair,
    choose_semantic_geometry,
    choose_task,
    point_for_iteration,
)


def record() -> dict[str, object]:
    return {
        "id": "example",
        "src_lang": "eng",
        "tgt_lang": "cmn",
        "source_glm": list(range(20)),
        "target_bicodec": list(range(100)),
        "bicodec_global": list(range(32)),
        "transcription_ids": [10, 11, 12],
        "translation_ids": [20, 21, 22, 23, 24],
    }


class CurriculumTest(unittest.TestCase):
    def test_schedule_is_normalized_and_expands_prefixes(self) -> None:
        early = point_for_iteration(1)
        late = point_for_iteration(12000)
        self.assertAlmostEqual(sum(early.probabilities), 1.0)
        self.assertAlmostEqual(sum(late.probabilities), 1.0)
        self.assertNotIn(0.25, early.prefix_ratios)
        self.assertIn(0.25, late.prefix_ratios)

    def test_choices_are_deterministic(self) -> None:
        point = point_for_iteration(5000)
        self.assertEqual(
            choose_task(point, sample_id="x", iteration=5000),
            choose_task(point, sample_id="x", iteration=5000),
        )
        short, long = choose_prefix_pair(point, sample_id="x", iteration=5000)
        self.assertLessEqual(short, long)
        text, cut, block = choose_semantic_geometry(
            sample_id="x", iteration=5000, semantic_length=100
        )
        self.assertTrue(0.0 < text <= 1.0)
        self.assertTrue(0 < cut < 100)
        self.assertTrue(0 < block <= 50)


class BuilderTest(unittest.TestCase):
    def test_streaming_prefix_has_no_future_source(self) -> None:
        value = builders.build_streaming_s2tt(record(), 0.25)
        encoded = c.encode_glm_semantic(list(range(5)))
        self.assertTrue(all(token in value.prompt_ids for token in encoded))
        self.assertNotIn(c.GLM_SEMANTIC_OFFSET + 19, value.prompt_ids)
        self.assertEqual(value.target_ids[-1], c.TOKEN_EOS)

    def test_replay_matches_phase3_protocol(self) -> None:
        quality = builders.build_replay(record(), "quality")
        performance = builders.build_replay(record(), "performance")
        self.assertEqual(quality.prompt_ids[:2], [c.TOKEN_TASK_S2S_TRANSLATION, c.TOKEN_SLOW_MODE])
        self.assertEqual(
            performance.prompt_ids[:2],
            [c.TOKEN_TASK_S2S_TRANSLATION, c.TOKEN_BALANCE_MODE],
        )
        self.assertIn(c.TOKEN_START_SEMANTIC, quality.target_ids)
        self.assertIn(c.TOKEN_START_SEMANTIC, performance.target_ids)

    def test_semantic_continuation_is_bounded(self) -> None:
        sample = builders.build_streaming_tts(
            record(), text_ratio=0.5, semantic_cut=40, block_size=25, history_tokens=20
        )
        semantic = [
            value - c.BICODEC_SEMANTIC_OFFSET
            for value in sample.target_ids
            if c.BICODEC_SEMANTIC_OFFSET
            <= value
            < c.BICODEC_SEMANTIC_OFFSET + c.BICODEC_SEMANTIC_SIZE
        ]
        self.assertEqual(semantic, list(range(40, 65)))

    def test_s2tt_outlier_is_bounded_without_future_source(self) -> None:
        value = record()
        value["source_glm"] = list(range(5000))
        value["translation_ids"] = list(range(100))
        bounded = builders.bounded_s2tt_record(value, 4096)
        sample = builders.build_streaming_s2tt(bounded, 1.0)
        teacher = builders.build_teacher_s2tt(bounded)
        self.assertLessEqual(len(sample.input_ids), 4096)
        self.assertLessEqual(len(teacher.input_ids), 4096)
        self.assertEqual(bounded["source_glm"], list(range(len(bounded["source_glm"]))))


if __name__ == "__main__":
    unittest.main()
