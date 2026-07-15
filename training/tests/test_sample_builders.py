import unittest

from training import constants_uniss as c
from training import sample_builders as b


def fake_text_encoder(text: str) -> list[int]:
    return [1000 + ord(ch) for ch in text]


class SampleBuildersTest(unittest.TestCase):
    def setUp(self):
        self.record = {
            "id": "sample-1",
            "src_lang": "eng",
            "tgt_lang": "cmn",
            "transcription": "hi",
            "translation": "你好",
            "source_glm": [0, 1, 16383],
            "source_bicodec": [0, 1, 8191],
            "target_bicodec": [2, 3, 8191],
            "bicodec_global": list(range(32)),
        }

    def test_asr_layout_matches_inference_prompt_style(self):
        sample = b.build_asr_sample(
            source_glm=self.record["source_glm"],
            bicodec_global=self.record["bicodec_global"],
            src_lang="eng",
            transcription="hi",
            text_encoder=fake_text_encoder,
        )
        self.assertEqual(sample.prompt_ids[0], c.TOKEN_TASK_ASR)
        self.assertEqual(sample.prompt_ids[1], c.TOKEN_ENG)
        self.assertEqual(sample.prompt_ids[2], c.TOKEN_START_GLOBAL)
        self.assertEqual(sample.prompt_ids[35], c.TOKEN_END_GLOBAL)
        self.assertEqual(sample.prompt_ids[36:39], c.encode_glm_semantic([0, 1, 16383]))
        self.assertEqual(sample.prompt_ids[-3:], [c.TOKEN_WRITE_GENERATE, c.TOKEN_ENG, c.TOKEN_START_CONTENT])
        self.assertEqual(sample.target_ids, [fake_text_encoder("hi")[0], fake_text_encoder("hi")[1], c.TOKEN_END_CONTENT, c.TOKEN_EOS])

    def test_quality_layout_has_cot_segments(self):
        sample = b.build_quality_sample(
            source_glm=self.record["source_glm"],
            bicodec_global=self.record["bicodec_global"],
            src_lang="eng",
            tgt_lang="cmn",
            transcription="hi",
            translation="你好",
            target_bicodec=self.record["target_bicodec"],
            text_encoder=fake_text_encoder,
        )
        self.assertEqual(sample.prompt_ids[:2], [c.TOKEN_TASK_S2S_TRANSLATION, c.TOKEN_SLOW_MODE])
        self.assertEqual(
            sample.prompt_ids[-5:],
            [
                c.TOKEN_WRITE_GENERATE,
                c.TOKEN_TASK_ASR,
                c.TOKEN_ENG,
                c.speed_token_id(1.0),
                c.TOKEN_START_CONTENT,
            ],
        )
        self.assertEqual(sample.target_ids[:3], [*fake_text_encoder("hi"), c.TOKEN_END_CONTENT])
        self.assertIn("quality_translation_text", sample.segment_spans)
        self.assertIn("quality_semantic", sample.segment_spans)
        translation_start, translation_end = sample.segment_spans["quality_translation_text"]
        self.assertEqual(sample.target_ids[translation_start:translation_end], fake_text_encoder("你好"))
        semantic_start, semantic_end = sample.segment_spans["quality_semantic"]
        self.assertEqual(
            sample.target_ids[semantic_start:semantic_end],
            c.encode_bicodec_semantic(self.record["target_bicodec"]),
        )
        self.assertEqual(sample.target_ids[-2:], [c.TOKEN_END_SEMANTIC, c.TOKEN_EOS])

    def test_performance_and_direct_layouts(self):
        performance = b.build_performance_sample(
            source_glm=self.record["source_glm"],
            bicodec_global=self.record["bicodec_global"],
            tgt_lang="cmn",
            translation="你好",
            target_bicodec=self.record["target_bicodec"],
            text_encoder=fake_text_encoder,
        )
        self.assertEqual(performance.prompt_ids[:2], [c.TOKEN_TASK_S2S_TRANSLATION, c.TOKEN_BALANCE_MODE])
        self.assertEqual(
            performance.prompt_ids[-5:],
            [c.TOKEN_WRITE_GENERATE, c.TOKEN_TASK_S2T_TRANSLATION, c.TOKEN_CMN, c.speed_token_id(), c.TOKEN_START_CONTENT],
        )
        self.assertEqual(performance.target_ids[:2], fake_text_encoder("你好"))
        self.assertEqual(performance.target_ids[-2:], [c.TOKEN_END_SEMANTIC, c.TOKEN_EOS])

        direct = b.build_direct_s2st_sample(
            source_glm=self.record["source_glm"],
            bicodec_global=self.record["bicodec_global"],
            tgt_lang="cmn",
            target_bicodec=self.record["target_bicodec"],
        )
        self.assertEqual(direct.prompt_ids[:2], [c.TOKEN_TASK_S2S_TRANSLATION, c.TOKEN_FAST_MODE])
        self.assertEqual(
            direct.prompt_ids[-5:],
            [c.TOKEN_WRITE_GENERATE, c.TOKEN_FAST_MODE, c.TOKEN_CMN, c.speed_token_id(), c.TOKEN_START_SEMANTIC],
        )
        self.assertEqual(direct.target_ids[-2:], [c.TOKEN_END_SEMANTIC, c.TOKEN_EOS])

    def test_phase_helpers(self):
        phase1 = b.build_phase1_samples_from_record(self.record, fake_text_encoder)
        self.assertEqual([sample.task for sample in phase1], ["asr", "s2tt", "tts"])

        phase2 = b.build_s2st_samples_from_record(
            self.record, fake_text_encoder, include_direct=True
        )
        self.assertEqual(
            [sample.task for sample in phase2], ["quality", "performance", "direct_s2st"]
        )

        phase3 = b.build_s2st_samples_from_record(
            self.record, fake_text_encoder, include_direct=False
        )
        self.assertEqual([sample.task for sample in phase3], ["quality", "performance"])

    def test_invalid_empty_text(self):
        with self.assertRaises(ValueError):
            b.build_asr_sample(
                source_glm=[0],
                bicodec_global=list(range(32)),
                src_lang="eng",
                transcription="",
                text_encoder=fake_text_encoder,
            )


if __name__ == "__main__":
    unittest.main()
