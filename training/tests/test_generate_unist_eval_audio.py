import unittest

from training import constants_uniss as c
from training import generate_unist_eval_audio as eval_audio


class GenerateUniSTEvalAudioTest(unittest.TestCase):
    def test_truncate_at_eos(self):
        self.assertEqual(eval_audio.truncate_at_eos([1, 2, c.TOKEN_EOS, 3]), [1, 2, c.TOKEN_EOS])
        self.assertEqual(eval_audio.truncate_at_eos([1, 2, 3]), [1, 2, 3])

    def test_extract_bicodec_semantic_values(self):
        ids = [
            1,
            c.BICODEC_SEMANTIC_SPAN.id_for(7),
            c.BICODEC_SEMANTIC_SPAN.id_for(8191),
            c.TOKEN_END_SEMANTIC,
        ]
        self.assertEqual(eval_audio.extract_bicodec_semantic_values(ids), [7, 8191])

    def test_clean_generated_text(self):
        text = "<|task_asr|> hello <|end_content|>"
        self.assertEqual(eval_audio.clean_generated_text(text), "hello")

    def test_safe_sample_name(self):
        name = eval_audio.safe_sample_name(3, "dataset/item:1", "quality")
        self.assertTrue(name.startswith("00003_quality_"))
        self.assertNotIn("/", name)
        self.assertNotIn(":", name)


if __name__ == "__main__":
    unittest.main()
