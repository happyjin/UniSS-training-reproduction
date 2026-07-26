import unittest

from evaluation.uniss_outputs import parse_generated_fields
from training import constants_uniss as c


class UniSSOutputsTest(unittest.TestCase):
    @staticmethod
    def decoder(ids):
        return " ".join(str(value) for value in ids)

    def test_parse_quality_fields(self):
        ids = [
            11,
            12,
            c.TOKEN_END_CONTENT,
            c.TOKEN_TASK_S2T_TRANSLATION,
            c.TOKEN_ENG,
            c.speed_token_id(1.0),
            c.TOKEN_START_CONTENT,
            21,
            22,
            c.TOKEN_END_CONTENT,
            c.TOKEN_START_SEMANTIC,
            c.bicodec_semantic_id(7),
            c.TOKEN_END_SEMANTIC,
            c.TOKEN_EOS,
        ]
        parsed = parse_generated_fields(ids, mode="quality", text_decoder=self.decoder)
        self.assertEqual(parsed["generated_transcription"], "11 12")
        self.assertEqual(parsed["generated_translation"], "21 22")
        self.assertEqual(parsed["semantic_values"], [7])
        self.assertTrue(parsed["has_semantic_end"])
        self.assertTrue(parsed["has_eos"])

    def test_parse_performance_fields(self):
        ids = [31, 32, c.TOKEN_END_CONTENT, c.TOKEN_START_SEMANTIC, c.bicodec_semantic_id(9)]
        parsed = parse_generated_fields(ids, mode="performance", text_decoder=self.decoder)
        self.assertIsNone(parsed["generated_transcription"])
        self.assertEqual(parsed["generated_translation"], "31 32")
        self.assertEqual(parsed["semantic_values"], [9])
        self.assertFalse(parsed["has_semantic_end"])


if __name__ == "__main__":
    unittest.main()
