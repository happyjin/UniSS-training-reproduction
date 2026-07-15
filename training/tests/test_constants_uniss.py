import unittest

from training import constants_uniss as c


class ConstantsUniSSTest(unittest.TestCase):
    def test_public_token_boundaries(self):
        self.assertEqual(c.VOCAB_SIZE, 180_407)
        self.assertEqual(c.bicodec_global_id(0), 151_665)
        self.assertEqual(c.bicodec_global_id(4095), 155_760)
        self.assertEqual(c.bicodec_semantic_id(0), 155_761)
        self.assertEqual(c.bicodec_semantic_id(8191), 163_952)
        self.assertEqual(c.glm_semantic_id(0), 163_953)
        self.assertEqual(c.glm_semantic_id(16383), 180_336)
        self.assertEqual(c.speed_token_id(1.0), 180_346)
        self.assertEqual(c.TOKEN_CMN, 180_372)
        self.assertEqual(c.TOKEN_ENG, 180_373)
        self.assertEqual(c.TOKEN_WRITE_GENERATE, 180_396)
        self.assertEqual(c.TOKEN_STREAMING_MODE, 180_406)

    def test_language_helpers(self):
        self.assertEqual(c.normalize_language("en"), "eng")
        self.assertEqual(c.normalize_language("zh_CN"), "cmn")
        self.assertEqual(c.language_token_id("eng"), c.TOKEN_ENG)
        self.assertEqual(c.language_token_id("cmn"), c.TOKEN_CMN)
        self.assertEqual(c.opposite_language("eng"), "cmn")
        self.assertEqual(c.opposite_language("zh"), "eng")

    def test_wrappers(self):
        global_values = list(range(32))
        wrapped_global = c.wrap_global_tokens(global_values)
        self.assertEqual(wrapped_global[0], c.TOKEN_START_GLOBAL)
        self.assertEqual(wrapped_global[-1], c.TOKEN_END_GLOBAL)
        self.assertEqual(wrapped_global[1], c.bicodec_global_id(0))
        self.assertEqual(wrapped_global[-2], c.bicodec_global_id(31))

        wrapped_semantic = c.wrap_semantic_tokens([0, 8191])
        self.assertEqual(wrapped_semantic[0], c.TOKEN_START_SEMANTIC)
        self.assertEqual(wrapped_semantic[1], c.bicodec_semantic_id(0))
        self.assertEqual(wrapped_semantic[2], c.bicodec_semantic_id(8191))
        self.assertEqual(wrapped_semantic[-2], c.TOKEN_END_SEMANTIC)
        self.assertEqual(wrapped_semantic[-1], c.TOKEN_EOS)

    def test_validation_errors(self):
        with self.assertRaises(ValueError):
            c.bicodec_global_id(4096)
        with self.assertRaises(ValueError):
            c.bicodec_semantic_id(-1)
        with self.assertRaises(ValueError):
            c.glm_semantic_id(16384)
        with self.assertRaises(ValueError):
            c.speed_token_id(3.6)
        with self.assertRaises(ValueError):
            c.encode_bicodec_global([0] * 31)
        with self.assertRaises(ValueError):
            c.language_token_id("fra")


if __name__ == "__main__":
    unittest.main()
