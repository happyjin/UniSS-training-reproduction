import unittest

import torch

from evaluation import vllm_generate


class VLLMGenerateTest(unittest.TestCase):
    def test_batched(self):
        self.assertEqual(list(vllm_generate.batched(range(5), 2)), [[0, 1], [2, 3], [4]])

    def test_resume_config_validation(self):
        config = {
            "model": "m",
            "manifest": "x",
            "modes": ["quality"],
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": -1,
            "repetition_penalty": 1.1,
            "max_new_tokens": 10,
            "seed": 1,
        }
        vllm_generate.validate_resume_config(config, dict(config))
        changed = dict(config)
        changed["seed"] = 2
        with self.assertRaises(ValueError):
            vllm_generate.validate_resume_config(config, changed)

    def test_padded_vocabulary_is_suppressed(self):
        logits = torch.zeros(8)
        processor = vllm_generate.SuppressPaddedVocabulary(5)
        output = processor([], logits)
        self.assertTrue(torch.isfinite(output[:5]).all())
        self.assertTrue(torch.isneginf(output[5:]).all())


if __name__ == "__main__":
    unittest.main()
