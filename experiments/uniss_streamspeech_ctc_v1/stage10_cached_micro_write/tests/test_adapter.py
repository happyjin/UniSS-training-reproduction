import unittest

import torch

from experiments.uniss_streamspeech_ctc_v1.stage10_cached_micro_write.adapter import (
    apply_repetition_penalty,
    block_collapsed_semantic,
    maximum_identical_run,
)
from training import constants_uniss as c


class AdapterHelperTest(unittest.TestCase):
    def test_repetition_penalty_changes_seen_positive_logit(self):
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        result = apply_repetition_penalty(logits, [1], 2.0)
        self.assertEqual(float(result[0, 1]), 1.0)
        self.assertEqual(float(result[0, 2]), 3.0)

    def test_blocks_long_semantic_run(self):
        token = c.BICODEC_SEMANTIC_OFFSET + 7
        logits = torch.zeros(1, c.VOCAB_SIZE)
        result = block_collapsed_semantic(
            logits, [c.TOKEN_START_SEMANTIC, *([token] * 6)]
        )
        self.assertTrue(torch.isneginf(result[0, token]))

    def test_maximum_run(self):
        self.assertEqual(maximum_identical_run([1, 1, 2, 2, 2, 1]), 3)


if __name__ == "__main__":
    unittest.main()
