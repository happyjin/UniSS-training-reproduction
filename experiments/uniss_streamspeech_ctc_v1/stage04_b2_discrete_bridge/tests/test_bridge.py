import sys
import unittest
from pathlib import Path

import torch


STAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STAGE))

from bridge import StraightThroughCodebookBridge, pool_frames, replace_embedding_span


class BridgeTest(unittest.TestCase):
    def test_pooling_preserves_partial_final_frame(self) -> None:
        hidden = torch.arange(15, dtype=torch.float32).reshape(1, 5, 3)
        pooled, lengths = pool_frames(hidden, torch.tensor([5]), factor=2)
        self.assertEqual(tuple(pooled.shape), (1, 3, 3))
        self.assertEqual(lengths.tolist(), [3])
        self.assertTrue(torch.equal(pooled[0, -1], hidden[0, -1]))

    def test_pooling_removes_nan_from_invalid_padding(self) -> None:
        hidden = torch.tensor([[[1.0], [2.0], [3.0], [float("nan")]]])
        pooled, lengths = pool_frames(hidden, torch.tensor([3]), factor=2)
        torch.testing.assert_close(pooled, torch.tensor([[[1.5], [3.0]]]))
        self.assertEqual(lengths.tolist(), [2])

    def test_hard_forward_keeps_gradient_to_projection(self) -> None:
        torch.manual_seed(1)
        bridge = StraightThroughCodebookBridge(
            encoder_dim=4,
            codebook=torch.randn(8, 6),
            qwen_glm_embeddings=torch.randn(8, 5),
            top_k=4,
            temperature=0.5,
        )
        output = bridge(torch.randn(2, 6, 4), torch.tensor([6, 5]))
        self.assertEqual(tuple(output.hard_code_ids.shape), (2, 3))
        self.assertEqual(tuple(output.qwen_speech_embeddings.shape), (2, 3, 5))
        output.qwen_speech_embeddings.square().mean().backward()
        self.assertGreater(float(bridge.projection.weight.grad.abs().sum()), 0.0)

    def test_embedding_replacement_is_local(self) -> None:
        tokens = torch.zeros(6, 3)
        speech = torch.ones(2, 3)
        result = replace_embedding_span(tokens, speech, span_start=2, speech_length=2)
        self.assertEqual(float(result.sum()), 6.0)
        self.assertEqual(float(result[:2].sum() + result[4:].sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
