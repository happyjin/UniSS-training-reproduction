from __future__ import annotations

import unittest

import torch
from torch import nn

from training.phase3_whisper_streamspeech_joint.phase3_batch import (
    build_policy_conditioned_phase3_batch,
    gather_target_hidden,
)


class Phase3BatchTest(unittest.TestCase):
    def test_batch_preserves_hard_source_and_masks_prediction_queries(self) -> None:
        embedding = nn.Embedding(180407, 4)
        record = {
            "id": "x",
            "tgt_lang": "cmn",
            "translation": "xy",
            "bicodec_global": list(range(32)),
            "target_bicodec": [1, 2, 3],
        }
        source = torch.tensor([[[10.0, 11.0, 12.0, 13.0], [20.0, 21.0, 22.0, 23.0]]])
        batch = build_policy_conditioned_phase3_batch(
            embedding_layer=embedding,
            text_encoder=lambda text: [1000 + ord(value) for value in text],
            records=[record],
            source_embeddings=source,
            source_lengths=torch.tensor([2]),
            g=torch.tensor([[0, 1]]),
        )
        start = int(batch.source_starts[0])
        torch.testing.assert_close(batch.inputs_embeds[0, start : start + 2], source[0])
        target = int(batch.target_starts[0])
        self.assertEqual(batch.labels[0, target : target + 2].tolist(), [1120, 1121])
        mask = batch.attention_mask[0, 0]
        self.assertEqual(float(mask[target - 1, start]), 0.0)
        self.assertLess(float(mask[target - 1, start + 1]), -1e20)

        hidden = torch.arange(batch.inputs_embeds.numel(), dtype=torch.float32).reshape_as(
            batch.inputs_embeds
        )
        gathered, lengths = gather_target_hidden(hidden, batch.target_positions)
        torch.testing.assert_close(gathered[0, :2], hidden[0, target : target + 2])
        self.assertEqual(lengths.tolist(), [2])


if __name__ == "__main__":
    unittest.main()
