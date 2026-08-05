from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from training.phase3_whisper_streamspeech_joint.config import (
    JointLossWeights,
    MultiChunkConfig,
)
from training.phase3_whisper_streamspeech_joint.losses import (
    NormalizedLoss,
    combine_joint_or_replay,
)
from training.phase3_whisper_streamspeech_joint.nar_bicodec_ctc import NARBiCodecCTC
from training.phase3_whisper_streamspeech_joint.phase3_ste_bridge import Phase3STEBridge
from training.phase3_whisper_streamspeech_joint.policy_mask import (
    build_g_from_ctc_logits,
    phase3_block_attention_allowed,
)
from training.phase3_whisper_streamspeech_joint.tokenizer_maps import (
    CompactCTCMap,
    build_compact_map,
)
from training.phase3_whisper_streamspeech_joint.whisper_multichunk import (
    chunk_causal_allowed,
    choose_chunk_ms,
)


class CoreTest(unittest.TestCase):
    def test_multichunk_mask_blocks_future_chunks_and_padding(self) -> None:
        allowed = chunk_causal_allowed(
            torch.tensor([7]),
            sequence_length=8,
            chunk_frames=2,
            right_context_frames=1,
        )
        self.assertTrue(bool(allowed[0, 0, 2]))
        self.assertFalse(bool(allowed[0, 0, 3]))
        self.assertFalse(bool(allowed[0, 0, 7]))
        self.assertFalse(bool(allowed[0, 7].any()))
        config = MultiChunkConfig()
        self.assertEqual(
            choose_chunk_ms(config, seed=7, sample_index=99),
            choose_chunk_ms(config, seed=7, sample_index=99),
        )

    def test_compact_qwen_map_roundtrip(self) -> None:
        value = build_compact_map("eng", [[9, 3, 9], [7]])
        self.assertEqual(value.compact_to_qwen, (3, 7, 9))
        self.assertEqual(value.decode(value.encode([9, 3])), [9, 3])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "map.json"
            value.save(path)
            self.assertEqual(CompactCTCMap.load(path), value)

    def test_ctc_policy_and_phase3_mask_hide_future_source(self) -> None:
        # Vocab: two labels plus blank=2. ASR events occur at frames 0 and 2;
        # target expected count supports token 1 at frame 0 and token 2 at 2.
        asr = torch.tensor(
            [[[9.0, -9.0, -9.0], [-9.0, -9.0, 9.0], [-9.0, 9.0, -9.0], [-9.0, -9.0, 9.0]]]
        )
        target = asr.clone()
        g = build_g_from_ctc_logits(
            asr,
            target,
            asr_blank_id=2,
            target_blank_id=2,
            target_lengths=torch.tensor([2]),
            encoder_lengths=torch.tensor([4]),
        )
        self.assertEqual(g.tolist(), [[0, 2]])
        allowed = phase3_block_attention_allowed(
            prefix_length=2,
            source_lengths=torch.tensor([4]),
            target_lengths=torch.tensor([2]),
            g=g,
        )
        target_start = 2 + 4
        self.assertTrue(bool(allowed[0, target_start, 2]))
        self.assertFalse(bool(allowed[0, target_start, 3]))
        self.assertTrue(bool(allowed[0, target_start + 1, 4]))
        self.assertFalse(bool(allowed[0, target_start + 1, 5]))

    def test_ste_is_hard_forward_with_continuous_gradient(self) -> None:
        bridge = Phase3STEBridge(
            2,
            3,
            codebook=torch.tensor([[0.0, 0.0], [1.0, 1.0]]),
            qwen_glm_embeddings=torch.tensor([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]]),
        )
        hidden = torch.tensor([[[0.9, 1.1]]], requires_grad=True)
        output = bridge(hidden)
        torch.testing.assert_close(output.embeddings, output.hard_embeddings)
        output.embeddings.sum().backward()
        self.assertGreater(float(hidden.grad.abs().sum()), 0)

    def test_nar_unit_ctc_geometry(self) -> None:
        model = NARBiCodecCTC(
            qwen_hidden_size=8,
            model_size=8,
            semantic_vocab_size=16,
            upsample_ratio=3,
            num_heads=2,
            t2u_layers=1,
            decoder_layers=1,
            dropout=0,
        )
        logits, lengths = model(torch.randn(2, 4, 8), torch.tensor([4, 2]))
        self.assertEqual(tuple(logits.shape), (2, 12, 17))
        self.assertEqual(lengths.tolist(), [12, 6])

    def test_joint_and_replay_are_exclusive(self) -> None:
        value = NormalizedLoss(torch.tensor(4.0), torch.tensor(2.0))
        total, metrics = combine_joint_or_replay(
            sample_kind="joint",
            weights=JointLossWeights(),
            bicodec_ctc=value,
            ar_s2tt=value,
            asr_ctc=value,
            nar_s2tt_ctc=value,
        )
        self.assertEqual(float(total), 34.0)
        self.assertEqual(set(metrics), {"bicodec_ctc", "ar_s2tt", "asr_ctc", "nar_s2tt_ctc"})
        replay, _ = combine_joint_or_replay(
            sample_kind="replay",
            weights=JointLossWeights(),
            phase3_replay=value,
        )
        self.assertEqual(float(replay), 1.0)
        with self.assertRaises(ValueError):
            combine_joint_or_replay(
                sample_kind="replay",
                weights=JointLossWeights(),
                bicodec_ctc=value,
                phase3_replay=value,
            )


if __name__ == "__main__":
    unittest.main()
