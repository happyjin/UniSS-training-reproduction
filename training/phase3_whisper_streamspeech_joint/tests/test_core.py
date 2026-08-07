from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from training.phase3_whisper_streamspeech_joint.config import (
    JointLossWeights,
    MultiChunkConfig,
)
from training.phase3_whisper_streamspeech_joint.losses import (
    NormalizedLoss,
    combine_joint_or_replay,
    ctc_normalized_loss,
)
from training.phase3_whisper_streamspeech_joint.model import (
    Phase3WhisperStreamSpeechJointModel,
)
from training.phase3_whisper_streamspeech_joint.nar_bicodec_ctc import NARBiCodecCTC
from training.phase3_whisper_streamspeech_joint.phase3_ste_bridge import Phase3STEBridge
from training.phase3_whisper_streamspeech_joint.policy_mask import (
    build_g_from_ctc_logits,
    packed_causal_attention_allowed,
    phase3_block_attention_allowed,
    phase3_prediction_attention_allowed,
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

    def test_prediction_mask_applies_g_to_next_token_query(self) -> None:
        allowed = phase3_prediction_attention_allowed(
            sequence_lengths=torch.tensor([10]),
            source_starts=torch.tensor([2]),
            source_lengths=torch.tensor([4]),
            target_starts=torch.tensor([8]),
            target_lengths=torch.tensor([2]),
            g=torch.tensor([[0, 2]]),
        )
        # Query 7 predicts target token 0 and can see only source frame 0.
        self.assertTrue(bool(allowed[0, 7, 2]))
        self.assertFalse(bool(allowed[0, 7, 3]))
        # Query 8 predicts target token 1 and can see source frames 0..2.
        self.assertTrue(bool(allowed[0, 8, 4]))
        self.assertFalse(bool(allowed[0, 8, 5]))

    def test_packed_replay_mask_has_no_cross_sample_attention(self) -> None:
        cu = torch.tensor([[0, 3, 5, 5, 5, 5]])
        allowed = packed_causal_attention_allowed(cu, 5)
        self.assertTrue(bool(allowed[0, 2, 0]))
        self.assertFalse(bool(allowed[0, 3, 2]))
        self.assertTrue(bool(allowed[0, 4, 3]))

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

    def test_topk_ste_is_hard_forward_and_masks_commitment_padding(self) -> None:
        bridge = Phase3STEBridge(
            2,
            3,
            codebook=torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]),
            qwen_glm_embeddings=torch.tensor(
                [[2.0, 3.0, 4.0], [5.0, 6.0, 7.0], [8.0, 9.0, 10.0]]
            ),
            surrogate="topk_soft",
            topk=2,
            temperature=0.5,
        )
        hidden = torch.tensor(
            [[[0.9, 1.1], [100.0, 100.0]]], requires_grad=True
        )
        output = bridge(hidden, torch.tensor([1]))
        torch.testing.assert_close(output.embeddings, output.hard_embeddings)
        self.assertLess(float(output.commitment_loss), 0.02)
        output.embeddings[:, :1].sum().backward()
        self.assertGreater(float(hidden.grad[:, :1].abs().sum()), 0)
        self.assertEqual(float(hidden.grad[:, 1:].abs().sum()), 0.0)

    def test_teacher_glm_alignment_uses_fixed_teacher_code(self) -> None:
        bridge = Phase3STEBridge(
            2,
            3,
            codebook=torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]),
            qwen_glm_embeddings=torch.tensor(
                [[2.0, 3.0, 4.0], [5.0, 6.0, 7.0], [8.0, 9.0, 10.0]]
            ),
            surrogate="topk_soft",
            topk=2,
            temperature=0.5,
            gradient_scale=0.1,
            teacher_temperature=0.5,
        )
        hidden = torch.tensor([[[0.9, 1.1], [2.0, 2.0]]], requires_grad=True)
        output = bridge(
            hidden,
            torch.tensor([2]),
            teacher_code_ids=torch.tensor([[1, 0]]),
            teacher_lengths=torch.tensor([2]),
        )
        self.assertAlmostEqual(float(output.teacher_agreement), 0.5)
        self.assertAlmostEqual(float(output.teacher_coverage), 1.0)
        self.assertGreater(float(output.teacher_commitment_loss), float(output.commitment_loss))
        (output.teacher_ce_loss + output.teacher_commitment_loss).backward()
        self.assertGreater(float(hidden.grad.abs().sum()), 0)

    def test_bridge_gradient_scale_reduces_surrogate_gradient(self) -> None:
        kwargs = dict(
            whisper_hidden_size=2,
            qwen_hidden_size=3,
            codebook=torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]),
            qwen_glm_embeddings=torch.tensor(
                [[2.0, 3.0, 4.0], [5.0, 6.0, 7.0], [8.0, 9.0, 10.0]]
            ),
            surrogate="topk_soft",
            topk=2,
            temperature=0.5,
        )
        full = Phase3STEBridge(**kwargs, gradient_scale=1.0)
        scaled = Phase3STEBridge(**kwargs, gradient_scale=0.1)
        hidden_full = torch.tensor([[[0.9, 1.1]]], requires_grad=True)
        hidden_scaled = hidden_full.detach().clone().requires_grad_(True)
        full(hidden_full).embeddings.sum().backward()
        scaled(hidden_scaled).embeddings.sum().backward()
        torch.testing.assert_close(hidden_scaled.grad, hidden_full.grad * 0.1)

    def test_joint_loss_weights_validate_bridge_commitment(self) -> None:
        weights = JointLossWeights(
            bridge_commitment=0.25,
            whisper_quantize=1.0,
            teacher_glm_ce=1.0,
            teacher_glm_commitment=1.0,
        )
        self.assertEqual(weights.bridge_commitment, 0.25)
        self.assertEqual(weights.whisper_quantize, 1.0)
        with self.assertRaises(ValueError):
            JointLossWeights(bridge_commitment=-0.1)

    def test_commitment_guard_uses_distributed_mean_not_rank_maximum(self) -> None:
        model = Phase3WhisperStreamSpeechJointModel.__new__(
            Phase3WhisperStreamSpeechJointModel
        )
        torch.nn.Module.__init__(model)
        model.max_bridge_commitment = 0.1
        model.max_bridge_commitment_ratio = None
        model.bridge_guard_baseline_microbatches = 0
        model.register_buffer("bridge_guard_baseline_sum", torch.zeros(()))
        model.register_buffer(
            "bridge_guard_baseline_count", torch.zeros((), dtype=torch.long)
        )

        def emulate_sum(value: torch.Tensor, *, op: object) -> None:
            del op
            value.fill_(0.20)

        prefix = "training.phase3_whisper_streamspeech_joint.model.dist"
        with (
            mock.patch(f"{prefix}.is_available", return_value=True),
            mock.patch(f"{prefix}.is_initialized", return_value=True),
            mock.patch(f"{prefix}.get_world_size", return_value=4),
            mock.patch(f"{prefix}.all_reduce", side_effect=emulate_sum),
        ):
            model._guard_bridge_commitment(torch.tensor(0.12), chunk_ms=960)

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

    def test_ctc_loss_excludes_infeasible_rows_without_target_shift(self) -> None:
        logits = torch.randn(2, 3, 5, requires_grad=True)
        targets = torch.tensor([1, 2, 3, 4, 1])
        loss, infeasible = ctc_normalized_loss(
            logits,
            targets,
            torch.tensor([3, 2]),
            torch.tensor([2, 3]),
            blank_id=4,
        )
        self.assertEqual(int(infeasible), 1)
        self.assertTrue(torch.isfinite(loss.mean))
        loss.mean.backward()
        self.assertGreater(float(logits.grad[0].abs().sum()), 0)
        self.assertEqual(float(logits.grad[1].abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
