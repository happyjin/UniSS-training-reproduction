from __future__ import annotations

import unittest

import torch

from experiments.uniss_phase3_runtime_parity_streaming_v2.frontend.cached_whispervq import (
    CachedBlockCausalWhisperVQ,
    block_causal_attention_mask,
)
from uniss.speech_tokenizer.glm4.configuration_whisper import WhisperVQConfig
from uniss.speech_tokenizer.glm4.modeling_whisper import WhisperVQEncoder


def _tiny_16_layer_whispervq() -> CachedBlockCausalWhisperVQ:
    config = WhisperVQConfig(
        d_model=32,
        encoder_layers=16,
        encoder_attention_heads=4,
        encoder_ffn_dim=64,
        num_mel_bins=8,
        max_source_positions=64,
        dropout=0.0,
        attention_dropout=0.0,
        activation_dropout=0.0,
        encoder_layerdrop=0.0,
        pooling_kernel_size=4,
        pooling_type="avg",
        pooling_position=16,
        quantize_vocab_size=97,
        quantize_position=16,
        quantize_encoder_only=True,
    )
    config._attn_implementation = "eager"
    encoder = WhisperVQEncoder(config).eval()
    return CachedBlockCausalWhisperVQ.from_whispervq_encoder(encoder).eval()


class CachedWhisperVQTest(unittest.TestCase):
    def test_mask_is_bidirectional_inside_block_and_causal_between_blocks(self) -> None:
        mask = block_causal_attention_mask(
            batch_size=1,
            sequence_length=18,
            block_frames=8,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )[0, 0]
        allowed = mask == 0
        self.assertTrue(bool(allowed[0, :8].all()))
        self.assertFalse(bool(allowed[0, 8:].any()))
        self.assertTrue(bool(allowed[7, :8].all()))
        self.assertTrue(bool(allowed[8, :16].all()))
        self.assertFalse(bool(allowed[8, 16:].any()))
        self.assertTrue(bool(allowed[17].all()))

    def test_full_and_cached_16_layer_outputs_have_exact_semantics(self) -> None:
        torch.manual_seed(17)
        model = _tiny_16_layer_whispervq()
        convolved = torch.randn(2, 24, 32)
        full = model.forward_full(convolved)

        state = None
        streamed_hidden = []
        streamed_quantized = []
        streamed_tokens = []
        for start in range(0, 24, 8):
            output = model.forward_chunk(
                convolved[:, start : start + 8],
                state,
                is_final=start == 16,
            )
            state = output.state
            streamed_hidden.append(output.pre_vq_hidden)
            streamed_quantized.append(output.quantized_hidden)
            streamed_tokens.append(output.token_ids)

        torch.testing.assert_close(
            full.pre_vq_hidden,
            torch.cat(streamed_hidden, dim=1),
            rtol=2e-5,
            atol=2e-6,
        )
        torch.testing.assert_close(
            full.quantized_hidden,
            torch.cat(streamed_quantized, dim=1),
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            full.token_ids,
            torch.cat(streamed_tokens, dim=1),
            rtol=0.0,
            atol=0.0,
        )
        self.assertIsNotNone(state)
        self.assertTrue(state.finalized)
        self.assertEqual(state.frames_seen, 24)
        self.assertEqual(len(state.layers), 16)
        self.assertTrue(all(layer.frames == 24 for layer in state.layers))

    def test_final_partial_block_matches_full_reference(self) -> None:
        torch.manual_seed(23)
        model = _tiny_16_layer_whispervq()
        convolved = torch.randn(1, 19, 32)
        full = model.forward_full(convolved)
        state = None
        pieces = []
        tokens = []
        for start, end, final in ((0, 8, False), (8, 16, False), (16, 19, True)):
            output = model.forward_chunk(convolved[:, start:end], state, is_final=final)
            state = output.state
            pieces.append(output.pre_vq_hidden)
            tokens.append(output.token_ids)
        torch.testing.assert_close(
            full.pre_vq_hidden,
            torch.cat(pieces, dim=1),
            rtol=2e-5,
            atol=2e-6,
        )
        torch.testing.assert_close(
            full.token_ids,
            torch.cat(tokens, dim=1),
            rtol=0.0,
            atol=0.0,
        )

    def test_later_blocks_cannot_change_a_committed_prefix(self) -> None:
        torch.manual_seed(31)
        model = _tiny_16_layer_whispervq()
        first = torch.randn(1, 24, 32)
        second = first.clone()
        second[:, 8:] = torch.randn_like(second[:, 8:]) * 7.0
        first_output = model.forward_full(first)
        second_output = model.forward_full(second)
        # One 160 ms block becomes two 80 ms WhisperVQ tokens after 4:1 pooling.
        torch.testing.assert_close(
            first_output.pre_vq_hidden[:, :2],
            second_output.pre_vq_hidden[:, :2],
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            first_output.token_ids[:, :2],
            second_output.token_ids[:, :2],
            rtol=0.0,
            atol=0.0,
        )

    def test_rejects_a_nonfinal_partial_block(self) -> None:
        model = _tiny_16_layer_whispervq()
        with self.assertRaisesRegex(ValueError, "short block"):
            model.forward_chunk(torch.randn(1, 4, 32), is_final=False)


if __name__ == "__main__":
    unittest.main()
