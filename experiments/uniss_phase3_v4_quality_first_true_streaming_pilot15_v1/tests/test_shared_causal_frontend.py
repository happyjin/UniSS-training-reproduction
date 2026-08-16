from __future__ import annotations

import numpy as np
import pytest
import torch

from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v1.stage00_baseline.shared_causal_frontend import (
    BLOCK_SAMPLES,
    SAMPLE_RATE,
    SharedCausalWhisperVQFrontend,
)
from uniss.speech_tokenizer.glm4.configuration_whisper import WhisperVQConfig
from uniss.speech_tokenizer.glm4.modeling_whisper import WhisperVQEncoder


def _tiny_frontend(max_positions: int = 64) -> SharedCausalWhisperVQFrontend:
    config = WhisperVQConfig(
        d_model=32,
        encoder_layers=16,
        encoder_attention_heads=4,
        encoder_ffn_dim=64,
        num_mel_bins=8,
        max_source_positions=max_positions,
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
        encoder_causal_convolution=True,
    )
    config._attn_implementation = "eager"
    encoder = WhisperVQEncoder(config).eval()
    generator = np.random.default_rng(7)
    filters = generator.random((201, 8), dtype=np.float32)
    return SharedCausalWhisperVQFrontend(
        encoder, filters, device=torch.device("cpu")
    ).eval()


def _stream(frontend: SharedCausalWhisperVQFrontend, waveform: torch.Tensor):
    state = None
    hidden = []
    quantized = []
    tokens = []
    reset_flags = []
    for start in range(0, waveform.numel(), BLOCK_SAMPLES):
        end = min(waveform.numel(), start + BLOCK_SAMPLES)
        step = frontend.push(
            waveform[start:end], state, is_final=end == waveform.numel()
        )
        state = step.state
        hidden.append(step.pre_vq_hidden)
        quantized.append(step.quantized_hidden)
        tokens.append(step.token_ids)
        reset_flags.append(step.encoder_reset_before_block)
    return (
        torch.cat(hidden, dim=1),
        torch.cat(quantized, dim=1),
        torch.cat(tokens, dim=1),
        state,
        reset_flags,
    )


def test_full_and_cached_share_exact_pcm_features_and_tokens() -> None:
    torch.manual_seed(11)
    frontend = _tiny_frontend()
    waveform = torch.randn(BLOCK_SAMPLES * 3 + 641) * 0.02
    full = frontend.forward_full_reference(waveform)
    hidden, quantized, tokens, state, _ = _stream(frontend, waveform)
    torch.testing.assert_close(full.pre_vq_hidden, hidden, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(full.quantized_hidden, quantized, rtol=0.0, atol=0.0)
    torch.testing.assert_close(full.token_ids, tokens, rtol=0.0, atol=0.0)
    assert full.valid_tokens == 7
    assert state.finalized


def test_recomputed_reference_and_cached_path_share_block_geometry() -> None:
    torch.manual_seed(12)
    frontend = _tiny_frontend()
    waveform = torch.randn(BLOCK_SAMPLES * 3 + 641) * 0.02
    recomputed = frontend.forward_recomputed_reference(waveform)
    hidden, quantized, tokens, _, _ = _stream(frontend, waveform)
    torch.testing.assert_close(
        recomputed.pre_vq_hidden, hidden, rtol=2e-5, atol=2e-6
    )
    torch.testing.assert_close(
        recomputed.quantized_hidden, quantized, rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(recomputed.token_ids, tokens, rtol=0.0, atol=0.0)


def test_future_block_perturbation_cannot_change_committed_prefix() -> None:
    torch.manual_seed(13)
    frontend = _tiny_frontend()
    original = torch.randn(BLOCK_SAMPLES * 4) * 0.01
    changed = original.clone()
    changed[BLOCK_SAMPLES * 2 :] = torch.randn_like(changed[BLOCK_SAMPLES * 2 :])
    first = frontend.forward_full_reference(original)
    second = frontend.forward_full_reference(changed)
    torch.testing.assert_close(
        first.pre_vq_hidden[:, :4], second.pre_vq_hidden[:, :4], rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        first.token_ids[:, :4], second.token_ids[:, :4], rtol=0.0, atol=0.0
    )


def test_position_reset_preserves_full_cached_parity() -> None:
    torch.manual_seed(17)
    # 64 positions permit exactly eight 160 ms blocks per segment.
    frontend = _tiny_frontend(max_positions=64)
    waveform = torch.randn(BLOCK_SAMPLES * 9 + 1200) * 0.01
    full = frontend.forward_full_reference(waveform)
    hidden, _, tokens, state, resets = _stream(frontend, waveform)
    torch.testing.assert_close(full.pre_vq_hidden, hidden, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(full.token_ids, tokens, rtol=0.0, atol=0.0)
    assert full.encoder_segments == 2
    assert state.encoder_resets == 1
    assert sum(resets) == 1


def test_position_reset_preserves_recomputed_cached_parity() -> None:
    torch.manual_seed(18)
    frontend = _tiny_frontend(max_positions=64)
    waveform = torch.randn(BLOCK_SAMPLES * 9 + 1200) * 0.01
    recomputed = frontend.forward_recomputed_reference(waveform)
    hidden, _, tokens, state, _ = _stream(frontend, waveform)
    torch.testing.assert_close(
        recomputed.pre_vq_hidden, hidden, rtol=2e-5, atol=2e-6
    )
    torch.testing.assert_close(recomputed.token_ids, tokens, rtol=0.0, atol=0.0)
    assert recomputed.encoder_segments == 2
    assert state.encoder_resets == 1


def test_partial_nonfinal_block_and_append_after_final_are_rejected() -> None:
    frontend = _tiny_frontend()
    with pytest.raises(ValueError, match="partial PCM"):
        frontend.push(torch.zeros(100), is_final=False)
    final = frontend.push(torch.zeros(100), is_final=True)
    with pytest.raises(ValueError, match="after frontend finalization"):
        frontend.push(torch.zeros(BLOCK_SAMPLES), final.state, is_final=True)


def test_training_extractor_keeps_conv_gradient_path() -> None:
    torch.manual_seed(19)
    frontend = _tiny_frontend()
    waveform = torch.randn(SAMPLE_RATE // 2) * 0.01
    convolved, valid_tokens = frontend.extract_convolved(waveform)
    convolved.square().mean().backward()
    assert valid_tokens == 7
    assert frontend.encoder_model.conv1.weight.grad is not None
    assert frontend.encoder_model.conv2.weight.grad is not None


def test_registered_frontend_buffers_follow_declared_device() -> None:
    frontend = _tiny_frontend()
    assert frontend.mel_filters.device == frontend.device
    assert frontend.stft_window.device == frontend.device
