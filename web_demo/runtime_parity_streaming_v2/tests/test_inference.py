from __future__ import annotations

import torch

from training import constants_uniss as c
from web_demo.runtime_parity_streaming_v2.inference import (
    _decode_continuation_choice,
    _decode_semantic_choice,
    _decode_text_choice,
    _quality_failures,
)


def _logits() -> torch.Tensor:
    return torch.full((1, c.VOCAB_SIZE), -100.0)


def test_text_decoder_compares_base_vocabulary_with_end_boundary() -> None:
    logits = _logits()
    logits[0, 42] = 5
    logits[0, c.TOKEN_END_CONTENT] = 4
    assert _decode_text_choice(logits) == 42
    logits[0, c.TOKEN_END_CONTENT] = 6
    assert _decode_text_choice(logits) == c.TOKEN_END_CONTENT


def test_semantic_decoder_requires_one_code_before_end() -> None:
    logits = _logits()
    logits[0, c.BICODEC_SEMANTIC_OFFSET + 9] = 5
    logits[0, c.TOKEN_END_SEMANTIC] = 7
    assert _decode_semantic_choice(logits, allow_end=False) == 9
    assert (
        _decode_semantic_choice(logits, allow_end=True)
        == c.TOKEN_END_SEMANTIC
    )


def test_continuation_decoder_selects_natural_eos_without_forcing() -> None:
    logits = _logits()
    logits[0, c.TOKEN_START_GLM] = 3
    logits[0, c.TOKEN_EOS] = 2
    choice, eos_probability = _decode_continuation_choice(logits)
    assert choice == "START_GLM"
    assert 0.26 < eos_probability < 0.27

    logits[0, c.TOKEN_EOS] = 4
    choice, eos_probability = _decode_continuation_choice(logits)
    assert choice == "EOS"
    assert 0.73 < eos_probability < 0.74


def test_strict_quality_gate_rejects_slow_or_incomplete_runtime() -> None:
    failures = _quality_failures(
        natural_writes=10,
        semantic_tokens=160,
        first_write_source_ms=640,
        source_duration_ms=4220,
        translation_audio_samples=32000,
        text_similarity=0.926,
        minimum_text_similarity=0.98,
        natural_eos=True,
        rtf=2.65,
        maximum_rtf=1.0,
        first_audio_wall_ms=1972.0,
        maximum_first_audio_wall_ms=1000.0,
    )
    assert "translation_text_similarity_below_0.98" in failures
    assert "rtf_not_below_1.00" in failures
    assert "first_audio_wall_not_below_1000ms" in failures
