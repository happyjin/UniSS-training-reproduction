from __future__ import annotations

import torch

from training import constants_uniss as c
from web_demo.runtime_parity_streaming_v2.inference import (
    _decode_semantic_choice,
    _decode_text_choice,
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
