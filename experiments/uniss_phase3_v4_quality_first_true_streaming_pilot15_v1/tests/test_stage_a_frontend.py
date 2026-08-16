from __future__ import annotations

import torch

from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v1.stage_a_causal_whisper_asr.training.frontend import (
    block_causal_allowed,
)


def test_block_causal_mask_has_no_future_chunk_visibility() -> None:
    allowed = block_causal_allowed(
        torch.tensor([16, 10]),
        sequence_length=16,
        block_frames=8,
    )
    assert allowed.shape == (2, 16, 16)
    assert bool(allowed[0, 0, :8].all())
    assert not bool(allowed[0, 0, 8:].any())
    assert bool(allowed[0, 8, :16].all())
    assert not bool(allowed[1, 0, 10:].any())


def test_padding_queries_have_one_safe_key_but_no_valid_loss_role() -> None:
    allowed = block_causal_allowed(
        torch.tensor([3]),
        sequence_length=8,
        block_frames=8,
    )
    assert allowed[0, 3:, 0].all()
    assert not allowed[0, 3:, 1:].any()
