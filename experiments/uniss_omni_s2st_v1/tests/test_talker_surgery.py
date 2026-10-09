"""Codebook layout, the re-draw, and the output mask."""

from __future__ import annotations

import types

import pytest
import torch
from torch import nn

from experiments.uniss_omni_s2st_v1.modeling.talker_surgery import (
    BICODEC_SEMANTIC_SIZE,
    apply_output_mask,
    read_codec_layout,
    retarget_talker_codebook,
    valid_output_mask,
)


def _config(**overrides):
    base = dict(
        vocab_size=8448,
        tts_codec_pad_token_id=8292,
        tts_codec_start_token_id=8293,
        tts_codec_end_token_id=8294,
        tts_codec_mask_token_id=8296,
    )
    base.update(overrides)
    return types.SimpleNamespace(**base)


def _talker(vocab=8448, embed_dim=16, head_dim=8):
    talker = nn.Module()
    talker.config = _config(vocab_size=vocab)
    embedding = nn.Embedding(vocab, embed_dim)
    talker.codec_head = nn.Linear(head_dim, vocab, bias=False)
    talker.get_input_embeddings = lambda: embedding
    return talker, embedding


def test_layout_reads_the_real_checkpoint_shape():
    layout = read_codec_layout(_config())
    assert layout.vocab_size == 8448
    assert layout.code_size == BICODEC_SEMANTIC_SIZE
    assert layout.special_ids == {"pad": 8292, "bos": 8293, "eos": 8294, "mask": 8296}


def test_a_special_token_inside_the_code_range_is_rejected():
    """Such a slot would be unreachable, silently shrinking the codebook."""
    with pytest.raises(ValueError, match="inside the first"):
        read_codec_layout(_config(tts_codec_end_token_id=42))


def test_a_codebook_larger_than_the_vocab_is_rejected():
    with pytest.raises(ValueError, match="exceeds talker vocab"):
        read_codec_layout(_config(vocab_size=1024))


def test_redraw_touches_the_code_rows_and_nothing_else():
    talker, embedding = _talker()
    reserved_before = embedding.weight[BICODEC_SEMANTIC_SIZE:].clone()
    head_reserved_before = talker.codec_head.weight[BICODEC_SEMANTIC_SIZE:].clone()
    codes_before = embedding.weight[:BICODEC_SEMANTIC_SIZE].clone()

    report = retarget_talker_codebook(talker, seed=7)

    assert report["untouched_rows"] == 8448 - BICODEC_SEMANTIC_SIZE
    assert torch.equal(embedding.weight[BICODEC_SEMANTIC_SIZE:], reserved_before)
    assert torch.equal(talker.codec_head.weight[BICODEC_SEMANTIC_SIZE:], head_reserved_before)
    assert not torch.equal(embedding.weight[:BICODEC_SEMANTIC_SIZE], codes_before)


def test_redraw_is_deterministic_for_a_seed():
    first, embed_a = _talker()
    second, embed_b = _talker()
    with torch.no_grad():
        embed_b.weight.copy_(embed_a.weight)
        second.codec_head.weight.copy_(first.codec_head.weight)

    retarget_talker_codebook(first, seed=11)
    retarget_talker_codebook(second, seed=11)
    assert torch.equal(embed_a.weight, embed_b.weight)
    assert torch.equal(first.codec_head.weight, second.codec_head.weight)


def test_head_scale_zero_flattens_the_code_rows():
    talker, _ = _talker()
    retarget_talker_codebook(talker, seed=3, head_scale=0.0)
    assert torch.count_nonzero(talker.codec_head.weight[:BICODEC_SEMANTIC_SIZE]) == 0
    # The reserved rows are not part of the draw at any scale.
    assert torch.count_nonzero(talker.codec_head.weight[BICODEC_SEMANTIC_SIZE:]) > 0


def test_head_scale_shrinks_the_draw_proportionally():
    talker, _ = _talker()
    retarget_talker_codebook(talker, seed=5, head_scale=1.0)
    full = talker.codec_head.weight[:BICODEC_SEMANTIC_SIZE].std().item()
    retarget_talker_codebook(talker, seed=5, head_scale=0.25)
    quarter = talker.codec_head.weight[:BICODEC_SEMANTIC_SIZE].std().item()
    assert quarter == pytest.approx(full * 0.25, rel=0.1)


def test_a_checkpoint_disagreeing_with_its_config_is_an_error():
    talker, _ = _talker(vocab=8448)
    talker.config = _config(vocab_size=9000)
    with pytest.raises(ValueError, match="does not match its config"):
        retarget_talker_codebook(talker)


def test_output_mask_admits_the_codes_and_eos_only():
    layout = read_codec_layout(_config())
    mask = valid_output_mask(layout)
    assert int(mask.sum()) == BICODEC_SEMANTIC_SIZE + 1
    assert mask[:BICODEC_SEMANTIC_SIZE].all()
    assert mask[8294] and not mask[8293] and not mask[8292] and not mask[8296]


def test_output_mask_rejects_an_unknown_special():
    with pytest.raises(KeyError):
        valid_output_mask(read_codec_layout(_config()), allow=("nope",))


def test_apply_output_mask_removes_the_masked_mass():
    layout = read_codec_layout(_config())
    mask = valid_output_mask(layout)
    logits = torch.zeros(1, 3, layout.vocab_size)
    probs = apply_output_mask(logits, mask).softmax(dim=-1)
    assert probs[..., 8293].max() == 0.0
    assert probs.sum(dim=-1).allclose(torch.ones(1, 3))


def test_apply_output_mask_checks_its_width():
    layout = read_codec_layout(_config())
    with pytest.raises(ValueError, match="slots"):
        apply_output_mask(torch.zeros(1, 2, 10), valid_output_mask(layout))
