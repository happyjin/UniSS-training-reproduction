"""Guards on seeding the Talker from a UniSS checkpoint."""

from __future__ import annotations

import json
import types

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from experiments.uniss_omni_s2st_v1.modeling.talker_transfer import (
    UNISS_SEMANTIC_OFFSET,
    TransferError,
    check_geometry,
    transfer_into_talker,
)

HIDDEN = 8
CODE_SIZE = 16
VOCAB = CODE_SIZE + 4


def _talker_config(**overrides):
    base = dict(
        hidden_size=HIDDEN,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        intermediate_size=16,
        vocab_size=VOCAB,
        tts_codec_pad_token_id=VOCAB - 4,
        tts_codec_start_token_id=VOCAB - 3,
        tts_codec_end_token_id=VOCAB - 2,
        tts_codec_mask_token_id=VOCAB - 1,
    )
    base.update(overrides)
    return types.SimpleNamespace(**base)


class _Talker(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = _talker_config()
        self.model = nn.Module()
        self.model.norm = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.model.embed_tokens = nn.Embedding(VOCAB, 32)
        self.codec_head = nn.Linear(HIDDEN, VOCAB, bias=False)
        self.thinker_to_talker_proj = nn.Linear(32, HIDDEN)


def _donor(tmp_path, **config_overrides):
    rows = UNISS_SEMANTIC_OFFSET + CODE_SIZE
    tensors = {
        "model.norm.weight": torch.full((HIDDEN, HIDDEN), 0.5),
        "model.embed_tokens.weight": torch.arange(
            rows * HIDDEN, dtype=torch.float32
        ).reshape(rows, HIDDEN),
    }
    save_file(tensors, str(tmp_path / "model.safetensors"))
    config = dict(
        hidden_size=HIDDEN,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        intermediate_size=16,
        tie_word_embeddings=True,
    )
    config.update(config_overrides)
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return tmp_path


def test_copies_the_shared_tensors_and_the_code_head(tmp_path):
    talker = _Talker()
    report = transfer_into_talker(talker, _donor(tmp_path), code_size=CODE_SIZE)
    assert report["head_rows"] == CODE_SIZE
    assert torch.allclose(talker.model.norm.weight, torch.full((HIDDEN, HIDDEN), 0.5))
    expected = torch.arange(
        UNISS_SEMANTIC_OFFSET * HIDDEN,
        (UNISS_SEMANTIC_OFFSET + CODE_SIZE) * HIDDEN,
        dtype=torch.float32,
    ).reshape(CODE_SIZE, HIDDEN)
    assert torch.allclose(talker.codec_head.weight[:CODE_SIZE], expected)


def test_the_reserved_head_rows_are_left_alone(tmp_path):
    talker = _Talker()
    before = talker.codec_head.weight[CODE_SIZE:].clone()
    transfer_into_talker(talker, _donor(tmp_path), code_size=CODE_SIZE)
    assert torch.equal(talker.codec_head.weight[CODE_SIZE:], before)


def test_the_omni_only_bridge_is_left_alone(tmp_path):
    """It maps 2,048 -> 896; the donor has no counterpart to copy."""
    talker = _Talker()
    before = talker.thinker_to_talker_proj.weight.clone()
    report = transfer_into_talker(talker, _donor(tmp_path), code_size=CODE_SIZE)
    assert torch.equal(talker.thinker_to_talker_proj.weight, before)
    assert "thinker_to_talker_proj.weight" in report["skipped"]


def test_the_code_embedding_is_left_alone(tmp_path):
    """Omni's is 2,048-dim against the donor's 896; copying would be wrong."""
    talker = _Talker()
    before = talker.model.embed_tokens.weight.clone()
    transfer_into_talker(talker, _donor(tmp_path), code_size=CODE_SIZE)
    assert torch.equal(talker.model.embed_tokens.weight, before)


def test_a_differently_shaped_donor_is_refused(tmp_path):
    talker = _Talker()
    with pytest.raises(TransferError, match="geometry differs"):
        transfer_into_talker(
            talker, _donor(tmp_path, num_hidden_layers=48), code_size=CODE_SIZE
        )


def test_an_untied_donor_is_refused(tmp_path):
    """The head is read off the embedding; untied, that is the wrong matrix."""
    talker = _Talker()
    with pytest.raises(TransferError, match="tie"):
        transfer_into_talker(
            talker, _donor(tmp_path, tie_word_embeddings=False), code_size=CODE_SIZE
        )


def test_a_donor_too_short_for_the_semantic_block_is_refused(tmp_path):
    talker = _Talker()
    save_file(
        {"model.embed_tokens.weight": torch.zeros(10, HIDDEN)},
        str(tmp_path / "model.safetensors"),
    )
    (tmp_path / "config.json").write_text(
        json.dumps(
            dict(
                hidden_size=HIDDEN,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
                intermediate_size=16,
                tie_word_embeddings=True,
            )
        ),
        encoding="utf-8",
    )
    with pytest.raises(TransferError, match="needs"):
        transfer_into_talker(talker, tmp_path, code_size=CODE_SIZE)


def test_check_geometry_accepts_a_matching_donor(tmp_path):
    assert check_geometry(_donor(tmp_path), _talker_config())["hidden_size"] == HIDDEN
