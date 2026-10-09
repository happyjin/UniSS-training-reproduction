"""Text-stream alignment and codec teacher forcing."""

from __future__ import annotations

import pytest
import torch

from experiments.uniss_omni_s2st_v1.modeling.stage0_assembly import (
    build_codec_sequence,
    build_text_stream,
    prefix_labels,
    prefix_mask,
)

EOS = torch.full((1, 1, 4), 9.0)
PAD = torch.full((1, 1, 4), 7.0)


def _stream(length, steps, batch=1):
    hidden = torch.ones(batch, length, 4)
    embeds = torch.ones(batch, length, 4) * 2
    return build_text_stream(hidden, embeds, steps=steps, eos_embed=EOS, pad_embed=PAD)


def test_conditioning_is_hidden_plus_embedding():
    """Omni sums both; conditioning on either alone is a different signal."""
    assert _stream(2, 2)[0, :, 0].tolist() == [3.0, 3.0]


def test_the_stream_runs_out_with_eos_then_pad():
    assert _stream(2, 5)[0, :, 0].tolist() == [3.0, 3.0, 9.0, 7.0, 7.0]


def test_exactly_one_step_past_the_text_is_the_eos():
    assert _stream(3, 4)[0, :, 0].tolist() == [3.0, 3.0, 3.0, 9.0]


def test_a_stream_shorter_than_its_text_is_truncated_not_padded():
    assert _stream(5, 2)[0, :, 0].tolist() == [3.0, 3.0]


def test_the_stream_is_built_for_every_row():
    assert _stream(2, 4, batch=3).shape == (3, 4, 4)


def test_mismatched_hidden_and_embeds_are_rejected():
    with pytest.raises(ValueError, match="must agree"):
        build_text_stream(
            torch.ones(1, 2, 4), torch.ones(1, 3, 4),
            steps=4, eos_embed=EOS, pad_embed=PAD,
        )


def test_inputs_are_bos_then_the_codes_shifted_right():
    inputs, labels, mask = build_codec_sequence(
        [torch.tensor([5, 6, 7])], bos_id=90, pad_id=91
    )
    assert inputs[0].tolist() == [90, 5, 6]
    assert labels[0].tolist() == [5, 6, 7]
    assert mask[0].tolist() == [1, 1, 1]


def test_padding_is_ignored_in_the_loss_and_the_mask():
    inputs, labels, mask = build_codec_sequence(
        [torch.tensor([5, 6, 7]), torch.tensor([8])], bos_id=90, pad_id=91
    )
    assert inputs[1].tolist() == [90, 91, 91]
    assert labels[1].tolist() == [8, -100, -100]
    assert mask[1].tolist() == [1, 0, 0]


def test_eos_is_appended_as_a_target_and_fed_as_the_last_code():
    inputs, labels, mask = build_codec_sequence(
        [torch.tensor([5, 6, 7])], bos_id=90, pad_id=91, eos_id=99
    )
    assert inputs[0].tolist() == [90, 5, 6, 7]
    assert labels[0].tolist() == [5, 6, 7, 99]
    assert mask[0].tolist() == [1, 1, 1, 1]


def test_eos_lands_at_each_rows_own_length():
    _, labels, mask = build_codec_sequence(
        [torch.tensor([5, 6, 7]), torch.tensor([8])],
        bos_id=90, pad_id=91, eos_id=99,
    )
    assert labels[1].tolist() == [8, 99, -100, -100]
    assert mask[1].tolist() == [1, 1, 0, 0]


def test_an_empty_batch_is_an_error():
    with pytest.raises(ValueError, match="no sequences"):
        build_codec_sequence([], bos_id=90, pad_id=91)


def test_prefix_labels_marks_the_prefix_unpredicted():
    labels = torch.tensor([[5, 6, -100]])
    assert prefix_labels(labels, 2).tolist() == [[-100, -100, 5, 6, -100]]


def test_prefix_mask_lets_the_prefix_be_attended():
    assert prefix_mask(torch.tensor([[1, 1, 0]]), 2).tolist() == [[1, 1, 1, 1, 0]]


def test_a_zero_width_prefix_changes_nothing():
    labels = torch.tensor([[5, 6]])
    mask = torch.tensor([[1, 0]])
    assert torch.equal(prefix_labels(labels, 0), labels)
    assert torch.equal(prefix_mask(mask, 0), mask)


def test_prefix_helpers_keep_the_batch():
    labels = torch.tensor([[5], [6], [7]])
    assert prefix_labels(labels, 4).shape == (3, 5)
    assert prefix_mask(torch.ones(3, 1, dtype=torch.long), 4).shape == (3, 5)
