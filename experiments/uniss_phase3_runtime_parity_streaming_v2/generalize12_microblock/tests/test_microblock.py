from __future__ import annotations

import types

import torch

from experiments.uniss_phase3_dense_aligned_streaming_pilot15_v1.data.packing import (
    ROLE_OBSERVED,
    ROLE_SEMANTIC,
)
from experiments.uniss_phase3_runtime_parity_streaming_v2.generalize12_microblock.microblock import (
    CausalMicroblockSemanticHead,
    _balanced_example_weights,
    extract_microblock_targets,
)
from training import constants_uniss as c


def test_extracts_runtime_identical_context_positions_and_natural_targets() -> None:
    hidden = torch.arange(12 * 3, dtype=torch.float32).reshape(12, 3)
    labels = torch.zeros(12, dtype=torch.long)
    roles = torch.full((12,), ROLE_OBSERVED, dtype=torch.long)
    mask = torch.zeros(12)
    # One six-unit semantic span.  Shifted position two is START_SEMANTIC's
    # hidden; shifted position six is the fourth semantic unit's hidden.
    labels[2:8] = c.BICODEC_SEMANTIC_OFFSET + torch.arange(10, 16)
    roles[2:8] = ROLE_SEMANTIC
    mask[2:8] = 1
    value = extract_microblock_targets(
        hidden, labels, roles, mask, block_size=4
    )
    assert value is not None
    torch.testing.assert_close(value.contexts, hidden[torch.tensor([2, 6])])
    assert value.targets.tolist() == [[10, 11, 12, 13], [14, 15, 0, 0]]
    assert value.content_mask.tolist() == [[True] * 4, [True, True, False, False]]
    assert value.lengths.tolist() == [4, 2]
    assert value.continue_targets.tolist() == [1, 0]
    assert value.final_mask.tolist() == [False, True]


def test_balancing_downweights_repeated_unit_without_extreme_gradients() -> None:
    targets = torch.tensor([[7, 7, 7, 7], [7, 9, 10, 11]])
    mask = torch.ones_like(targets, dtype=torch.bool)
    weights = _balanced_example_weights(targets, mask, classes=16)
    assert float(weights[0, 0]) < float(weights[1, 1])
    assert float(weights.min()) >= 0.5
    assert float(weights.max()) <= 4.0


def test_decode_uses_four_units_for_continue_and_natural_final_length() -> None:
    head = CausalMicroblockSemanticHead(8, block_size=4)
    context = torch.zeros(1, 8)
    embeddings = torch.zeros(c.BICODEC_SEMANTIC_OFFSET + c.BICODEC_SEMANTIC_SIZE, 8)
    chosen = torch.tensor([3, 4, 5, 6])

    def content_logits(self, context, word_embedding_weight, *, teacher_targets=None):
        del word_embedding_weight, teacher_targets
        logits = torch.full((context.shape[0], 4, c.BICODEC_SEMANTIC_SIZE), -10.0)
        logits[0, torch.arange(4), chosen] = 10.0
        return logits

    head.content_logits = types.MethodType(content_logits, head)
    head.continuation_logits = types.MethodType(
        lambda self, value: torch.tensor([[0.0, 1.0]]), head
    )
    units, continued = head.decode(context, embeddings)
    assert units == (3, 4, 5, 6)
    assert continued

    head.continuation_logits = types.MethodType(
        lambda self, value: torch.tensor([[1.0, 0.0]]), head
    )
    head.length_logits = types.MethodType(
        lambda self, value: torch.tensor([[0.0, 2.0, 0.0, 0.0]]), head
    )
    units, continued = head.decode(context, embeddings)
    assert units == (3, 4)
    assert not continued
