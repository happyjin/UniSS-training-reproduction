from __future__ import annotations

import torch

from experiments.uniss_phase3_runtime_parity_streaming_v2.overfit5.semantic_block import (
    END_CLASS,
    ParallelSemanticBlockHead,
)
from training import constants_uniss as c


def test_parallel_semantic_targets_include_natural_end() -> None:
    torch.manual_seed(7)
    hidden_size = 16
    head = ParallelSemanticBlockHead(hidden_size, maximum_semantic_tokens=4)
    hidden = torch.randn(8, hidden_size)
    labels = torch.tensor(
        [0, c.BICODEC_SEMANTIC_OFFSET + 4, c.BICODEC_SEMANTIC_OFFSET + 5, 0,
         c.BICODEC_SEMANTIC_OFFSET + 9, 0, 0, 0]
    )
    roles = torch.tensor([0, 3, 3, 0, 3, 0, 0, 0])
    mask = (roles == 3).float()
    embeddings = torch.randn(c.VOCAB_SIZE, hidden_size)
    output = head.training_output(hidden, labels, roles, mask, embeddings)
    assert output.blocks.item() == 2
    assert output.term.denominator.item() == 11
    assert torch.isfinite(output.term.mean)


def test_decode_requires_model_selected_end() -> None:
    head = ParallelSemanticBlockHead(4, maximum_semantic_tokens=3)

    def fake_forward(context, weight):
        del context, weight
        logits = torch.full((1, 4, END_CLASS + 1), -10.0)
        logits[0, 0, 17] = 10.0
        logits[0, 1, 23] = 10.0
        logits[0, 2, END_CLASS] = 10.0
        return logits

    head.forward = fake_forward  # type: ignore[method-assign]
    values, natural_end = head.decode(torch.zeros(4), torch.zeros(1, 4))
    assert values == (17, 23)
    assert natural_end is True

