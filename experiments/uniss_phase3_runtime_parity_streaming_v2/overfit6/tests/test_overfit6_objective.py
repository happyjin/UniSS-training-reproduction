from __future__ import annotations

import torch

from experiments.uniss_phase3_runtime_parity_streaming_v2.overfit6.pretrain_overfit6 import (
    UntiedParallelSemanticBlockHead,
)
from training import constants_uniss as c


def test_untied_head_can_memorize_codes_and_natural_length() -> None:
    torch.manual_seed(11)
    head = UntiedParallelSemanticBlockHead(16, maximum_semantic_tokens=4)
    optimizer = torch.optim.AdamW(head.parameters(), lr=3e-2, weight_decay=0.0)
    hidden = torch.randn(5, 16)
    labels = torch.tensor(
        [c.BICODEC_SEMANTIC_OFFSET + 10, c.BICODEC_SEMANTIC_OFFSET + 20, 0, 0, 0]
    )
    roles = torch.tensor([3, 3, 0, 0, 0])
    mask = (roles == 3).float()
    dummy_embeddings = torch.empty(0)
    for _ in range(80):
        output = head.training_output(
            hidden, labels, roles, mask, dummy_embeddings
        )
        optimizer.zero_grad()
        output.term.mean.backward()
        optimizer.step()
    output = head.training_output(hidden, labels, roles, mask, dummy_embeddings)
    assert output.token_accuracy.item() == 1.0
    assert output.end_accuracy.item() == 1.0
    assert output.length_mae.item() == 0.0

