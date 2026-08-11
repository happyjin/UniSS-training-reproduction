from __future__ import annotations

import torch

from experiments.uniss_phase3_runtime_parity_streaming_v2.overfit7.natural_length import (
    NaturalLengthParallelSemanticBlockHead,
)
from training import constants_uniss as c


def test_content_and_length_posteriors_memorize_without_forced_end() -> None:
    torch.manual_seed(17)
    head = NaturalLengthParallelSemanticBlockHead(
        16, maximum_semantic_tokens=4
    )
    optimizer = torch.optim.AdamW(head.parameters(), lr=3e-2, weight_decay=0.0)
    hidden = torch.randn(12, 16)
    labels = torch.zeros(12, dtype=torch.long)
    roles = torch.zeros(12, dtype=torch.long)
    labels[1:3] = torch.tensor(
        [c.BICODEC_SEMANTIC_OFFSET + 10, c.BICODEC_SEMANTIC_OFFSET + 20]
    )
    roles[1:3] = 3
    labels[6:10] = torch.tensor(
        [
            c.BICODEC_SEMANTIC_OFFSET + 30,
            c.BICODEC_SEMANTIC_OFFSET + 40,
            c.BICODEC_SEMANTIC_OFFSET + 50,
            c.BICODEC_SEMANTIC_OFFSET + 60,
        ]
    )
    roles[6:10] = 3
    mask = (roles == 3).float()
    for _ in range(120):
        output = head.training_output(hidden, labels, roles, mask, torch.empty(0))
        loss = output.content_term.mean + 2.0 * output.length_term.mean
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    output = head.training_output(hidden, labels, roles, mask, torch.empty(0))
    assert output.token_accuracy.item() == 1.0
    assert output.length_accuracy.item() == 1.0
    assert output.length_mae.item() == 0.0
    codes, natural_end = head.decode(hidden[1], torch.empty(0))
    assert natural_end
    assert codes == (10, 20)
