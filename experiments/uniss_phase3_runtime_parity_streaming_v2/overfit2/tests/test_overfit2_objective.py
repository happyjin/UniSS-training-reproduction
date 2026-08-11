from __future__ import annotations

import torch

from experiments.uniss_phase3_dense_aligned_streaming_pilot15_v1.data.packing import (
    ROLE_ACTION,
    ROLE_BOUNDARY,
    ROLE_SEMANTIC,
    ROLE_TEXT,
)
from experiments.uniss_phase3_runtime_parity_streaming_v2.overfit2.pretrain_overfit2 import (
    OVERFIT_WEIGHTS,
    trajectory_token_weights,
)
from training import constants_uniss as c


def test_terminal_and_content_roles_are_strengthened() -> None:
    labels = torch.tensor([7, 8, 9, c.TOKEN_END_CONTENT, c.TOKEN_END_SEMANTIC, c.TOKEN_EOS])
    roles = torch.tensor(
        [ROLE_ACTION, ROLE_TEXT, ROLE_SEMANTIC, ROLE_BOUNDARY, ROLE_BOUNDARY, ROLE_BOUNDARY]
    )
    mask = torch.ones(6)
    main, boundary = trajectory_token_weights(labels, roles, mask)
    assert main.tolist() == [4.0, 8.0, 2.0, 2.0, 2.0, 2.0]
    assert boundary.tolist() == [0.0, 0.0, 0.0, 4.0, 6.0, 12.0]


def test_overfit2_keeps_phase3_anchor_but_prioritizes_grammar() -> None:
    assert OVERFIT_WEIGHTS["phase3_replay"] > 0
    assert OVERFIT_WEIGHTS["boundary_continuity"] > OVERFIT_WEIGHTS["phase3_replay"]
    assert OVERFIT_WEIGHTS["deadline_survival"] == 0
