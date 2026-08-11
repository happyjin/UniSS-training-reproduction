from __future__ import annotations

import torch

from experiments.uniss_phase3_dense_aligned_streaming_pilot15_v1.data.packing import (
    ROLE_ACTION,
    ROLE_BOUNDARY,
    ROLE_SEMANTIC,
    ROLE_TEXT,
)
from experiments.uniss_phase3_runtime_parity_streaming_v2.overfit4.pretrain_overfit4 import (
    OVERFIT4_WEIGHTS,
    trajectory_token_weights,
)
from training import constants_uniss as c


def test_content_consolidation_retains_all_grammar_targets() -> None:
    labels = torch.tensor(
        [7, 8, 9, c.TOKEN_END_CONTENT, c.TOKEN_END_SEMANTIC, c.TOKEN_START_GLM, c.TOKEN_EOS]
    )
    roles = torch.tensor(
        [
            ROLE_ACTION,
            ROLE_TEXT,
            ROLE_SEMANTIC,
            ROLE_BOUNDARY,
            ROLE_BOUNDARY,
            ROLE_BOUNDARY,
            ROLE_BOUNDARY,
        ]
    )
    main, boundary = trajectory_token_weights(labels, roles, torch.ones(7))
    assert main.tolist() == [4.0, 16.0, 4.0, 1.0, 1.0, 1.0, 1.0]
    assert boundary.tolist() == [0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 2.0]


def test_boundary_term_is_regularizer_not_dominant_objective() -> None:
    assert 0 < OVERFIT4_WEIGHTS["boundary_continuity"]
    assert OVERFIT4_WEIGHTS["boundary_continuity"] < OVERFIT4_WEIGHTS["interleaved_trajectory"]
