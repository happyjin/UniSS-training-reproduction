from __future__ import annotations

import torch

from experiments.uniss_phase3_dense_aligned_streaming_pilot15_v1.data.packing import ROLE_BOUNDARY
from experiments.uniss_phase3_runtime_parity_streaming_v2.overfit3.pretrain_overfit3 import (
    trajectory_token_weights,
)
from training import constants_uniss as c


def test_continue_and_stop_boundaries_have_equal_weight() -> None:
    labels = torch.tensor([c.TOKEN_START_GLM, c.TOKEN_EOS])
    roles = torch.tensor([ROLE_BOUNDARY, ROLE_BOUNDARY])
    _, boundary = trajectory_token_weights(labels, roles, torch.ones(2))
    assert boundary.tolist() == [8.0, 8.0]
