#!/usr/bin/env python3
"""Megatron overfit v3: teach START_GLM continuation as well as final EOS."""

from __future__ import annotations

from collections import OrderedDict

import torch

import experiments.uniss_phase3_dense_aligned_streaming_pilot15_v1.training.pretrain_dense_aligned_megatron as dense
import experiments.uniss_phase3_runtime_parity_streaming_v2.overfit2.pretrain_overfit2 as v2
from experiments.uniss_phase3_dense_aligned_streaming_pilot15_v1.data.packing import (
    ROLE_ACTION,
    ROLE_BOUNDARY,
    ROLE_SEMANTIC,
    ROLE_TEXT,
)
from training import constants_uniss as c


OVERFIT3_WEIGHTS = OrderedDict(v2.OVERFIT_WEIGHTS)
OVERFIT3_WEIGHTS["boundary_continuity"] = 4.0


def trajectory_token_weights(
    labels: torch.Tensor,
    token_roles: torch.Tensor,
    loss_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    active = (loss_mask > 0).float()
    main = torch.zeros_like(loss_mask, dtype=torch.float32)
    main = torch.where(token_roles == ROLE_ACTION, 4.0, main)
    main = torch.where(token_roles == ROLE_TEXT, 8.0, main)
    main = torch.where(token_roles == ROLE_SEMANTIC, 2.0, main)
    main = torch.where(token_roles == ROLE_BOUNDARY, 2.0, main) * active

    boundary = (token_roles == ROLE_BOUNDARY).float() * active * 2.0
    boundary = torch.where(labels == c.TOKEN_END_CONTENT, 4.0 * active, boundary)
    boundary = torch.where(labels == c.TOKEN_END_SEMANTIC, 6.0 * active, boundary)
    # Equal emphasis prevents the terminal token from winning at the first
    # source-finished WRITE.  Context must decide whether to drain or stop.
    boundary = torch.where(labels == c.TOKEN_START_GLM, 8.0 * active, boundary)
    boundary = torch.where(labels == c.TOKEN_EOS, 8.0 * active, boundary)
    return main, boundary


def main() -> None:
    v2.trajectory_token_weights = trajectory_token_weights
    v2.OVERFIT_WEIGHTS = OVERFIT3_WEIGHTS
    dense.base.TrueSubsecondObjective = v2.RuntimeParityOverfit2Objective
    dense._distributed_dense_objective = v2.distributed_overfit2_objective
    dense.main()


if __name__ == "__main__":
    main()
