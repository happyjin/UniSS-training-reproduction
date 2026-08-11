#!/usr/bin/env python3
"""Megatron overfit v4: retain natural boundaries while consolidating content.

Overfit v3 solved the START_GLM/EOS continuation decision, but its very large
rare-boundary mass left the final drain text and semantic block underfit.  V4
starts from the completed v3 checkpoint, keeps every grammar target active,
and shifts capacity back to the actual text/audio continuation.
"""

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


OVERFIT4_WEIGHTS = OrderedDict(v2.OVERFIT_WEIGHTS)
OVERFIT4_WEIGHTS["boundary_continuity"] = 0.5


def trajectory_token_weights(
    labels: torch.Tensor,
    token_roles: torch.Tensor,
    loss_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prioritize drain content without removing any learned grammar target."""

    active = (loss_mask > 0).float()
    main = torch.zeros_like(loss_mask, dtype=torch.float32)
    main = torch.where(token_roles == ROLE_ACTION, 4.0, main)
    main = torch.where(token_roles == ROLE_TEXT, 16.0, main)
    main = torch.where(token_roles == ROLE_SEMANTIC, 4.0, main)
    main = torch.where(token_roles == ROLE_BOUNDARY, 1.0, main) * active

    boundary = (token_roles == ROLE_BOUNDARY).float() * active
    boundary = torch.where(labels == c.TOKEN_END_CONTENT, 2.0 * active, boundary)
    boundary = torch.where(labels == c.TOKEN_END_SEMANTIC, 2.0 * active, boundary)
    boundary = torch.where(labels == c.TOKEN_START_GLM, 2.0 * active, boundary)
    boundary = torch.where(labels == c.TOKEN_EOS, 2.0 * active, boundary)
    return main, boundary


def main() -> None:
    v2.trajectory_token_weights = trajectory_token_weights
    v2.OVERFIT_WEIGHTS = OVERFIT4_WEIGHTS
    dense.base.TrueSubsecondObjective = v2.RuntimeParityOverfit2Objective
    dense._distributed_dense_objective = v2.distributed_overfit2_objective
    dense.main()


if __name__ == "__main__":
    main()
