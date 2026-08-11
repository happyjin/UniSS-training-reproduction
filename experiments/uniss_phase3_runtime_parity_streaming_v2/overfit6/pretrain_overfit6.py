#!/usr/bin/env python3
"""Megatron overfit v6: frozen v4 plus an untied parallel classifier."""

from __future__ import annotations

import torch
from torch import nn

import experiments.uniss_phase3_dense_aligned_streaming_pilot15_v1.training.pretrain_dense_aligned_megatron as dense
import experiments.uniss_phase3_runtime_parity_streaming_v2.overfit2.pretrain_overfit2 as v2
import experiments.uniss_phase3_runtime_parity_streaming_v2.overfit5.pretrain_overfit5 as v5
from experiments.uniss_phase3_runtime_parity_streaming_v2.overfit4.pretrain_overfit4 import (
    trajectory_token_weights,
)
from experiments.uniss_phase3_runtime_parity_streaming_v2.overfit5.semantic_block import (
    END_CLASS,
    ParallelSemanticBlockHead,
)


class UntiedParallelSemanticBlockHead(ParallelSemanticBlockHead):
    """Use a trainable classifier instead of fixed Phase3 embedding rows."""

    def __init__(self, hidden_size: int, *, maximum_semantic_tokens: int = 24) -> None:
        super().__init__(
            hidden_size,
            maximum_semantic_tokens=maximum_semantic_tokens,
            end_loss_weight=1.0,
        )
        self.output_projection = nn.Linear(hidden_size, END_CLASS + 1)
        nn.init.normal_(self.output_projection.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.output_projection.bias)
        for parameter in self.output_projection.parameters():
            parameter.uniss_lr_new_heads = True

    def forward(self, context, word_embedding_weight):
        del word_embedding_weight
        if context.ndim != 2 or context.shape[-1] != self.hidden_size:
            raise ValueError("semantic block context must be [blocks,hidden]")
        slots = self.slot_embeddings.weight.to(context.dtype).unsqueeze(0)
        hidden = self.context_projection(context).unsqueeze(1) + slots
        hidden = hidden + self.hidden_projection(hidden)
        hidden = self.output_norm(hidden)
        return self.output_projection(hidden)


class RuntimeParityOverfit6Objective(v5.RuntimeParityOverfit5Objective):
    def __init__(self, hidden_size: int, codebook_weight: torch.Tensor, **kwargs) -> None:
        super().__init__(hidden_size, codebook_weight, **kwargs)
        self.semantic_block_head = UntiedParallelSemanticBlockHead(
            hidden_size, maximum_semantic_tokens=24
        )


def freeze_v4_except_parallel_head() -> None:
    original = dense.base.augment_native_gpt

    def augment_and_freeze(model, args):
        summary = original(model, args)
        trainable = 0
        for name, parameter in model.named_parameters():
            keep = "true_subsecond_objective.semantic_block_head." in name
            parameter.requires_grad_(keep)
            if keep:
                parameter.uniss_lr_new_heads = True
                trainable += parameter.numel()
        if trainable <= 0:
            raise RuntimeError("overfit6 freeze policy found no semantic-head parameters")
        model._overfit6_trainable_parameters = trainable
        return summary

    dense.base.augment_native_gpt = augment_and_freeze


def main() -> None:
    v2.trajectory_token_weights = trajectory_token_weights
    dense.base.TrueSubsecondObjective = RuntimeParityOverfit6Objective
    dense._distributed_dense_objective = v5.distributed_overfit5_objective
    dense._dense_output_processor = v5.dense_output_processor
    dense.METRIC_NAMES = v5.V5_METRIC_NAMES
    dense.base.METRIC_NAMES = v5.V5_METRIC_NAMES
    freeze_v4_except_parallel_head()
    dense.main()


if __name__ == "__main__":
    main()

