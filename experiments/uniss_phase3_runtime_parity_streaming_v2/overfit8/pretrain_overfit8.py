#!/usr/bin/env python3
"""Megatron overfit v8: preserve v6 content and learn only natural length."""

from __future__ import annotations

import experiments.uniss_phase3_dense_aligned_streaming_pilot15_v1.training.pretrain_dense_aligned_megatron as dense
import experiments.uniss_phase3_runtime_parity_streaming_v2.overfit2.pretrain_overfit2 as v2
import experiments.uniss_phase3_runtime_parity_streaming_v2.overfit7.pretrain_overfit7 as v7
from experiments.uniss_phase3_runtime_parity_streaming_v2.overfit4.pretrain_overfit4 import (
    trajectory_token_weights,
)


def is_v8_trainable_parameter(name: str) -> bool:
    return "true_subsecond_objective.semantic_block_head.length_head." in name


def freeze_except_natural_length() -> None:
    original = dense.base.augment_native_gpt

    def augment_and_freeze(model, args):
        summary = original(model, args)
        trainable = 0
        for name, parameter in model.named_parameters():
            keep = is_v8_trainable_parameter(name)
            parameter.requires_grad_(keep)
            if keep:
                parameter.uniss_lr_new_heads = True
                trainable += parameter.numel()
        if trainable <= 0:
            raise RuntimeError("overfit8 freeze policy found no length-head parameters")
        model._overfit8_trainable_parameters = trainable
        return summary

    dense.base.augment_native_gpt = augment_and_freeze


def main() -> None:
    v2.trajectory_token_weights = trajectory_token_weights
    dense.base.TrueSubsecondObjective = v7.RuntimeParityOverfit7Objective
    dense._distributed_dense_objective = v7.distributed_overfit7_objective
    dense._dense_output_processor = v7.dense_output_processor
    dense.METRIC_NAMES = v7.V7_METRIC_NAMES
    dense.base.METRIC_NAMES = v7.V7_METRIC_NAMES
    freeze_except_natural_length()
    dense.main()


if __name__ == "__main__":
    main()
