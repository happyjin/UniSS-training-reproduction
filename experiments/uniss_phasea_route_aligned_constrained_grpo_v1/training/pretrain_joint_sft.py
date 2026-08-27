#!/usr/bin/env python3
"""Short ASR/MT/TTS route-aligned SFT warm-up using the existing E2E pool."""

from __future__ import annotations

import types

from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.training.task_samples import (
    LOSS_ASR,
    LOSS_BOUNDARY,
    LOSS_EOS,
    LOSS_MT,
    LOSS_SEMANTIC,
)
from experiments.uniss_stagea_quality_first_joint_grpo_v1.model.dual_lora import (
    DualLoRAController,
)
from experiments.uniss_stagea_quality_first_joint_grpo_v1.training import (
    pretrain_megatron as base,
)


_ORIGINAL_VALIDATE = base.validate_args


def validate_route_aligned_sft(args) -> None:
    if args.joint_mode != "sft" or not bool(args.joint_smoke):
        raise ValueError("route-aligned warm-up requires --joint-mode sft --joint-smoke")
    requested = int(args.train_iters)
    if not 2 <= requested <= 128:
        raise ValueError("route-aligned warm-up supports 2..128 updates")
    args.train_iters = 2
    try:
        _ORIGINAL_VALIDATE(args)
    finally:
        args.train_iters = requested


def attach_route_aligned_sft(model, *, allow_missing_teachers: bool = False) -> None:
    base.e2e.attach_e2e_forward(
        model, allow_missing_teachers=allow_missing_teachers
    )
    wrapped = model.forward
    controller = model.quality_grpo_lora
    if not isinstance(controller, DualLoRAController):
        raise TypeError("missing dual LoRA controller")

    def forward_with_route(self, *args, e2e_batch=None, **kwargs):
        if e2e_batch is None:
            raise ValueError("missing route-aligned SFT sidecar batch")
        kinds = e2e_batch["loss_kinds"]
        route = (
            (kinds == LOSS_ASR)
            | (kinds == LOSS_MT)
            | (kinds == LOSS_SEMANTIC)
            | (kinds == LOSS_BOUNDARY)
            | (kinds == LOSS_EOS)
        ).reshape(-1)
        controller.set_active_mask(route)
        e2e_batch["grpo_training"] = False
        e2e_batch["grpo_reference_ready"] = controller.reference_ready.to(
            device=e2e_batch["tokens"].device
        )
        e2e_batch["grpo_reference_anchor"] = controller.reference_anchor()
        e2e_batch["grpo_policy_update_rms"] = controller.policy_update_rms()
        with controller.use("policy"):
            return wrapped(*args, e2e_batch=e2e_batch, **kwargs)

    model.forward = types.MethodType(forward_with_route, model)


def main() -> None:
    base.validate_args = validate_route_aligned_sft
    base.attach_joint_forward = attach_route_aligned_sft
    base.main()


if __name__ == "__main__":
    main()

