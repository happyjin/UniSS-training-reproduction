#!/usr/bin/env python3
"""Isolated Megatron Stage A v4 discrete-identity entrypoint."""

from __future__ import annotations

from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v2.stage_a_causal_whisper_asr.training import (
    pretrain_stage_a_megatron as v2_entrypoint,
)
from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v4.stage_a_causal_whisper_asr.training.objective import (
    DIAGNOSTIC_NAMES,
    TERM_NAMES,
    StageAObjective,
    chunk_pair_for_progress,
    distributed_stage_a_objective,
)


def curriculum_group_multiplier(group, progress: float) -> float:
    """Adapt only Whisper top layers; preserve bottom acoustic geometry."""

    if not 0.0 <= progress <= 1.0:
        raise ValueError("invalid Stage A v4 LR progress")
    if group.get("uniss_stage_a_qwen") or group.get("uniss_stage_a_qwen_io"):
        return 1.0 if progress >= 0.05 else 0.0
    if group.get("uniss_stage_a_whisper_top"):
        return 1.0 if progress >= 0.10 else 0.0
    if group.get("uniss_stage_a_whisper_bottom") or group.get(
        "uniss_stage_a_whisper_conv"
    ):
        return 0.0
    return 1.0


def install_v4_overrides() -> None:
    """Patch the inherited native runtime only inside the v4 process."""

    native = v2_entrypoint.implementation
    native.StageAObjective = StageAObjective
    native.DIAGNOSTIC_NAMES = DIAGNOSTIC_NAMES
    native.TERM_NAMES = TERM_NAMES
    native.chunk_pair_for_progress = chunk_pair_for_progress
    native.distributed_stage_a_objective = distributed_stage_a_objective
    native.curriculum_group_multiplier = curriculum_group_multiplier
    native.METRIC_NAMES = (
        *TERM_NAMES,
        *DIAGNOSTIC_NAMES,
        *native.CURRICULUM_METRICS,
    )


def main() -> None:
    install_v4_overrides()
    v2_entrypoint.main()


if __name__ == "__main__":
    main()


__all__ = ["curriculum_group_multiplier", "install_v4_overrides", "main"]

