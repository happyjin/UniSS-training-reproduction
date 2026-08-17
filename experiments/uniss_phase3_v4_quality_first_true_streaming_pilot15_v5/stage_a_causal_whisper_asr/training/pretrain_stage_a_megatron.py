#!/usr/bin/env python3
"""Isolated Megatron Stage A v5 causal-code adapter entrypoint."""

from __future__ import annotations

from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v2.stage_a_causal_whisper_asr.training import (
    pretrain_stage_a_megatron as v2_entrypoint,
)
from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v5.stage_a_causal_whisper_asr.training.objective import (
    DIAGNOSTIC_NAMES,
    TERM_NAMES,
    StageAObjective,
    chunk_pair_for_progress,
    distributed_stage_a_objective,
)


def curriculum_group_multiplier(group, progress: float) -> float:
    """Train new heads/adapter while preserving all Whisper parameters."""

    if not 0.0 <= progress <= 1.0:
        raise ValueError("invalid Stage A v5 LR progress")
    if group.get("uniss_stage_a_qwen") or group.get("uniss_stage_a_qwen_io"):
        return 1.0 if progress >= 0.05 else 0.0
    if (
        group.get("uniss_stage_a_whisper_top")
        or group.get("uniss_stage_a_whisper_bottom")
        or group.get("uniss_stage_a_whisper_conv")
    ):
        return 0.0
    return 1.0


def install_v5_overrides() -> None:
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
    install_v5_overrides()
    v2_entrypoint.main()


if __name__ == "__main__":
    main()


__all__ = ["curriculum_group_multiplier", "install_v5_overrides", "main"]

