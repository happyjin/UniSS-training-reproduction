#!/usr/bin/env python3
"""Megatron Stage A V9 with bridge freeze and stronger blank margin."""

from __future__ import annotations

from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v2.stage_a_causal_whisper_asr.training import (
    pretrain_stage_a_megatron as v2_entrypoint,
)
from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v7.stage_a_causal_whisper_asr.training import (
    pretrain_stage_a_megatron as v7_entrypoint,
)
from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v9.stage_a_causal_whisper_asr.training import (
    objective as v9_objective,
)


V9_DECISION_MARGIN_SCALE = v9_objective.DECISION_MARGIN_SCALE


def curriculum_group_multiplier(group, progress: float) -> float:
    """Freeze bridge/adapter exactly after the 127-update optimizer horizon."""

    value = v7_entrypoint.curriculum_group_multiplier(group, progress)
    if group.get("uniss_stage_a_bridge") and progress >= 1.0:
        return 0.0
    return value


def install_v9_overrides() -> None:
    v7_entrypoint.install_v7_overrides()
    native = v2_entrypoint.implementation
    native.StageAObjective = v9_objective.StageAObjective
    native.DIAGNOSTIC_NAMES = v9_objective.DIAGNOSTIC_NAMES
    native.TERM_NAMES = v9_objective.TERM_NAMES
    native.chunk_pair_for_progress = v9_objective.chunk_pair_for_progress
    native.distributed_stage_a_objective = (
        v9_objective.distributed_stage_a_objective
    )
    native.curriculum_group_multiplier = curriculum_group_multiplier
    native.METRIC_NAMES = (
        *v9_objective.TERM_NAMES,
        *v9_objective.DIAGNOSTIC_NAMES,
        *native.CURRICULUM_METRICS,
    )


def main() -> None:
    install_v9_overrides()
    v2_entrypoint.main()


if __name__ == "__main__":
    main()


__all__ = [
    "curriculum_group_multiplier",
    "install_v9_overrides",
    "main",
    "V9_DECISION_MARGIN_SCALE",
]
