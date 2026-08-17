#!/usr/bin/env python3
"""Megatron Stage A V9 entrypoint."""

from __future__ import annotations

from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v2.stage_a_causal_whisper_asr.training import (
    pretrain_stage_a_megatron as v2_entrypoint,
)
from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v7.stage_a_causal_whisper_asr.training import (
    pretrain_stage_a_megatron as v7_entrypoint,
)
from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v9.stage_a_causal_whisper_asr.training.objective import (
    DIAGNOSTIC_NAMES,
    TERM_NAMES,
    StageAObjective,
    chunk_pair_for_progress,
    distributed_stage_a_objective,
)


def install_v9_overrides() -> None:
    v7_entrypoint.install_v7_overrides()
    native = v2_entrypoint.implementation
    native.StageAObjective = StageAObjective
    native.DIAGNOSTIC_NAMES = DIAGNOSTIC_NAMES
    native.TERM_NAMES = TERM_NAMES
    native.chunk_pair_for_progress = chunk_pair_for_progress
    native.distributed_stage_a_objective = distributed_stage_a_objective
    native.METRIC_NAMES = (
        *TERM_NAMES,
        *DIAGNOSTIC_NAMES,
        *native.CURRICULUM_METRICS,
    )


def main() -> None:
    install_v9_overrides()
    v2_entrypoint.main()


if __name__ == "__main__":
    main()
