#!/usr/bin/env python3
"""E2E continuation with the continue / content-end / repetition terms installed.

The established trainer imports its objective entry points and builds its metric
name contract at import time, so this entrypoint rebinds four module attributes
on it and then delegates to its ``main``.  The three weights arrive through the
environment, so the launcher keeps passing exactly the arguments the established
``run_e2e_megatron.sh`` builds.  Nothing in
``uniss_phase3_v4_e2e_simuls2st_pilot15_v1`` is modified.
"""

from __future__ import annotations

import functools
import json
import os

import experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.training.pretrain_e2e_megatron as trainer
from experiments.uniss_phase3_e2e_continue_end_v1.training.objective_ext import (
    distributed_with_continue_end,
    extended_objective_metric_names,
    flattened_with_continue_end,
)

ENV_CONTINUE_WEIGHT = "UNISS_E2E_CONTINUE_AFTER_FRAGMENT_WEIGHT"
ENV_CONTINUE_MARGIN = "UNISS_E2E_CONTINUE_AFTER_FRAGMENT_LOGIT_MARGIN"
ENV_CONTENT_END_WEIGHT = "UNISS_E2E_CONTENT_END_MARGIN_WEIGHT"
ENV_CONTENT_END_MARGIN = "UNISS_E2E_CONTENT_END_LOGIT_MARGIN"
ENV_REPETITION_WEIGHT = "UNISS_E2E_REPETITION_WEIGHT"
ENV_REPETITION_WINDOW = "UNISS_E2E_REPETITION_WINDOW"

DEFAULT_REPETITION_WINDOW = 8


def _number(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    value = float(raw) if raw else float(default)
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    return value


def resolve_extension() -> dict[str, float]:
    continue_weight = _number(ENV_CONTINUE_WEIGHT, 0.0)
    content_end_weight = _number(ENV_CONTENT_END_WEIGHT, 0.0)
    repetition = _number(ENV_REPETITION_WEIGHT, 0.0)
    window = int(_number(ENV_REPETITION_WINDOW, DEFAULT_REPETITION_WINDOW))
    if window < 1:
        raise ValueError(f"{ENV_REPETITION_WINDOW} must be at least 1")
    if continue_weight == 0.0 and content_end_weight == 0.0 and repetition == 0.0:
        raise ValueError(
            "this entrypoint exists to add the continue, content-end and "
            "repetition terms; set one of "
            f"{ENV_CONTINUE_WEIGHT} / {ENV_CONTENT_END_WEIGHT} / "
            f"{ENV_REPETITION_WEIGHT} above zero or run the established launcher"
        )
    continue_margin = _number(ENV_CONTINUE_MARGIN, 1.0)
    content_end_margin = _number(ENV_CONTENT_END_MARGIN, 2.0)
    if continue_weight > 0.0 and continue_margin <= 0.0:
        raise ValueError(
            f"{ENV_CONTINUE_MARGIN} must be positive: the probe measured the "
            "decision at a median -2.88, so a zero margin supervises nothing"
        )
    if content_end_weight > 0.0 and content_end_margin <= 0.0:
        raise ValueError(f"{ENV_CONTENT_END_MARGIN} must be positive")
    return {
        "continue_after_fragment": continue_weight,
        "continue_after_fragment_logit_margin": continue_margin,
        "content_end_margin": content_end_weight,
        "content_end_logit_margin": content_end_margin,
        "repetition_penalty": repetition,
        "repetition_window": float(window),
    }


def install(configuration: dict[str, float]) -> None:
    trainer.flattened_e2e_objective = functools.partial(
        flattened_with_continue_end,
        continue_after_fragment_logit_margin=configuration[
            "continue_after_fragment_logit_margin"
        ],
        content_end_logit_margin=configuration["content_end_logit_margin"],
        repetition_window=int(configuration["repetition_window"]),
    )
    trainer.distributed_e2e_objective = functools.partial(
        distributed_with_continue_end,
        continue_after_fragment=configuration["continue_after_fragment"],
        content_end_margin=configuration["content_end_margin"],
        repetition_penalty=configuration["repetition_penalty"],
    )
    trainer.OBJECTIVE_METRIC_NAMES = extended_objective_metric_names(
        trainer.OBJECTIVE_METRIC_NAMES
    )
    trainer.METRIC_NAMES = (
        *trainer.OBJECTIVE_METRIC_NAMES,
        *trainer.DIAGNOSTIC_NAMES,
    )


def main() -> None:
    configuration = resolve_extension()
    install(configuration)
    print(
        json.dumps(
            {"objective_extension": configuration, "terms_added": 3},
            sort_keys=True,
        ),
        flush=True,
    )
    trainer.main()


if __name__ == "__main__":
    main()
