#!/usr/bin/env python3
"""E2E continuation with the speak-decision and repetition terms installed.

The established trainer imports its objective entry points and builds its metric
name contract at import time, so this entrypoint rebinds four module attributes
on it and then delegates to its ``main``:

* ``flattened_e2e_objective`` / ``distributed_e2e_objective`` -> the wrappers in
  ``objective_ext``, which call the established functions for the twenty
  established terms and add three of their own;
* ``OBJECTIVE_METRIC_NAMES`` / ``METRIC_NAMES`` -> the same contracts with the
  new names appended, because the trainer asserts metric order twice
  (``pretrain_e2e_megatron.py:1944`` and ``:1959``) and ``loss_func`` reads the
  module global at call time.

The two new weights arrive through the environment rather than new CLI flags, so
the launcher keeps passing exactly the arguments the established
``run_e2e_megatron.sh`` builds.  Nothing in
``uniss_phase3_v4_e2e_simuls2st_pilot15_v1`` is modified.
"""

from __future__ import annotations

import functools
import json
import os

import experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.training.pretrain_e2e_megatron as trainer
from experiments.uniss_phase3_e2e_speak_decision_v1.training.objective_ext import (
    DEFAULT_REPETITION_WINDOW,
    distributed_with_speak_decision,
    extended_objective_metric_names,
    flattened_with_speak_decision,
)


ENV_SPEAK_WEIGHT = "UNISS_E2E_SPEAK_DECISION_WEIGHT"
ENV_SPEAK_MARGIN = "UNISS_E2E_SPEAK_DECISION_LOGIT_MARGIN"
ENV_REPETITION_WEIGHT = "UNISS_E2E_REPETITION_WEIGHT"
ENV_REPETITION_WINDOW = "UNISS_E2E_REPETITION_WINDOW"


def _number(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    value = float(raw) if raw else float(default)
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    return value


def resolve_extension() -> dict[str, float]:
    speak = _number(ENV_SPEAK_WEIGHT, 0.0)
    repetition = _number(ENV_REPETITION_WEIGHT, 0.0)
    window = int(_number(ENV_REPETITION_WINDOW, DEFAULT_REPETITION_WINDOW))
    if window < 1:
        raise ValueError(f"{ENV_REPETITION_WINDOW} must be at least 1")
    if speak == 0.0 and repetition == 0.0:
        raise ValueError(
            "this entrypoint exists to add the speak decision and repetition "
            f"terms; set {ENV_SPEAK_WEIGHT} or {ENV_REPETITION_WEIGHT} above zero "
            "or run the established launcher instead"
        )
    return {
        "speak_decision": speak,
        "speak_decision_logit_margin": _number(ENV_SPEAK_MARGIN, 1.0),
        "repetition_penalty": repetition,
        "repetition_window": float(window),
    }


def install(configuration: dict[str, float]) -> None:
    trainer.flattened_e2e_objective = functools.partial(
        flattened_with_speak_decision,
        speak_decision_logit_margin=configuration["speak_decision_logit_margin"],
        repetition_window=int(configuration["repetition_window"]),
    )
    trainer.distributed_e2e_objective = functools.partial(
        distributed_with_speak_decision,
        speak_decision=configuration["speak_decision"],
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
