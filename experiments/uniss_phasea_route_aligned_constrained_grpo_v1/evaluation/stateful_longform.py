#!/usr/bin/env python3
"""Run the existing Runtime-v2 evaluator with adapter enabled for ASR too."""

from experiments.uniss_phasea_stateful_longepisode_rl_v1.evaluation import (
    stateful_longform as base,
)


def main() -> None:
    base._route_for_prompt = lambda _prompt: True
    base.main()


if __name__ == "__main__":
    main()

