#!/usr/bin/env python3
"""Run the established coverage rollout with the content-first state loader."""

from experiments.uniss_phase3_content_first_joint_s2st_v1.runtime.model_loader import (
    load_content_first_models,
)
from experiments.uniss_phasea_coverage_constrained_grpo_v3.training import (
    rollout as established_rollout,
)


def main() -> None:
    # The coverage/reward/trace implementation stays immutable.  Only its
    # historical Stage-A loader is replaced with the checkpoint-compatible
    # content-first loader for this isolated experiment.
    established_rollout._load_models = load_content_first_models
    established_rollout.main()


if __name__ == "__main__":
    main()

