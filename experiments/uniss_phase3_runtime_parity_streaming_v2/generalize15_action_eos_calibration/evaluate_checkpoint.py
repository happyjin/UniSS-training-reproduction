#!/usr/bin/env python3
"""Strict real-PCM evaluation for Generalize15 checkpoints."""

from __future__ import annotations

import json
from pathlib import Path

import web_demo.runtime_parity_streaming_v2.evaluate_checkpoint as v2
from experiments.uniss_phase3_runtime_parity_streaming_v2.generalize15_action_eos_calibration.inference import (
    CalibratedMicroblockRuntimeGenerator,
)
from experiments.uniss_phase3_runtime_parity_streaming_v2.generalize15_action_eos_calibration.model_loader import (
    load_runtime_models,
)
from web_demo.runtime_parity_streaming_v5.evaluate_checkpoint import (
    WarmedBiCodecTokenizer,
)


def evaluate(args):
    v2.load_runtime_models = load_runtime_models
    v2.NaturalRuntimeParityGenerator = CalibratedMicroblockRuntimeGenerator
    v2.BiCodecTokenizer = WarmedBiCodecTokenizer
    summary = v2.evaluate(args)
    summary["runtime_training"] = {
        "version": "generalize15_action_eos_calibration_v1",
        "initial_checkpoint": "generalize14_dagger_prefix_canary_iter_0000050",
        "trainable_scope": "action_and_continuation_heads_only",
        "content_parameters_frozen": True,
        "wait_false_positive_weight": 2.0,
        "bounded_model_prefix_rollin": 0.10,
        "learned_natural_continuation_head": True,
        "forced_write": False,
        "forced_eos": False,
        "oracle_semantic_length": False,
        "forced_semantic_length": False,
        "safety_ceiling_is_failure": True,
    }
    (Path(args.output) / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


if __name__ == "__main__":
    evaluate(v2.parse_args())

