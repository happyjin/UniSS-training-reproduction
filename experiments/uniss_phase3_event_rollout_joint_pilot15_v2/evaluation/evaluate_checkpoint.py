#!/usr/bin/env python3
"""Strict real-PCM evaluation for repaired fixed15 event-rollout checkpoints."""

from __future__ import annotations

import json
from pathlib import Path

import web_demo.runtime_parity_streaming_v2.evaluate_checkpoint as runtime_eval
from experiments.uniss_phase3_event_rollout_joint_pilot15_v1.evaluation.model_loader import (
    load_runtime_models,
)
from experiments.uniss_phase3_runtime_parity_streaming_v2.generalize15_action_eos_calibration.inference import (
    CalibratedMicroblockRuntimeGenerator,
)
from web_demo.runtime_parity_streaming_v5.evaluate_checkpoint import (
    WarmedBiCodecTokenizer,
)


def evaluate(args):
    """Run the shared exact runtime while recording the repaired v2 provenance."""

    runtime_eval.load_runtime_models = load_runtime_models
    runtime_eval.NaturalRuntimeParityGenerator = CalibratedMicroblockRuntimeGenerator
    runtime_eval.BiCodecTokenizer = WarmedBiCodecTokenizer
    summary = runtime_eval.evaluate(args)
    summary["schema_version"] = "uniss_event_rollout_fixed15_pcm_evaluation_v2"
    summary["runtime_training"] = {
        "version": "uniss_phase3_event_rollout_joint_pilot15_v2",
        "repair": "trainable_causal_frontend",
        "initial_checkpoint": "phase3_v4_iter_0009075",
        "data_scope": "unist_train_shards_00000_00014_only",
        "timing_classification": "pseudo_oracle_alignment",
        "natural_exact_timing": False,
        "persistent_kv": True,
        "event_rollout_recovery": True,
        "learned_wait_write": True,
        "learned_text": True,
        "learned_semantic_microblocks": True,
        "learned_continuation_eos": True,
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
    evaluate(runtime_eval.parse_args())

