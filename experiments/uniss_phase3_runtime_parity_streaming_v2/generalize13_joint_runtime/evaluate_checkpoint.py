#!/usr/bin/env python3
"""Strict real-PCM evaluation for the v13 joint runtime experiment."""

from __future__ import annotations

import json
from pathlib import Path

import web_demo.runtime_parity_streaming_v2.evaluate_checkpoint as v2
from web_demo.runtime_parity_streaming_v5.evaluate_checkpoint import (
    WarmedBiCodecTokenizer,
)
from web_demo.runtime_parity_streaming_v12.inference import (
    MicroblockRuntimeGenerator,
)
from web_demo.runtime_parity_streaming_v13.model_loader import load_runtime_models


def evaluate(args):
    v2.load_runtime_models = load_runtime_models
    v2.NaturalRuntimeParityGenerator = MicroblockRuntimeGenerator
    v2.BiCodecTokenizer = WarmedBiCodecTokenizer
    summary = v2.evaluate(args)
    summary["runtime_training"] = {
        "version": "generalize13_joint_runtime_v1",
        "initial_checkpoint": "generalize12_microblock_canary_iter_0000200",
        "trainable_scope": "qwen_lora_action_support_safe_commit_microblock",
        "frozen_scope": "phase3_base_embeddings_output_frontend",
        "forced_write": False,
        "oracle_semantic_length": False,
        "forced_semantic_length": False,
        "forced_microblock_count": False,
        "safety_ceiling_is_failure": True,
    }
    (Path(args.output) / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


if __name__ == "__main__":
    evaluate(v2.parse_args())
