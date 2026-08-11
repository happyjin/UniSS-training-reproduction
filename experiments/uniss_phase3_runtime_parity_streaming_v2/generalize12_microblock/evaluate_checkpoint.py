#!/usr/bin/env python3
"""Strict real-PCM evaluation for the v12 microblock experiment."""

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
from web_demo.runtime_parity_streaming_v12.model_loader import load_runtime_models


def evaluate(args):
    v2.load_runtime_models = load_runtime_models
    v2.NaturalRuntimeParityGenerator = MicroblockRuntimeGenerator
    v2.BiCodecTokenizer = WarmedBiCodecTokenizer
    summary = v2.evaluate(args)
    summary["runtime_training"] = {
        "version": "generalize12_microblock_v1",
        "base_checkpoint": "dense_aligned_pilot15_iter_0002151",
        "trainable_scope": "causal_microblock_semantic_head_only",
        "microblock_size": 4,
        "within_microblock": "causal_teacher_forcing_train_free_running_runtime",
        "between_microblocks": "predicted_units_committed_to_main_qwen_kv",
        "semantic_classifier": "tied_frozen_phase3_embedding_rows",
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
