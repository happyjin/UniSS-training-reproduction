#!/usr/bin/env python3
"""Evaluate generalize10 with natural v9 fused streaming inference."""

from __future__ import annotations

import json
from pathlib import Path

import web_demo.runtime_parity_streaming_v2.evaluate_checkpoint as v2
from web_demo.runtime_parity_streaming_v5.evaluate_checkpoint import (
    WarmedBiCodecTokenizer,
)
from web_demo.runtime_parity_streaming_v7.model_loader import load_runtime_models
from web_demo.runtime_parity_streaming_v9.inference import (
    FusedSemanticRuntimeGenerator,
)


def evaluate(args):
    v2.load_runtime_models = load_runtime_models
    v2.NaturalRuntimeParityGenerator = FusedSemanticRuntimeGenerator
    v2.BiCodecTokenizer = WarmedBiCodecTokenizer
    summary = v2.evaluate(args)
    summary["runtime_training"] = {
        "version": "generalize10_v1",
        "train_exact_runtime_sessions": 128,
        "held_out_exact_runtime_sessions": 32,
        "base_checkpoint": "dense_aligned_pilot15_iter_0002151",
        "trainable_scope": "natural_length_parallel_semantic_head_only",
        "forced_write": False,
        "oracle_semantic_length": False,
        "forced_semantic_length": False,
    }
    (Path(args.output) / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


if __name__ == "__main__":
    evaluate(v2.parse_args())
