#!/usr/bin/env python3
"""Strictly evaluate v8 weights through the lossless fused-commit v9 runtime."""

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
    summary["runtime_optimization"] = {
        "semantic_generation": "frozen_v8_natural_content_and_length_posterior",
        "semantic_commit": "semantic_codes_plus_natural_end_single_forward_v1",
        "maximum_semantic_tokens_per_write": 24,
        "natural_length_support": [1, 24],
        "codec_kernel_prewarm": True,
        "frozen_v8_model_weights": True,
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
