#!/usr/bin/env python3
"""Export and strictly evaluate the v8 length-only checkpoint."""

from __future__ import annotations

import json
from pathlib import Path

import web_demo.runtime_parity_streaming_v2.evaluate_checkpoint as v2
from web_demo.runtime_parity_streaming_v5.evaluate_checkpoint import (
    WarmedBiCodecTokenizer,
)
from web_demo.runtime_parity_streaming_v5.inference import (
    ParallelSemanticRuntimeGenerator,
)
from web_demo.runtime_parity_streaming_v7.model_loader import load_runtime_models


def evaluate(args):
    v2.load_runtime_models = load_runtime_models
    v2.NaturalRuntimeParityGenerator = ParallelSemanticRuntimeGenerator
    v2.BiCodecTokenizer = WarmedBiCodecTokenizer
    summary = v2.evaluate(args)
    summary["runtime_optimization"] = {
        "semantic_generation": "frozen_v6_content_plus_length_only_posterior_v1",
        "maximum_semantic_tokens_per_write": 24,
        "natural_length_support": [1, 24],
        "codec_kernel_prewarm": True,
        "frozen_v6_parallel_content": True,
        "frozen_v4_policy_text_and_frontend": True,
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
