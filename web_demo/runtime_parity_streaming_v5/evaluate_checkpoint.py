#!/usr/bin/env python3
"""Export and strictly evaluate the v5 parallel-semantic checkpoint."""

from __future__ import annotations

import json

import torch

import web_demo.runtime_parity_streaming_v2.evaluate_checkpoint as v2
from uniss.speech_tokenizer.bicodec.bicodec_tokenizer import BiCodecTokenizer
from web_demo.runtime_parity_streaming_v5.inference import (
    ParallelSemanticRuntimeGenerator,
)
from web_demo.runtime_parity_streaming_v5.model_loader import load_runtime_models


class WarmedBiCodecTokenizer(BiCodecTokenizer):
    """Warm stateless decoder kernels before strict wall-clock measurement."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        with torch.inference_mode():
            speaker = torch.zeros((1, 32), dtype=torch.long, device=self.device)
            semantic = torch.zeros((1, 20), dtype=torch.long, device=self.device)
            self.detokenize(speaker, semantic)
            if torch.cuda.is_available() and torch.device(self.device).type == "cuda":
                torch.cuda.synchronize(torch.device(self.device))


def evaluate(args):
    v2.load_runtime_models = load_runtime_models
    v2.NaturalRuntimeParityGenerator = ParallelSemanticRuntimeGenerator
    v2.BiCodecTokenizer = WarmedBiCodecTokenizer
    summary = v2.evaluate(args)
    summary["runtime_optimization"] = {
        "semantic_generation": "natural_length_parallel_block_v1",
        "maximum_semantic_tokens_per_write": 24,
        "codec_kernel_prewarm": True,
        "forced_write": False,
        "forced_semantic_length": False,
    }
    output = args.output
    from pathlib import Path
    (Path(output) / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


if __name__ == "__main__":
    evaluate(v2.parse_args())

