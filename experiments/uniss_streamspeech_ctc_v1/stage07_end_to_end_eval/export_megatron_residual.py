#!/usr/bin/env python3
"""Export a small auditable Stage06 residual artifact from a Megatron checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from checkpoint_io import BIAS_KEY, SCALE_KEY, WEIGHT_KEY, checkpoint_iteration, load_residual_tensors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input-dim", type=int, default=768)
    parser.add_argument("--output-dim", type=int, default=896)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite residual export: {args.output}")
    tensors = load_residual_tensors(
        args.checkpoint_dir,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": "uniss_streamspeech_stage06_megatron_residual_v1",
            "source_checkpoint": str(args.checkpoint_dir.resolve()),
            "iteration": checkpoint_iteration(args.checkpoint_dir),
            "residual": {
                "weight": tensors[WEIGHT_KEY],
                "bias": tensors[BIAS_KEY],
                "scale": tensors[SCALE_KEY],
            },
        },
        args.output,
    )
    print(args.output)


if __name__ == "__main__":
    main()
