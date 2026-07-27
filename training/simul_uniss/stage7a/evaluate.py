"""Evaluate one Stage7A action-head checkpoint on fixed action samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from evaluation.io_utils import write_json
from training.simul_uniss.distributed import DistributedContext
from training.simul_uniss.stage7a.train import (
    CHECKPOINT_SCHEMA,
    dtype_from_name,
    evaluate_policy,
    load_base_and_head,
)


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    distributed = DistributedContext.initialize(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA:
        raise ValueError(
            f"unexpected checkpoint schema: {checkpoint.get('schema_version')}"
        )
    model_path = Path(args.model or checkpoint["base_model"])
    model, head = load_base_and_head(
        model_path,
        device=distributed.device,
        dtype=dtype_from_name(args.dtype),
        attention_implementation=args.attention_implementation,
    )
    head.load_state_dict(checkpoint["action_head"])
    head.to(distributed.device).eval()
    metrics = evaluate_policy(
        model,
        head,
        samples_path=Path(args.samples),
        distributed=distributed,
        max_sequence_length=args.max_sequence_length,
        max_batch_tokens=args.max_batch_tokens,
        max_batch_size=args.max_batch_size,
        limit_records=args.limit_records,
        chunk_ms=args.chunk_ms,
    )
    result = {
        "schema_version": "simul_uniss_stage7a_action_evaluation_v1",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "model": str(model_path.resolve()),
        "samples": str(Path(args.samples).resolve()),
        "world_size": distributed.world_size,
        "metrics": metrics,
    }
    if distributed.is_main:
        write_json(Path(args.output), result)
        print(json.dumps(result, sort_keys=True))
    distributed.barrier()
    distributed.close()
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument(
        "--attention-implementation",
        choices=("eager", "sdpa", "flash_attention_2"),
        default="flash_attention_2",
    )
    parser.add_argument("--max-sequence-length", type=int, default=18_000)
    parser.add_argument("--max-batch-tokens", type=int, default=131_072)
    parser.add_argument("--max-batch-size", type=int, default=256)
    parser.add_argument("--limit-records", type=int, default=0)
    parser.add_argument("--chunk-ms", type=float, default=640.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    evaluate(parse_args(argv))


if __name__ == "__main__":
    main()
