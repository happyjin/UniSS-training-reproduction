#!/usr/bin/env python3
"""Export only trained LoRA tensors from a Megatron torch-dist checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from safetensors.torch import load_file, save_file
from torch.distributed.checkpoint import FileSystemReader


SCHEMA_VERSION = "uniss_phase3_prefix_streaming_v3_lora_adapter_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_lora_state(checkpoint: Path) -> dict[str, torch.Tensor]:
    metadata = FileSystemReader(str(checkpoint)).read_metadata()
    keys = sorted(
        key
        for key in metadata.state_dict_metadata
        if key.startswith("qwen.") and (".lora_A." in key or ".lora_B." in key)
    )
    if len(keys) != 96:
        raise ValueError(f"expected 96 q/v LoRA tensors, found {len(keys)}")
    state = {
        key: torch.empty(
            tuple(metadata.state_dict_metadata[key].size),
            dtype=metadata.state_dict_metadata[key].properties.dtype,
        )
        for key in keys
    }
    dcp.load(state_dict=state, checkpoint_id=str(checkpoint))
    return {
        key.removeprefix("qwen."): value.detach().cpu().contiguous()
        for key, value in state.items()
    }


def export_adapter(
    checkpoint: Path,
    base_model: Path,
    output: Path,
    selection_manifest: Path,
) -> dict[str, object]:
    required = [checkpoint / ".metadata", base_model / "config.json", selection_manifest]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing export input: {missing}")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty adapter output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    state = checkpoint_lora_state(checkpoint)
    weights = output / "adapter_model.safetensors"
    save_file(state, weights)
    reloaded = load_file(weights)
    if set(reloaded) != set(state):
        raise AssertionError("safetensors key verification failed")
    for key in state:
        if not torch.equal(state[key], reloaded[key]):
            raise AssertionError(f"safetensors value verification failed: {key}")
    selection = json.loads(selection_manifest.read_text(encoding="utf-8"))
    config = {
        "schema_version": SCHEMA_VERSION,
        "base_model": str(base_model.resolve()),
        "source_checkpoint": str(checkpoint.resolve()),
        "selected_iteration": int(selection["selected_iteration"]),
        "rank": 16,
        "alpha": 32.0,
        "scaling": 2.0,
        "dropout": 0.05,
        "target_modules": ["q_proj", "v_proj"],
        "tensor_count": len(state),
        "parameter_count": sum(value.numel() for value in state.values()),
        "dtype": sorted({str(value.dtype) for value in state.values()}),
        "weights": {
            "file": weights.name,
            "bytes": weights.stat().st_size,
            "sha256": sha256(weights),
        },
        "checkpoint_metadata_sha256": sha256(checkpoint / ".metadata"),
        "selection_manifest": selection,
    }
    (output / "adapter_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = export_adapter(
        args.checkpoint, args.base_model, args.output, args.selection_manifest
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

