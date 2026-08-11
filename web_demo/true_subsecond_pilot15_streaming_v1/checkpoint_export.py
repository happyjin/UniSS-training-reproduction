#!/usr/bin/env python3
"""Export the pilot15 Megatron LoRA and streaming sidecars without base weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Mapping

import torch
import torch.distributed.checkpoint as dcp
from safetensors.torch import load_file, save_file
from torch.distributed.checkpoint import FileSystemReader


SCHEMA_VERSION = "uniss_true_subsecond_runtime_export_v2"
LORA_PREFIX = "true_subsecond_lora."
OBJECTIVE_PREFIX = "true_subsecond_objective."

QWEN_HIDDEN_SIZE = 896
QWEN_ATTENTION_HEADS = 14
QWEN_QUERY_GROUPS = 2
QWEN_HEAD_DIM = 64
QWEN_LORA_RANK = 32


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _metadata_tensor(metadata, key: str) -> torch.Tensor:
    value = metadata.state_dict_metadata[key]
    return torch.empty(tuple(value.size), dtype=value.properties.dtype)


def checkpoint_state(checkpoint: Path) -> dict[str, torch.Tensor]:
    metadata = FileSystemReader(str(checkpoint)).read_metadata()
    keys = sorted(
        key
        for key in metadata.state_dict_metadata
        if key.startswith((LORA_PREFIX, OBJECTIVE_PREFIX))
        and key != f"{OBJECTIVE_PREFIX}codebook.weight"
    )
    lora_keys = [key for key in keys if key.startswith(LORA_PREFIX)]
    objective_keys = [key for key in keys if key.startswith(OBJECTIVE_PREFIX)]
    if len(lora_keys) != 144:
        raise ValueError(f"expected 144 native LoRA tensors, found {len(lora_keys)}")
    if not objective_keys:
        raise ValueError("checkpoint contains no true-subsecond objective tensors")
    state = {key: _metadata_tensor(metadata, key) for key in keys}
    dcp.load(state_dict=state, checkpoint_id=str(checkpoint))
    return {key: value.detach().cpu().contiguous() for key, value in state.items()}


def _native_pair(state: Mapping[str, torch.Tensor], layer: int, module: str):
    prefix = (
        f"{LORA_PREFIX}branches.decoder__layers__{layer}__"
        f"{module.replace('.', '__')}"
    )
    return state[f"{prefix}.lora_a"], state[f"{prefix}.lora_b"]


def split_megatron_gqa_lora_b(
    fused: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert Megatron's query-group-interleaved QKV rows to HF Q/K/V rows.

    Megatron stores every query group as ``q...q, k, v``.  Qwen2-0.5B has
    fourteen query heads and two KV groups, so the native 1152 rows are laid
    out as two 576-row groups rather than one contiguous Q block followed by K
    and V.  Hugging Face exposes three independent projections and therefore
    needs the rows gathered by semantic branch.
    """

    expected = (
        QWEN_QUERY_GROUPS
        * (QWEN_ATTENTION_HEADS // QWEN_QUERY_GROUPS + 2)
        * QWEN_HEAD_DIM,
        QWEN_LORA_RANK,
    )
    if tuple(fused.shape) != expected:
        raise ValueError(f"unexpected fused GQA LoRA-B shape {tuple(fused.shape)}")
    queries_per_group = QWEN_ATTENTION_HEADS // QWEN_QUERY_GROUPS
    group_width = (queries_per_group + 2) * QWEN_HEAD_DIM
    grouped = fused.reshape(QWEN_QUERY_GROUPS, group_width, QWEN_LORA_RANK)
    query_width = queries_per_group * QWEN_HEAD_DIM
    query = grouped[:, :query_width].reshape(
        QWEN_ATTENTION_HEADS * QWEN_HEAD_DIM, QWEN_LORA_RANK
    )
    key = grouped[:, query_width : query_width + QWEN_HEAD_DIM].reshape(
        QWEN_QUERY_GROUPS * QWEN_HEAD_DIM, QWEN_LORA_RANK
    )
    value = grouped[:, query_width + QWEN_HEAD_DIM :].reshape(
        QWEN_QUERY_GROUPS * QWEN_HEAD_DIM, QWEN_LORA_RANK
    )
    return tuple(item.contiguous() for item in (query, key, value))


def interleave_hf_gqa_outputs(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
) -> torch.Tensor:
    """Rebuild Megatron's fused output layout from HF projection outputs."""

    batch_shape = query.shape[:-1]
    if query.shape[-1] != QWEN_ATTENTION_HEADS * QWEN_HEAD_DIM:
        raise ValueError("unexpected HF query width")
    if key.shape != (*batch_shape, QWEN_QUERY_GROUPS * QWEN_HEAD_DIM):
        raise ValueError("unexpected HF key width")
    if value.shape != key.shape:
        raise ValueError("unexpected HF value width")
    query = query.reshape(*batch_shape, QWEN_QUERY_GROUPS, -1)
    key = key.reshape(*batch_shape, QWEN_QUERY_GROUPS, QWEN_HEAD_DIM)
    value = value.reshape(*batch_shape, QWEN_QUERY_GROUPS, QWEN_HEAD_DIM)
    return torch.cat((query, key, value), dim=-1).reshape(*batch_shape, -1)


def map_native_lora_to_hf(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Split Megatron fused QKV/SwiGLU branches into exact HF Qwen branches."""

    output: dict[str, torch.Tensor] = {}

    def add(layer: int, target: str, a: torch.Tensor, b: torch.Tensor) -> None:
        prefix = f"model.layers.{layer}.{target}"
        output[f"{prefix}.lora_A.weight"] = a.clone().contiguous()
        output[f"{prefix}.lora_B.weight"] = b.clone().contiguous()

    for layer in range(24):
        qkv_a, qkv_b = _native_pair(
            state, layer, "self_attention.linear_qkv"
        )
        q_b, k_b, v_b = split_megatron_gqa_lora_b(qkv_b)
        add(layer, "self_attn.q_proj", qkv_a, q_b)
        add(layer, "self_attn.k_proj", qkv_a, k_b)
        add(layer, "self_attn.v_proj", qkv_a, v_b)
        proj_a, proj_b = _native_pair(
            state, layer, "self_attention.linear_proj"
        )
        add(layer, "self_attn.o_proj", proj_a, proj_b)

    for layer in range(12, 24):
        fc1_a, fc1_b = _native_pair(state, layer, "mlp.linear_fc1")
        if tuple(fc1_b.shape) != (9728, 32):
            raise ValueError(f"unexpected layer {layer} fused FC1 shape {tuple(fc1_b.shape)}")
        gate_b, up_b = torch.split(fc1_b, (4864, 4864), dim=0)
        add(layer, "mlp.gate_proj", fc1_a, gate_b)
        add(layer, "mlp.up_proj", fc1_a, up_b)
        fc2_a, fc2_b = _native_pair(state, layer, "mlp.linear_fc2")
        add(layer, "mlp.down_proj", fc2_a, fc2_b)

    if len(output) != 264:
        raise AssertionError(f"expected 264 mapped HF tensors, found {len(output)}")
    return output


def objective_state(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        key.removeprefix(OBJECTIVE_PREFIX): value.clone().contiguous()
        for key, value in state.items()
        if key.startswith(OBJECTIVE_PREFIX)
    }


def verify_fused_mapping(
    native: Mapping[str, torch.Tensor], mapped: Mapping[str, torch.Tensor]
) -> None:
    generator = torch.Generator().manual_seed(20260811)
    value = torch.randn(3, 896, generator=generator, dtype=torch.float32)
    scale = 2.0
    for layer in (0, 12, 23):
        a, b = _native_pair(native, layer, "self_attention.linear_qkv")
        expected = (value @ a.float().t()) @ b.float().t() * scale
        pieces = []
        for target in ("q_proj", "k_proj", "v_proj"):
            prefix = f"model.layers.{layer}.self_attn.{target}"
            branch = (value @ mapped[f"{prefix}.lora_A.weight"].float().t())
            branch = branch @ mapped[f"{prefix}.lora_B.weight"].float().t() * scale
            pieces.append(branch)
        actual = interleave_hf_gqa_outputs(*pieces)
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    for layer in (12, 23):
        a, b = _native_pair(native, layer, "mlp.linear_fc1")
        expected = (value @ a.float().t()) @ b.float().t() * scale
        pieces = []
        for target in ("gate_proj", "up_proj"):
            prefix = f"model.layers.{layer}.mlp.{target}"
            branch = (value @ mapped[f"{prefix}.lora_A.weight"].float().t())
            branch = branch @ mapped[f"{prefix}.lora_B.weight"].float().t() * scale
            pieces.append(branch)
        torch.testing.assert_close(expected, torch.cat(pieces, dim=-1), rtol=0, atol=0)


def _save_verified(path: Path, state: Mapping[str, torch.Tensor]) -> None:
    save_file(dict(state), path)
    reloaded = load_file(path)
    if set(reloaded) != set(state):
        raise AssertionError(f"safetensors key mismatch in {path}")
    for key, value in state.items():
        if not torch.equal(value, reloaded[key]):
            raise AssertionError(f"safetensors value mismatch for {key}")


def export_runtime(checkpoint: Path, base_model: Path, output: Path) -> dict[str, object]:
    required = (checkpoint / ".metadata", base_model / "config.json")
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing export inputs: {missing}")
    manifest_path = output / "manifest.json"
    if output.exists() and any(output.iterdir()):
        if not manifest_path.is_file():
            raise FileExistsError(f"refusing to overwrite non-empty export: {output}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for name in ("adapter_model.safetensors", "objective_model.safetensors"):
            path = output / name
            if not path.is_file() or sha256(path) != manifest["files"][name]["sha256"]:
                raise ValueError(f"existing export failed checksum validation: {path}")
        return manifest

    output.mkdir(parents=True, exist_ok=False)
    native = checkpoint_state(checkpoint)
    adapter = map_native_lora_to_hf(native)
    objective = objective_state(native)
    verify_fused_mapping(native, adapter)
    adapter_path = output / "adapter_model.safetensors"
    objective_path = output / "objective_model.safetensors"
    _save_verified(adapter_path, adapter)
    _save_verified(objective_path, objective)
    match = re.fullmatch(r"iter_(\d+)", checkpoint.name)
    if match is None:
        raise ValueError(f"checkpoint directory does not encode an iteration: {checkpoint}")
    selected_iteration = int(match.group(1))
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "source_checkpoint": str(checkpoint.resolve()),
        "base_model": str(base_model.resolve()),
        "selected_iteration": selected_iteration,
        "rank": 32,
        "alpha": 64.0,
        "scale": 2.0,
        "qkv_layout": "megatron_query_group_interleaved_to_hf_qkv",
        "native_lora_tensor_count": 144,
        "hf_lora_tensor_count": len(adapter),
        "objective_tensor_count": len(objective),
        "frontend_supervision": {
            "chunk_ms": 160,
            "right_context_ms": 80,
            "input": "WhisperVQ codebook IDs",
        },
        "files": {},
        "checkpoint_metadata_sha256": sha256(checkpoint / ".metadata"),
    }
    for path in (adapter_path, objective_path):
        manifest["files"][path.name] = {
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--base-model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        json.dumps(
            export_runtime(args.checkpoint, args.base_model, args.output),
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
