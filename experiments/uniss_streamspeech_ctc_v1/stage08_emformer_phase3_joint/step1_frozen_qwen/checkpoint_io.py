"""Selective inference loading from Stage08 Step1 Megatron checkpoints."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import torch
import torch.distributed.checkpoint as dcp


ITERATION_PATTERN = re.compile(r"^iter_(\d+)$")


def checkpoint_iteration(checkpoint_dir: str | Path) -> int:
    path = Path(checkpoint_dir)
    match = ITERATION_PATTERN.fullmatch(path.name)
    if match is None:
        raise ValueError(f"checkpoint directory must be named iter_XXXXXXXX: {path}")
    return int(match.group(1))


def inference_tensor_names(model: Any, unfreeze_encoder_layers: int = 4) -> list[str]:
    layers = model.endpoint.base.encoder.emformer_layers
    if not 1 <= unfreeze_encoder_layers <= len(layers):
        raise ValueError(
            f"unfreeze_encoder_layers must be in [1,{len(layers)}], got "
            f"{unfreeze_encoder_layers}"
        )
    first = len(layers) - unfreeze_encoder_layers
    prefixes = [
        *(f"endpoint.base.encoder.emformer_layers.{index}." for index in range(first, len(layers))),
        "endpoint.base.output_norm.",
        "residual.",
    ]
    names = [
        name
        for name in model.state_dict()
        if any(name.startswith(prefix) for prefix in prefixes) or name == "residual_scale"
    ]
    if not names:
        raise ValueError("no Stage08 inference tensors selected")
    return sorted(names)


def trainable_tensor_names(model: Any) -> list[str]:
    """Return exactly the parameters optimized by the Step1 model."""

    names = sorted(name for name, value in model.named_parameters() if value.requires_grad)
    if not names:
        raise ValueError("no trainable Stage08 tensors selected")
    return names


def load_step1_inference_into_model(
    model: Any,
    checkpoint_dir: str | Path,
    *,
    unfreeze_encoder_layers: int = 4,
) -> dict[str, Any]:
    """Load only the updated Emformer tail, output norm and B1 residual."""

    path = Path(checkpoint_dir)
    if not path.is_dir():
        raise FileNotFoundError(f"missing Megatron checkpoint directory: {path}")
    if not (path / ".metadata").is_file():
        raise FileNotFoundError(f"missing torch_dist metadata: {path / '.metadata'}")
    iteration = checkpoint_iteration(path)
    destination = model.state_dict()
    names = inference_tensor_names(model, unfreeze_encoder_layers)
    state = {
        f"joint.{name}": torch.empty_like(destination[name], device="cpu")
        for name in names
    }
    dcp.load(state, checkpoint_id=path)
    with torch.no_grad():
        for name in names:
            value = state[f"joint.{name}"]
            if not torch.isfinite(value).all():
                raise FloatingPointError(f"non-finite tensor joint.{name} in {path}")
            destination[name].copy_(
                value.to(device=destination[name].device, dtype=destination[name].dtype)
            )
    return {
        "schema_version": "uniss_streamspeech_stage08_step1_inference_v1",
        "checkpoint_dir": str(path.resolve()),
        "iteration": iteration,
        "loaded_tensors": len(names),
        "unfreeze_encoder_layers": unfreeze_encoder_layers,
        "residual_scale": float(model.residual_scale.detach().cpu()),
        "residual_weight_norm": float(model.residual.weight.detach().float().norm().cpu()),
    }


def load_step1_trainable_into_model(
    model: Any,
    checkpoint_dir: str | Path,
) -> dict[str, Any]:
    """Initialize a fresh optimizer run from every Step1 trainable tensor."""

    path = Path(checkpoint_dir)
    if not path.is_dir():
        raise FileNotFoundError(f"missing Megatron checkpoint directory: {path}")
    if not (path / ".metadata").is_file():
        raise FileNotFoundError(f"missing torch_dist metadata: {path / '.metadata'}")
    iteration = checkpoint_iteration(path)
    destination = model.state_dict()
    names = trainable_tensor_names(model)
    state = {
        f"joint.{name}": torch.empty_like(destination[name], device="cpu")
        for name in names
    }
    dcp.load(state, checkpoint_id=path)
    with torch.no_grad():
        for name in names:
            value = state[f"joint.{name}"]
            if not torch.isfinite(value).all():
                raise FloatingPointError(f"non-finite tensor joint.{name} in {path}")
            destination[name].copy_(
                value.to(device=destination[name].device, dtype=destination[name].dtype)
            )
    return {
        "schema_version": "uniss_streamspeech_stage08_step1_trainable_v1",
        "checkpoint_dir": str(path.resolve()),
        "iteration": iteration,
        "loaded_tensors": len(names),
        "trainable_parameters": sum(
            value.numel() for value in model.parameters() if value.requires_grad
        ),
        "residual_weight_norm": float(model.residual.weight.detach().float().norm().cpu()),
    }
