"""Safe residual-only loading from Stage06 Megatron torch_dist checkpoints."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import torch
import torch.distributed.checkpoint as dcp


WEIGHT_KEY = "bridge.residual.weight"
BIAS_KEY = "bridge.residual.bias"
SCALE_KEY = "bridge.residual_scale"
ITERATION_PATTERN = re.compile(r"^iter_(\d+)$")


def checkpoint_iteration(checkpoint_dir: str | Path) -> int:
    path = Path(checkpoint_dir)
    match = ITERATION_PATTERN.fullmatch(path.name)
    if match is None:
        raise ValueError(f"checkpoint directory must be named iter_XXXXXXXX: {path}")
    return int(match.group(1))


def _validate_checkpoint_dir(checkpoint_dir: str | Path) -> Path:
    path = Path(checkpoint_dir)
    if not path.is_dir():
        raise FileNotFoundError(f"missing Megatron checkpoint directory: {path}")
    if not (path / ".metadata").is_file():
        raise FileNotFoundError(f"missing torch_dist metadata: {path / '.metadata'}")
    checkpoint_iteration(path)
    return path


def load_residual_tensors(
    checkpoint_dir: str | Path,
    *,
    output_dim: int,
    input_dim: int,
) -> dict[str, torch.Tensor]:
    """Load only the B1 residual tensors, never the frozen Qwen/base weights."""

    path = _validate_checkpoint_dir(checkpoint_dir)
    state = {
        WEIGHT_KEY: torch.empty(output_dim, input_dim, dtype=torch.float32),
        BIAS_KEY: torch.empty(output_dim, dtype=torch.float32),
        SCALE_KEY: torch.empty((), dtype=torch.float32),
    }
    dcp.load(state, checkpoint_id=path)
    for name, tensor in state.items():
        if not torch.isfinite(tensor).all():
            raise FloatingPointError(f"non-finite tensor {name} in {path}")
    if float(state[SCALE_KEY]) <= 0:
        raise ValueError(f"residual scale must be positive in {path}")
    return state


def load_residual_into_model(model: Any, checkpoint_dir: str | Path) -> dict[str, Any]:
    """Populate a FrozenB2ResidualBridge-compatible model and return provenance."""

    weight = model.residual.weight
    bias = model.residual.bias
    tensors = load_residual_tensors(
        checkpoint_dir,
        output_dim=weight.shape[0],
        input_dim=weight.shape[1],
    )
    with torch.no_grad():
        weight.copy_(tensors[WEIGHT_KEY].to(device=weight.device, dtype=weight.dtype))
        bias.copy_(tensors[BIAS_KEY].to(device=bias.device, dtype=bias.dtype))
        model.residual_scale.copy_(
            tensors[SCALE_KEY].to(
                device=model.residual_scale.device, dtype=model.residual_scale.dtype
            )
        )
    return {
        "schema_version": "uniss_streamspeech_stage06_megatron_residual_v1",
        "checkpoint_dir": str(Path(checkpoint_dir).resolve()),
        "iteration": checkpoint_iteration(checkpoint_dir),
        "residual_scale": float(model.residual_scale.detach().cpu()),
        "weight_norm": float(model.residual.weight.detach().float().norm().cpu()),
        "bias_norm": float(model.residual.bias.detach().float().norm().cpu()),
    }
