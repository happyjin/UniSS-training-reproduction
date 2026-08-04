"""Selective LoRA loading from Stage08 Step2 Megatron checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.distributed.checkpoint as dcp

from experiments.uniss_streamspeech_ctc_v1.stage08_emformer_phase3_joint.step1_frozen_qwen.checkpoint_io import (
    checkpoint_iteration,
)

from experiments.uniss_streamspeech_ctc_v1.stage08_emformer_phase3_joint.step2_qwen_lora_replay_v1.lora import (
    lora_tensor_names,
)


def load_step2_lora_into_qwen(qwen: Any, checkpoint_dir: str | Path) -> dict[str, Any]:
    path = Path(checkpoint_dir)
    if not path.is_dir():
        raise FileNotFoundError(f"missing Step2 Megatron checkpoint: {path}")
    if not (path / ".metadata").is_file():
        raise FileNotFoundError(f"missing torch_dist metadata: {path / '.metadata'}")
    destination = qwen.state_dict()
    names = lora_tensor_names(qwen)
    state = {
        f"qwen.{name}": torch.empty_like(destination[name], device="cpu")
        for name in names
    }
    dcp.load(state, checkpoint_id=path)
    with torch.no_grad():
        for name in names:
            value = state[f"qwen.{name}"]
            if not torch.isfinite(value).all():
                raise FloatingPointError(f"non-finite LoRA tensor qwen.{name} in {path}")
            destination[name].copy_(
                value.to(device=destination[name].device, dtype=destination[name].dtype)
            )
    return {
        "schema_version": "uniss_streamspeech_stage08_step2_lora_v1",
        "checkpoint_dir": str(path.resolve()),
        "iteration": checkpoint_iteration(path),
        "loaded_tensors": len(names),
        "trainable_parameters": sum(
            value.numel() for value in qwen.parameters() if value.requires_grad
        ),
    }
