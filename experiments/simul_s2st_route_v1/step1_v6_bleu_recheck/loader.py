"""Read joint-V6 Megatron checkpoints into a single-process inference model.

Joint V6 saves with the Megatron ``torch_dist`` backend, so an ``iter_XXXXXXXX`` directory
holds sharded ``.distcp`` files whose model tensors are prefixed ``joint.`` by the Megatron
``Composite`` wrapper. ``torch.distributed.checkpoint.load`` fills a caller-supplied state
dict, which lets a single process pull exactly the tensors it can place — no Megatron
launch, no distributed init.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemReader

CHECKPOINT_PREFIX = "joint."
ITERATION_PATTERN = re.compile(r"^iter_(\d+)$")


@dataclass
class LoadReport:
    checkpoint_dir: str
    iteration: int
    loaded_tensors: int
    missing_in_checkpoint: list[str]
    unused_in_checkpoint: int

    def to_dict(self) -> dict[str, object]:
        return dict(self.__dict__)


def checkpoint_iteration(checkpoint_dir: str | Path) -> int:
    name = Path(checkpoint_dir).name
    match = ITERATION_PATTERN.fullmatch(name)
    if match is None:
        raise ValueError(f"checkpoint directory must be named iter_XXXXXXXX: {name}")
    return int(match.group(1))


def checkpoint_tensor_metadata(checkpoint_dir: str | Path) -> dict[str, Any]:
    return dict(FileSystemReader(Path(checkpoint_dir)).read_metadata().state_dict_metadata)


def load_joint_checkpoint(model: Any, checkpoint_dir: str | Path) -> LoadReport:
    """Copy every model tensor the checkpoint provides, in place, on the model's devices.

    Megatron trains this model under ``--bf16``, so the saved tensors are bfloat16 even
    where the module keeps a float32 buffer. Staging tensors therefore mirror the saved
    dtype and shape, and the final ``copy_`` casts back to whatever the module holds.
    """

    path = Path(checkpoint_dir)
    if not path.is_dir():
        raise FileNotFoundError(f"missing Megatron checkpoint directory: {path}")
    if not (path / ".metadata").is_file():
        raise FileNotFoundError(f"missing torch_dist metadata: {path / '.metadata'}")

    available = checkpoint_tensor_metadata(path)
    destination = model.state_dict()
    wanted: list[str] = []
    missing: list[str] = []
    for name, value in destination.items():
        entry = available.get(f"{CHECKPOINT_PREFIX}{name}")
        if entry is None:
            missing.append(name)
            continue
        if tuple(entry.size) != tuple(value.shape):
            raise ValueError(
                f"shape mismatch for {name}: model {tuple(value.shape)} vs "
                f"checkpoint {tuple(entry.size)}"
            )
        wanted.append(name)
    if not wanted:
        raise RuntimeError(f"checkpoint {path} shares no tensor names with the model")

    state = {
        f"{CHECKPOINT_PREFIX}{name}": torch.empty(
            tuple(available[f"{CHECKPOINT_PREFIX}{name}"].size),
            dtype=available[f"{CHECKPOINT_PREFIX}{name}"].properties.dtype,
            device="cpu",
        )
        for name in wanted
    }
    dcp.load(state, checkpoint_id=path)
    with torch.no_grad():
        for name in wanted:
            value = state[f"{CHECKPOINT_PREFIX}{name}"]
            if value.is_floating_point() and not torch.isfinite(value).all():
                raise FloatingPointError(f"non-finite tensor {CHECKPOINT_PREFIX}{name} in {path}")
            target = destination[name]
            target.copy_(value.to(device=target.device, dtype=target.dtype))
    model_keys = {f"{CHECKPOINT_PREFIX}{name}" for name in destination}
    return LoadReport(
        checkpoint_dir=str(path.resolve()),
        iteration=checkpoint_iteration(path),
        loaded_tensors=len(wanted),
        missing_in_checkpoint=missing,
        unused_in_checkpoint=len(
            [
                name
                for name in available
                if name.startswith(CHECKPOINT_PREFIX) and name not in model_keys
            ]
        ),
    )


@torch.no_grad()
def backbone_drift(joint_qwen: Any, reference_qwen: Any) -> dict[str, object]:
    """How far the checkpoint's own Qwen moved from the frozen Phase3 export.

    Stage A freezes Qwen and Stage B trains it at a very small learning rate, so this
    separates "the frontend changed" from "the backend also changed" without needing a
    second generation pass.
    """

    reference = reference_qwen.state_dict()
    worst_name = ""
    worst = 0.0
    changed = 0
    compared = 0
    for name, value in joint_qwen.state_dict().items():
        other = reference.get(name)
        if other is None or other.shape != value.shape or not value.is_floating_point():
            continue
        compared += 1
        delta = float((value.float() - other.float().to(value.device)).abs().max())
        if delta > 0:
            changed += 1
        if delta > worst:
            worst = delta
            worst_name = name
    return {
        "compared_tensors": compared,
        "changed_tensors": changed,
        "max_abs_delta": worst,
        "max_abs_delta_tensor": worst_name,
    }
