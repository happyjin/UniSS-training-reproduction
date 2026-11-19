"""Isolated prototype for codec EOS weighting and history corruption.

This module intentionally has no dependency on the production trainer/model so
the proposed behavior can be validated before changing the training pipeline.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class CodecTrainingOptions:
    """Two opt-in training knobs; defaults preserve the current behavior."""

    codec_eos_loss_weight: float = 1.0
    codec_history_corruption_prob: float = 0.0
    seed: int = 20260727

    def __post_init__(self) -> None:
        if self.codec_eos_loss_weight <= 0:
            raise ValueError("codec_eos_loss_weight must be > 0")
        if not 0.0 <= self.codec_history_corruption_prob <= 1.0:
            raise ValueError("codec_history_corruption_prob must be in [0, 1]")

    @classmethod
    def from_json(cls, path: str | Path) -> "CodecTrainingOptions":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        unknown = set(payload) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown codec training option(s): {sorted(unknown)}")
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def weighted_codec_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    eos_id: int,
    eos_weight: float,
) -> torch.Tensor:
    """Cross entropy with MiniMind-style EOS numerator weighting.

    The denominator remains the number of valid labels. Therefore weight=1 is
    exactly ordinary mean cross entropy, while weight=10 multiplies the EOS
    position's logit gradient by 10 without shrinking non-EOS gradients.
    """
    if logits.shape[:-1] != labels.shape:
        raise ValueError(
            f"logits/labels shape mismatch: {tuple(logits.shape)} vs {tuple(labels.shape)}")
    if eos_weight <= 0:
        raise ValueError("eos_weight must be > 0")

    flat_logits = logits.reshape(-1, logits.size(-1))
    flat_labels = labels.reshape(-1)
    valid = flat_labels.ne(-100)
    valid_count = valid.sum()
    if int(valid_count.item()) == 0:
        return flat_logits.sum() * 0.0

    per_token = F.cross_entropy(
        flat_logits,
        flat_labels,
        ignore_index=-100,
        reduction="none",
    )
    weights = torch.ones_like(per_token)
    weights = torch.where(flat_labels.eq(eos_id), weights * float(eos_weight), weights)
    return (per_token * weights * valid).sum() / valid_count


def corrupt_codec_history(
    codec_input_ids: torch.Tensor,
    codec_labels: torch.Tensor,
    *,
    probability: float,
    acoustic_vocab_size: int = 2048,
    seed: int = 20260727,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Replace a fraction of supervised acoustic history tokens.

    Only input ids in ``[0, acoustic_vocab_size)`` whose aligned label is valid
    are eligible. Control/prefix tokens are never touched and labels are never
    modified. Randomness is generated on CPU for deterministic CPU/GPU tests.
    """
    if codec_input_ids.shape != codec_labels.shape:
        raise ValueError("codec_input_ids and codec_labels must have the same shape")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in [0, 1]")
    if acoustic_vocab_size <= 1:
        raise ValueError("acoustic_vocab_size must be > 1")

    result = codec_input_ids.clone()
    eligible = (
        codec_input_ids.ge(0)
        & codec_input_ids.lt(acoustic_vocab_size)
        & codec_labels.ne(-100)
    )
    if probability == 0.0 or not bool(eligible.any()):
        return result, torch.zeros_like(codec_input_ids, dtype=torch.bool)

    cpu_generator = torch.Generator(device="cpu")
    cpu_generator.manual_seed(int(seed))
    sampled = torch.rand(codec_input_ids.shape, generator=cpu_generator).lt(probability)
    corruption_mask_cpu = sampled & eligible.detach().cpu()
    replacements = torch.randint(
        0,
        acoustic_vocab_size,
        codec_input_ids.shape,
        generator=cpu_generator,
        dtype=torch.long,
    )
    original_cpu = codec_input_ids.detach().cpu()
    replacements = torch.where(
        replacements.eq(original_cpu),
        (replacements + 1) % acoustic_vocab_size,
        replacements,
    )
    corruption_mask = corruption_mask_cpu.to(codec_input_ids.device)
    result[corruption_mask] = replacements.to(codec_input_ids.device)[corruption_mask]
    return result, corruption_mask


def eos_training_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    eos_id: int,
) -> dict[str, float | int]:
    """Return the two most actionable teacher-forced EOS metrics."""
    if logits.shape[:-1] != labels.shape:
        raise ValueError("logits and labels do not align")
    predictions = logits.argmax(dim=-1)
    boundary = labels.eq(eos_id)
    non_boundary = labels.ne(-100) & labels.ne(eos_id)
    boundary_count = int(boundary.sum().item())
    non_boundary_count = int(non_boundary.sum().item())
    return {
        "eos_boundary_top1_acc": (
            float(predictions[boundary].eq(eos_id).float().mean().item())
            if boundary_count else 0.0
        ),
        "premature_eos_top1_rate": (
            float(predictions[non_boundary].eq(eos_id).float().mean().item())
            if non_boundary_count else 0.0
        ),
        "eos_boundary_count": boundary_count,
        "non_boundary_count": non_boundary_count,
    }
