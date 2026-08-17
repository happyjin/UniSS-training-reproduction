"""Stage A v2 objective with active full-vocabulary same-prefix KL."""

from __future__ import annotations

from collections import OrderedDict
from typing import Mapping

import torch
import torch.distributed as dist
from torch.nn import functional as F

from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v1.stage_a_causal_whisper_asr.training import (
    objective as v1,
)


DIAGNOSTIC_NAMES = (*v1.DIAGNOSTIC_NAMES, "offline_teacher_kl_tokens")
TERM_NAMES = v1.TERM_NAMES
DEFAULT_WEIGHTS = v1.DEFAULT_WEIGHTS
LossTerm = v1.LossTerm
StageAObjectiveOutput = v1.StageAObjectiveOutput


class StageAObjective(v1.StageAObjective):
    def __init__(self, *args, teacher_temperature: float = 1.5, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if teacher_temperature <= 0:
            raise ValueError("teacher KL temperature must be positive")
        self.teacher_temperature = float(teacher_temperature)

    def _teacher_kl(
        self,
        logits: torch.Tensor,
        batch: Mapping[str, torch.Tensor],
        anchor: torch.Tensor,
        *,
        original_seq_length: int,
    ) -> LossTerm:
        del anchor
        required = (
            "teacher_batch",
            "teacher_positions",
            "teacher_reference_labels",
            "teacher_indices",
            "teacher_probabilities",
            "teacher_mask",
        )
        missing = [name for name in required if name not in batch]
        if missing:
            raise KeyError(f"same-prefix teacher batch fields are missing: {missing}")
        flat = batch["teacher_batch"].long() * original_seq_length + batch[
            "teacher_positions"
        ].long()
        indices = batch["teacher_indices"].long()
        probabilities = batch["teacher_probabilities"].float()
        mask = batch["teacher_mask"].bool()
        if not flat.numel() or not bool(mask.any()):
            raise ValueError("same-prefix teacher denominator is zero")
        if indices.ndim != 2 or probabilities.shape != indices.shape or mask.shape != indices.shape:
            raise ValueError("same-prefix teacher top-k tensors differ")
        if len(flat) != len(indices):
            raise ValueError("same-prefix teacher position count differs")
        if int(flat.min()) < 0 or int(flat.max()) >= len(logits):
            raise ValueError("same-prefix teacher position exceeds student logits")
        if int(indices.min()) < 0 or int(indices.max()) >= logits.shape[-1]:
            raise ValueError("same-prefix teacher candidate exceeds vocabulary")
        temperature = self.teacher_temperature
        student_log = F.log_softmax(logits[flat].float() / temperature, dim=-1)
        selected_log = student_log.gather(1, indices)
        teacher = probabilities.masked_fill(~mask, 0.0)
        teacher = teacher / teacher.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        values = (
            teacher
            * (teacher.clamp_min(1e-8).log() - selected_log)
        ).sum(dim=-1) * (temperature**2)
        active = mask.any(dim=-1)
        return v1._values_term(values, active)

    def compute(self, *args, **kwargs) -> StageAObjectiveOutput:
        logits = args[1] if len(args) > 1 else kwargs["logits"]
        labels = args[2] if len(args) > 2 else kwargs["labels"]
        batch = args[5] if len(args) > 5 else kwargs["batch"]
        original_seq_length = kwargs["original_seq_length"]
        flat = batch["teacher_batch"].long() * original_seq_length + batch[
            "teacher_positions"
        ].long()
        if not torch.equal(
            labels.long()[flat], batch["teacher_reference_labels"].long()
        ):
            raise ValueError("same-prefix teacher labels differ from current Stage A labels")
        output = super().compute(*args, **kwargs)
        diagnostics = OrderedDict(output.diagnostics)
        diagnostics["offline_teacher_kl_tokens"] = output.terms[
            "offline_teacher_kl"
        ].denominator.detach().float()
        return StageAObjectiveOutput(output.terms, diagnostics, output.decoder_input)


def distributed_stage_a_objective(
    output: StageAObjectiveOutput,
    *,
    weights: Mapping[str, float] = DEFAULT_WEIGHTS,
) -> tuple[torch.Tensor, OrderedDict[str, torch.Tensor]]:
    if tuple(output.terms) != TERM_NAMES or tuple(weights) != TERM_NAMES:
        raise ValueError("Stage A v2 objective term order changed")
    numerators = torch.stack([output.terms[name].numerator for name in TERM_NAMES])
    denominators = torch.stack(
        [output.terms[name].denominator.to(numerators.dtype) for name in TERM_NAMES]
    )
    global_numerators = numerators.detach().clone()
    global_denominators = denominators.detach().clone()
    world_size = 1
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(global_numerators)
        dist.all_reduce(global_denominators)
        world_size = dist.get_world_size()
    if not bool((global_denominators[TERM_NAMES.index("offline_teacher_kl")] > 0).item()):
        raise ValueError("global same-prefix teacher denominator is zero")
    active = global_denominators > 0
    local_means = torch.where(
        active,
        world_size * numerators / global_denominators.clamp_min(1),
        numerators * 0.0,
    )
    scales = numerators.new_tensor([float(weights[name]) for name in TERM_NAMES])
    total = (local_means * scales).sum()
    global_means = torch.where(
        active,
        global_numerators / global_denominators.clamp_min(1),
        global_numerators * 0.0,
    )
    metrics = OrderedDict(
        (name, global_means[index]) for index, name in enumerate(TERM_NAMES)
    )
    diagnostics = torch.stack(
        [output.diagnostics[name].detach().float() for name in DIAGNOSTIC_NAMES]
    )
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(diagnostics)
        diagnostics /= dist.get_world_size()
    metrics.update(
        (name, diagnostics[index]) for index, name in enumerate(DIAGNOSTIC_NAMES)
    )
    return total, metrics


__all__ = [
    "DEFAULT_WEIGHTS",
    "DIAGNOSTIC_NAMES",
    "StageAObjective",
    "TERM_NAMES",
    "distributed_stage_a_objective",
]
