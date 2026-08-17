"""V8 objective aligned with long-horizon blank and geometry gates."""

from __future__ import annotations

import math
from collections import OrderedDict
from typing import Mapping

import torch
from torch.nn import functional as F

from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v1.stage_a_causal_whisper_asr.training.frontend import (
    CausalWhisperOutput,
)
from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v5.stage_a_causal_whisper_asr.training import (
    objective as v5,
)


BLANK_POSTERIOR_TARGET = 0.20
ALLOWED_BLANK_FRACTION = 0.20
NONBLANK_MARGIN = 0.05
DECISION_MARGIN_SCALE = 0.05

TERM_NAMES = v5.TERM_NAMES
DIAGNOSTIC_NAMES = v5.DIAGNOSTIC_NAMES
DEFAULT_WEIGHTS = OrderedDict(v5.DEFAULT_WEIGHTS)
DEFAULT_WEIGHTS["codebook_commitment"] = 0.30
DEFAULT_WEIGHTS["codebook_identity_ce"] = 0.50
DEFAULT_WEIGHTS["code_adapter_residual"] = 0.05

LossTerm = v5.LossTerm
StageAObjectiveOutput = v5.StageAObjectiveOutput
chunk_pair_for_progress = v5.chunk_pair_for_progress


def ctc_seed_strength(progress: float) -> float:
    """Preserve a small non-blank anchor throughout the long hold."""

    if not 0.0 <= progress <= 1.0:
        raise ValueError("invalid Stage A v8 progress")
    if progress <= 0.10:
        return 1.0
    if progress >= 0.40:
        return 0.10
    return 0.10 + 0.90 * (0.40 - progress) / 0.30


def decision_margin_penalty(
    logits: torch.Tensor,
    input_lengths: torch.Tensor,
    blank_id: int,
    *,
    allowed_blank_fraction: float = ALLOWED_BLANK_FRACTION,
    margin: float = NONBLANK_MARGIN,
) -> torch.Tensor:
    """Penalize blank winning on more than the allowed frame fraction."""

    if logits.ndim != 3 or input_lengths.ndim != 1:
        raise ValueError("invalid V8 CTC decision-margin geometry")
    if not 0.0 <= allowed_blank_fraction < 1.0 or margin < 0.0:
        raise ValueError("invalid V8 CTC decision-margin settings")
    if not 0 <= blank_id < logits.shape[-1] or len(input_lengths) != len(logits):
        raise ValueError("invalid V8 CTC blank id or lengths")

    values, ids = logits.float().topk(k=2, dim=-1)
    best_nonblank = torch.where(ids[..., 0] == blank_id, values[..., 1], values[..., 0])
    blank_advantage = logits.float()[..., blank_id] - best_nonblank
    rows: list[torch.Tensor] = []
    for row in range(len(logits)):
        frames = int(input_lengths[row].item())
        if not 0 < frames <= logits.shape[1]:
            raise ValueError("invalid V8 CTC frame length")
        required_nonblank = max(
            1,
            int(math.ceil((1.0 - allowed_blank_fraction) * frames)),
        )
        easiest = blank_advantage[row, :frames].topk(
            k=required_nonblank,
            largest=False,
        ).values
        rows.append(F.relu(easiest + margin).square().mean())
    return torch.stack(rows)


class StageAObjective(v5.StageAObjective):
    """V5 objective with persistent non-blank and decision-aligned guards."""

    def _ctc_terms(
        self,
        output: CausalWhisperOutput,
        batch: Mapping[str, torch.Tensor],
    ) -> tuple[
        LossTerm,
        LossTerm,
        LossTerm,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        logits = self.ctc_head(output.frame_hidden)
        log_probs = logits.float().log_softmax(dim=-1).transpose(0, 1)
        input_lengths = output.frame_lengths.long()
        target_lengths = batch["ctc_lengths"].long()
        losses = F.ctc_loss(
            log_probs,
            batch["ctc_ids"].long(),
            input_lengths,
            target_lengths,
            blank=self.ctc_blank_id,
            reduction="none",
            zero_infinity=True,
        )
        ctc = LossTerm(losses.sum(), target_lengths.sum().to(losses.dtype))

        positions = torch.arange(logits.shape[1], device=logits.device)[None, :]
        valid = positions < input_lengths[:, None]
        blank_argmax = (logits.argmax(dim=-1) == self.ctc_blank_id) & valid
        blank_ratio = blank_argmax.sum().float() / valid.sum().clamp_min(1)
        blank_probability = logits.float().softmax(dim=-1)[..., self.ctc_blank_id]
        global_blank_posterior = (
            blank_probability.masked_fill(~valid, 0.0).sum()
            / valid.sum().clamp_min(1)
        )

        progress_value = batch.get("training_progress")
        progress = (
            float(progress_value.detach().item())
            if isinstance(progress_value, torch.Tensor)
            else 0.0
        )
        strength = logits.new_tensor(ctc_seed_strength(progress)).float()
        margin_rows = decision_margin_penalty(
            logits,
            input_lengths,
            self.ctc_blank_id,
        )
        seed_values: list[torch.Tensor] = []
        budget_values: list[torch.Tensor] = []
        budget_targets: list[torch.Tensor] = []
        for row in range(int(logits.shape[0])):
            frames = int(input_lengths[row].item())
            targets = int(target_lengths[row].item())
            if frames <= 0 or targets <= 0:
                raise ValueError("Stage A V8 CTC row has no input frames or targets")
            target = batch["ctc_ids"][row, :targets].long()
            anchor = (
                (torch.arange(targets, device=logits.device, dtype=torch.float32) + 0.5)
                * float(frames)
                / float(targets)
            ).floor().long().clamp_max(frames - 1)
            seed_values.append(
                F.cross_entropy(logits[row, anchor].float(), target, reduction="none")
            )
            row_blank = blank_probability[row, :frames].mean()
            budget = logits.new_tensor(BLANK_POSTERIOR_TARGET)
            posterior_penalty = (row_blank - budget).clamp_min(0.0).square()
            budget_values.append(
                posterior_penalty + DECISION_MARGIN_SCALE * margin_rows[row]
            )
            budget_targets.append(budget)

        seed = torch.cat(seed_values)
        monotonic_seed = LossTerm(
            seed.sum() * strength,
            target_lengths.sum().to(seed.dtype),
        )
        budget_tensor = torch.stack(budget_values)
        blank_budget = LossTerm(
            budget_tensor.sum(),
            budget_tensor.new_tensor(float(len(budget_values))),
        )
        budget_target = torch.stack(budget_targets).mean().detach()
        return (
            ctc,
            monotonic_seed,
            blank_budget,
            blank_ratio,
            global_blank_posterior.detach(),
            budget_target,
            strength.detach(),
            input_lengths.sum().float(),
        )


def distributed_stage_a_objective(
    output: StageAObjectiveOutput,
    *,
    weights: Mapping[str, float] = DEFAULT_WEIGHTS,
):
    return v5.distributed_stage_a_objective(output, weights=weights)


__all__ = [
    "ALLOWED_BLANK_FRACTION",
    "BLANK_POSTERIOR_TARGET",
    "DECISION_MARGIN_SCALE",
    "DEFAULT_WEIGHTS",
    "DIAGNOSTIC_NAMES",
    "NONBLANK_MARGIN",
    "StageAObjective",
    "TERM_NAMES",
    "chunk_pair_for_progress",
    "ctc_seed_strength",
    "decision_margin_penalty",
    "distributed_stage_a_objective",
]
