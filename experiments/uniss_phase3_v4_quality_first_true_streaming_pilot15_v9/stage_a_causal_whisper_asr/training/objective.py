"""V8 objective preserved exactly except for a stronger blank margin."""

from __future__ import annotations

from collections import OrderedDict
from typing import Mapping

import torch
from torch.nn import functional as F

from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v1.stage_a_causal_whisper_asr.training.frontend import (
    CausalWhisperOutput,
)
from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v8.stage_a_causal_whisper_asr.training import (
    objective as v8,
)


BLANK_POSTERIOR_TARGET = v8.BLANK_POSTERIOR_TARGET
ALLOWED_BLANK_FRACTION = v8.ALLOWED_BLANK_FRACTION
NONBLANK_MARGIN = v8.NONBLANK_MARGIN
DECISION_MARGIN_SCALE = 0.20

TERM_NAMES = v8.TERM_NAMES
DIAGNOSTIC_NAMES = v8.DIAGNOSTIC_NAMES
DEFAULT_WEIGHTS = OrderedDict(v8.DEFAULT_WEIGHTS)

LossTerm = v8.LossTerm
StageAObjectiveOutput = v8.StageAObjectiveOutput
chunk_pair_for_progress = v8.chunk_pair_for_progress
ctc_seed_strength = v8.ctc_seed_strength


class StageAObjective(v8.StageAObjective):
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
        margin_rows = v8.decision_margin_penalty(
            logits,
            input_lengths,
            self.ctc_blank_id,
            allowed_blank_fraction=ALLOWED_BLANK_FRACTION,
            margin=NONBLANK_MARGIN,
        )
        seed_values: list[torch.Tensor] = []
        budget_values: list[torch.Tensor] = []
        budget_targets: list[torch.Tensor] = []
        for row in range(int(logits.shape[0])):
            frames = int(input_lengths[row].item())
            targets = int(target_lengths[row].item())
            if frames <= 0 or targets <= 0:
                raise ValueError("Stage A V9 CTC row has no input frames or targets")
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
        return (
            ctc,
            monotonic_seed,
            blank_budget,
            blank_ratio,
            global_blank_posterior.detach(),
            torch.stack(budget_targets).mean().detach(),
            strength.detach(),
            input_lengths.sum().float(),
        )


def distributed_stage_a_objective(
    output: StageAObjectiveOutput,
    *,
    weights: Mapping[str, float] = DEFAULT_WEIGHTS,
):
    return v8.distributed_stage_a_objective(output, weights=weights)


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
    "distributed_stage_a_objective",
]
