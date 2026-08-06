"""Numerically explicit joint/replay loss accounting."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from .config import JointLossWeights


@dataclass
class NormalizedLoss:
    numerator: torch.Tensor
    denominator: torch.Tensor

    @property
    def mean(self) -> torch.Tensor:
        return self.numerator / self.denominator.clamp_min(1)


def ctc_normalized_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    input_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    *,
    blank_id: int,
) -> tuple[NormalizedLoss, torch.Tensor]:
    """CTC summed over targets plus an explicit infeasible-sample count."""

    pieces = torch.split(targets, [int(value) for value in target_lengths.tolist()])
    # CTC needs one extra encoder frame between every pair of consecutive
    # identical labels.  Checking only target_length <= input_length lets
    # PyTorch return +inf for otherwise well-formed rows such as [a, a] with
    # two encoder frames.  Treat those rows as explicitly infeasible, just as
    # StreamSpeech-style short chunks require.
    repeated = torch.stack(
        [
            (piece[1:] == piece[:-1]).sum()
            if len(piece) > 1
            else target_lengths.new_zeros(())
            for piece in pieces
        ]
    )
    required_input_lengths = target_lengths + repeated.to(target_lengths.dtype)
    infeasible = required_input_lengths > input_lengths
    feasible_rows = torch.nonzero(~infeasible, as_tuple=False).flatten()
    if not len(feasible_rows):
        zero = logits.sum() * 0.0
        return NormalizedLoss(zero, zero.detach().new_ones(())), infeasible.sum()
    safe_targets = torch.cat([pieces[int(row)] for row in feasible_rows.tolist()])
    safe_target_lengths = target_lengths[feasible_rows]
    safe_input_lengths = input_lengths[feasible_rows]
    safe_logits = logits[feasible_rows]
    numerator = F.ctc_loss(
        safe_logits.float().log_softmax(-1).transpose(0, 1),
        safe_targets,
        safe_input_lengths,
        safe_target_lengths,
        blank=blank_id,
        reduction="sum",
        zero_infinity=False,
    )
    denominator = safe_target_lengths.sum().to(numerator.dtype).clamp_min(1)
    return NormalizedLoss(numerator, denominator), infeasible.sum()


def masked_ce_normalized(logits: torch.Tensor, labels: torch.Tensor) -> NormalizedLoss:
    if logits.shape[:-1] != labels.shape:
        raise ValueError("logit and label geometry differ")
    valid = labels != -100
    numerator = F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        ignore_index=-100,
        reduction="sum",
    )
    return NormalizedLoss(numerator, valid.sum().to(numerator.dtype).clamp_min(1))


def combine_joint_or_replay(
    *,
    sample_kind: str,
    weights: JointLossWeights,
    bicodec_ctc: NormalizedLoss | None = None,
    ar_s2tt: NormalizedLoss | None = None,
    asr_ctc: NormalizedLoss | None = None,
    nar_s2tt_ctc: NormalizedLoss | None = None,
    phase3_replay: NormalizedLoss | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Make joint and exact-replay microbatches mutually exclusive."""

    if sample_kind not in {"joint", "replay"}:
        raise ValueError("sample_kind must be joint or replay")
    components = {
        "bicodec_ctc": bicodec_ctc,
        "ar_s2tt": ar_s2tt,
        "asr_ctc": asr_ctc,
        "nar_s2tt_ctc": nar_s2tt_ctc,
        "phase3_replay": phase3_replay,
    }
    if sample_kind == "joint":
        required = components.copy()
        required.pop("phase3_replay")
        if any(value is None for value in required.values()) or phase3_replay is not None:
            raise ValueError("joint samples require exactly the four StreamSpeech losses")
        means = {name: value.mean for name, value in required.items() if value is not None}
        total = (
            weights.bicodec_ctc * means["bicodec_ctc"]
            + weights.ar_s2tt * means["ar_s2tt"]
            + weights.asr_ctc * means["asr_ctc"]
            + weights.nar_s2tt_ctc * means["nar_s2tt_ctc"]
        )
    else:
        if phase3_replay is None or any(
            components[name] is not None
            for name in ("bicodec_ctc", "ar_s2tt", "asr_ctc", "nar_s2tt_ctc")
        ):
            raise ValueError("replay samples require only exact Phase3 replay loss")
        means = {"phase3_replay": phase3_replay.mean}
        total = weights.phase3_replay * means["phase3_replay"]
    return total, means
