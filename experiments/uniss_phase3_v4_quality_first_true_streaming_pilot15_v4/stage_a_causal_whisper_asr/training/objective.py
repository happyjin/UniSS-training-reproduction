"""Stage A v4 objective with direct discrete WhisperVQ identity supervision."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Mapping

import torch
import torch.distributed as dist
from torch.nn import functional as F

from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v1.stage_a_causal_whisper_asr.training import (
    objective as v1,
)
from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v1.stage_a_causal_whisper_asr.training.frontend import (
    CausalWhisperOutput,
)
from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v3.stage_a_causal_whisper_asr.training import (
    objective as v3,
)


TERM_NAMES = (*v3.TERM_NAMES, "codebook_identity_ce")
DIAGNOSTIC_NAMES = (*v3.DIAGNOSTIC_NAMES, "teacher_code_margin")
DEFAULT_WEIGHTS = OrderedDict(
    (*v3.DEFAULT_WEIGHTS.items(), ("codebook_identity_ce", 0.30))
)

LossTerm = v1.LossTerm
StageAObjectiveOutput = v1.StageAObjectiveOutput


@dataclass(frozen=True)
class StageAPrepared(v3.StageAPrepared):
    codebook_identity_ce: LossTerm
    teacher_code_margin: torch.Tensor


def chunk_pair_for_progress(progress: float, update: int) -> tuple[int, int]:
    """Expose 160 ms early enough to learn it while retaining larger anchors."""

    if not 0.0 <= progress <= 1.0 or update < 0:
        raise ValueError("invalid Stage A v4 curriculum position")
    if progress < 0.10:
        choices = (1280,)
    elif progress < 0.25:
        choices = (1280, 960)
    elif progress < 0.45:
        choices = (960, 640)
    elif progress < 0.65:
        choices = (640, 320)
    elif progress < 0.85:
        choices = (640, 320, 160)
    else:
        choices = (320, 160)
    return choices[update % len(choices)], max(choices)


class StageAObjective(v3.StageAObjective):
    """V3 anti-collapse objective plus full-codebook identity classification."""

    def __init__(
        self,
        *args,
        codebook_identity_temperature: float = 0.07,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if codebook_identity_temperature <= 0:
            raise ValueError("codebook identity temperature must be positive")
        self.codebook_identity_temperature = float(codebook_identity_temperature)

    def _aligned_hidden_and_teacher(
        self,
        pooled_hidden: torch.Tensor,
        pooled_lengths: torch.Tensor,
        batch: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_rows: list[torch.Tensor] = []
        teacher_rows: list[torch.Tensor] = []
        for row in range(int(pooled_hidden.shape[0])):
            length = int(batch["glm_lengths"][row].item())
            causal_length = int(pooled_lengths[row].item())
            waveform_samples = int(batch["waveform_lengths"][row].item())
            hidden = pooled_hidden[row, :causal_length]
            if causal_length == length:
                pass
            elif v1.terminal_codec_extension_deficit_samples(
                waveform_samples, causal_length, length
            ) is not None:
                hidden = torch.cat((hidden, hidden[-1:]), dim=0)
            else:
                raise ValueError("codebook identity sees invalid GLM coverage")
            teacher = batch["glm_ids"][row, :length].long()
            if int(teacher.min()) < 0 or int(teacher.max()) >= len(self.codebook):
                raise ValueError("teacher GLM code exceeds the immutable codebook")
            hidden_rows.append(hidden.float())
            teacher_rows.append(teacher)
        return torch.cat(hidden_rows), torch.cat(teacher_rows)

    def _codebook_identity(
        self,
        pooled_hidden: torch.Tensor,
        pooled_lengths: torch.Tensor,
        batch: Mapping[str, torch.Tensor],
    ) -> tuple[LossTerm, torch.Tensor]:
        hidden, teacher = self._aligned_hidden_and_teacher(
            pooled_hidden, pooled_lengths, batch
        )
        current = F.normalize(hidden, dim=-1)
        codebook = F.normalize(
            self.codebook.detach().to(device=hidden.device, dtype=torch.float32),
            dim=-1,
        )
        logits = current @ codebook.t()
        losses = F.cross_entropy(
            logits / self.codebook_identity_temperature,
            teacher,
            reduction="none",
        )
        with torch.no_grad():
            top_values, top_ids = logits.topk(k=2, dim=-1)
            best_other = torch.where(
                top_ids[:, 0] == teacher,
                top_values[:, 1],
                top_values[:, 0],
            )
            teacher_value = logits.gather(1, teacher[:, None]).squeeze(1)
            margin = (teacher_value - best_other).mean()
        return (
            LossTerm(losses.sum(), losses.new_tensor(float(len(losses)))),
            margin.detach(),
        )

    def prepare(
        self,
        decoder_input: torch.Tensor,
        word_embedding_weight: torch.Tensor,
        batch: Mapping[str, torch.Tensor],
        *,
        original_seq_length: int,
        chunk_ms: int,
        consistency_chunk_ms: int,
    ) -> StageAPrepared:
        waveform = batch["waveform"]
        waveform_lengths = batch["waveform_lengths"]
        output: CausalWhisperOutput = self.frontend(
            waveform, waveform_lengths, chunk_ms=chunk_ms
        )
        corrected, causal_ids, agreement, residual_rms, terminal_extensions = (
            self._inject_causal_glm(
                decoder_input,
                word_embedding_weight,
                output.pooled_hidden,
                output.pooled_lengths,
                batch,
                original_seq_length=original_seq_length,
            )
        )
        (
            ctc,
            monotonic_seed,
            blank_budget,
            blank_ratio,
            blank_posterior,
            blank_budget_target,
            seed_strength,
            input_frames,
        ) = self._ctc_terms(output, batch)
        commitment, teacher_code_cosine = self._codebook_commitment(
            output.pooled_hidden,
            output.pooled_lengths,
            batch,
        )
        identity_ce, teacher_code_margin = self._codebook_identity(
            output.pooled_hidden,
            output.pooled_lengths,
            batch,
        )

        if consistency_chunk_ms == chunk_ms:
            hidden_consistency = v1._zero_term(output.frame_hidden)
        else:
            with torch.no_grad():
                reference: CausalWhisperOutput = self.frontend(
                    waveform,
                    waveform_lengths,
                    chunk_ms=consistency_chunk_ms,
                )
            mask = v1.stable_multichunk_mask(
                waveform_lengths,
                sequence_length=output.frame_hidden.shape[1],
                first_chunk_ms=chunk_ms,
                second_chunk_ms=consistency_chunk_ms,
            )
            squared = (
                output.frame_hidden.float() - reference.frame_hidden.float()
            ).square()
            expanded = mask[:, :, None].expand_as(squared)
            hidden_consistency = v1._values_term(
                squared.reshape(-1), expanded.reshape(-1)
            )

        cache_full = v1._zero_term(output.frame_hidden)
        return StageAPrepared(
            decoder_input=corrected,
            ctc=ctc,
            hidden_chunk_consistency=hidden_consistency,
            cache_full_consistency=cache_full,
            ctc_blank_ratio=blank_ratio.detach(),
            causal_glm_agreement=agreement.detach(),
            bridge_residual_rms=residual_rms.detach(),
            causal_glm_tokens=causal_ids.new_tensor(causal_ids.numel()).float(),
            causal_glm_terminal_extensions=terminal_extensions.detach(),
            ctc_input_frames=input_frames.detach(),
            ctc_monotonic_seed=monotonic_seed,
            ctc_blank_budget=blank_budget,
            codebook_commitment=commitment,
            ctc_blank_posterior=blank_posterior,
            ctc_blank_budget_target=blank_budget_target,
            ctc_seed_strength=seed_strength,
            teacher_code_cosine=teacher_code_cosine,
            codebook_identity_ce=identity_ce,
            teacher_code_margin=teacher_code_margin,
        )

    def compute(self, *args, **kwargs) -> StageAObjectiveOutput:
        prepared = args[0] if args else kwargs["prepared"]
        output = super().compute(*args, **kwargs)
        terms = OrderedDict(output.terms)
        terms["codebook_identity_ce"] = prepared.codebook_identity_ce
        diagnostics = OrderedDict(output.diagnostics)
        diagnostics["teacher_code_margin"] = prepared.teacher_code_margin
        return StageAObjectiveOutput(terms, diagnostics, output.decoder_input)


def distributed_stage_a_objective(
    output: StageAObjectiveOutput,
    *,
    weights: Mapping[str, float] = DEFAULT_WEIGHTS,
) -> tuple[torch.Tensor, OrderedDict[str, torch.Tensor]]:
    if tuple(output.terms) != TERM_NAMES or tuple(weights) != TERM_NAMES:
        raise ValueError("Stage A v4 objective term order changed")
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
    teacher_index = TERM_NAMES.index("offline_teacher_kl")
    if not bool((global_denominators[teacher_index] > 0).item()):
        raise ValueError("global same-prefix teacher denominator is zero")
    identity_index = TERM_NAMES.index("codebook_identity_ce")
    if not bool((global_denominators[identity_index] > 0).item()):
        raise ValueError("global codebook identity denominator is zero")
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
    "chunk_pair_for_progress",
    "distributed_stage_a_objective",
]

