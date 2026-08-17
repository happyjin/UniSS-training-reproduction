"""Stage A v3 objective with CTC and WhisperVQ anti-collapse constraints."""

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
from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v2.stage_a_causal_whisper_asr.training import (
    objective as v2,
)


TERM_NAMES = (
    *v2.TERM_NAMES,
    "ctc_monotonic_seed",
    "ctc_blank_budget",
    "codebook_commitment",
)

DIAGNOSTIC_NAMES = (
    *v2.DIAGNOSTIC_NAMES,
    "ctc_blank_posterior",
    "ctc_blank_budget_target",
    "ctc_seed_strength",
    "teacher_code_cosine",
)

DEFAULT_WEIGHTS = OrderedDict(
    (
        *v2.DEFAULT_WEIGHTS.items(),
        ("ctc_monotonic_seed", 0.30),
        ("ctc_blank_budget", 20.0),
        ("codebook_commitment", 0.10),
    )
)

LossTerm = v1.LossTerm
StageAObjectiveOutput = v1.StageAObjectiveOutput


@dataclass(frozen=True)
class StageAPrepared(v1.StageAPrepared):
    ctc_monotonic_seed: LossTerm
    ctc_blank_budget: LossTerm
    codebook_commitment: LossTerm
    ctc_blank_posterior: torch.Tensor
    ctc_blank_budget_target: torch.Tensor
    ctc_seed_strength: torch.Tensor
    teacher_code_cosine: torch.Tensor


def ctc_seed_strength(progress: float) -> float:
    """Keep explicit non-blank anchors early, then hand control to pure CTC."""

    if not 0.0 <= progress <= 1.0:
        raise ValueError("invalid Stage A progress")
    if progress <= 0.10:
        return 1.0
    if progress >= 0.40:
        return 0.0
    return (0.40 - progress) / 0.30


class StageAObjective(v2.StageAObjective):
    """Same-prefix teacher objective plus explicit non-blank/code constraints."""

    def __init__(self, *args, ctc_initial_blank_bias: float = -2.0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if ctc_initial_blank_bias >= 0:
            raise ValueError("the repaired CTC blank bias must be negative")
        with torch.no_grad():
            self.ctc_head.bias[self.ctc_blank_id] = float(ctc_initial_blank_bias)
        self.ctc_initial_blank_bias = float(ctc_initial_blank_bias)

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
        seed_values: list[torch.Tensor] = []
        budget_values: list[torch.Tensor] = []
        budget_targets: list[torch.Tensor] = []
        for row in range(int(logits.shape[0])):
            frames = int(input_lengths[row].item())
            targets = int(target_lengths[row].item())
            if frames <= 0 or targets <= 0:
                raise ValueError("Stage A CTC row has no input frames or targets")
            target = batch["ctc_ids"][row, :targets].long()
            # Uniform monotonic anchors are deliberately weak pseudo alignment.
            # They prevent the fresh head from choosing the all-blank basin; the
            # exact CTC objective remains responsible for alignment discovery.
            anchor = (
                (torch.arange(targets, device=logits.device, dtype=torch.float32) + 0.5)
                * float(frames)
                / float(targets)
            ).floor().long().clamp_max(frames - 1)
            seed_values.append(
                F.cross_entropy(logits[row, anchor].float(), target, reduction="none")
            )
            row_blank = blank_probability[row, :frames].mean()
            target_density = logits.new_tensor(float(targets) / float(frames))
            budget = (1.0 - 0.5 * target_density).clamp(0.60, 0.95)
            budget_values.append((row_blank - budget).clamp_min(0.0).square())
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

    def _codebook_commitment(
        self,
        pooled_hidden: torch.Tensor,
        pooled_lengths: torch.Tensor,
        batch: Mapping[str, torch.Tensor],
    ) -> tuple[LossTerm, torch.Tensor]:
        values: list[torch.Tensor] = []
        cosines: list[torch.Tensor] = []
        codebook = self.codebook.detach().to(
            device=pooled_hidden.device,
            dtype=torch.float32,
        )
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
                raise ValueError("codebook commitment sees invalid GLM coverage")
            teacher_ids = batch["glm_ids"][row, :length].long()
            if int(teacher_ids.min()) < 0 or int(teacher_ids.max()) >= len(codebook):
                raise ValueError("teacher GLM code exceeds the immutable codebook")
            target = F.embedding(teacher_ids, codebook)
            current = hidden.float()
            values.append((current - target).square().mean(dim=-1))
            cosines.append(F.cosine_similarity(current, target, dim=-1))
        commitment = torch.cat(values)
        cosine = torch.cat(cosines)
        return (
            LossTerm(commitment.sum(), commitment.new_tensor(float(len(commitment)))),
            cosine.mean().detach(),
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
        )

    def compute(self, *args, **kwargs) -> StageAObjectiveOutput:
        prepared = args[0] if args else kwargs["prepared"]
        output = super().compute(*args, **kwargs)
        terms = OrderedDict(output.terms)
        terms["ctc_monotonic_seed"] = prepared.ctc_monotonic_seed
        terms["ctc_blank_budget"] = prepared.ctc_blank_budget
        terms["codebook_commitment"] = prepared.codebook_commitment
        diagnostics = OrderedDict(output.diagnostics)
        diagnostics["ctc_blank_posterior"] = prepared.ctc_blank_posterior
        diagnostics["ctc_blank_budget_target"] = prepared.ctc_blank_budget_target
        diagnostics["ctc_seed_strength"] = prepared.ctc_seed_strength
        diagnostics["teacher_code_cosine"] = prepared.teacher_code_cosine
        return StageAObjectiveOutput(terms, diagnostics, output.decoder_input)


def distributed_stage_a_objective(
    output: StageAObjectiveOutput,
    *,
    weights: Mapping[str, float] = DEFAULT_WEIGHTS,
) -> tuple[torch.Tensor, OrderedDict[str, torch.Tensor]]:
    if tuple(output.terms) != TERM_NAMES or tuple(weights) != TERM_NAMES:
        raise ValueError("Stage A v3 objective term order changed")
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
    "StageAPrepared",
    "TERM_NAMES",
    "ctc_seed_strength",
    "distributed_stage_a_objective",
]

