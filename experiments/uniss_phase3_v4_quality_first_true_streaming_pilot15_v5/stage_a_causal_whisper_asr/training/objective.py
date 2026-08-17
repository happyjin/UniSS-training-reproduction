"""Stage A v5 objective with a zero-initialized causal-code adapter."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Mapping

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F

from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v1.stage_a_causal_whisper_asr.training import (
    objective as v1,
)
from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v1.stage_a_causal_whisper_asr.training.frontend import (
    CausalWhisperOutput,
)
from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v4.stage_a_causal_whisper_asr.training import (
    objective as v4,
)


TERM_NAMES = (*v4.TERM_NAMES, "code_adapter_residual")
DIAGNOSTIC_NAMES = (*v4.DIAGNOSTIC_NAMES, "code_adapter_rms")
DEFAULT_WEIGHTS = OrderedDict(
    (*v4.DEFAULT_WEIGHTS.items(), ("code_adapter_residual", 0.01))
)

LossTerm = v1.LossTerm
StageAObjectiveOutput = v1.StageAObjectiveOutput
chunk_pair_for_progress = v4.chunk_pair_for_progress


@dataclass(frozen=True)
class StageAPrepared(v4.StageAPrepared):
    code_adapter_residual: LossTerm
    code_adapter_rms: torch.Tensor


class ResidualCodeAdapter(nn.Module):
    """Low-rank zero-output adapter that leaves Phase3 exact at initialization."""

    def __init__(self, hidden_size: int, rank: int = 128) -> None:
        super().__init__()
        if hidden_size <= 0 or rank <= 0 or rank > hidden_size:
            raise ValueError("invalid causal-code adapter geometry")
        self.norm = nn.LayerNorm(hidden_size)
        self.down = nn.Linear(hidden_size, rank, bias=False)
        self.up = nn.Linear(rank, hidden_size, bias=False)
        nn.init.zeros_(self.up.weight)
        for parameter in self.parameters():
            parameter.uniss_stage_a_bridge = True

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        residual = self.up(F.silu(self.down(self.norm(hidden)))).to(hidden.dtype)
        return hidden + residual, residual


class StageAObjective(v4.StageAObjective):
    """Freeze Whisper geometry and learn only a post-pooling code correction."""

    def __init__(self, *args, code_adapter_rank: int = 128, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        hidden_size = int(getattr(self.frontend, "hidden_size"))
        self.code_adapter = ResidualCodeAdapter(hidden_size, code_adapter_rank)

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
        adapted_hidden, adapter_residual = self.code_adapter(output.pooled_hidden)
        corrected, causal_ids, agreement, residual_rms, terminal_extensions = (
            self._inject_causal_glm(
                decoder_input,
                word_embedding_weight,
                adapted_hidden,
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
            adapted_hidden,
            output.pooled_lengths,
            batch,
        )
        identity_ce, teacher_code_margin = self._codebook_identity(
            adapted_hidden,
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

        residual_values = adapter_residual.float().square()
        adapter_term = LossTerm(
            residual_values.sum(),
            residual_values.new_tensor(float(residual_values.numel())),
        )
        adapter_rms = residual_values.mean().sqrt().detach()
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
            code_adapter_residual=adapter_term,
            code_adapter_rms=adapter_rms,
        )

    def compute(self, *args, **kwargs) -> StageAObjectiveOutput:
        prepared = args[0] if args else kwargs["prepared"]
        output = super().compute(*args, **kwargs)
        terms = OrderedDict(output.terms)
        terms["code_adapter_residual"] = prepared.code_adapter_residual
        diagnostics = OrderedDict(output.diagnostics)
        diagnostics["code_adapter_rms"] = prepared.code_adapter_rms
        return StageAObjectiveOutput(terms, diagnostics, output.decoder_input)


def distributed_stage_a_objective(
    output: StageAObjectiveOutput,
    *,
    weights: Mapping[str, float] = DEFAULT_WEIGHTS,
) -> tuple[torch.Tensor, OrderedDict[str, torch.Tensor]]:
    if tuple(output.terms) != TERM_NAMES or tuple(weights) != TERM_NAMES:
        raise ValueError("Stage A v5 objective term order changed")
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
    for required in ("offline_teacher_kl", "codebook_identity_ce"):
        index = TERM_NAMES.index(required)
        if not bool((global_denominators[index] > 0).item()):
            raise ValueError(f"global {required} denominator is zero")
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
    "ResidualCodeAdapter",
    "StageAObjective",
    "TERM_NAMES",
    "chunk_pair_for_progress",
    "distributed_stage_a_objective",
]

