"""Stage A causal-ASR objective around the native Phase3 decoder."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Mapping

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F

from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v1.stage_a_causal_whisper_asr.packing import (
    LOSS_CAUSAL_FULL_ASR,
    LOSS_OFFLINE_ASR_REPLAY,
    LOSS_PHASE3_REPLAY,
    LOSS_STREAMING_ASR,
)
from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v1.stage_a_causal_whisper_asr.training.frontend import (
    CausalWhisperOutput,
    FRAME_SAMPLES,
    block_padded_frame_lengths,
)
from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v1.stage00_baseline.shared_causal_frontend import (
    TOKEN_HOP_SAMPLES,
)
from training import constants_uniss as c


TERM_NAMES = (
    "ar_asr",
    "source_ctc",
    "offline_teacher_kl",
    "hidden_chunk_consistency",
    "cache_full_consistency",
    "offline_asr_replay",
    "phase3_replay",
)

DIAGNOSTIC_NAMES = (
    "ctc_blank_ratio",
    "causal_glm_agreement",
    "bridge_residual_rms",
    "causal_glm_tokens",
    "causal_glm_terminal_extensions",
    "ctc_target_tokens",
    "ctc_input_frames",
    "ar_asr_tokens",
    "offline_asr_replay_tokens",
    "phase3_replay_tokens",
    "disabled_acoustics",
)

DEFAULT_WEIGHTS = OrderedDict(
    (
        ("ar_asr", 1.00),
        ("source_ctc", 0.30),
        ("offline_teacher_kl", 0.20),
        ("hidden_chunk_consistency", 0.10),
        ("cache_full_consistency", 0.10),
        ("offline_asr_replay", 1.00),
        ("phase3_replay", 1.00),
    )
)


@dataclass(frozen=True)
class LossTerm:
    numerator: torch.Tensor
    denominator: torch.Tensor

    @property
    def mean(self) -> torch.Tensor:
        return self.numerator / self.denominator.clamp_min(1)


@dataclass(frozen=True)
class StageAObjectiveOutput:
    terms: OrderedDict[str, LossTerm]
    diagnostics: OrderedDict[str, torch.Tensor]
    decoder_input: torch.Tensor


@dataclass(frozen=True)
class StageAPrepared:
    decoder_input: torch.Tensor
    ctc: LossTerm
    hidden_chunk_consistency: LossTerm
    cache_full_consistency: LossTerm
    ctc_blank_ratio: torch.Tensor
    causal_glm_agreement: torch.Tensor
    bridge_residual_rms: torch.Tensor
    causal_glm_tokens: torch.Tensor
    causal_glm_terminal_extensions: torch.Tensor
    ctc_input_frames: torch.Tensor


def _zero_term(anchor: torch.Tensor) -> LossTerm:
    zero = anchor.sum() * 0.0
    return LossTerm(zero, zero.detach())


def _values_term(values: torch.Tensor, mask: torch.Tensor) -> LossTerm:
    weights = mask.to(dtype=values.dtype)
    return LossTerm((values * weights).sum(), weights.sum())


def chunk_pair_for_progress(progress: float, update: int) -> tuple[int, int]:
    """Deterministic multi-chunk curriculum and its larger consistency view."""

    if not 0.0 <= progress <= 1.0 or update < 0:
        raise ValueError("invalid Stage A curriculum position")
    if progress < 0.10:
        choices = (1280,)
    elif progress < 0.30:
        choices = (1280, 960)
    elif progress < 0.60:
        choices = (960, 640)
    elif progress < 0.85:
        choices = (640, 320)
    else:
        choices = (320, 160)
    selected = choices[update % len(choices)]
    return selected, max(choices)


def stable_multichunk_mask(
    waveform_lengths: torch.Tensor,
    *,
    sequence_length: int,
    first_chunk_ms: int,
    second_chunk_ms: int,
) -> torch.Tensor:
    """Frames whose complete visible block endpoint is identical in two views."""

    if first_chunk_ms % 20 or second_chunk_ms % 20:
        raise ValueError("Stage A chunk sizes must be multiples of 20 ms")
    valid = block_padded_frame_lengths(waveform_lengths)
    positions = torch.arange(sequence_length, device=waveform_lengths.device)
    first = first_chunk_ms // 20
    second = second_chunk_ms // 20
    first_end = ((positions // first) + 1) * first
    second_end = ((positions // second) + 1) * second
    first_end = torch.minimum(first_end[None, :], valid[:, None])
    second_end = torch.minimum(second_end[None, :], valid[:, None])
    return (
        (positions[None, :] < valid[:, None])
        & (first_end == second_end)
    )


def terminal_codec_extension_deficit_samples(
    waveform_samples: int,
    causal_tokens: int,
    packed_tokens: int,
) -> int | None:
    """Return the audited terminal PCM deficit for one safe GLM extension.

    Released UniST GLM tokens and the BiCodec-reconstructed PCM are separate
    codec views of the same utterance.  The formal Stage A audit found exactly
    two one-token terminal discrepancies: PCM ending on the final causal-token
    boundary, or one 20-ms Whisper frame before that boundary.  No wider
    tolerance is authorized here.
    """

    waveform_samples = int(waveform_samples)
    causal_tokens = int(causal_tokens)
    packed_tokens = int(packed_tokens)
    if waveform_samples <= 0 or causal_tokens <= 0:
        return None
    expected_causal = (
        waveform_samples + TOKEN_HOP_SAMPLES - 1
    ) // TOKEN_HOP_SAMPLES
    if causal_tokens != expected_causal or packed_tokens != causal_tokens + 1:
        return None
    deficit = causal_tokens * TOKEN_HOP_SAMPLES - waveform_samples
    return deficit if deficit in (0, FRAME_SAMPLES) else None


def _batch_string(batch: Mapping[str, object], key: str, row: int) -> str:
    values = batch.get(key)
    if isinstance(values, (list, tuple)) and 0 <= row < len(values):
        return str(values[row])
    return "unknown"


class StageAObjective(nn.Module):
    """Causal WhisperVQ replacement, byte CTC, replay, and AR-ASR losses."""

    def __init__(
        self,
        frontend: nn.Module,
        *,
        qwen_hidden_size: int,
        ctc_output_size: int = 257,
        ctc_blank_id: int = 256,
        glm_semantic_offset: int = c.GLM_SEMANTIC_OFFSET,
        nearest_code_block: int = 512,
    ) -> None:
        super().__init__()
        if qwen_hidden_size <= 0 or ctc_output_size <= 1:
            raise ValueError("invalid Stage A objective geometry")
        if not 0 <= ctc_blank_id < ctc_output_size:
            raise ValueError("Stage A CTC blank is outside the output inventory")
        if nearest_code_block <= 0:
            raise ValueError("nearest-code block must be positive")
        self.frontend = frontend
        hidden_size = int(getattr(frontend, "hidden_size"))
        self.ctc_head = nn.Linear(hidden_size, ctc_output_size)
        self.bridge_norm = nn.LayerNorm(hidden_size)
        self.bridge_projection = nn.Linear(hidden_size, qwen_hidden_size, bias=False)
        nn.init.zeros_(self.bridge_projection.weight)
        self.ctc_blank_id = int(ctc_blank_id)
        self.glm_semantic_offset = int(glm_semantic_offset)
        self.nearest_code_block = int(nearest_code_block)
        if hasattr(frontend, "tag_learning_rate_groups"):
            frontend.tag_learning_rate_groups()
        for parameter in self.ctc_head.parameters():
            parameter.uniss_stage_a_new_head = True
        for module in (self.bridge_norm, self.bridge_projection):
            for parameter in module.parameters():
                parameter.uniss_stage_a_bridge = True

    @property
    def codebook(self) -> torch.Tensor:
        value = getattr(self.frontend, "codebook")
        if not isinstance(value, torch.Tensor) or value.ndim != 2:
            raise TypeError("Stage A frontend codebook is malformed")
        return value

    def _nearest_codes(self, hidden: torch.Tensor) -> torch.Tensor:
        codebook = self.codebook.to(device=hidden.device, dtype=torch.float32)
        flat = hidden.reshape(-1, hidden.shape[-1]).float()
        code_norm = codebook.square().sum(dim=1)
        pieces: list[torch.Tensor] = []
        for start in range(0, len(flat), self.nearest_code_block):
            values = flat[start : start + self.nearest_code_block]
            distances = (
                values.square().sum(dim=1, keepdim=True)
                + code_norm.unsqueeze(0)
                - 2.0 * values @ codebook.t()
            )
            pieces.append(distances.argmin(dim=1))
        return torch.cat(pieces).reshape(hidden.shape[:-1])

    def _inject_causal_glm(
        self,
        decoder_input: torch.Tensor,
        word_embedding_weight: torch.Tensor,
        pooled_hidden: torch.Tensor,
        pooled_lengths: torch.Tensor,
        batch: Mapping[str, torch.Tensor],
        *,
        original_seq_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if decoder_input.ndim != 3 or decoder_input.shape[1] != 1:
            raise ValueError("Stage A packed decoder input must be [tokens,1,hidden]")
        corrected = decoder_input.clone()
        causal_ids: list[torch.Tensor] = []
        teacher_ids: list[torch.Tensor] = []
        residuals: list[torch.Tensor] = []
        terminal_extensions = 0
        for row in range(int(pooled_hidden.shape[0])):
            length = int(batch["glm_lengths"][row].item())
            causal_length = int(pooled_lengths[row].item())
            waveform_samples = int(batch["waveform_lengths"][row].item())
            hidden = pooled_hidden[row, :causal_length]
            if causal_length == length:
                pass
            elif terminal_codec_extension_deficit_samples(
                waveform_samples, causal_length, length
            ) is not None:
                # Released GLM and reconstructed BiCodec PCM occasionally
                # differ by one terminal codec slot. Repeating the final
                # already-visible causal state fills only that audited slot;
                # it does not expose future audio and preserves Phase3 shape.
                hidden = torch.cat((hidden, hidden[-1:]), dim=0)
                terminal_extensions += 1
            else:
                raise ValueError(
                    "causal WhisperVQ token count differs from packed GLM coverage: "
                    f"{causal_length} vs {length}; "
                    f"sample_id={_batch_string(batch, 'acoustic_sample_ids', row)} "
                    f"source_audio={_batch_string(batch, 'source_audio_paths', row)} "
                    f"waveform_samples={waveform_samples} "
                    f"waveform_mod_80ms={waveform_samples % TOKEN_HOP_SAMPLES}"
                )
            codes = self._nearest_codes(hidden)
            qwen_ids = codes + self.glm_semantic_offset
            if int(qwen_ids.max()) >= int(word_embedding_weight.shape[0]):
                raise ValueError("causal GLM token exceeds native Qwen vocabulary")
            base = F.embedding(qwen_ids, word_embedding_weight)
            residual = self.bridge_projection(self.bridge_norm(hidden)).to(base.dtype)
            positions = batch["glm_positions"][row, :length].long()
            packed_row = int(batch["acoustic_batch"][row].item())
            flattened = packed_row * original_seq_length + positions
            if int(flattened.max()) >= int(corrected.shape[0]):
                raise ValueError("Stage A GLM injection exceeds packed decoder input")
            corrected[:, 0].index_copy_(0, flattened, base + residual)
            causal_ids.append(codes)
            teacher_ids.append(batch["glm_ids"][row, :length].long())
            residuals.append(residual.float())
        causal = torch.cat(causal_ids)
        teacher = torch.cat(teacher_ids)
        residual = torch.cat(residuals)
        agreement = (causal == teacher).float().mean()
        residual_rms = residual.square().mean().sqrt()
        extension_count = causal.new_tensor(terminal_extensions).float()
        return corrected, causal, agreement, residual_rms, extension_count

    def _ctc_term(
        self,
        output: CausalWhisperOutput,
        batch: Mapping[str, torch.Tensor],
    ) -> tuple[LossTerm, torch.Tensor, torch.Tensor]:
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
        term = LossTerm(losses.sum(), target_lengths.sum().to(losses.dtype))
        positions = torch.arange(logits.shape[1], device=logits.device)[None, :]
        valid = positions < input_lengths[:, None]
        blank = (logits.argmax(dim=-1) == self.ctc_blank_id) & valid
        blank_ratio = blank.sum().float() / valid.sum().clamp_min(1)
        return term, blank_ratio, input_lengths.sum().float()

    @staticmethod
    def _teacher_kl(
        logits: torch.Tensor,
        batch: Mapping[str, torch.Tensor],
        anchor: torch.Tensor,
        *,
        original_seq_length: int,
    ) -> LossTerm:
        required = (
            "teacher_batch",
            "teacher_positions",
            "teacher_indices",
            "teacher_probabilities",
            "teacher_mask",
        )
        if any(name not in batch for name in required):
            return _zero_term(anchor)
        flat = batch["teacher_batch"].long() * original_seq_length + batch[
            "teacher_positions"
        ].long()
        indices = batch["teacher_indices"].long()
        probabilities = batch["teacher_probabilities"].float()
        mask = batch["teacher_mask"].bool()
        if not flat.numel():
            return _zero_term(anchor)
        selected = logits[flat].float().gather(1, indices)
        selected = selected.masked_fill(~mask, torch.finfo(selected.dtype).min)
        student_log = selected.log_softmax(dim=-1)
        teacher = probabilities.masked_fill(~mask, 0.0)
        teacher = teacher / teacher.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        values = (teacher * (teacher.clamp_min(1e-8).log() - student_log)).sum(dim=-1)
        active = mask.any(dim=-1)
        return _values_term(values, active)

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
        ctc, blank_ratio, input_frames = self._ctc_term(output, batch)

        if consistency_chunk_ms == chunk_ms:
            hidden_consistency = _zero_term(output.frame_hidden)
        else:
            with torch.no_grad():
                reference: CausalWhisperOutput = self.frontend(
                    waveform,
                    waveform_lengths,
                    chunk_ms=consistency_chunk_ms,
                )
            mask = stable_multichunk_mask(
                waveform_lengths,
                sequence_length=output.frame_hidden.shape[1],
                first_chunk_ms=chunk_ms,
                second_chunk_ms=consistency_chunk_ms,
            )
            values = (output.frame_hidden.float() - reference.frame_hidden.float()).square()
            expanded = mask[:, :, None].expand_as(values)
            hidden_consistency = _values_term(values.reshape(-1), expanded.reshape(-1))

        # The current-weight cached/full path is mathematically identical and
        # is enforced by the real-checkpoint external parity gate.  Keeping an
        # anchored zero term avoids doubling every formal step for a loss whose
        # correct value is exactly zero; any non-zero runtime mismatch blocks
        # training before this objective is authorized.
        cache_full = _zero_term(output.frame_hidden)
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
        )

    def compute(
        self,
        prepared: StageAPrepared,
        logits: torch.Tensor,
        labels: torch.Tensor,
        loss_mask: torch.Tensor,
        loss_kinds: torch.Tensor,
        batch: Mapping[str, torch.Tensor],
        *,
        original_seq_length: int,
    ) -> StageAObjectiveOutput:
        token_values = F.cross_entropy(
            logits.float(), labels.long(), reduction="none"
        )
        active = loss_mask > 0
        ar_mask = active & (
            (loss_kinds == LOSS_STREAMING_ASR)
            | (loss_kinds == LOSS_CAUSAL_FULL_ASR)
        )
        offline_mask = active & (loss_kinds == LOSS_OFFLINE_ASR_REPLAY)
        phase3_mask = active & (loss_kinds == LOSS_PHASE3_REPLAY)
        ar = _values_term(token_values, ar_mask)
        offline = _values_term(token_values, offline_mask)
        phase3 = _values_term(token_values, phase3_mask)
        teacher = self._teacher_kl(
            logits,
            batch,
            token_values,
            original_seq_length=original_seq_length,
        )
        terms = OrderedDict(
            (
                ("ar_asr", ar),
                ("source_ctc", prepared.ctc),
                ("offline_teacher_kl", teacher),
                ("hidden_chunk_consistency", prepared.hidden_chunk_consistency),
                ("cache_full_consistency", prepared.cache_full_consistency),
                ("offline_asr_replay", offline),
                ("phase3_replay", phase3),
            )
        )
        diagnostics = OrderedDict(
            (
                ("ctc_blank_ratio", prepared.ctc_blank_ratio),
                ("causal_glm_agreement", prepared.causal_glm_agreement),
                ("bridge_residual_rms", prepared.bridge_residual_rms),
                ("causal_glm_tokens", prepared.causal_glm_tokens),
                (
                    "causal_glm_terminal_extensions",
                    prepared.causal_glm_terminal_extensions,
                ),
                ("ctc_target_tokens", batch["ctc_lengths"].sum().detach().float()),
                ("ctc_input_frames", prepared.ctc_input_frames),
                ("ar_asr_tokens", ar.denominator.detach().float()),
                ("offline_asr_replay_tokens", offline.denominator.detach().float()),
                ("phase3_replay_tokens", phase3.denominator.detach().float()),
                ("disabled_acoustics", batch["disabled_acoustics"].sum().detach().float()),
            )
        )
        return StageAObjectiveOutput(terms, diagnostics, prepared.decoder_input)

    def forward(
        self,
        decoder_input: torch.Tensor,
        word_embedding_weight: torch.Tensor,
        logits: torch.Tensor,
        labels: torch.Tensor,
        loss_mask: torch.Tensor,
        loss_kinds: torch.Tensor,
        batch: Mapping[str, torch.Tensor],
        *,
        original_seq_length: int,
        chunk_ms: int,
        consistency_chunk_ms: int,
    ) -> StageAObjectiveOutput:
        prepared = self.prepare(
            decoder_input,
            word_embedding_weight,
            batch,
            original_seq_length=original_seq_length,
            chunk_ms=chunk_ms,
            consistency_chunk_ms=consistency_chunk_ms,
        )
        return self.compute(
            prepared,
            logits,
            labels,
            loss_mask,
            loss_kinds,
            batch,
            original_seq_length=original_seq_length,
        )


def distributed_stage_a_objective(
    output: StageAObjectiveOutput,
    *,
    weights: Mapping[str, float] = DEFAULT_WEIGHTS,
) -> tuple[torch.Tensor, OrderedDict[str, torch.Tensor]]:
    if tuple(output.terms) != TERM_NAMES or tuple(weights) != TERM_NAMES:
        raise ValueError("Stage A objective term order changed")
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
    "LossTerm",
    "StageAObjective",
    "StageAObjectiveOutput",
    "StageAPrepared",
    "TERM_NAMES",
    "chunk_pair_for_progress",
    "distributed_stage_a_objective",
    "stable_multichunk_mask",
]
