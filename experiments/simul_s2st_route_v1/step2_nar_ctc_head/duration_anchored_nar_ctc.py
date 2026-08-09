"""Duration-anchored causal NAR CTC head for BiCodec semantic generation.

Replaces the V6 ``NARBiCodecCTC`` geometry:

* Frame budget comes from source audio duration (``frames_per_second``), not
  ``text_length * upsample_ratio``. BiCodec is a fixed 50 Hz codec; Step 2a
  measured that frames-per-source-second is 2.2× tighter than frames-per-text-token.
* Both the text-to-unit encoder and the unit decoder are causal and padding-aware.
  The V6 head left the T2U encoder bidirectional and padding-blind, which is
  disqualifying for streaming.
* Text encodings are linearly interpolated onto the duration frame timeline so each
  sample can have its own frame length without a shared integer upsample ratio.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def causal_mask(length: int, device: torch.device) -> torch.Tensor:
    return torch.triu(
        torch.ones(length, length, dtype=torch.bool, device=device), diagonal=1
    )


def adjacent_repeats(tokens: torch.Tensor) -> torch.Tensor:
    """Count CTC-required extra frames from consecutive identical labels."""

    if tokens.ndim != 1:
        raise ValueError("tokens must be a 1-D sequence")
    if tokens.numel() <= 1:
        return tokens.new_zeros((), dtype=torch.long)
    return (tokens[1:] == tokens[:-1]).sum()


def required_ctc_frames(unit_length: int, repeats: int) -> int:
    return int(unit_length) + int(repeats)


def duration_frame_lengths(
    duration_ms: torch.Tensor,
    *,
    frames_per_second: float,
    unit_lengths: torch.Tensor | None = None,
    unit_repeats: torch.Tensor | None = None,
    min_frames: int = 1,
    max_frames: int | None = None,
) -> torch.Tensor:
    """Ceil(duration_s * fps), then raise to the CTC feasibility floor."""

    if duration_ms.ndim != 1:
        raise ValueError("duration_ms must be [B]")
    if frames_per_second <= 0:
        raise ValueError("frames_per_second must be positive")
    frames = torch.ceil(duration_ms.float() * (frames_per_second / 1000.0)).long()
    frames = frames.clamp_min(min_frames)
    if unit_lengths is not None:
        if unit_repeats is None:
            raise ValueError("unit_repeats required with unit_lengths")
        frames = torch.maximum(frames, unit_lengths + unit_repeats)
    if max_frames is not None:
        frames = frames.clamp_max(int(max_frames))
    return frames


def expand_text_to_frames(
    encoded: torch.Tensor,
    text_lengths: torch.Tensor,
    frame_lengths: torch.Tensor,
) -> torch.Tensor:
    """Linearly interpolate each sample's text encodings onto its frame timeline."""

    if encoded.ndim != 3:
        raise ValueError("encoded must be [B, T, H]")
    if text_lengths.shape != encoded.shape[:1] or frame_lengths.shape != encoded.shape[:1]:
        raise ValueError("length tensors must match batch")
    batch, _, hidden = encoded.shape
    maximum = int(frame_lengths.max().item()) if batch else 0
    output = encoded.new_zeros(batch, maximum, hidden)
    for row in range(batch):
        text = int(text_lengths[row].item())
        frames = int(frame_lengths[row].item())
        if text <= 0 or frames <= 0:
            continue
        source = encoded[row, :text]
        if text == 1 or frames == 1:
            output[row, :frames] = source[:1].expand(frames, -1)
            continue
        positions = torch.linspace(0, text - 1, frames, device=encoded.device)
        left = positions.long()
        right = (left + 1).clamp_max(text - 1)
        weight = (positions - left.float()).unsqueeze(-1)
        output[row, :frames] = (1.0 - weight) * source[left] + weight * source[right]
    return output


class DurationAnchoredCausalNARCTC(nn.Module):
    """Causal text → duration-frame → BiCodec-unit CTC head."""

    def __init__(
        self,
        *,
        qwen_hidden_size: int = 896,
        model_size: int = 512,
        semantic_vocab_size: int = 8192,
        frames_per_second: float = 75.0,
        num_heads: int = 8,
        t2u_layers: int = 2,
        decoder_layers: int = 2,
        dropout: float = 0.1,
        max_frames: int = 1500,
    ) -> None:
        super().__init__()
        if frames_per_second <= 0:
            raise ValueError("frames_per_second must be positive")
        if max_frames <= 0:
            raise ValueError("max_frames must be positive")
        if model_size % num_heads:
            raise ValueError("model_size must be divisible by num_heads")
        self.frames_per_second = float(frames_per_second)
        self.max_frames = int(max_frames)
        self.blank_id = int(semantic_vocab_size)
        self.speaker_vocab_size = int(semantic_vocab_size)
        # WhisperVQ / GLM discrete source codes used in Phase3 joint manifests.
        self.source_glm_vocab_size = 16384
        self.input_projection = nn.Linear(qwen_hidden_size, model_size)
        # BiCodec global speaker tokens (same code space as semantic units).
        self.speaker_embed = nn.Embedding(self.speaker_vocab_size, model_size)
        self.speaker_proj = nn.Linear(model_size, model_size)
        self.source_glm_embed = nn.Embedding(self.source_glm_vocab_size, model_size)
        self.source_glm_proj = nn.Linear(model_size, model_size)
        encoder_layer = nn.TransformerEncoderLayer(
            model_size,
            num_heads,
            dim_feedforward=4 * model_size,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.t2u_encoder = nn.TransformerEncoder(encoder_layer, num_layers=t2u_layers)
        decoder_layer = nn.TransformerEncoderLayer(
            model_size,
            num_heads,
            dim_feedforward=4 * model_size,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.unit_decoder = nn.TransformerEncoder(decoder_layer, num_layers=decoder_layers)
        self.output = nn.Linear(model_size, semantic_vocab_size + 1)

    def frame_lengths_for(
        self,
        duration_ms: torch.Tensor,
        *,
        unit_lengths: torch.Tensor | None = None,
        unit_repeats: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return duration_frame_lengths(
            duration_ms,
            frames_per_second=self.frames_per_second,
            unit_lengths=unit_lengths,
            unit_repeats=unit_repeats,
            max_frames=self.max_frames,
        )

    def _speaker_condition(
        self,
        speaker_ids: torch.Tensor | None,
        speaker_lengths: torch.Tensor | None,
        *,
        batch: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        if speaker_ids is None:
            return None
        if speaker_ids.ndim != 2 or speaker_ids.shape[0] != batch:
            raise ValueError("speaker_ids must be [B,S]")
        if speaker_lengths is None:
            speaker_lengths = torch.full(
                (batch,), speaker_ids.shape[1], dtype=torch.long, device=device
            )
        # Clamp out-of-range pads; real codes are in [0, vocab).
        safe = speaker_ids.clamp(0, self.speaker_vocab_size - 1)
        embedded = self.speaker_embed(safe)
        positions = torch.arange(safe.shape[1], device=device)
        mask = positions[None, :] < speaker_lengths.to(device)[:, None]
        denom = mask.float().sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = (embedded * mask.unsqueeze(-1).float()).sum(dim=1) / denom
        return self.speaker_proj(pooled)

    def _source_glm_frames(
        self,
        source_glm: torch.Tensor | None,
        source_glm_lengths: torch.Tensor | None,
        frame_lengths: torch.Tensor,
    ) -> torch.Tensor | None:
        """Interpolate discrete source GLM codes onto the NAR frame grid."""

        if source_glm is None:
            return None
        if source_glm.ndim != 2 or source_glm.shape[0] != frame_lengths.shape[0]:
            raise ValueError("source_glm must be [B,S]")
        if source_glm_lengths is None:
            source_glm_lengths = torch.full(
                (source_glm.shape[0],),
                source_glm.shape[1],
                dtype=torch.long,
                device=source_glm.device,
            )
        safe = source_glm.clamp(0, self.source_glm_vocab_size - 1)
        embedded = self.source_glm_embed(safe)
        positions = torch.arange(safe.shape[1], device=safe.device)
        padding = positions[None, :] >= source_glm_lengths.to(safe.device)[:, None]
        embedded = embedded.masked_fill(padding.unsqueeze(-1), 0.0)
        expanded = expand_text_to_frames(embedded, source_glm_lengths, frame_lengths)
        return self.source_glm_proj(expanded)

    def forward(
        self,
        text_hidden: torch.Tensor,
        text_lengths: torch.Tensor,
        duration_ms: torch.Tensor,
        *,
        unit_lengths: torch.Tensor | None = None,
        unit_repeats: torch.Tensor | None = None,
        speaker_ids: torch.Tensor | None = None,
        speaker_lengths: torch.Tensor | None = None,
        source_glm: torch.Tensor | None = None,
        source_glm_lengths: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if text_hidden.ndim != 3 or text_lengths.shape != text_hidden.shape[:1]:
            raise ValueError("invalid text hidden/length geometry")
        if duration_ms.shape != text_hidden.shape[:1]:
            raise ValueError("duration_ms must match batch")
        frame_lengths = self.frame_lengths_for(
            duration_ms, unit_lengths=unit_lengths, unit_repeats=unit_repeats
        )
        projected = self.input_projection(text_hidden)
        text_positions = torch.arange(projected.shape[1], device=projected.device)
        text_padding = text_positions[None, :] >= text_lengths[:, None]
        encoded = self.t2u_encoder(
            projected,
            mask=causal_mask(projected.shape[1], projected.device),
            src_key_padding_mask=text_padding,
        )
        # Zero padded text positions before interpolation so they cannot leak into frames.
        encoded = encoded.masked_fill(text_padding.unsqueeze(-1), 0.0)
        expanded = expand_text_to_frames(encoded, text_lengths, frame_lengths)
        speaker = self._speaker_condition(
            speaker_ids,
            speaker_lengths,
            batch=expanded.shape[0],
            device=expanded.device,
        )
        if speaker is not None:
            expanded = expanded + speaker[:, None, :]
        glm_frames = self._source_glm_frames(source_glm, source_glm_lengths, frame_lengths)
        if glm_frames is not None:
            expanded = expanded + glm_frames
        frame_positions = torch.arange(expanded.shape[1], device=expanded.device)
        frame_padding = frame_positions[None, :] >= frame_lengths[:, None]
        decoded = self.unit_decoder(
            expanded,
            mask=causal_mask(expanded.shape[1], expanded.device),
            src_key_padding_mask=frame_padding,
        )
        return self.output(decoded), frame_lengths

    @torch.no_grad()
    def greedy_decode(self, logits: torch.Tensor, lengths: torch.Tensor) -> list[list[int]]:
        outputs: list[list[int]] = []
        for row, length in enumerate(lengths.tolist()):
            ids = logits[row, : int(length)].argmax(dim=-1).tolist()
            collapsed: list[int] = []
            previous = None
            for value in ids:
                if value != previous and value != self.blank_id:
                    collapsed.append(int(value))
                previous = value
            outputs.append(collapsed)
        return outputs
