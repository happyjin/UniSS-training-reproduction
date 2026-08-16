"""Gradient-capable shared-causal WhisperVQ frontend for Stage A."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from transformers import WhisperFeatureExtractor

from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v1.stage00_baseline.shared_causal_frontend import (
    BLOCK_SAMPLES,
    FRAMES_PER_BLOCK,
    MEL_HOP,
    N_FFT,
    PCM_LEFT_CONTEXT,
    SAMPLE_RATE,
)
from uniss.speech_tokenizer.glm4.utils import load_quantize_encoder


FRAME_SAMPLES = SAMPLE_RATE * 20 // 1000
POOLING_FRAMES = 4


def block_padded_frame_lengths(waveform_lengths: torch.Tensor) -> torch.Tensor:
    """Frames that deployment evaluates, including the final padded block.

    The cached Stage 00 runtime pads a partial final 160-ms PCM block and runs
    all eight encoder frames before slicing the two 80-ms outputs.  Training
    must use the same attention geometry; masking at the raw PCM duration
    would change the final pooled hidden state for non-block-aligned audio.
    """

    if waveform_lengths.ndim != 1 or int(waveform_lengths.min()) <= 0:
        raise ValueError("waveform lengths must be a positive vector")
    return (
        torch.div(
            waveform_lengths + BLOCK_SAMPLES - 1,
            BLOCK_SAMPLES,
            rounding_mode="floor",
        )
        * FRAMES_PER_BLOCK
    )


def block_causal_allowed(
    valid_frames: torch.Tensor,
    *,
    sequence_length: int,
    block_frames: int,
) -> torch.Tensor:
    if valid_frames.ndim != 1 or sequence_length <= 0 or block_frames <= 0:
        raise ValueError("invalid block-causal attention geometry")
    positions = torch.arange(sequence_length, device=valid_frames.device)
    query_valid = positions[None, :] < valid_frames[:, None]
    key_valid = query_valid
    query_block = torch.div(positions[:, None], block_frames, rounding_mode="floor")
    key_block = torch.div(positions[None, :], block_frames, rounding_mode="floor")
    temporal = key_block <= query_block
    allowed = (
        temporal.unsqueeze(0)
        & query_valid[:, :, None]
        & key_valid[:, None, :]
    )
    # Avoid all-masked softmax rows for padding queries. Their outputs are
    # explicitly zeroed after every layer and never enter any loss.
    padding_query = ~query_valid
    allowed = allowed | (padding_query[:, :, None] & (positions[None, None, :] == 0))
    return allowed


@dataclass(frozen=True)
class CausalWhisperOutput:
    frame_hidden: torch.Tensor
    frame_lengths: torch.Tensor
    pooled_hidden: torch.Tensor
    pooled_lengths: torch.Tensor


class TrainableSharedCausalWhisperVQ(nn.Module):
    """Released WhisperVQ weights under the exact Stage 00 PCM frontend."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        self.model_path = str(Path(model_path).resolve())
        extractor = WhisperFeatureExtractor.from_pretrained(
            self.model_path,
            local_files_only=True,
        )
        if int(extractor.n_fft) != N_FFT or int(extractor.hop_length) != MEL_HOP:
            raise ValueError("Whisper feature geometry differs from Stage 00")
        self.encoder = load_quantize_encoder(self.model_path)
        pooling_position = int(self.encoder.config.pooling_position)
        if pooling_position != len(self.encoder.layers):
            raise ValueError("Stage A requires VQ pooling after the final pre-VQ layer")
        if int(self.encoder.config.pooling_kernel_size) != POOLING_FRAMES:
            raise ValueError("Stage A requires four-frame WhisperVQ pooling")
        if str(self.encoder.config.pooling_type) != "avg":
            raise ValueError("Stage A currently requires average WhisperVQ pooling")
        self.gradient_checkpointing = bool(gradient_checkpointing)
        filters = torch.as_tensor(extractor.mel_filters, dtype=torch.float32)
        if tuple(filters.shape) != (N_FFT // 2 + 1, self.encoder.conv1.in_channels):
            raise ValueError("Whisper mel filters differ from Stage 00")
        self.register_buffer("mel_filters", filters, persistent=False)
        self.register_buffer("stft_window", torch.hann_window(N_FFT), persistent=False)

        # Code identity and post-VQ geometry are immutable. The training path
        # does not call the historical EMA quantizer mutation code.
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        for layer in self.encoder.layers:
            for parameter in layer.parameters():
                parameter.requires_grad_(True)
        for module in (self.encoder.conv1, self.encoder.conv2):
            for parameter in module.parameters():
                parameter.requires_grad_(True)
        self.encoder.codebook.weight.requires_grad_(False)
        if self.encoder.pooling_layer is not None:
            for parameter in self.encoder.pooling_layer.parameters():
                parameter.requires_grad_(False)

    @property
    def hidden_size(self) -> int:
        return int(self.encoder.config.d_model)

    @property
    def codebook(self) -> torch.Tensor:
        return self.encoder.codebook.weight

    def tag_learning_rate_groups(self) -> None:
        pooling_position = int(self.encoder.config.pooling_position)
        top_start = max(0, pooling_position - 8)
        for parameter in self.encoder.conv1.parameters():
            parameter.uniss_stage_a_whisper_bottom = True
        for parameter in self.encoder.conv2.parameters():
            parameter.uniss_stage_a_whisper_bottom = True
        for index, layer in enumerate(self.encoder.layers[:pooling_position]):
            attribute = (
                "uniss_stage_a_whisper_top"
                if index >= top_start
                else "uniss_stage_a_whisper_bottom"
            )
            for parameter in layer.parameters():
                setattr(parameter, attribute, True)

    def _extract_convolved(
        self,
        waveform: torch.Tensor,
        waveform_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if waveform.ndim != 2 or waveform_lengths.shape != waveform.shape[:1]:
            raise ValueError("Stage A waveform must be [B,S] with [B] lengths")
        if int(waveform_lengths.min()) <= 0:
            raise ValueError("Stage A waveform lengths must be positive")
        maximum = int(waveform_lengths.max().item())
        blocks = math.ceil(maximum / BLOCK_SAMPLES)
        padded_samples = blocks * BLOCK_SAMPLES
        if waveform.shape[1] < padded_samples:
            waveform = F.pad(waveform, (0, padded_samples - waveform.shape[1]))
        else:
            waveform = waveform[:, :padded_samples]
        batch = waveform.shape[0]
        dtype = self.encoder.conv1.weight.dtype
        pcm_tail = torch.zeros(
            batch, PCM_LEFT_CONTEXT, dtype=torch.float32, device=waveform.device
        )
        mel_tail = torch.zeros(
            batch,
            self.encoder.conv1.in_channels,
            2,
            dtype=dtype,
            device=waveform.device,
        )
        conv1_tail = torch.zeros(
            batch,
            self.encoder.conv1.out_channels,
            2,
            dtype=dtype,
            device=waveform.device,
        )
        pieces: list[torch.Tensor] = []
        window = self.stft_window.to(device=waveform.device)
        filters = self.mel_filters.to(device=waveform.device)
        for block_index in range(blocks):
            current = waveform[
                :, block_index * BLOCK_SAMPLES : (block_index + 1) * BLOCK_SAMPLES
            ].float()
            analysis = torch.cat((pcm_tail, current), dim=-1)
            spectrum = torch.stft(
                analysis,
                N_FFT,
                MEL_HOP,
                window=window,
                center=False,
                return_complex=True,
            )
            magnitudes = spectrum.abs().square()
            mel = torch.einsum("fm,bft->bmt", filters, magnitudes)
            log_mel = mel.clamp_min(1e-10).log10()
            local_peak = log_mel.amax(dim=(1, 2), keepdim=True)
            log_mel = torch.maximum(log_mel, local_peak - 8.0)
            log_mel = ((log_mel + 4.0) / 4.0).to(dtype)
            pcm_tail = torch.cat((pcm_tail, current), dim=-1)[:, -PCM_LEFT_CONTEXT:]

            combined = torch.cat((mel_tail, log_mel), dim=-1)
            conv1 = F.conv1d(
                combined,
                self.encoder.conv1.weight,
                self.encoder.conv1.bias,
                stride=self.encoder.conv1.stride,
            )
            mel_tail = combined[:, :, -2:]
            conv1 = F.gelu(conv1)
            combined = torch.cat((conv1_tail, conv1), dim=-1)
            conv2 = F.conv1d(
                combined,
                self.encoder.conv2.weight,
                self.encoder.conv2.bias,
                stride=self.encoder.conv2.stride,
            )
            conv1_tail = combined[:, :, -2:]
            convolved = F.gelu(conv2).transpose(1, 2).contiguous()
            if int(convolved.shape[1]) != FRAMES_PER_BLOCK:
                raise AssertionError("one Stage A block must produce eight 20-ms frames")
            pieces.append(convolved)
        frame_lengths = torch.div(
            waveform_lengths + FRAME_SAMPLES - 1,
            FRAME_SAMPLES,
            rounding_mode="floor",
        ).clamp_max(blocks * FRAMES_PER_BLOCK)
        attention_frame_lengths = block_padded_frame_lengths(
            waveform_lengths
        ).clamp_max(blocks * FRAMES_PER_BLOCK)
        return torch.cat(pieces, dim=1), frame_lengths, attention_frame_lengths

    def _run_layer(
        self,
        layer: nn.Module,
        hidden: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        def forward(value: torch.Tensor, attention: torch.Tensor) -> torch.Tensor:
            return layer(
                value,
                attention,
                layer_head_mask=None,
                output_attentions=False,
            )[0]

        if self.gradient_checkpointing and self.training:
            return checkpoint(forward, hidden, mask, use_reentrant=False)
        return forward(hidden, mask)

    def forward(
        self,
        waveform: torch.Tensor,
        waveform_lengths: torch.Tensor,
        *,
        chunk_ms: int,
    ) -> CausalWhisperOutput:
        if chunk_ms <= 0 or chunk_ms % 160:
            raise ValueError("Stage A chunk must be a positive multiple of 160 ms")
        convolved, frame_lengths, attention_frame_lengths = self._extract_convolved(
            waveform, waveform_lengths
        )
        sequence = int(convolved.shape[1])
        if sequence > int(self.encoder.embed_positions.num_embeddings):
            raise ValueError("Stage A utterance exceeds Whisper absolute positions")
        positions = self.encoder.embed_positions.weight[:sequence].to(
            device=convolved.device,
            dtype=convolved.dtype,
        )
        hidden = convolved + positions
        allowed = block_causal_allowed(
            attention_frame_lengths,
            sequence_length=sequence,
            block_frames=chunk_ms // 20,
        )
        mask = torch.zeros_like(allowed, dtype=hidden.dtype).masked_fill(
            ~allowed,
            torch.finfo(hidden.dtype).min,
        ).unsqueeze(1)
        valid = (
            torch.arange(sequence, device=hidden.device)[None, :]
            < attention_frame_lengths[:, None]
        )
        for layer in self.encoder.layers[: int(self.encoder.config.pooling_position)]:
            hidden = self._run_layer(layer, hidden, mask)
            hidden = hidden.masked_fill(~valid[:, :, None], 0.0)
        pooled = F.avg_pool1d(
            hidden.transpose(1, 2),
            kernel_size=POOLING_FRAMES,
            stride=POOLING_FRAMES,
            ceil_mode=True,
        ).transpose(1, 2).contiguous()
        pooled_lengths = torch.div(
            frame_lengths + POOLING_FRAMES - 1,
            POOLING_FRAMES,
            rounding_mode="floor",
        )
        pooled_valid = (
            torch.arange(pooled.shape[1], device=pooled.device)[None, :]
            < pooled_lengths[:, None]
        )
        pooled = pooled.masked_fill(~pooled_valid[:, :, None], 0.0)
        return CausalWhisperOutput(hidden, frame_lengths, pooled, pooled_lengths)


__all__ = [
    "CausalWhisperOutput",
    "TrainableSharedCausalWhisperVQ",
    "block_causal_allowed",
]
