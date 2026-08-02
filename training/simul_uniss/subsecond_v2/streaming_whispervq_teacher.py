"""Frozen WhisperVQ clone with bounded chunk-causal attention and hidden export."""

from __future__ import annotations

import types
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
import torchaudio
from transformers import WhisperFeatureExtractor

from uniss.speech_tokenizer.glm4.utils import load_quantize_encoder


@dataclass(frozen=True)
class StreamingTeacherOutput:
    tokens: torch.Tensor
    pre_vq_hidden: torch.Tensor


def chunk_right_attention_mask(
    attention_mask: torch.Tensor,
    *,
    chunk_frames: int,
    right_context_frames: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Allow all history plus the current chunk and bounded right context."""

    if chunk_frames <= 0 or right_context_frames < 0:
        raise ValueError("invalid chunk/right-context geometry")
    _, sequence_length = attention_mask.shape
    queries = torch.arange(sequence_length, device=attention_mask.device).reshape(-1, 1)
    keys = torch.arange(sequence_length, device=attention_mask.device).reshape(1, -1)
    chunk_end = (
        torch.div(queries, chunk_frames, rounding_mode="floor") + 1
    ) * chunk_frames - 1
    allowed = keys <= chunk_end + right_context_frames
    allowed = allowed.unsqueeze(0) & attention_mask[:, None, :].bool()
    additive = (~allowed).to(dtype) * torch.finfo(dtype).min
    return additive.unsqueeze(1)


class StreamingWhisperVQTeacher:
    """Read-only WhisperVQ weights under a streaming-compatible attention mask."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str = "cuda:0",
        chunk_ms: int = 160,
        right_context_ms: int = 80,
    ) -> None:
        if chunk_ms % 20 or right_context_ms % 20:
            raise ValueError("WhisperVQ chunk geometry must be a multiple of 20 ms")
        self.device = torch.device(device)
        self.chunk_ms = int(chunk_ms)
        self.right_context_ms = int(right_context_ms)
        self.pre_pool_chunk_frames = chunk_ms // 20
        self.pre_pool_right_frames = right_context_ms // 20
        self.model = load_quantize_encoder(str(model_path)).to(self.device).eval()
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(str(model_path))
        self.model.config.quantize_causal_block_size = self.pre_pool_chunk_frames
        self.model._streaming_pre_pool_chunk_frames = self.pre_pool_chunk_frames
        self.model._streaming_pre_pool_right_frames = self.pre_pool_right_frames

        def bounded_mask(encoder, attention_mask, block_size=50):
            scale = encoder._streaming_pre_pool_chunk_frames / max(1, block_size)
            right = round(encoder._streaming_pre_pool_right_frames / scale)
            return chunk_right_attention_mask(
                attention_mask,
                chunk_frames=block_size,
                right_context_frames=right,
                dtype=encoder.dtype,
            )

        self.model.get_block_causal_attention_mask = types.MethodType(
            bounded_mask, self.model
        )

    @staticmethod
    def _audio_tuple(
        value: str | Path | torch.Tensor | tuple[torch.Tensor, int],
    ) -> tuple[torch.Tensor, int]:
        if isinstance(value, torch.Tensor):
            waveform, sample_rate = value, 16_000
        elif isinstance(value, tuple):
            waveform, sample_rate = value
        else:
            waveform, sample_rate = torchaudio.load(str(value))
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        waveform = waveform[:1]
        if sample_rate != 16_000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16_000)
        return waveform.cpu(), 16_000

    @torch.inference_mode()
    def encode(
        self,
        audio: Sequence[str | Path | torch.Tensor | tuple[torch.Tensor, int]],
    ) -> list[StreamingTeacherOutput]:
        if not audio:
            return []
        prepared = [self._audio_tuple(value) for value in audio]
        arrays = [waveform[0].numpy() for waveform, _ in prepared]
        pooling = self.model.config.pooling_kernel_size or 1
        stride = (
            self.model.conv1.stride[0]
            * self.model.conv2.stride[0]
            * pooling
            * self.feature_extractor.hop_length
        )
        features = self.feature_extractor(
            arrays,
            sampling_rate=16_000,
            return_attention_mask=True,
            return_tensors="pt",
            padding="longest",
            pad_to_multiple_of=stride,
        ).to(self.device)
        captured: list[torch.Tensor] = []

        def capture_pooling(_module, _inputs, output):
            captured.append(output.detach())

        handle = self.model.pooling_layer.register_forward_hook(capture_pooling)
        try:
            outputs = self.model(**features)
        finally:
            handle.remove()
        if len(captured) != 1:
            raise RuntimeError(f"expected one pooling capture, found {len(captured)}")
        pre_vq = captured[0].permute(0, 2, 1).contiguous()
        tokens = outputs.quantized_token_ids
        mask = features.attention_mask[:, :: self.model.conv1.stride[0] * self.model.conv2.stride[0]]
        mask = mask[:, ::pooling].bool()
        if mask.shape != tokens.shape or pre_vq.shape[:2] != tokens.shape:
            raise RuntimeError(
                f"streaming teacher shape mismatch: hidden={tuple(pre_vq.shape)}, "
                f"tokens={tuple(tokens.shape)}, mask={tuple(mask.shape)}"
            )
        return [
            StreamingTeacherOutput(
                tokens=tokens[index][mask[index]].detach().cpu(),
                pre_vq_hidden=pre_vq[index][mask[index]].detach().cpu(),
            )
            for index in range(len(audio))
        ]
