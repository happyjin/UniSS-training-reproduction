"""Trainable original WhisperVQ frontend with GPU log-Mel and multi-chunk masks."""

from __future__ import annotations

import math
import types
from functools import partial
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from transformers import WhisperFeatureExtractor

from uniss.speech_tokenizer.glm4.utils import load_quantize_encoder

from .config import MultiChunkConfig
from .whisper_multichunk import additive_attention_mask, chunk_causal_allowed


class WhisperGPUFeatureExtractor(nn.Module):
    """Torch implementation matching Transformers Whisper preprocessing."""

    def __init__(self, model_path: str | Path) -> None:
        super().__init__()
        extractor = WhisperFeatureExtractor.from_pretrained(
            str(model_path), local_files_only=True
        )
        self.n_fft = int(extractor.n_fft)
        self.hop_length = int(extractor.hop_length)
        self.register_buffer(
            "mel_filters", torch.from_numpy(extractor.mel_filters).float(), persistent=False
        )
        self.register_buffer(
            "window", torch.hann_window(self.n_fft), persistent=False
        )

    def forward(
        self,
        waveform: torch.Tensor,
        waveform_lengths: torch.Tensor,
        *,
        pad_to_multiple_of: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if waveform.ndim != 2 or waveform_lengths.shape != waveform.shape[:1]:
            raise ValueError("waveform must be [B,S] and lengths [B]")
        if pad_to_multiple_of <= 0:
            raise ValueError("pad_to_multiple_of must be positive")
        maximum = max(self.n_fft, int(waveform_lengths.max().item()))
        padded_samples = math.ceil(maximum / pad_to_multiple_of) * pad_to_multiple_of
        if waveform.shape[1] < padded_samples:
            waveform = F.pad(waveform, (0, padded_samples - waveform.shape[1]))
        elif waveform.shape[1] > padded_samples:
            waveform = waveform[:, :padded_samples]
        stft = torch.stft(
            waveform.float(),
            self.n_fft,
            self.hop_length,
            window=self.window.to(waveform.device),
            return_complex=True,
        )
        magnitudes = stft[..., :-1].abs().square()
        mel = self.mel_filters.to(waveform.device).t() @ magnitudes
        log_spec = mel.clamp_min(1e-10).log10()
        maximum_value = log_spec.amax(dim=(1, 2), keepdim=True)
        log_spec = torch.maximum(log_spec, maximum_value - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        feature_lengths = torch.div(
            waveform_lengths + self.hop_length - 1,
            self.hop_length,
            rounding_mode="floor",
        ).clamp_max(log_spec.shape[-1])
        positions = torch.arange(log_spec.shape[-1], device=waveform.device)
        attention_mask = positions[None, :] < feature_lengths[:, None]
        return log_spec, attention_mask.long()


@dataclass
class WhisperJointOutput:
    pre_vq_hidden: torch.Tensor
    token_lengths: torch.Tensor
    quantized_token_ids: torch.Tensor
    quantize_loss: torch.Tensor


class TrainableMultiChunkWhisperVQ(nn.Module):
    """The unchanged Phase3 WhisperVQ weights under a selectable chunk mask."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        chunk_config: MultiChunkConfig | None = None,
    ) -> None:
        super().__init__()
        self.model_path = str(Path(model_path).resolve())
        self.chunk_config = chunk_config or MultiChunkConfig()
        self.feature_extractor = WhisperGPUFeatureExtractor(model_path)
        self.encoder = load_quantize_encoder(str(model_path))
        if self.encoder.pooling_layer is None or self.encoder.config.pooling_kernel_size is None:
            raise ValueError("Phase3 WhisperVQ must contain its historical pooling layer")
        self.encoder.codebook.weight.requires_grad_(False)
        self._pre_pool_chunk_frames: int | None = None
        self._pre_pool_right_frames = self.chunk_config.right_context_frames

        def bounded_mask(encoder, attention_mask, block_size=50):
            if self._pre_pool_chunk_frames is None:
                valid = attention_mask.sum(dim=1)
                allowed = chunk_causal_allowed(
                    valid,
                    sequence_length=attention_mask.shape[1],
                    chunk_frames=None,
                    right_context_frames=0,
                )
            else:
                scale = self._pre_pool_chunk_frames / max(1, int(block_size))
                right = round(self._pre_pool_right_frames / scale)
                valid = attention_mask.sum(dim=1)
                allowed = chunk_causal_allowed(
                    valid,
                    sequence_length=attention_mask.shape[1],
                    chunk_frames=int(block_size),
                    right_context_frames=right,
                )
            return additive_attention_mask(allowed, encoder.dtype)

        self.encoder.get_block_causal_attention_mask = types.MethodType(
            bounded_mask, self.encoder
        )

    @property
    def hidden_size(self) -> int:
        return int(self.encoder.config.d_model)

    @property
    def codebook(self) -> torch.Tensor:
        return self.encoder.codebook.weight

    def set_chunk(self, chunk_ms: int | None) -> None:
        frames = self.chunk_config.frames(chunk_ms)
        self._pre_pool_chunk_frames = frames
        self.encoder.config.quantize_causal_block_size = frames

    def configure_gradient_checkpointing(self, enabled: bool) -> None:
        self.encoder.gradient_checkpointing = bool(enabled)
        if enabled:
            # ``load_quantize_encoder`` constructs the bare WhisperVQEncoder
            # instead of its Hugging Face PreTrainedModel parent.  The parent
            # normally injects this callable when
            # ``gradient_checkpointing_enable`` is used; keep that behavior
            # local to the new trainable wrapper so historical inference code
            # remains untouched.
            self.encoder._gradient_checkpointing_func = partial(
                checkpoint, use_reentrant=True
            )

    def forward(
        self,
        waveform: torch.Tensor,
        waveform_lengths: torch.Tensor,
        *,
        chunk_ms: int | None,
    ) -> WhisperJointOutput:
        self.set_chunk(chunk_ms)
        pooling = int(self.encoder.config.pooling_kernel_size)
        sample_multiple = (
            int(self.encoder.conv1.stride[0])
            * int(self.encoder.conv2.stride[0])
            * pooling
            * self.feature_extractor.hop_length
        )
        features, attention = self.feature_extractor(
            waveform, waveform_lengths, pad_to_multiple_of=sample_multiple
        )
        captures: list[torch.Tensor] = []

        def capture_pooling(_module, _inputs, output):
            captures.append(output.permute(0, 2, 1).contiguous())

        handle = self.encoder.pooling_layer.register_forward_hook(capture_pooling)
        try:
            output = self.encoder(
                input_features=features,
                attention_mask=attention,
                return_dict=True,
            )
        finally:
            handle.remove()
        if len(captures) != 1:
            raise RuntimeError(f"expected one pre-VQ pooling capture, got {len(captures)}")
        token_mask = attention[:, :: int(self.encoder.conv1.stride[0])]
        token_mask = token_mask[:, :: int(self.encoder.conv2.stride[0])]
        token_mask = token_mask[:, ::pooling]
        token_lengths = token_mask.sum(dim=1).long()
        if captures[0].shape[:2] != output.quantized_token_ids.shape:
            raise RuntimeError("Whisper pre-VQ/token geometry mismatch")
        quantize_loss = self.encoder.quantize_loss
        if quantize_loss is None:
            quantize_loss = captures[0].sum() * 0.0
        return WhisperJointOutput(
            pre_vq_hidden=captures[0],
            token_lengths=token_lengths,
            quantized_token_ids=output.quantized_token_ids,
            quantize_loss=quantize_loss,
        )

    def tag_learning_rate_groups(self) -> None:
        """Mark params for the isolated Megatron optimizer override."""

        for parameter in self.encoder.parameters():
            if parameter.requires_grad:
                parameter.uniss_lr_whisper_bottom = True
        halfway = len(self.encoder.layers) // 2
        for layer in self.encoder.layers[halfway:]:
            for parameter in layer.parameters():
                if parameter.requires_grad:
                    parameter.uniss_lr_whisper_bottom = False
                    parameter.uniss_lr_whisper_top = True
