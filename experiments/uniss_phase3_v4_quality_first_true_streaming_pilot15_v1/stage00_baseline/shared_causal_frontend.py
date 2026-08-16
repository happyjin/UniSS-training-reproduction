"""One causal PCM frontend shared by Stage A data generation and deployment.

The historical offline Whisper feature extractor uses centered STFT windows
and utterance-global peak normalization. Those operations cannot be reproduced
when PCM arrives online. This module therefore defines the single authoritative
Stage A frontend:

* 16 kHz mono PCM arrives in 160 ms blocks;
* STFT uses ``center=False`` and a 240-sample causal waveform tail;
* log-mel normalization uses only the current arrived block;
* both released causal convolutions keep their exact two-frame left states;
* the first 16 WhisperVQ layers use 160 ms block-causal attention;
* encoder K/V is reset before the absolute-position table would overflow.

``extract_convolved`` is gradient-capable and is the training-data/training
entry point. ``forward_full_reference`` and ``push`` consume those exact same
features, providing independent full block-mask and cached execution paths.
No historical frontend file is modified.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from experiments.uniss_phase3_runtime_parity_streaming_v2.frontend.cached_whispervq import (
    CachedBlockCausalWhisperVQ,
    CachedWhisperVQOutput,
    CachedWhisperVQState,
)


SAMPLE_RATE = 16_000
BLOCK_MS = 160
BLOCK_SAMPLES = SAMPLE_RATE * BLOCK_MS // 1000
TOKEN_HOP_MS = 80
TOKEN_HOP_SAMPLES = SAMPLE_RATE * TOKEN_HOP_MS // 1000
N_FFT = 400
MEL_HOP = 160
PCM_LEFT_CONTEXT = N_FFT - MEL_HOP
FRAMES_PER_BLOCK = BLOCK_MS // 20
TOKENS_PER_BLOCK = BLOCK_MS // TOKEN_HOP_MS
FRONTEND_SCHEMA = "uniss_shared_causal_whispervq_v1"


@dataclass(frozen=True)
class CausalAcousticState:
    """State required before the Transformer encoder."""

    pcm_tail: torch.Tensor
    mel_tail: torch.Tensor
    conv1_tail: torch.Tensor


@dataclass(frozen=True)
class SharedFrontendState:
    """Append-only state for one streaming utterance."""

    acoustic: CausalAcousticState
    encoder: CachedWhisperVQState | None
    samples_seen: int
    blocks_seen: int
    encoder_resets: int
    finalized: bool = False


@dataclass(frozen=True)
class AcousticBlock:
    """One padded 160 ms acoustic block and its valid token count."""

    convolved_hidden: torch.Tensor
    valid_samples: int
    valid_tokens: int
    state: CausalAcousticState


@dataclass(frozen=True)
class SharedFrontendOutput:
    """New committed WhisperVQ values produced by one PCM push."""

    source_end_ms: int
    pre_vq_hidden: torch.Tensor
    quantized_hidden: torch.Tensor
    token_ids: torch.Tensor
    state: SharedFrontendState
    encoder_reset_before_block: bool
    is_final: bool


@dataclass(frozen=True)
class FullReferenceOutput:
    """Full block-mask reference over one or more position-table segments."""

    pre_vq_hidden: torch.Tensor
    quantized_hidden: torch.Tensor
    token_ids: torch.Tensor
    convolved_hidden: torch.Tensor
    valid_tokens: int
    encoder_segments: int


class SharedCausalWhisperVQFrontend(nn.Module):
    """Authoritative causal frontend for both training and online inference."""

    def __init__(
        self,
        whisper_encoder: nn.Module,
        mel_filters: np.ndarray | torch.Tensor,
        *,
        device: str | torch.device,
    ) -> None:
        super().__init__()
        self.encoder_model = whisper_encoder
        self.device = torch.device(device)
        self.encoder_model.to(self.device)
        self.cached_encoder = CachedBlockCausalWhisperVQ.from_whispervq_encoder(
            self.encoder_model,
            chunk_ms=BLOCK_MS,
            frame_ms=20,
            right_context_ms=0,
        ).to(self.device)
        filters = torch.as_tensor(mel_filters, dtype=torch.float32)
        if filters.ndim != 2 or filters.shape[0] != N_FFT // 2 + 1:
            raise ValueError(
                f"mel_filters must be [{N_FFT // 2 + 1}, mel_bins], "
                f"got {tuple(filters.shape)}"
            )
        if int(filters.shape[1]) != int(self.encoder_model.conv1.in_channels):
            raise ValueError("mel filter count differs from WhisperVQ conv1 input")
        self.register_buffer(
            "mel_filters", filters.to(self.device), persistent=False
        )
        self.register_buffer(
            "stft_window",
            torch.hann_window(N_FFT, dtype=torch.float32, device=self.device),
            persistent=False,
        )
        self.max_blocks_per_encoder_segment = (
            int(self.cached_encoder.position_embedding.num_embeddings) // FRAMES_PER_BLOCK
        )
        if self.max_blocks_per_encoder_segment <= 0:
            raise ValueError("WhisperVQ position table cannot hold one causal block")

    @property
    def model_dtype(self) -> torch.dtype:
        return self.encoder_model.conv1.weight.dtype

    @property
    def maximum_segment_ms(self) -> int:
        return self.max_blocks_per_encoder_segment * BLOCK_MS

    def initial_acoustic_state(self, batch_size: int = 1) -> CausalAcousticState:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        hidden = int(self.encoder_model.conv1.out_channels)
        mel = int(self.encoder_model.conv1.in_channels)
        return CausalAcousticState(
            pcm_tail=torch.zeros(batch_size, PCM_LEFT_CONTEXT, device=self.device),
            mel_tail=torch.zeros(batch_size, mel, 2, device=self.device, dtype=self.model_dtype),
            conv1_tail=torch.zeros(
                batch_size, hidden, 2, device=self.device, dtype=self.model_dtype
            ),
        )

    def initial_state(self) -> SharedFrontendState:
        return SharedFrontendState(
            acoustic=self.initial_acoustic_state(),
            encoder=None,
            samples_seen=0,
            blocks_seen=0,
            encoder_resets=0,
            finalized=False,
        )

    def _pcm_tensor(
        self, pcm: Sequence[float] | np.ndarray | torch.Tensor
    ) -> torch.Tensor:
        value = torch.as_tensor(pcm, dtype=torch.float32, device=self.device)
        if value.ndim == 1:
            value = value.unsqueeze(0)
        if value.ndim != 2:
            raise ValueError("PCM must have shape [samples] or [batch, samples]")
        if not 0 < int(value.shape[-1]) <= BLOCK_SAMPLES:
            raise ValueError(f"PCM block must contain 1..{BLOCK_SAMPLES} samples")
        if not bool(torch.isfinite(value).all()):
            raise ValueError("PCM block contains non-finite samples")
        return value

    def _causal_log_mel(
        self, block: torch.Tensor, pcm_tail: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.stft_window.device != block.device or self.mel_filters.device != block.device:
            raise RuntimeError("causal frontend buffers and PCM are on different devices")
        actual = int(block.shape[-1])
        padded = F.pad(block, (0, BLOCK_SAMPLES - actual))
        analysis = torch.cat((pcm_tail.to(block.dtype), padded), dim=-1)
        spectrum = torch.stft(
            analysis,
            N_FFT,
            MEL_HOP,
            window=self.stft_window,
            center=False,
            return_complex=True,
        )
        magnitudes = spectrum.abs().square()
        if int(magnitudes.shape[-1]) != BLOCK_MS // 10:
            raise AssertionError("one 160 ms block must produce sixteen mel frames")
        mel = torch.einsum("fm,bft->bmt", self.mel_filters, magnitudes)
        log_mel = torch.clamp(mel, min=1e-10).log10()
        # This block-local maximum is available at the end of the same 160 ms
        # block. It introduces no cross-block future dependency and is shared
        # exactly by training extraction, full reference, and cached runtime.
        local_peak = log_mel.amax(dim=(1, 2), keepdim=True)
        log_mel = torch.maximum(log_mel, local_peak - 8.0)
        log_mel = (log_mel + 4.0) / 4.0
        next_tail = torch.cat((pcm_tail, block), dim=-1)[:, -PCM_LEFT_CONTEXT:]
        return log_mel, next_tail

    @staticmethod
    def _causal_conv_block(
        module: nn.Module, current: torch.Tensor, tail: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        combined = torch.cat((tail.to(dtype=current.dtype), current), dim=-1)
        output = F.conv1d(
            combined,
            module.weight,
            module.bias,
            stride=module.stride,
            dilation=module.dilation,
            groups=module.groups,
        )
        return output, combined[:, :, -2:]

    def extract_block(
        self,
        pcm: Sequence[float] | np.ndarray | torch.Tensor,
        state: CausalAcousticState | None = None,
        *,
        is_final: bool = False,
        detach_state: bool = False,
    ) -> AcousticBlock:
        """Extract one block; gradients can flow through Whisper convolutions."""

        state = state or self.initial_acoustic_state()
        block = self._pcm_tensor(pcm)
        if int(block.shape[0]) != int(state.pcm_tail.shape[0]):
            raise ValueError("PCM batch size differs from acoustic state")
        if int(block.shape[-1]) != BLOCK_SAMPLES and not is_final:
            raise ValueError("a partial PCM block is only legal at source EOS")
        log_mel, pcm_tail = self._causal_log_mel(block, state.pcm_tail)
        conv1, mel_tail = self._causal_conv_block(
            self.encoder_model.conv1, log_mel.to(self.model_dtype), state.mel_tail
        )
        conv1 = F.gelu(conv1)
        conv2, conv1_tail = self._causal_conv_block(
            self.encoder_model.conv2, conv1, state.conv1_tail
        )
        convolved = F.gelu(conv2).transpose(1, 2).contiguous()
        if tuple(convolved.shape[1:2]) != (FRAMES_PER_BLOCK,):
            raise AssertionError(
                f"one PCM block must produce {FRAMES_PER_BLOCK} encoder frames, "
                f"got {int(convolved.shape[1])}"
            )
        valid_samples = int(block.shape[-1])
        valid_tokens = (
            TOKENS_PER_BLOCK
            if valid_samples == BLOCK_SAMPLES
            else max(1, math.ceil(valid_samples / TOKEN_HOP_SAMPLES))
        )
        next_state = CausalAcousticState(pcm_tail, mel_tail, conv1_tail)
        if detach_state:
            next_state = CausalAcousticState(
                *(value.detach() for value in (
                    next_state.pcm_tail,
                    next_state.mel_tail,
                    next_state.conv1_tail,
                ))
            )
        return AcousticBlock(
            convolved_hidden=convolved,
            valid_samples=valid_samples,
            valid_tokens=valid_tokens,
            state=next_state,
        )

    def extract_convolved(
        self,
        pcm: Sequence[float] | np.ndarray | torch.Tensor,
        *,
        detach_state: bool = False,
    ) -> tuple[torch.Tensor, int]:
        """Extract the exact padded block sequence used by both execution paths."""

        waveform = torch.as_tensor(pcm, dtype=torch.float32, device=self.device)
        if waveform.ndim != 1 or not int(waveform.numel()):
            raise ValueError("full PCM must be a non-empty mono waveform")
        acoustic = self.initial_acoustic_state()
        pieces: list[torch.Tensor] = []
        valid_tokens = 0
        for start in range(0, int(waveform.numel()), BLOCK_SAMPLES):
            end = min(int(waveform.numel()), start + BLOCK_SAMPLES)
            block = self.extract_block(
                waveform[start:end],
                acoustic,
                is_final=end == int(waveform.numel()),
                detach_state=detach_state,
            )
            acoustic = block.state
            pieces.append(block.convolved_hidden)
            valid_tokens += block.valid_tokens
        return torch.cat(pieces, dim=1), valid_tokens

    def _require_eval(self) -> None:
        if self.training or self.encoder_model.training or self.cached_encoder.training:
            raise RuntimeError("full/cached parity execution requires eval()")

    @staticmethod
    def _run_layer_recomputed_blocks(
        layer: nn.Module, hidden: torch.Tensor, block_frames: int
    ) -> torch.Tensor:
        """Block-causal layer reference that never retains a K/V cache.

        Projected history is rebuilt from the complete layer input on every
        invocation. Projections and attention matmuls retain deployment block
        geometry, avoiding the expected GEMM reduction drift of a single giant
        masked matrix while remaining independent of persistent cached state.
        """

        normalized_blocks = [
            layer.self_attn_layer_norm(hidden[:, start : start + block_frames])
            for start in range(0, int(hidden.shape[1]), block_frames)
        ]
        key_blocks: list[torch.Tensor] = []
        value_blocks: list[torch.Tensor] = []
        output_blocks: list[torch.Tensor] = []
        attention = layer.self_attn
        batch = int(hidden.shape[0])
        for block_index, normalized in enumerate(normalized_blocks):
            frames = int(normalized.shape[1])
            queries = attention._shape(
                attention.q_proj(normalized) * attention.scaling, frames, batch
            )
            key_blocks.append(
                attention._shape(attention.k_proj(normalized), frames, batch)
            )
            value_blocks.append(
                attention._shape(attention.v_proj(normalized), frames, batch)
            )
            keys = torch.cat(key_blocks, dim=2)
            values = torch.cat(value_blocks, dim=2)
            weights = torch.matmul(queries, keys.transpose(2, 3))
            probabilities = F.softmax(weights, dim=-1)
            attended = torch.matmul(probabilities, values)
            attended = attended.transpose(1, 2).reshape(
                batch, frames, attention.embed_dim
            )
            attended = attention.out_proj(attended)
            residual = hidden[
                :, block_index * block_frames : block_index * block_frames + frames
            ]
            block_hidden = residual + attended
            residual = block_hidden
            block_hidden = layer.final_layer_norm(block_hidden)
            block_hidden = layer.activation_fn(layer.fc1(block_hidden))
            block_hidden = layer.fc2(block_hidden)
            output_blocks.append(residual + block_hidden)
        return torch.cat(output_blocks, dim=1)

    def _forward_recomputed_segment(
        self, convolved_hidden: torch.Tensor
    ) -> CachedWhisperVQOutput:
        self.cached_encoder._validate_input(convolved_hidden)
        hidden = self.cached_encoder._add_positions(convolved_hidden, 0)
        for layer in self.cached_encoder.layers:
            hidden = self._run_layer_recomputed_blocks(
                layer, hidden, self.cached_encoder.block_frames
            )
        pooled = self.cached_encoder._pool(hidden)
        quantized, token_ids = self.cached_encoder._quantize(pooled)
        return CachedWhisperVQOutput(pooled, quantized, token_ids)

    @torch.inference_mode()
    def forward_recomputed_reference(
        self, pcm: Sequence[float] | np.ndarray | torch.Tensor
    ) -> FullReferenceOutput:
        """Reference with no persistent K/V cache and deployment block geometry."""

        self._require_eval()
        convolved, valid_tokens = self.extract_convolved(pcm, detach_state=True)
        frames_per_segment = self.max_blocks_per_encoder_segment * FRAMES_PER_BLOCK
        hidden_parts: list[torch.Tensor] = []
        quantized_parts: list[torch.Tensor] = []
        token_parts: list[torch.Tensor] = []
        for start in range(0, int(convolved.shape[1]), frames_per_segment):
            output = self._forward_recomputed_segment(
                convolved[:, start : start + frames_per_segment]
            )
            hidden_parts.append(output.pre_vq_hidden)
            quantized_parts.append(output.quantized_hidden)
            token_parts.append(output.token_ids)
        hidden = torch.cat(hidden_parts, dim=1)[:, :valid_tokens]
        quantized = torch.cat(quantized_parts, dim=1)[:, :valid_tokens]
        tokens = torch.cat(token_parts, dim=1)[:, :valid_tokens]
        return FullReferenceOutput(
            pre_vq_hidden=hidden,
            quantized_hidden=quantized,
            token_ids=tokens,
            convolved_hidden=convolved,
            valid_tokens=valid_tokens,
            encoder_segments=len(hidden_parts),
        )

    @torch.inference_mode()
    def forward_full_reference(
        self, pcm: Sequence[float] | np.ndarray | torch.Tensor
    ) -> FullReferenceOutput:
        """Run a single large masked matrix per position-safe segment.

        This remains a useful semantic diagnostic, but CUDA GEMM reduction
        order differs from the 8-frame cached path. The strict FP32 deployment
        gate therefore uses :meth:`forward_recomputed_reference` and records
        this single-mask result separately as a numerical-drift diagnostic.
        """

        self._require_eval()
        convolved, valid_tokens = self.extract_convolved(pcm, detach_state=True)
        frames_per_segment = self.max_blocks_per_encoder_segment * FRAMES_PER_BLOCK
        hidden_parts: list[torch.Tensor] = []
        quantized_parts: list[torch.Tensor] = []
        token_parts: list[torch.Tensor] = []
        for start in range(0, int(convolved.shape[1]), frames_per_segment):
            output = self.cached_encoder.forward_full(
                convolved[:, start : start + frames_per_segment]
            )
            hidden_parts.append(output.pre_vq_hidden)
            quantized_parts.append(output.quantized_hidden)
            token_parts.append(output.token_ids)
        hidden = torch.cat(hidden_parts, dim=1)[:, :valid_tokens]
        quantized = torch.cat(quantized_parts, dim=1)[:, :valid_tokens]
        tokens = torch.cat(token_parts, dim=1)[:, :valid_tokens]
        return FullReferenceOutput(
            pre_vq_hidden=hidden,
            quantized_hidden=quantized,
            token_ids=tokens,
            convolved_hidden=convolved,
            valid_tokens=valid_tokens,
            encoder_segments=len(hidden_parts),
        )

    @torch.inference_mode()
    def push(
        self,
        pcm: Sequence[float] | np.ndarray | torch.Tensor,
        state: SharedFrontendState | None = None,
        *,
        is_final: bool = False,
    ) -> SharedFrontendOutput:
        """Consume one PCM block using K/V cache and automatic position reset."""

        self._require_eval()
        state = state or self.initial_state()
        if state.finalized:
            raise ValueError("cannot append PCM after frontend finalization")
        acoustic = self.extract_block(
            pcm, state.acoustic, is_final=is_final, detach_state=True
        )
        encoder_state = state.encoder
        reset = False
        resets = state.encoder_resets
        if (
            encoder_state is not None
            and encoder_state.frames_seen + FRAMES_PER_BLOCK
            > self.cached_encoder.position_embedding.num_embeddings
        ):
            encoder_state = None
            reset = True
            resets += 1
        output = self.cached_encoder.forward_chunk(
            acoustic.convolved_hidden,
            encoder_state,
            is_final=is_final,
        )
        keep = acoustic.valid_tokens
        samples_seen = state.samples_seen + acoustic.valid_samples
        next_state = SharedFrontendState(
            acoustic=acoustic.state,
            encoder=output.state.detach() if output.state is not None else None,
            samples_seen=samples_seen,
            blocks_seen=state.blocks_seen + 1,
            encoder_resets=resets,
            finalized=bool(is_final),
        )
        return SharedFrontendOutput(
            source_end_ms=int(round(samples_seen * 1000 / SAMPLE_RATE)),
            pre_vq_hidden=output.pre_vq_hidden[:, :keep].detach(),
            quantized_hidden=output.quantized_hidden[:, :keep].detach(),
            token_ids=output.token_ids[:, :keep].detach(),
            state=next_state,
            encoder_reset_before_block=reset,
            is_final=bool(is_final),
        )


__all__ = [
    "AcousticBlock",
    "BLOCK_MS",
    "BLOCK_SAMPLES",
    "CausalAcousticState",
    "FRAMES_PER_BLOCK",
    "FRONTEND_SCHEMA",
    "FullReferenceOutput",
    "N_FFT",
    "PCM_LEFT_CONTEXT",
    "SAMPLE_RATE",
    "SharedCausalWhisperVQFrontend",
    "SharedFrontendOutput",
    "SharedFrontendState",
    "TOKEN_HOP_MS",
    "TOKEN_HOP_SAMPLES",
]
