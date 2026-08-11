"""True 160 ms PCM frontend with causal STFT/conv state and WhisperVQ K/V cache."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch.nn import functional as F

from .cached_whispervq import (
    CachedBlockCausalWhisperVQ,
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


@dataclass(frozen=True)
class CachedAudioFrontendState:
    pcm_tail: torch.Tensor
    mel_tail: torch.Tensor
    conv1_tail: torch.Tensor
    encoder: CachedWhisperVQState | None
    samples_seen: int
    blocks_seen: int
    finalized: bool = False


@dataclass(frozen=True)
class CachedAudioFrontendStep:
    source_end_ms: int
    new_tokens: tuple[int, ...]
    pre_vq_hidden: torch.Tensor
    state: CachedAudioFrontendState
    is_final: bool


class StreamingCachedWhisperVQFrontend:
    """Shared data/deployment frontend with no future PCM dependency.

    Each call consumes at most one 160 ms block.  Log-mel extraction uses a
    causal 240-sample waveform tail and ``center=False``.  The two released
    causal convolution layers retain their exact two-frame left contexts.
    Encoder attention is block-causal: all history plus the complete current
    160 ms block, with no later-block right context.
    """

    def __init__(
        self,
        whisper_encoder,
        mel_filters: np.ndarray | torch.Tensor,
        *,
        device: str | torch.device,
    ) -> None:
        self.encoder_model = whisper_encoder.eval()
        self.device = torch.device(device)
        self.encoder_model.to(self.device)
        self.cached = CachedBlockCausalWhisperVQ.from_whispervq_encoder(
            self.encoder_model,
            chunk_ms=BLOCK_MS,
            frame_ms=20,
            right_context_ms=0,
        ).to(self.device).eval()
        filters = torch.as_tensor(mel_filters, dtype=torch.float32)
        if filters.ndim != 2 or filters.shape[0] != N_FFT // 2 + 1:
            raise ValueError(
                f"mel_filters must be [{N_FFT // 2 + 1},mel], got {tuple(filters.shape)}"
            )
        self.mel_filters = filters.to(self.device)
        self.window = torch.hann_window(N_FFT, device=self.device)

    @property
    def device_dtype(self) -> torch.dtype:
        return self.encoder_model.conv1.weight.dtype

    def initial_state(self) -> CachedAudioFrontendState:
        hidden = int(self.encoder_model.conv1.out_channels)
        mel = int(self.encoder_model.conv1.in_channels)
        return CachedAudioFrontendState(
            pcm_tail=torch.zeros(1, PCM_LEFT_CONTEXT, device=self.device),
            mel_tail=torch.zeros(1, mel, 2, device=self.device),
            conv1_tail=torch.zeros(1, hidden, 2, device=self.device),
            encoder=None,
            samples_seen=0,
            blocks_seen=0,
            finalized=False,
        )

    def _pcm_tensor(self, pcm: Sequence[float] | np.ndarray | torch.Tensor) -> torch.Tensor:
        value = torch.as_tensor(pcm, dtype=torch.float32, device=self.device).reshape(1, -1)
        if not 0 < value.shape[-1] <= BLOCK_SAMPLES:
            raise ValueError(f"PCM block must contain 1..{BLOCK_SAMPLES} samples")
        if not bool(torch.isfinite(value).all()):
            raise ValueError("PCM block contains non-finite samples")
        return value

    def _log_mel(self, block: torch.Tensor, pcm_tail: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        actual = block.shape[-1]
        padded = F.pad(block, (0, BLOCK_SAMPLES - actual))
        analysis = torch.cat((pcm_tail, padded), dim=-1)
        spectrum = torch.stft(
            analysis,
            N_FFT,
            MEL_HOP,
            window=self.window,
            center=False,
            return_complex=True,
        )
        magnitudes = spectrum.abs().square()
        if magnitudes.shape[-1] != BLOCK_MS // 10:
            raise AssertionError(
                f"one PCM block must produce 16 mel frames, got {magnitudes.shape[-1]}"
            )
        mel = torch.matmul(self.mel_filters.T, magnitudes[0]).unsqueeze(0)
        log_mel = torch.clamp(mel, min=1e-10).log10()
        # Per-block normalization is intentional: it is bounded to audio that
        # has arrived and therefore cannot revise a committed past block.
        log_mel = torch.maximum(log_mel, log_mel.amax(dim=(1, 2), keepdim=True) - 8.0)
        log_mel = (log_mel + 4.0) / 4.0
        next_tail = torch.cat((pcm_tail, block), dim=-1)[:, -PCM_LEFT_CONTEXT:]
        return log_mel, next_tail

    @staticmethod
    def _causal_conv_block(module, current: torch.Tensor, tail: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        combined = torch.cat((tail.to(current.dtype), current), dim=-1)
        output = F.conv1d(
            combined,
            module.weight,
            module.bias,
            stride=module.stride,
            dilation=module.dilation,
            groups=module.groups,
        )
        next_tail = combined[:, :, -2:].detach()
        return output, next_tail

    @torch.inference_mode()
    def push(
        self,
        pcm: Sequence[float] | np.ndarray | torch.Tensor,
        state: CachedAudioFrontendState | None = None,
        *,
        is_final: bool = False,
    ) -> CachedAudioFrontendStep:
        state = state or self.initial_state()
        if state.finalized:
            raise ValueError("cannot append PCM after frontend finalization")
        block = self._pcm_tensor(pcm)
        if block.shape[-1] != BLOCK_SAMPLES and not is_final:
            raise ValueError("a partial PCM block is only legal at source EOS")
        log_mel, pcm_tail = self._log_mel(block, state.pcm_tail)
        conv1, mel_tail = self._causal_conv_block(
            self.encoder_model.conv1, log_mel.to(self.device_dtype), state.mel_tail
        )
        conv1 = F.gelu(conv1)
        conv2, conv1_tail = self._causal_conv_block(
            self.encoder_model.conv2, conv1, state.conv1_tail
        )
        convolved = F.gelu(conv2).transpose(1, 2).contiguous()
        if convolved.shape[1] != 8:
            raise AssertionError(f"one block must produce 8 encoder frames, got {convolved.shape[1]}")
        output = self.cached.forward_chunk(
            convolved,
            state.encoder,
            is_final=is_final,
        )
        keep = 2
        if is_final and block.shape[-1] < BLOCK_SAMPLES:
            keep = max(1, math.ceil(block.shape[-1] / TOKEN_HOP_SAMPLES))
        tokens = tuple(int(value) for value in output.token_ids[0, :keep].tolist())
        samples_seen = state.samples_seen + int(block.shape[-1])
        next_state = CachedAudioFrontendState(
            pcm_tail=pcm_tail.detach(),
            mel_tail=mel_tail.detach(),
            conv1_tail=conv1_tail.detach(),
            encoder=output.state.detach() if output.state is not None else None,
            samples_seen=samples_seen,
            blocks_seen=state.blocks_seen + 1,
            finalized=bool(is_final),
        )
        return CachedAudioFrontendStep(
            source_end_ms=int(round(samples_seen * 1000 / SAMPLE_RATE)),
            new_tokens=tokens,
            pre_vq_hidden=output.pre_vq_hidden[:, :keep].detach(),
            state=next_state,
            is_final=bool(is_final),
        )


__all__ = [
    "BLOCK_MS",
    "BLOCK_SAMPLES",
    "CachedAudioFrontendState",
    "CachedAudioFrontendStep",
    "SAMPLE_RATE",
    "StreamingCachedWhisperVQFrontend",
    "TOKEN_HOP_MS",
]
