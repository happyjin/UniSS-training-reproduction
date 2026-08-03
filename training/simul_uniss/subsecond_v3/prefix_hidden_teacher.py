"""Exact prefix-80 WhisperVQ teacher with aligned pre-VQ hidden export."""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
import torchaudio
from transformers import WhisperFeatureExtractor

from uniss.speech_tokenizer.glm4.utils import load_quantize_encoder


@dataclass(frozen=True)
class PrefixTeacherOutput:
    tokens: torch.Tensor
    pre_vq_hidden: torch.Tensor


class ExactPrefixWhisperVQTeacher:
    """Frozen released WhisperVQ evaluated on finite visible prefixes.

    Unlike the streaming clone, this class does not replace the teacher's
    attention mask.  Each input is an exact prefix ending at
    ``committed_ms + lookahead_ms``.  A pooling hook exports the vector that is
    passed directly to the frozen VQ codebook.
    """

    def __init__(self, model_path: str | Path, *, device: str = "cuda:0") -> None:
        self.device = torch.device(device)
        self.model = load_quantize_encoder(str(model_path)).to(self.device).eval()
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(str(model_path))

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
    ) -> list[PrefixTeacherOutput]:
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
        convolution_stride = self.model.conv1.stride[0] * self.model.conv2.stride[0]
        mask = features.attention_mask[:, ::convolution_stride]
        mask = mask[:, ::pooling].bool()
        if mask.shape != tokens.shape or pre_vq.shape[:2] != tokens.shape:
            raise RuntimeError(
                "prefix teacher shape mismatch: "
                f"hidden={tuple(pre_vq.shape)}, tokens={tuple(tokens.shape)}, "
                f"mask={tuple(mask.shape)}"
            )
        return [
            PrefixTeacherOutput(
                tokens=tokens[index][mask[index]].detach().cpu(),
                pre_vq_hidden=pre_vq[index][mask[index]].detach().cpu(),
            )
            for index in range(len(audio))
        ]


@torch.inference_mode()
def build_exact_prefix_hidden_targets(
    teacher: ExactPrefixWhisperVQTeacher,
    waveform: torch.Tensor,
    token_end_ms: Sequence[int],
    *,
    chunk_ms: int,
    lookahead_ms: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return append-only token, stability and matching exact-prefix hidden targets."""

    duration_ms = int(round(waveform.shape[-1] / 16))
    commit_ends = list(range(chunk_ms, duration_ms + chunk_ms, chunk_ms))
    commit_ends = list(dict.fromkeys(min(value, duration_ms) for value in commit_ends))
    prefixes: list[tuple[torch.Tensor, int]] = []
    for committed_ms in commit_ends:
        visible_ms = min(duration_ms, committed_ms + lookahead_ms)
        visible_samples = max(400, min(waveform.shape[-1], visible_ms * 16))
        prefixes.append((waveform[..., :visible_samples], 16_000))
    predictions = teacher.encode(prefixes)
    tokens: list[int] = []
    stability: list[int] = []
    hidden: list[torch.Tensor] = []
    for tick, committed_ms in enumerate(commit_ends):
        required = bisect.bisect_right(token_end_ms, committed_ms)
        while len(tokens) < required:
            index = len(tokens)
            source_tick = next(
                (
                    candidate
                    for candidate in range(tick, len(predictions))
                    if index < len(predictions[candidate].tokens)
                ),
                None,
            )
            if source_tick is None:
                break
            output = predictions[source_tick]
            value = int(output.tokens[index])
            later = [
                predictions[min(len(predictions) - 1, source_tick + delta)].tokens
                for delta in (0, 1, 2)
            ]
            stable = all(index < len(current) and int(current[index]) == value for current in later)
            tokens.append(value)
            stability.append(int(stable))
            hidden.append(output.pre_vq_hidden[index].float())
        if len(tokens) < required:
            break
    dimension = int(teacher.model.codebook.weight.shape[1])
    hidden_tensor = (
        torch.stack(hidden) if hidden else torch.zeros(0, dimension, dtype=torch.float32)
    )
    return (
        torch.tensor(tokens, dtype=torch.int32),
        torch.tensor(stability, dtype=torch.uint8),
        hidden_tensor,
    )
