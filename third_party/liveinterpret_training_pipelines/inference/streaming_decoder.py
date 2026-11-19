"""Fixed-window compiled adapter for the installed Qwen3-TTS 12Hz decoder."""

from __future__ import annotations

import time
from typing import Any, Callable, Optional, Tuple

import numpy as np
import torch


class CompiledStreamingDecoder:
    """Compile one strict, fixed-shape 12Hz streaming decode window."""

    def __init__(
        self,
        codec_tokenizer: Any,
        window_frames: int,
        compile_mode: str = "reduce-overhead",
        *,
        compile_fn: Callable[..., Any] = torch.compile,
    ) -> None:
        self.window_frames = int(window_frames)
        if self.window_frames < 1:
            raise ValueError("window_frames must be positive")
        self.compile_mode = compile_mode
        tokenizer = codec_tokenizer.tokenizer
        model = getattr(tokenizer, "model", None)
        if model is None:
            raise TypeError(
                "Expected codec_tokenizer.tokenizer.model for Qwen3-TTS"
            )
        model_type = (
            model.get_model_type()
            if hasattr(model, "get_model_type")
            else getattr(getattr(model, "config", None), "model_type", None)
        )
        if model_type != "qwen3_tts_tokenizer_12hz":
            raise ValueError(
                "Compiled streaming decoder requires the 12Hz tokenizer; "
                f"got {model_type!r}"
            )
        decoder = getattr(model, "decoder", None)
        if decoder is None or not callable(getattr(decoder, "forward", None)):
            raise TypeError(
                "Expected Qwen3-TTS tokenizer model.decoder.forward"
            )
        parameter = next(decoder.parameters(), None)
        if parameter is None:
            raise TypeError("Tokenizer decoder has no parameters")
        self.device = parameter.device
        # An injected compiler is the narrow CPU test seam. Production always
        # uses torch.compile and therefore enforces the CUDA-only contract.
        self._production_compile = compile_fn is torch.compile
        if self._production_compile and self.device.type != "cuda":
            raise ValueError(
                "Compiled streaming decoder requires decoder parameters on CUDA"
            )
        self.decoder = decoder
        self.num_quantizers = int(decoder.config.num_quantizers)
        self.upsample_rate = int(model.get_decode_upsample_rate())
        self.sample_rate = int(model.get_output_sample_rate())
        if self.upsample_rate < 1 or self.sample_rate < 1:
            raise ValueError("Tokenizer decoder returned invalid audio rates")
        self._compiled_forward = compile_fn(
            decoder.forward,
            mode=compile_mode,
            fullgraph=False,
            dynamic=False,
        )
        self._warmup_ms: Optional[float] = None

    def _invoke(self, codes: torch.LongTensor) -> torch.Tensor:
        if self._production_compile and self.compile_mode == "reduce-overhead":
            torch.compiler.cudagraph_mark_step_begin()
        waveform = self._compiled_forward(codes)
        if not torch.is_tensor(waveform):
            raise TypeError(
                "Tokenizer decoder.forward must return a waveform tensor"
            )
        return waveform

    def warmup(self) -> float:
        """Compile/warm the exact production shape and return milliseconds."""
        if self._warmup_ms is not None:
            return self._warmup_ms
        codes = torch.zeros(
            1,
            self.num_quantizers,
            self.window_frames,
            dtype=torch.long,
            device=self.device,
        )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started_at = time.perf_counter()
        with torch.inference_mode():
            self._invoke(codes)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self._warmup_ms = (time.perf_counter() - started_at) * 1000.0
        return self._warmup_ms

    def decode(self, codes: torch.Tensor) -> Tuple[np.ndarray, int]:
        """Right-pad ``[T,Q]`` codes, decode, and trim future padding."""
        codes = torch.as_tensor(codes, dtype=torch.long, device=self.device)
        if codes.ndim != 2:
            raise ValueError(
                f"Expected codec window [T,Q], got shape={tuple(codes.shape)}"
            )
        frames, quantizers = codes.shape
        if quantizers != self.num_quantizers:
            raise ValueError(
                f"Expected {self.num_quantizers} codec groups, got {quantizers}"
            )
        if frames > self.window_frames:
            raise ValueError(
                f"Codec window has {frames} frames, maximum is "
                f"{self.window_frames}"
            )
        if frames == 0:
            return np.empty(0, dtype=np.float32), self.sample_rate
        codes = torch.where(codes >= 2148, torch.zeros_like(codes), codes)
        codes = codes.clamp(0, 2047)
        padded = torch.zeros(
            1,
            self.num_quantizers,
            self.window_frames,
            dtype=torch.long,
            device=self.device,
        )
        padded[:, :, :frames] = codes.transpose(0, 1)
        with torch.inference_mode():
            waveform = self._invoke(padded)
        valid_samples = frames * self.upsample_rate
        waveform = waveform.reshape(waveform.shape[0], -1)[
            0, :valid_samples
        ]
        return (
            waveform.detach().to(torch.float32).cpu().numpy(),
            self.sample_rate,
        )


__all__ = ["CompiledStreamingDecoder"]
