"""Configuration for the isolated Stage11 audio runtime."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class Stage11Config:
    speech_tokenizer_path: Path = ROOT / "pretrained_models/UniSS"
    output_root: Path = ROOT / "eval_outputs/uniss_streamspeech_ctc_v1/stage11_streaming_audio_v1"
    max_write_tokens: int = 384
    codec_left_context_tokens: int = 50
    codec_holdback_tokens: int = 5
    codec_overlap_ms: float = 80.0
    semantic_unique_ratio_min: float = 0.10
    semantic_max_run: int = 16

    def validate(self) -> None:
        for name in ("glm4_tokenizer", "bicodec"):
            if not (self.speech_tokenizer_path / name).is_dir():
                raise FileNotFoundError(
                    f"missing Stage11 tokenizer component: {self.speech_tokenizer_path / name}"
                )
        if self.max_write_tokens <= 0:
            raise ValueError("max_write_tokens must be positive")
        if not 0.0 <= self.semantic_unique_ratio_min <= 1.0:
            raise ValueError("semantic_unique_ratio_min must be in [0,1]")
