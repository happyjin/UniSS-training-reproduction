"""Validated configuration objects for the joint model."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MultiChunkConfig:
    """WhisperVQ chunk choices expressed in milliseconds.

    ``None`` is the offline/full-context choice. Whisper encoder frames are
    20 ms before the historical UniSS pooling layer.
    """

    chunk_ms: tuple[int | None, ...] = (320, 640, 960, 1280, None)
    right_context_ms: int = 80
    encoder_frame_ms: int = 20

    def __post_init__(self) -> None:
        if not self.chunk_ms:
            raise ValueError("chunk_ms must contain at least one choice")
        if self.encoder_frame_ms <= 0:
            raise ValueError("encoder_frame_ms must be positive")
        if self.right_context_ms < 0 or self.right_context_ms % self.encoder_frame_ms:
            raise ValueError("right context must be a non-negative frame multiple")
        finite = [value for value in self.chunk_ms if value is not None]
        if any(value <= 0 or value % self.encoder_frame_ms for value in finite):
            raise ValueError("finite chunks must be positive encoder-frame multiples")
        if len(set(self.chunk_ms)) != len(self.chunk_ms):
            raise ValueError("chunk choices must be unique")

    @property
    def right_context_frames(self) -> int:
        return self.right_context_ms // self.encoder_frame_ms

    def frames(self, chunk_ms: int | None) -> int | None:
        if chunk_ms not in self.chunk_ms:
            raise ValueError(f"unconfigured chunk: {chunk_ms}")
        return None if chunk_ms is None else chunk_ms // self.encoder_frame_ms


@dataclass(frozen=True)
class JointLossWeights:
    """StreamSpeech weights plus exact Phase3 replay protection."""

    bicodec_ctc: float = 1.0
    ar_s2tt: float = 8.0
    asr_ctc: float = 4.0
    nar_s2tt_ctc: float = 4.0
    phase3_replay: float = 0.5
    bridge_commitment: float = 0.0
    replay_probability: float = 0.20

    def __post_init__(self) -> None:
        values = (
            self.bicodec_ctc,
            self.ar_s2tt,
            self.asr_ctc,
            self.nar_s2tt_ctc,
            self.phase3_replay,
            self.bridge_commitment,
        )
        if any(value < 0 for value in values):
            raise ValueError("loss weights must be non-negative")
        if not 0 <= self.replay_probability < 1:
            raise ValueError("replay_probability must be in [0, 1)")
