"""Frozen server-side configuration for the Stage13 research demo."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


DEMO_ROOT = Path(__file__).resolve().parent
REPO_ROOT = DEMO_ROOT.parents[1]


@dataclass(frozen=True)
class Stage13Config:
    repo_root: Path = REPO_ROOT
    demo_root: Path = DEMO_ROOT
    output_root: Path = DEMO_ROOT / "runtime_outputs"
    log_root: Path = DEMO_ROOT / "runtime_logs"
    fixed_speaker_manifest: Path = REPO_ROOT / "data/processed/simul_uniss_subsecond_v1/pilot_15shard/stage_a_source/stage_a_source_manifest.jsonl"
    device: str = "cuda:0"
    chunk_ms: int = 160
    max_upload_bytes: int = 100 * 1024 * 1024
    min_audio_seconds: float = 0.5
    max_audio_seconds: float = 30.0
    max_microphone_seconds: float = 45.0
    queue_size: int = 4

    @classmethod
    def from_env(cls) -> "Stage13Config":
        return cls(
            device=os.environ.get("UNISS_STAGE13_DEVICE", "cuda:0"),
            max_audio_seconds=float(os.environ.get("UNISS_STAGE13_MAX_AUDIO_SECONDS", "30")),
            max_microphone_seconds=float(os.environ.get("UNISS_STAGE13_MIC_SECONDS", "45")),
        )

    def validate(self) -> None:
        if not self.fixed_speaker_manifest.is_file():
            raise FileNotFoundError(f"missing fixed speaker manifest: {self.fixed_speaker_manifest}")
        if self.chunk_ms != 160:
            raise ValueError("Stage13 preserves the trained 160 ms Emformer segment")
        if not 0 < self.min_audio_seconds <= self.max_audio_seconds:
            raise ValueError("invalid upload duration limits")
        if self.queue_size <= 0:
            raise ValueError("queue_size must be positive")

    def fixed_speaker_tokens(self) -> list[int]:
        with self.fixed_speaker_manifest.open("r", encoding="utf-8") as handle:
            row = json.loads(next(handle))
        values = [int(value) for value in row["bicodec_global"]]
        if len(values) != 32:
            raise ValueError("fixed speaker record does not contain 32 global tokens")
        return values

    @property
    def model_label(self) -> str:
        return "Stage09 CTC + Stage10 Step2-LoRA KV-cache + Stage11 BiCodec"
