"""Frozen configuration for the isolated five-minute demo."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


DEMO_ROOT = Path(__file__).resolve().parent
REPO_ROOT = DEMO_ROOT.parents[1]


@dataclass(frozen=True)
class LongFormDemoConfig:
    repo_root: Path = REPO_ROOT
    demo_root: Path = DEMO_ROOT
    adapter_dir: Path = (
        REPO_ROOT
        / "checkpoints/exported_adapters/uniss_phase3_prefix_streaming_full198_joint_v3_iter_0008000_lora_v1"
    )
    speech_tokenizer_dir: Path = REPO_ROOT / "pretrained_models/UniSS"
    output_root: Path = DEMO_ROOT / "runtime_outputs"
    device: str = "cuda:0"
    max_audio_seconds: float = 305.0
    max_upload_bytes: int = 150 * 1024 * 1024
    target_window_seconds: float = 25.0
    minimum_window_seconds: float = 18.0
    maximum_window_seconds: float = 30.0
    boundary_search_seconds: float = 5.0
    minimum_retry_seconds: float = 4.0
    output_ttl_hours: float = 48.0
    queue_max_size: int = 2

    @classmethod
    def from_env(cls) -> "LongFormDemoConfig":
        return cls(
            device=os.environ.get("UNISS_PREFIX_LONGFORM_DEVICE", "cuda:0"),
            max_audio_seconds=float(
                os.environ.get("UNISS_PREFIX_LONGFORM_MAX_AUDIO_SECONDS", "305")
            ),
        )

    def validate_assets(self) -> None:
        required = (
            self.adapter_dir / "adapter_config.json",
            self.adapter_dir / "adapter_model.safetensors",
            self.speech_tokenizer_dir / "glm4_tokenizer",
            self.speech_tokenizer_dir / "bicodec",
        )
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(f"missing frozen long-form demo assets: {missing}")
        if not (
            0
            < self.minimum_window_seconds
            <= self.target_window_seconds
            <= self.maximum_window_seconds
            <= 30.0
        ):
            raise ValueError("expected 0 < minimum <= target <= maximum <= 30 seconds")
