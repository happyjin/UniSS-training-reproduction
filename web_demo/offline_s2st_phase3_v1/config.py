"""Frozen configuration for the isolated Phase3 offline S2ST demo."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


DEMO_ROOT = Path(__file__).resolve().parent
REPO_ROOT = DEMO_ROOT.parents[1]


@dataclass(frozen=True)
class DemoConfig:
    """Server-side-only configuration.

    Model paths are deliberately not accepted from browser requests.
    """

    repo_root: Path = REPO_ROOT
    demo_root: Path = DEMO_ROOT
    model_path: Path = REPO_ROOT / "checkpoints/exported_hf/qwen0p5b_phase3_unist198_iter_0009075_hf"
    speech_tokenizer_path: Path = REPO_ROOT / "pretrained_models/UniSS"
    output_root: Path = DEMO_ROOT / "runtime_outputs"
    device: str = "cuda:0"
    mode: str = "quality"
    task_name: str = "Quality"
    max_upload_bytes: int = 50 * 1024 * 1024
    max_audio_seconds: float = 60.0
    max_chunk_seconds: float = 30.0
    min_audio_seconds: float = 0.25
    chunk_silence_seconds: float = 0.12
    max_new_tokens: int = 1500
    temperature: float = 0.7
    top_p: float = 0.8
    repetition_penalty: float = 1.1
    seed: int = 20260726
    output_ttl_hours: float = 24.0

    @classmethod
    def from_env(cls) -> "DemoConfig":
        return cls(
            device=os.environ.get("UNISS_DEMO_DEVICE", "cuda:0"),
            max_audio_seconds=float(os.environ.get("UNISS_DEMO_MAX_AUDIO_SECONDS", "60")),
            output_ttl_hours=float(os.environ.get("UNISS_DEMO_OUTPUT_TTL_HOURS", "24")),
        )

    def validate(self) -> None:
        if self.mode != "quality" or self.task_name != "Quality":
            raise ValueError("This demo is frozen to Phase3 Quality mode")
        required_model_files = (
            self.model_path / "config.json",
            self.model_path / "model.safetensors",
            self.model_path / "tokenizer.json",
            self.model_path / "export_manifest.json",
        )
        missing = [str(path) for path in required_model_files if not path.is_file()]
        for name in ("glm4_tokenizer", "bicodec"):
            path = self.speech_tokenizer_path / name
            if not path.is_dir():
                missing.append(str(path))
        if missing:
            raise FileNotFoundError(f"Missing frozen Phase3 demo assets: {missing}")
        if self.max_audio_seconds <= 0 or self.max_chunk_seconds <= 0:
            raise ValueError("Audio duration limits must be positive")
        if self.max_chunk_seconds > self.max_audio_seconds:
            raise ValueError("max_chunk_seconds cannot exceed max_audio_seconds")

    @property
    def model_label(self) -> str:
        return "Phase3 full198 iter_0009075"
