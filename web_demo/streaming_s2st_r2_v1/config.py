"""Frozen server-side configuration for the isolated streaming S2ST demo."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


DEMO_ROOT = Path(__file__).resolve().parent
REPO_ROOT = DEMO_ROOT.parents[1]
MODEL_ROOT = (
    REPO_ROOT
    / "checkpoints/exported_hf/simul_uniss_stage7a_reward_v2_15shard_v1"
)


@dataclass(frozen=True)
class StreamingDemoConfig:
    """Configuration values that browser requests are not allowed to override."""

    repo_root: Path = REPO_ROOT
    demo_root: Path = DEMO_ROOT
    primary_model_path: Path = MODEL_ROOT / "r2_explicit_latency_best_hf"
    fallback_model_path: Path = MODEL_ROOT / "r3_bilingual_adaptive_best_hf"
    speech_tokenizer_path: Path = REPO_ROOT / "pretrained_models/UniSS"
    output_root: Path = DEMO_ROOT / "runtime_outputs"
    log_root: Path = DEMO_ROOT / "runtime_logs"
    device: str = "cuda:0"
    model_name: str = "r2"
    chunk_ms: int = 640
    stable_prefix_holdback_tokens: int = 2
    codec_left_context_tokens: int = 50
    codec_holdback_tokens: int = 5
    codec_overlap_ms: float = 80.0
    max_write_tokens: int = 700
    max_model_len: int = 32768
    training_context_limit: int = 18000
    repetition_penalty: float = 1.1
    max_upload_bytes: int = 100 * 1024 * 1024
    min_audio_seconds: float = 0.5
    max_audio_seconds: float = 60.0
    microphone_max_audio_seconds: float = 90.0
    output_ttl_hours: float = 24.0
    queue_max_size: int = 4
    seed: int = 20260730

    @classmethod
    def from_env(cls) -> "StreamingDemoConfig":
        model_name = os.environ.get("UNISS_STREAMING_MODEL", "r2").strip().lower()
        return cls(
            device=os.environ.get("UNISS_STREAMING_DEVICE", "cuda:0"),
            model_name=model_name,
            max_audio_seconds=float(
                os.environ.get("UNISS_STREAMING_MAX_AUDIO_SECONDS", "60")
            ),
            microphone_max_audio_seconds=float(
                os.environ.get("UNISS_STREAMING_MIC_MAX_AUDIO_SECONDS", "90")
            ),
            output_ttl_hours=float(
                os.environ.get("UNISS_STREAMING_OUTPUT_TTL_HOURS", "24")
            ),
        )

    @property
    def model_path(self) -> Path:
        if self.model_name == "r2":
            return self.primary_model_path
        if self.model_name == "r3":
            return self.fallback_model_path
        raise ValueError(f"Unsupported frozen model selection: {self.model_name!r}")

    @property
    def model_label(self) -> str:
        manifest = self.export_manifest()
        step = manifest.get("action_checkpoint_step", "unknown")
        variant = (
            "R2 explicit-latency" if self.model_name == "r2" else "R3 bilingual-adaptive"
        )
        return f"Stage7A Reward-v2 {variant} step {step}"

    def export_manifest(self) -> dict[str, object]:
        path = self.model_path / "stage7a_export_manifest.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def validate(self) -> None:
        if self.model_name not in {"r2", "r3"}:
            raise ValueError("UNISS_STREAMING_MODEL must be r2 or r3")
        required_model_files = (
            self.model_path / "config.json",
            self.model_path / "model.safetensors",
            self.model_path / "tokenizer.json",
            self.model_path / "stage7a_export_manifest.json",
        )
        missing = [str(path) for path in required_model_files if not path.is_file()]
        for name in ("glm4_tokenizer", "bicodec"):
            path = self.speech_tokenizer_path / name
            if not path.is_dir():
                missing.append(str(path))
        if missing:
            raise FileNotFoundError(f"Missing frozen streaming demo assets: {missing}")
        if self.chunk_ms <= 0 or self.chunk_ms % 20:
            raise ValueError("chunk_ms must be a positive multiple of 20 ms")
        if not 0 < self.min_audio_seconds <= self.max_audio_seconds:
            raise ValueError("invalid upload audio duration limits")
        if self.max_audio_seconds > self.microphone_max_audio_seconds:
            raise ValueError("microphone limit cannot be shorter than upload limit")
        if not 0 < self.training_context_limit < self.max_model_len:
            raise ValueError("training context limit must be below max_model_len")
        if self.queue_max_size < 1:
            raise ValueError("queue_max_size must be positive")
