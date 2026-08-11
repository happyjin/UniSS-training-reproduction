"""Frozen paths and runtime limits for the isolated pilot15 streaming demo."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


DEMO_ROOT = Path(__file__).resolve().parent
REPO_ROOT = DEMO_ROOT.parents[1]


@dataclass(frozen=True)
class DemoConfig:
    repo_root: Path = REPO_ROOT
    demo_root: Path = DEMO_ROOT
    checkpoint: Path = (
        REPO_ROOT / "checkpoints/uniss_true_subsecond_pilot15_epoch1_v3/iter_0000350"
    )
    base_model: Path = (
        REPO_ROOT / "checkpoints/exported_hf/qwen0p5b_phase3_unist198_iter_0009075_hf"
    )
    exported_runtime: Path = (
        REPO_ROOT
        / "checkpoints/exported_adapters/uniss_true_subsecond_pilot15_iter_0000350_runtime_v2"
    )
    speech_tokenizer_dir: Path = REPO_ROOT / "pretrained_models/UniSS"
    whispervq_dir: Path = REPO_ROOT / "pretrained_models/UniSS/glm4_tokenizer"
    output_root: Path = DEMO_ROOT / "runtime_outputs"
    device: str = "cuda:0"
    decision_chunk_ms: int = 320
    acoustic_chunk_ms: int = 160
    acoustic_right_context_ms: int = 80
    frontend_window_ms: int = 4_800
    soft_deadline_ms: int = 640
    hard_deadline_ms: int = 800
    max_audio_seconds: float = 305.0
    max_upload_bytes: int = 160 * 1024 * 1024
    semantic_block_tokens: int = 12
    max_text_tokens_per_write: int = 8
    semantic_history_tokens: int = 200
    speaker_warmup_ms: int = 3_200
    speaker_vad_frame_ms: int = 20
    speaker_vad_min_rms: float = 0.006
    seed: int = 20260811

    @classmethod
    def from_env(cls) -> "DemoConfig":
        return cls(
            device=os.environ.get("UNISS_TRUE_STREAMING_DEVICE", "cuda:0"),
            decision_chunk_ms=int(
                os.environ.get("UNISS_TRUE_STREAMING_CHUNK_MS", "320")
            ),
        )

    def validate_assets(self, *, require_export: bool = False) -> None:
        required = [
            self.checkpoint / ".metadata",
            self.base_model / "config.json",
            self.whispervq_dir / "model.safetensors",
            self.speech_tokenizer_dir / "bicodec/BiCodec/model.safetensors",
        ]
        if require_export:
            required.extend(
                [
                    self.exported_runtime / "adapter_model.safetensors",
                    self.exported_runtime / "objective_model.safetensors",
                    self.exported_runtime / "manifest.json",
                ]
            )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"missing frozen streaming assets: {missing}")
        if self.decision_chunk_ms not in {320, 480, 640}:
            raise ValueError("decision_chunk_ms must be 320, 480 or 640")
        if self.acoustic_chunk_ms != 160 or self.acoustic_right_context_ms != 80:
            raise ValueError("checkpoint supervision is frozen at 160ms + 80ms")
        if self.frontend_window_ms < self.decision_chunk_ms + self.acoustic_right_context_ms:
            raise ValueError("frontend window is shorter than one observable decision")
