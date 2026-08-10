"""Frozen server configuration; browser users may only select direction/chunk."""

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
    adapter_dir: Path = (
        REPO_ROOT
        / "checkpoints/exported_adapters/uniss_phase3_prefix_streaming_full198_joint_v3_iter_0008000_lora_v1"
    )
    speech_tokenizer_dir: Path = REPO_ROOT / "pretrained_models/UniSS"
    output_root: Path = DEMO_ROOT / "runtime_outputs"
    device: str = "cuda:0"
    max_audio_seconds: float = 60.0
    queue_max_size: int = 4

    @classmethod
    def from_env(cls) -> "DemoConfig":
        return cls(
            device=os.environ.get("UNISS_PREFIX_STREAMING_DEVICE", "cuda:0"),
            max_audio_seconds=float(
                os.environ.get("UNISS_PREFIX_STREAMING_MAX_AUDIO_SECONDS", "60")
            ),
        )

    def validate(self) -> None:
        required = (
            self.adapter_dir / "adapter_config.json",
            self.adapter_dir / "adapter_model.safetensors",
            self.speech_tokenizer_dir / "glm4_tokenizer",
            self.speech_tokenizer_dir / "bicodec",
        )
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(f"missing frozen demo assets: {missing}")

