"""Frozen configuration for the Student-v2 + R2 streaming stereo demo."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from web_demo.streaming_s2st_r2_v1.config import StreamingDemoConfig


DEMO_ROOT = Path(__file__).resolve().parent
REPO_ROOT = DEMO_ROOT.parents[1]


@dataclass(frozen=True)
class StudentV2StreamingConfig(StreamingDemoConfig):
    demo_root: Path = DEMO_ROOT
    repo_root: Path = REPO_ROOT
    output_root: Path = DEMO_ROOT / "runtime_outputs"
    log_root: Path = DEMO_ROOT / "runtime_logs"
    student_checkpoint_path: Path = (
        REPO_ROOT
        / "checkpoints/simul_uniss_subsecond_v2/"
        "stage_b_v2_prefix80_finetune_100k_v1/best.pt"
    )
    frontend_feed_ms: int = 160
    frontend_right_context_ms: int = 80
    # Keep the audited R2 controller cadence unchanged.  Student-v2 consumes
    # native 160 ms PCM internally and accumulates emissions for each R2 tick.
    chunk_ms: int = 640

    @classmethod
    def from_env(cls) -> "StudentV2StreamingConfig":
        return cls(
            device=os.environ.get("UNISS_STUDENT_V2_DEMO_DEVICE", "cuda:0"),
            max_audio_seconds=float(
                os.environ.get("UNISS_STUDENT_V2_DEMO_MAX_AUDIO_SECONDS", "60")
            ),
            microphone_max_audio_seconds=float(
                os.environ.get("UNISS_STUDENT_V2_DEMO_MIC_MAX_AUDIO_SECONDS", "90")
            ),
            output_ttl_hours=float(
                os.environ.get("UNISS_STUDENT_V2_DEMO_OUTPUT_TTL_HOURS", "24")
            ),
        )

    @property
    def model_label(self) -> str:
        return "Stage-B-v2 prefix-80 Student + Stage7A Reward-v2 R2 controller"

    def validate(self) -> None:
        super().validate()
        if not self.student_checkpoint_path.is_file():
            raise FileNotFoundError(self.student_checkpoint_path)
        if self.frontend_feed_ms <= 0 or self.frontend_feed_ms % 40:
            raise ValueError("frontend_feed_ms must be a positive multiple of 40 ms")
        if self.frontend_right_context_ms != 80:
            raise ValueError("the frozen Student-v2 checkpoint uses 80 ms right context")
