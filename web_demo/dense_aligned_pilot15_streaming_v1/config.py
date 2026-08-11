"""Frozen paths for the validation-best dense-aligned streaming demo."""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

from web_demo.true_subsecond_pilot15_streaming_v1.config import (
    DemoConfig as BaseDemoConfig,
)


DEMO_ROOT = Path(__file__).resolve().parent
REPO_ROOT = DEMO_ROOT.parents[1]
CHECKPOINT = (
    REPO_ROOT
    / "checkpoints/uniss_phase3_dense_aligned_streaming_pilot15_v1/iter_0000500"
)
EXPORTED_RUNTIME = (
    REPO_ROOT
    / "checkpoints/exported_adapters/"
    "uniss_phase3_dense_aligned_streaming_pilot15_iter_0000500_runtime_v2"
)


def load_config() -> BaseDemoConfig:
    """Return an isolated config while preserving the verified causal runtime."""

    base = BaseDemoConfig.from_env()
    return replace(
        base,
        demo_root=DEMO_ROOT,
        checkpoint=Path(
            os.environ.get("UNISS_DENSE_STREAMING_CHECKPOINT", str(CHECKPOINT))
        ),
        exported_runtime=Path(
            os.environ.get(
                "UNISS_DENSE_STREAMING_EXPORT", str(EXPORTED_RUNTIME)
            )
        ),
        output_root=DEMO_ROOT / "runtime_outputs",
        device=os.environ.get("UNISS_DENSE_STREAMING_DEVICE", "cuda:0"),
        decision_chunk_ms=int(
            os.environ.get("UNISS_DENSE_STREAMING_CHUNK_MS", "320")
        ),
    )

