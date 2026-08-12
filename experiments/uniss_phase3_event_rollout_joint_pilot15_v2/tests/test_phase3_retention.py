from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.uniss_phase3_event_rollout_joint_pilot15_v2.evaluation.merge_phase3_retention import (
    merge,
)


def _write(path: Path, rows) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _row(sample_id: str, mode: str):
    return {
        "id": sample_id,
        "mode": mode,
        "generated_translation": "hello",
        "semantic_token_count": 8,
        "has_eos": True,
        "audio_path": "/audio.wav",
        "audio_finite": True,
        "audio_non_silent_fraction": 0.5,
        "generation_seconds": 1.0,
        "error": None,
    }


def test_retention_merge_requires_paired_systems(tmp_path: Path) -> None:
    part = tmp_path / "part.jsonl"
    _write(part, [_row("a", "phase3_v4"), _row("a", "streaming_adapter")])
    report = merge([part], tmp_path / "output")
    assert report["paired_complete"] is True
    assert report["samples"] == 1
    assert report["groups"]["streaming_adapter"]["playable_audio_rate"] == 1.0


def test_retention_merge_rejects_unpaired_sample(tmp_path: Path) -> None:
    part = tmp_path / "part.jsonl"
    _write(part, [_row("a", "phase3_v4")])
    with pytest.raises(ValueError, match="unpaired"):
        merge([part], tmp_path / "output")
