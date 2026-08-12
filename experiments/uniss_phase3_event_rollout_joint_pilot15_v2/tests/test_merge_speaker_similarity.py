from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.uniss_phase3_event_rollout_joint_pilot15_v2.evaluation.merge_speaker_similarity import (
    merge,
)


def _write(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_merge_speaker_similarity_proves_coverage(tmp_path: Path) -> None:
    input_path = tmp_path / "results.jsonl"
    _write(
        input_path,
        [
            {"id": "a", "mode": "exact_runtime", "audio_path": "/a.wav", "fixed_speaker_reference_audio_path": "/r.wav"},
            {"id": "b", "mode": "exact_runtime", "audio_path": "/b.wav", "fixed_speaker_reference_audio_path": "/r.wav"},
        ],
    )
    part = tmp_path / "part.jsonl"
    _write(
        part,
        [
            {"id": "a", "mode": "exact_runtime", "src_lang": "eng", "tgt_lang": "cmn", "speaker_similarity": 0.8},
            {"id": "b", "mode": "exact_runtime", "src_lang": "cmn", "tgt_lang": "eng", "speaker_similarity": 0.7},
        ],
    )
    report = merge(input_path, [part], tmp_path / "output")
    assert report["coverage"]["complete"] is True
    assert report["scored_count"] == 2


def test_merge_speaker_similarity_rejects_missing_row(tmp_path: Path) -> None:
    input_path = tmp_path / "results.jsonl"
    _write(
        input_path,
        [{"id": "a", "mode": "exact_runtime", "audio_path": "/a.wav", "fixed_speaker_reference_audio_path": "/r.wav"}],
    )
    with pytest.raises(ValueError, match="coverage mismatch"):
        merge(input_path, [], tmp_path / "output")
