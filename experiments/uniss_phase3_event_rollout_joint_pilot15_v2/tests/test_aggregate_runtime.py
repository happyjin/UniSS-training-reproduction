from __future__ import annotations

import pytest

from experiments.uniss_phase3_event_rollout_joint_pilot15_v2.evaluation.aggregate_runtime import (
    aggregate,
)


def _sample(sample_id: str, *, first_write: int = 480, safe: int = 320):
    return {
        "sample_id": sample_id,
        "src_lang": "eng",
        "tgt_lang": "cmn",
        "source_duration_ms": 1600,
        "target_text": "你好",
        "generated_text": "你好",
        "events": [
            {"action": "WAIT", "source_end_ms": 160, "wall_end_ms": 200},
            {"action": "WRITE", "source_end_ms": first_write, "wall_end_ms": 700},
            {"action": "WRITE", "source_end_ms": 800, "wall_end_ms": 1050},
        ],
        "natural_writes": 2,
        "forced_writes": 0,
        "committed_revision_violations": 0,
        "first_write_source_ms": first_write,
        "first_audio_source_ms": first_write,
        "first_write_wall_ms": 700,
        "first_audio_wall_ms": 700,
        "maximum_compute_backlog_ms": 250,
        "source_finished_before_first_write": False,
        "natural_eos": True,
        "rtf": 0.5,
        "quality_passed": True,
        "oracle_first_safe_write_ms": safe,
        "translation_audio_samples": 16000,
        "translation_audio_sample_rate": 16000,
        "translation_audio_finite": True,
        "severe_semantic_collapse": False,
    }


def test_aggregate_keeps_first_useful_audio_not_evaluable() -> None:
    report, rows = aggregate([{"checkpoint": "/ckpt", "samples": [_sample("a")]}])
    group = report["groups"]["all"]
    assert group["natural_write_sample_rate"] == 1.0
    assert group["first_write_premature_rate"] == 0.0
    assert group["first_useful_audio_wall_ms"] == "not_evaluable_until_prefix_asr"
    assert rows[0]["translation_ref"] == "你好"
    assert rows[0]["audio_duration_seconds"] == 1.0


def test_aggregate_detects_premature_first_write() -> None:
    report, _ = aggregate([{"samples": [_sample("a", first_write=160, safe=320)]}])
    assert report["groups"]["all"]["first_write_premature_rate"] == 1.0


def test_aggregate_rejects_duplicate_samples() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        aggregate([{"samples": [_sample("a")]}, {"samples": [_sample("a")] }])
