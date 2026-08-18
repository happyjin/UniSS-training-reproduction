from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.data.gold_trajectory import (
    build_gold_trajectory,
)
from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.data.schema import (
    E2ETrajectory,
    validate_trajectory,
)


DIGEST_A = hashlib.sha256(b"v1").hexdigest()
DIGEST_B = hashlib.sha256(b"phase3").hexdigest()


def _record(audio: Path) -> dict[str, object]:
    return {
        "id": "sample-1",
        "src_lang": "eng",
        "tgt_lang": "cmn",
        "transcription": "Good morning everyone.",
        "translation": "早上好，大家。",
        "source_audio": str(audio),
        "source_duration_ms": 1000,
        "source_words": [
            {"text": "Good", "start_ms": 0, "end_ms": 240},
            {"text": "morning", "start_ms": 240, "end_ms": 560},
            {"text": "everyone", "start_ms": 560, "end_ms": 960},
        ],
        "source_glm": [1, 2, 3, 4],
        "source_glm_end_ms": [160, 400, 720, 960],
        "target_words": [
            {"text": "早上好", "start_ms": 0, "end_ms": 300},
            {"text": "大家", "start_ms": 300, "end_ms": 700},
        ],
        "target_support": [
            {"alignment_confidence": 0.95},
            {"alignment_confidence": 0.90},
        ],
        "target_bicodec": [10, 11, 12, 13, 14, 15],
        "bicodec_global": list(range(32)),
        "micro_write_events": [
            {
                "micro_write_index": 0,
                "text": "早上好",
                "target_word_start": 0,
                "target_word_end": 1,
                "semantic_start": 0,
                "semantic_end": 2,
                "semantic_count": 2,
                "support_end_ms": 240,
                "safe_if_source_ms_gte": 320,
                "future_monotonic_support": True,
            },
            {
                "micro_write_index": 1,
                "text": "大家",
                "target_word_start": 1,
                "target_word_end": 2,
                "semantic_start": 2,
                "semantic_end": 6,
                "semantic_count": 4,
                "support_end_ms": 960,
                "safe_if_source_ms_gte": 1000,
                "future_monotonic_support": True,
            },
        ],
        "formal_a45_pass": True,
        "formal_a68_pass": True,
    }


def test_build_is_lossless_and_has_prefinal_write(tmp_path: Path) -> None:
    audio = tmp_path / "audio.flac"
    audio.write_bytes(b"synthetic-audio-container")
    trajectory = build_gold_trajectory(
        _record(audio),
        split="train",
        source_manifest=str(tmp_path / "source.jsonl"),
        source_manifest_record=7,
        v1_checkpoint_sha256=DIGEST_A,
        phase3_teacher_sha256=DIGEST_B,
        hash_audio=True,
    )
    metrics = validate_trajectory(trajectory, require_audio_hash=True)
    assert trajectory.events[-1].gold_source_prefix == "Good morning everyone"
    assert trajectory.events[-1].target_text_prefix == "早上好大家"
    assert metrics["target_semantic_tokens"] == 6
    assert metrics["prefinal_target_writes"] == 1
    assert trajectory.source_audio_sha256 == hashlib.sha256(audio.read_bytes()).hexdigest()


def test_serialized_round_trip(tmp_path: Path) -> None:
    audio = tmp_path / "audio.flac"
    audio.write_bytes(b"audio")
    original = build_gold_trajectory(
        _record(audio),
        split="valid",
        source_manifest=str(tmp_path / "source.jsonl"),
        source_manifest_record=0,
        v1_checkpoint_sha256=DIGEST_A,
        phase3_teacher_sha256=DIGEST_B,
    )
    recovered = E2ETrajectory.from_mapping(original.to_mapping())
    assert recovered == original
    validate_trajectory(recovered)


def test_semantic_gap_is_rejected(tmp_path: Path) -> None:
    audio = tmp_path / "audio.flac"
    audio.write_bytes(b"audio")
    record = _record(audio)
    record["micro_write_events"][1]["semantic_start"] = 3  # type: ignore[index]
    with pytest.raises(ValueError, match="gap, overlap"):
        build_gold_trajectory(
            record,
            split="train",
            source_manifest=str(tmp_path / "source.jsonl"),
            source_manifest_record=0,
            v1_checkpoint_sha256=DIGEST_A,
            phase3_teacher_sha256=DIGEST_B,
        )

def test_future_target_leak_is_rejected(tmp_path: Path) -> None:
    audio = tmp_path / "audio.flac"
    audio.write_bytes(b"audio")
    record = _record(audio)
    record["micro_write_events"][0]["support_end_ms"] = 500  # type: ignore[index]
    with pytest.raises(ValueError, match="future target content"):
        build_gold_trajectory(
            record,
            split="train",
            source_manifest=str(tmp_path / "source.jsonl"),
            source_manifest_record=0,
            v1_checkpoint_sha256=DIGEST_A,
            phase3_teacher_sha256=DIGEST_B,
        )
