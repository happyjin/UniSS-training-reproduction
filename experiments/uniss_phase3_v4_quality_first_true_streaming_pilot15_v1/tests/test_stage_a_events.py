from __future__ import annotations

import pytest

from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v1.stage_a_causal_whisper_asr.audit_data import (
    _ranges,
)
from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v1.stage_a_causal_whisper_asr.events import (
    MAX_EMPTY_EVENT_GAP_MS,
    build_asr_event_session,
)


def record(language: str = "eng") -> dict[str, object]:
    words = (
        [
            {"text": "Good", "start_ms": 80, "end_ms": 320},
            {"text": "morning", "start_ms": 400, "end_ms": 880},
            {"text": "everyone", "start_ms": 2400, "end_ms": 3040},
        ]
        if language == "eng"
        else [
            {"text": "大家", "start_ms": 80, "end_ms": 320},
            {"text": "早上好", "start_ms": 400, "end_ms": 880},
            {"text": "朋友", "start_ms": 2400, "end_ms": 3040},
        ]
    )
    return {
        "id": f"sample-{language}",
        "src_lang": language,
        "source_duration_ms": 3300,
        "source_words": words,
        "source_glm": list(range(42)),
        "source_glm_end_ms": [80 * (index + 1) for index in range(42)],
    }


@pytest.mark.parametrize(
    ("language", "expected"),
    [("eng", "Good morning everyone"), ("cmn", "大家早上好朋友")],
)
def test_events_are_append_only_and_cover_text_glm(language: str, expected: str) -> None:
    session = build_asr_event_session(record(language))
    assert session.normalized_transcript == expected
    assert session.events[-1].is_final
    assert session.events[-1].committed_text == expected
    assert session.events[0].glm_start == 0
    assert session.events[-1].glm_end == 42
    assert session.prefinal_text_commit
    for previous, current in zip(session.events, session.events[1:]):
        assert previous.word_end == current.word_start
        assert previous.glm_end == current.glm_start
        assert current.committed_text.startswith(previous.committed_text)
        assert current.source_end_ms - previous.source_end_ms <= MAX_EMPTY_EVENT_GAP_MS


def test_event_builder_rejects_nonmonotonic_alignment() -> None:
    value = record()
    value["source_words"] = [
        {"text": "a", "start_ms": 300, "end_ms": 500},
        {"text": "b", "start_ms": 100, "end_ms": 200},
    ]
    with pytest.raises(ValueError, match="non-monotonic"):
        build_asr_event_session(value)


def test_parallel_ranges_cover_every_record_once() -> None:
    ranges = _ranges(17, 4)
    values = [index for start, stop in ranges for index in range(start, stop)]
    assert values == list(range(17))
    assert len(ranges) == 4

