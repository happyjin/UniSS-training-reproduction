from __future__ import annotations

from web_demo.runtime_parity_streaming_v2.tests.test_session import RecordingBackend
from web_demo.runtime_parity_streaming_v5.inference import (
    ParallelSemanticPromptSession,
)


def test_end_text_captures_start_semantic_hidden() -> None:
    backend = RecordingBackend()
    session = ParallelSemanticPromptSession(
        backend, target_lang="zh", speaker_global=list(range(32))
    )
    session.begin_tick([1, 2])
    session.begin_write()
    session.append_text_ids([7])
    result = session.end_text_with_hidden()
    assert result.last_hidden is not None
