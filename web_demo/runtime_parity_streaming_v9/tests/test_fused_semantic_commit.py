from __future__ import annotations

import pytest

from web_demo.runtime_parity_streaming_v2.tests.test_session import RecordingBackend
from web_demo.runtime_parity_streaming_v5.inference import (
    ParallelSemanticPromptSession,
)
from web_demo.runtime_parity_streaming_v9.inference import (
    FusedSemanticPromptSession,
)


def _prepare(session):
    session.begin_tick([3, 5])
    session.begin_write()
    session.append_text_ids([101, 102])
    session.end_text_with_hidden()
    return session


def test_fused_commit_is_transcript_and_cache_equivalent() -> None:
    old_backend = RecordingBackend()
    new_backend = RecordingBackend()
    old = _prepare(
        ParallelSemanticPromptSession(
            old_backend, target_lang="cmn", speaker_global=tuple(range(32))
        )
    )
    new = _prepare(
        FusedSemanticPromptSession(
            new_backend, target_lang="cmn", speaker_global=tuple(range(32))
        )
    )

    old.append_semantic_codes([7, 8, 9])
    old_tick = old.finish_write()
    new_tick = new.commit_semantic_block([7, 8, 9])

    assert new.transcript == old.transcript
    assert new.past_key_values == old.past_key_values
    assert new.phase == old.phase
    assert new.committed_ticks == old.committed_ticks
    assert new_tick.continuation_logits == old_tick.continuation_logits
    assert len(new_backend.calls) == len(old_backend.calls) - 1


def test_fused_commit_rejects_empty_block_without_mutating_session() -> None:
    session = _prepare(
        FusedSemanticPromptSession(
            RecordingBackend(), target_lang="cmn", speaker_global=tuple(range(32))
        )
    )
    before = session.transcript
    with pytest.raises(ValueError, match="at least one semantic"):
        session.commit_semantic_block([])
    assert session.transcript == before
