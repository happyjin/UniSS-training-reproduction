from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import pytest

from experiments.uniss_phase3_dense_aligned_streaming_pilot15_v1.data.packing import (
    build_session_token_sample,
)
from experiments.uniss_phase3_dense_aligned_streaming_pilot15_v1.data.schema import (
    DenseEvent,
    DenseSession,
)
from training import constants_uniss as c
from web_demo.runtime_parity_streaming_v2.session import (
    KVAppendResult,
    PersistentPromptSession,
    SessionPhase,
)


@dataclass(frozen=True)
class Call:
    kind: str
    values: tuple[int, ...]
    incoming_cache: tuple[int, ...] | None
    capture_last_hidden: bool


class RecordingBackend:
    """CPU-only backend whose cache is exactly the committed token transcript."""

    def __init__(self) -> None:
        self.calls: list[Call] = []

    def append_token_ids(
        self,
        token_ids: Sequence[int],
        *,
        past_key_values: Any,
        capture_last_hidden: bool = False,
    ) -> KVAppendResult:
        incoming = past_key_values
        values = tuple(int(value) for value in token_ids)
        self.calls.append(Call("tokens", values, incoming, capture_last_hidden))
        cache = (*(() if incoming is None else incoming), *values)
        hidden = ("hidden_at", len(cache) - 1) if capture_last_hidden else None
        return KVAppendResult(
            past_key_values=cache,
            logits=("logits_after", cache[-1]),
            last_hidden=hidden,
        )

    def append_source_codes(
        self,
        source_codes: Sequence[int],
        canonical_token_ids: Sequence[int],
        *,
        past_key_values: Any,
    ) -> KVAppendResult:
        incoming = past_key_values
        codes = tuple(int(value) for value in source_codes)
        canonical = tuple(int(value) for value in canonical_token_ids)
        assert canonical == tuple(c.encode_glm_semantic(codes))
        self.calls.append(Call("source", canonical, incoming, False))
        cache = (*incoming, *canonical)
        return KVAppendResult(past_key_values=cache)


class FusedRecordingBackend(RecordingBackend):
    fuse_ticks = True

    def append_tick(
        self,
        source_codes: Sequence[int],
        canonical_token_ids: Sequence[int],
        *,
        past_key_values: Any,
    ) -> KVAppendResult:
        codes = tuple(int(value) for value in source_codes)
        canonical = tuple(int(value) for value in canonical_token_ids)
        assert canonical == tuple(c.encode_glm_semantic(codes))
        values = (c.TOKEN_START_GLM, *canonical, c.TOKEN_END_GLM)
        self.calls.append(Call("fused_tick", values, past_key_values, True))
        cache = (*(() if past_key_values is None else past_key_values), *values)
        return KVAppendResult(
            past_key_values=cache,
            logits=("logits_after", c.TOKEN_END_GLM),
            last_hidden=("hidden_at", len(cache) - 1),
        )


def _session(backend: RecordingBackend | None = None) -> PersistentPromptSession:
    return PersistentPromptSession(
        backend or RecordingBackend(),
        target_lang="cmn",
        speaker_global=tuple(range(32)),
    )


def test_wait_and_write_are_committed_to_one_persistent_cache() -> None:
    backend = RecordingBackend()
    session = _session(backend)
    header = session.transcript
    assert c.TOKEN_START_GLM not in header

    observation = session.begin_tick([10, 11])
    assert observation.last_hidden == ("hidden_at", len(session.transcript) - 1)
    first = session.commit_wait()
    assert first.action == "WAIT"
    assert first.continuation_logits == (
        "logits_after",
        c.TOKEN_WAIT_READ,
    )

    session.begin_tick([12])
    second = session.commit_write([101, 102], [20, 21, 22])
    assert second.action == "WRITE"
    assert second.text_ids == (101, 102)
    assert second.semantic_codes == (20, 21, 22)
    assert second.continuation_logits == (
        "logits_after",
        c.TOKEN_END_SEMANTIC,
    )

    expected = (
        *header,
        c.TOKEN_START_GLM,
        *c.encode_glm_semantic([10, 11]),
        c.TOKEN_END_GLM,
        c.TOKEN_WAIT_READ,
        c.TOKEN_START_GLM,
        *c.encode_glm_semantic([12]),
        c.TOKEN_END_GLM,
        c.TOKEN_WRITE_GENERATE,
        c.TOKEN_CMN,
        c.speed_token_id(1.0),
        c.TOKEN_START_CONTENT,
        101,
        102,
        c.TOKEN_END_CONTENT,
        c.TOKEN_START_SEMANTIC,
        *c.encode_bicodec_semantic([20, 21, 22]),
        c.TOKEN_END_SEMANTIC,
    )
    assert session.transcript == expected
    assert session.past_key_values == expected

    # Every backend call receives the cache produced by the immediately
    # preceding call: no branch observation or rebuilt history is involved.
    committed: tuple[int, ...] | None = None
    for call in backend.calls:
        assert call.incoming_cache == committed
        committed = (*(() if committed is None else committed), *call.values)


def test_policy_hidden_is_captured_at_every_end_glm() -> None:
    backend = RecordingBackend()
    session = _session(backend)
    first = session.begin_tick([])
    session.commit_wait()
    second = session.begin_tick([7])
    session.commit_wait()

    assert session.transcript[first.action_prediction_position] == c.TOKEN_END_GLM
    assert session.transcript[second.action_prediction_position] == c.TOKEN_END_GLM
    hidden_calls = [call for call in backend.calls if call.capture_last_hidden]
    assert [call.values for call in hidden_calls] == [
        (c.TOKEN_END_GLM,),
        (c.TOKEN_END_GLM,),
    ]


def test_fused_tick_preserves_transcript_and_action_position() -> None:
    backend = FusedRecordingBackend()
    session = _session(backend)
    observation = session.begin_tick([7, 8])
    session.commit_wait()
    assert session.transcript[observation.action_prediction_position] == c.TOKEN_END_GLM
    assert backend.calls[-2].kind == "fused_tick"
    assert backend.calls[-2].values == (
        c.TOKEN_START_GLM,
        *c.encode_glm_semantic([7, 8]),
        c.TOKEN_END_GLM,
    )


def test_state_machine_rejects_skipped_or_overlapping_ticks() -> None:
    session = _session()
    with pytest.raises(RuntimeError, match="action_pending"):
        session.commit_wait()

    session.begin_tick([1])
    with pytest.raises(RuntimeError, match="current phase is action_pending"):
        session.begin_tick([2])
    with pytest.raises(ValueError, match="at least one semantic"):
        session.commit_write([100], [])
    assert session.phase is SessionPhase.ACTION_PENDING

    session.commit_wait()
    session.begin_tick([2])
    session.begin_write()
    with pytest.raises(RuntimeError, match="current phase is write_text"):
        session.commit_wait()
    session.end_text()
    with pytest.raises(RuntimeError, match="at least one semantic"):
        session.finish_write()


def test_runtime_transcript_exactly_matches_dense_training_builder() -> None:
    events = (
        DenseEvent(
            event_index=0,
            wall_time_ms=160,
            source_end_ms=160,
            visible_source_token_end=2,
            action="READ",
            playback_buffer_before_ms=0,
            playback_buffer_after_ms=0,
            support_bucket=0,
            safe_pending_count=0,
        ),
        DenseEvent(
            event_index=1,
            wall_time_ms=320,
            source_end_ms=320,
            visible_source_token_end=3,
            action="WRITE",
            playback_buffer_before_ms=0,
            playback_buffer_after_ms=60,
            support_bucket=1,
            safe_pending_count=1,
            text_delta="A",
            target_word_start=0,
            target_word_end=1,
            semantic_start=0,
            semantic_end=3,
            target_audio_start_ms=0,
            target_audio_end_ms=60,
            earliest_safe_ms=320,
            final_write=True,
            source_finished=True,
        ),
    )
    dense = DenseSession(
        sample_id="parity-sample",
        source_manifest="unused.jsonl",
        source_index=0,
        split="valid",
        src_lang="eng",
        tgt_lang="cmn",
        source_duration_ms=320,
        target_duration_ms=60,
        source_glm_length=3,
        target_semantic_length=3,
        target_word_count=1,
        target_text="A",
        speaker_global=tuple(range(32)),
        events=events,
    )
    formal = {
        "id": "parity-sample",
        "source_glm": [10, 11, 12],
        "target_bicodec": [20, 21, 22],
    }
    encode = lambda text: [101] if text else []
    training = build_session_token_sample(dense, formal, encode)

    runtime = _session()
    cursor = 0
    for event in dense.events:
        visible = event.visible_source_token_end
        runtime.begin_tick(formal["source_glm"][cursor:visible])
        cursor = visible
        if event.action == "READ":
            runtime.commit_wait()
        else:
            runtime.commit_write(
                encode(event.text_delta),
                formal["target_bicodec"][event.semantic_start : event.semantic_end],
            )
    runtime.finish_session()

    # Training stores the standard next-token-shifted pair; restore the final
    # unshifted EOS to compare the exact causal transcript.
    training_full = (*training.tokens, training.labels[-1])
    assert runtime.transcript == training_full
    assert runtime.past_key_values == training_full
    assert [tick.action_prediction_position for tick in runtime.committed_ticks] == [
        int(annotation["action_position"]) for annotation in training.annotations
    ]


def test_session_cannot_mutate_after_eos() -> None:
    session = _session()
    session.begin_tick([1])
    session.commit_wait()
    transcript = session.finish_session()
    assert transcript[-1] == c.TOKEN_EOS
    assert session.phase is SessionPhase.CLOSED
    with pytest.raises(RuntimeError, match="current phase is closed"):
        session.begin_tick([2])
