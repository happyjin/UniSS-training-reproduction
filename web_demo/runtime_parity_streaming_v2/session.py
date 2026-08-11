"""Strict training/runtime prompt parity over one persistent Qwen KV cache.

The dense training sequence is an append-only transcript::

    header
    START_GLM source_delta END_GLM WAIT
    START_GLM source_delta END_GLM WRITE lang speed
        START_CONTENT text_delta END_CONTENT
        START_SEMANTIC semantic_delta END_SEMANTIC
    ...

The historical runtime diverged from that transcript: it kept one open GLM
envelope, observed policy on a temporary cache branch, and reconstructed a
compressed target history.  This module makes that divergence impossible by
using a state machine and by replacing the persistent cache after *every*
append.  Source codes may still use continuous frontend-adapted embeddings;
``KVBackend.append_source_codes`` receives both raw codes and their canonical
token IDs so its cache length and the training transcript remain identical.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal, Protocol, Sequence

from training import constants_uniss as c


Action = Literal["WAIT", "WRITE"]


@dataclass(frozen=True)
class KVAppendResult:
    """Result of appending one contiguous block to the persistent model cache."""

    past_key_values: Any
    logits: Any = None
    last_hidden: Any = None


class KVBackend(Protocol):
    """Minimal model-specific bridge required by :class:`PersistentPromptSession`.

    A real backend normally calls Qwen with ``use_cache=True``.  Token blocks
    use ``input_ids``.  Source blocks use the causal frontend adapter plus Qwen
    ``inputs_embeds`` while preserving one cache position per canonical GLM
    token ID.
    """

    def append_token_ids(
        self,
        token_ids: Sequence[int],
        *,
        past_key_values: Any,
        capture_last_hidden: bool = False,
    ) -> KVAppendResult: ...

    def append_source_codes(
        self,
        source_codes: Sequence[int],
        canonical_token_ids: Sequence[int],
        *,
        past_key_values: Any,
    ) -> KVAppendResult: ...


class SessionPhase(str, Enum):
    READY = "ready"
    ACTION_PENDING = "action_pending"
    WRITE_TEXT = "write_text"
    WRITE_SEMANTIC = "write_semantic"
    CLOSED = "closed"


@dataclass(frozen=True)
class TickObservation:
    """Policy observation at the training-identical ``END_GLM`` position."""

    event_index: int
    source_codes: tuple[int, ...]
    transcript_start: int
    action_prediction_position: int
    last_hidden: Any


@dataclass(frozen=True)
class CommittedTick:
    """One immutable tick after its actual action/output has entered main KV."""

    event_index: int
    source_codes: tuple[int, ...]
    action: Action
    text_ids: tuple[int, ...]
    semantic_codes: tuple[int, ...]
    transcript_start: int
    action_prediction_position: int
    transcript_end: int


@dataclass
class _PendingTick:
    event_index: int
    source_codes: tuple[int, ...]
    transcript_start: int
    action_prediction_position: int
    text_ids: list[int]
    semantic_codes: list[int]


def build_session_header(
    target_lang: str,
    speaker_global: Sequence[int],
    *,
    speed: float = 1.0,
) -> tuple[int, ...]:
    """Return the exact dense-training header, without an open GLM envelope."""

    normalized = c.normalize_language(target_lang)
    global_codes = tuple(int(value) for value in speaker_global)
    if len(global_codes) != 32:
        raise ValueError("speaker_global must contain exactly 32 tokens")
    return (
        c.TOKEN_TASK_STREAMING_S2ST,
        c.TOKEN_STREAMING_MODE,
        c.TOKEN_DYNAMIC_MODE,
        c.language_token_id(normalized),
        c.speed_token_id(speed),
        *c.wrap_global_tokens(global_codes),
    )


class PersistentPromptSession:
    """Append one runtime session to exactly one evolving KV cache.

    The public methods mirror the only legal grammar.  In particular, callers
    cannot start the next source tick until the current WAIT or complete WRITE
    has been committed.  ``transcript`` is the canonical discrete sequence
    used for parity audits even when source positions were fed as embeddings.
    """

    def __init__(
        self,
        backend: KVBackend,
        *,
        target_lang: str,
        speaker_global: Sequence[int],
        speed: float = 1.0,
    ) -> None:
        self.backend = backend
        self.target_lang = c.normalize_language(target_lang)
        self.speed = float(speed)
        self.header = build_session_header(
            self.target_lang, speaker_global, speed=self.speed
        )
        self._transcript: list[int] = []
        self._past_key_values: Any = None
        self._phase = SessionPhase.READY
        self._pending: _PendingTick | None = None
        self._ticks: list[CommittedTick] = []
        self._append_token_ids(self.header)

    @property
    def phase(self) -> SessionPhase:
        return self._phase

    @property
    def transcript(self) -> tuple[int, ...]:
        return tuple(self._transcript)

    @property
    def past_key_values(self) -> Any:
        return self._past_key_values

    @property
    def committed_ticks(self) -> tuple[CommittedTick, ...]:
        return tuple(self._ticks)

    def _require_phase(self, expected: SessionPhase) -> None:
        if self._phase is not expected:
            raise RuntimeError(
                f"operation requires phase {expected.value}, current phase is "
                f"{self._phase.value}"
            )

    @staticmethod
    def _validate_source_codes(values: Sequence[int]) -> tuple[int, ...]:
        codes = tuple(int(value) for value in values)
        for value in codes:
            c.validate_range(value, c.GLM_SEMANTIC_SIZE, "source_glm")
        return codes

    @staticmethod
    def _validate_text_ids(values: Sequence[int]) -> tuple[int, ...]:
        token_ids = tuple(int(value) for value in values)
        for value in token_ids:
            if not 0 <= value <= c.QWEN_BASE_VOCAB_END:
                raise ValueError(
                    f"text token {value} is outside the base Qwen vocabulary"
                )
        return token_ids

    @staticmethod
    def _validate_semantic_codes(values: Sequence[int]) -> tuple[int, ...]:
        codes = tuple(int(value) for value in values)
        for value in codes:
            c.validate_range(value, c.BICODEC_SEMANTIC_SIZE, "target_bicodec")
        return codes

    def _append_token_ids(
        self, values: Sequence[int], *, capture_last_hidden: bool = False
    ) -> KVAppendResult:
        token_ids = tuple(int(value) for value in values)
        if not token_ids:
            raise ValueError("cannot append an empty token block")
        for token_id in token_ids:
            c.validate_token_id(token_id)
        result = self.backend.append_token_ids(
            token_ids,
            past_key_values=self._past_key_values,
            capture_last_hidden=capture_last_hidden,
        )
        self._past_key_values = result.past_key_values
        self._transcript.extend(token_ids)
        return result

    def begin_tick(self, source_codes: Sequence[int]) -> TickObservation:
        """Commit ``START_GLM + delta + END_GLM`` and expose policy hidden state."""

        self._require_phase(SessionPhase.READY)
        codes = self._validate_source_codes(source_codes)
        event_index = len(self._ticks)
        transcript_start = len(self._transcript)
        self._append_token_ids((c.TOKEN_START_GLM,))
        if codes:
            canonical = tuple(c.encode_glm_semantic(codes))
            result = self.backend.append_source_codes(
                codes,
                canonical,
                past_key_values=self._past_key_values,
            )
            self._past_key_values = result.past_key_values
            self._transcript.extend(canonical)
        result = self._append_token_ids(
            (c.TOKEN_END_GLM,), capture_last_hidden=True
        )
        action_position = len(self._transcript) - 1
        self._pending = _PendingTick(
            event_index=event_index,
            source_codes=codes,
            transcript_start=transcript_start,
            action_prediction_position=action_position,
            text_ids=[],
            semantic_codes=[],
        )
        self._phase = SessionPhase.ACTION_PENDING
        return TickObservation(
            event_index=event_index,
            source_codes=codes,
            transcript_start=transcript_start,
            action_prediction_position=action_position,
            last_hidden=result.last_hidden,
        )

    def _finish_tick(self, action: Action) -> CommittedTick:
        if self._pending is None:
            raise RuntimeError("no pending tick")
        tick = CommittedTick(
            event_index=self._pending.event_index,
            source_codes=self._pending.source_codes,
            action=action,
            text_ids=tuple(self._pending.text_ids),
            semantic_codes=tuple(self._pending.semantic_codes),
            transcript_start=self._pending.transcript_start,
            action_prediction_position=self._pending.action_prediction_position,
            transcript_end=len(self._transcript),
        )
        self._ticks.append(tick)
        self._pending = None
        self._phase = SessionPhase.READY
        return tick

    def commit_wait(self) -> CommittedTick:
        """Commit the model's actual WAIT token to the persistent main cache."""

        self._require_phase(SessionPhase.ACTION_PENDING)
        self._append_token_ids((c.TOKEN_WAIT_READ,))
        return self._finish_tick("WAIT")

    def begin_write(self) -> KVAppendResult:
        """Commit WRITE and the fixed text prefix; return logits for first text token."""

        self._require_phase(SessionPhase.ACTION_PENDING)
        result = self._append_token_ids(
            (
                c.TOKEN_WRITE_GENERATE,
                c.language_token_id(self.target_lang),
                c.speed_token_id(self.speed),
                c.TOKEN_START_CONTENT,
            )
        )
        self._phase = SessionPhase.WRITE_TEXT
        return result

    def append_text_ids(self, values: Sequence[int]) -> KVAppendResult:
        """Commit generated text IDs and return logits after the last committed ID."""

        self._require_phase(SessionPhase.WRITE_TEXT)
        token_ids = self._validate_text_ids(values)
        if not token_ids:
            raise ValueError("cannot append an empty text block")
        result = self._append_token_ids(token_ids)
        assert self._pending is not None
        self._pending.text_ids.extend(token_ids)
        return result

    def end_text(self) -> KVAppendResult:
        """Commit the text boundary and semantic start marker to main KV."""

        self._require_phase(SessionPhase.WRITE_TEXT)
        result = self._append_token_ids(
            (c.TOKEN_END_CONTENT, c.TOKEN_START_SEMANTIC)
        )
        self._phase = SessionPhase.WRITE_SEMANTIC
        return result

    def append_semantic_codes(self, values: Sequence[int]) -> KVAppendResult:
        """Commit generated BiCodec semantic codes to main KV."""

        self._require_phase(SessionPhase.WRITE_SEMANTIC)
        codes = self._validate_semantic_codes(values)
        if not codes:
            raise ValueError("cannot append an empty semantic block")
        result = self._append_token_ids(c.encode_bicodec_semantic(codes))
        assert self._pending is not None
        self._pending.semantic_codes.extend(codes)
        return result

    def finish_write(self) -> CommittedTick:
        """Close a non-empty semantic block and finish the current WRITE tick."""

        self._require_phase(SessionPhase.WRITE_SEMANTIC)
        assert self._pending is not None
        if not self._pending.semantic_codes:
            raise RuntimeError("WRITE must commit at least one semantic code")
        self._append_token_ids((c.TOKEN_END_SEMANTIC,))
        return self._finish_tick("WRITE")

    def commit_write(
        self, text_ids: Sequence[int], semantic_codes: Sequence[int]
    ) -> CommittedTick:
        """Atomically validate, then append one complete training-identical WRITE."""

        self._require_phase(SessionPhase.ACTION_PENDING)
        text = self._validate_text_ids(text_ids)
        semantic = self._validate_semantic_codes(semantic_codes)
        if not semantic:
            raise ValueError("WRITE must contain at least one semantic code")
        self.begin_write()
        if text:
            self.append_text_ids(text)
        self.end_text()
        self.append_semantic_codes(semantic)
        return self.finish_write()

    def finish_session(self) -> tuple[int, ...]:
        """Append EOS after a complete tick and permanently close the session."""

        self._require_phase(SessionPhase.READY)
        if not self._ticks:
            raise RuntimeError("cannot finish a session without any committed ticks")
        self._append_token_ids((c.TOKEN_EOS,))
        self._phase = SessionPhase.CLOSED
        return self.transcript


__all__ = [
    "Action",
    "CommittedTick",
    "KVAppendResult",
    "KVBackend",
    "PersistentPromptSession",
    "SessionPhase",
    "TickObservation",
    "build_session_header",
]
