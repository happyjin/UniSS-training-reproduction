"""Lossless runtime events and recovery from exact model-induced states.

V2 merged semantic-only top-level WRITEs into an earlier lexical WRITE.  That
changed both the action distribution and the semantic generation horizon.  V3
keeps every packed WRITE exactly as authored.  Empty text deltas are legal;
empty semantic deltas are not legal oracle targets.

Recovery is represented by a generated prefix plus an oracle suffix.  The
first supervised label is therefore predicted from the exact model state at
an action, text, semantic, semantic-END, or event-continuation divergence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from experiments.uniss_phase3_dense_aligned_streaming_pilot15_v1.data.packing import (
    PACK_SCHEMA,
    ROLE_ACTION,
    ROLE_BOUNDARY,
    ROLE_OBSERVED,
    ROLE_SEMANTIC,
    ROLE_TEXT,
)
from training import constants_uniss as c


RECOVERY_SCHEMA = "uniss_event_rollout_recovery_v3"
DIVERGENCE_KINDS = (
    "action",
    "text_token",
    "text_end",
    "semantic_token",
    "semantic_end",
    "event_continuation",
)


@dataclass(frozen=True)
class OracleEvent:
    event_index: int
    source_codes: tuple[int, ...]
    source_block: tuple[int, ...]
    action: str
    outcome_tokens: tuple[int, ...]
    outcome_roles: tuple[int, ...]
    continuation_token: int
    source_finished: bool
    previous_committed_length: int
    stable_target_length: int
    support_bucket: int
    chunk_end_ms: int
    soft_deadline_ms: int
    hard_deadline_ms: int
    deadline_forced: bool
    deadline_loss_enabled: bool
    playback_buffer_ms: int

    def __post_init__(self) -> None:
        if self.action not in {"WAIT", "WRITE"}:
            raise ValueError("oracle action must be WAIT or WRITE")
        if not self.source_block or self.source_block[0] != c.TOKEN_START_GLM:
            raise ValueError("source block must begin with START_GLM")
        if self.source_block[-1] != c.TOKEN_END_GLM:
            raise ValueError("source block must end with END_GLM")
        if len(self.outcome_tokens) != len(self.outcome_roles):
            raise ValueError("oracle outcome token/role lengths differ")
        expected = c.TOKEN_WAIT_READ if self.action == "WAIT" else c.TOKEN_WRITE_GENERATE
        if not self.outcome_tokens or self.outcome_tokens[0] != expected:
            raise ValueError("oracle outcome does not begin with its action token")
        if self.outcome_roles[0] != ROLE_ACTION:
            raise ValueError("oracle action token must have ROLE_ACTION")
        if self.continuation_token not in {c.TOKEN_START_GLM, c.TOKEN_EOS}:
            raise ValueError("continuation target must be START_GLM or EOS")


@dataclass(frozen=True)
class OracleSession:
    sample_id: str
    target_lang: str
    speaker_global: tuple[int, ...]
    header: tuple[int, ...]
    events: tuple[OracleEvent, ...]
    full_translation_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.sample_id or not self.events:
            raise ValueError("oracle session must have an ID and events")
        if len(self.speaker_global) != 32:
            raise ValueError("oracle speaker must contain 32 global codes")
        if [event.event_index for event in self.events] != list(range(len(self.events))):
            raise ValueError("oracle event indices are not contiguous")
        if self.events[-1].continuation_token != c.TOKEN_EOS:
            raise ValueError("last oracle event must supervise EOS")
        if any(event.continuation_token == c.TOKEN_EOS for event in self.events[:-1]):
            raise ValueError("only the final oracle event may supervise EOS")


@dataclass(frozen=True)
class ParsedWrite:
    text_ids: tuple[int, ...]
    semantic_codes: tuple[int, ...]


@dataclass(frozen=True)
class GeneratedTick:
    """One runtime tick, including whether grammar boundaries were natural."""

    action: str
    text_ids: tuple[int, ...] = ()
    semantic_codes: tuple[int, ...] = ()
    natural_text_end: bool = True
    natural_semantic_end: bool = True
    choose_eos: bool = False

    def __post_init__(self) -> None:
        if self.action not in {"WAIT", "WRITE"}:
            raise ValueError("generated action must be WAIT or WRITE")
        if self.action == "WAIT" and (self.text_ids or self.semantic_codes):
            raise ValueError("WAIT cannot carry a generated payload")
        for token in self.text_ids:
            if not 0 <= int(token) <= c.QWEN_BASE_VOCAB_END:
                raise ValueError("generated text escaped the base Qwen vocabulary")
        for code in self.semantic_codes:
            c.validate_range(int(code), c.BICODEC_SEMANTIC_SIZE, "target_bicodec")

    @property
    def grammar_valid(self) -> bool:
        return bool(
            self.action == "WAIT"
            or (
                self.semantic_codes
                and self.natural_text_end
                and self.natural_semantic_end
            )
        )


@dataclass(frozen=True)
class RecoveryPoint:
    event_index: int
    kind: str
    generated_text_prefix: int = 0
    generated_semantic_prefix: int = 0
    include_complete_generated_outcome: bool = False
    contains_corruption: bool = False

    def __post_init__(self) -> None:
        if self.kind not in DIVERGENCE_KINDS:
            raise ValueError(f"unsupported divergence kind: {self.kind}")
        if self.generated_text_prefix < 0 or self.generated_semantic_prefix < 0:
            raise ValueError("generated recovery prefix lengths must be non-negative")


@dataclass(frozen=True)
class RolloutTrace:
    sample_id: str
    transcript: tuple[int, ...]
    action_prefixes: tuple[tuple[int, ...], ...]
    source_positions: tuple[tuple[int, int], ...]
    generated_ticks: tuple[GeneratedTick, ...]
    stopped_early: bool

    def first_divergence(self, oracle: OracleSession) -> RecoveryPoint | None:
        for index, generated in enumerate(self.generated_ticks):
            point = first_tick_divergence(oracle.events[index], generated)
            if point is not None:
                return RecoveryPoint(index, **point)
        if len(self.generated_ticks) < len(oracle.events):
            return RecoveryPoint(len(self.generated_ticks), "action")
        return None


@dataclass(frozen=True)
class RecoveryExample:
    sample_id: str
    event_index: int
    divergence_kind: str
    tokens: tuple[int, ...]
    labels: tuple[int, ...]
    loss_mask: tuple[int, ...]
    token_roles: tuple[int, ...]
    position_ids: tuple[int, ...]
    frontend_positions: tuple[int, ...]
    frontend_ids: tuple[int, ...]
    recovery_position: int
    generated_prefix_length: int
    corrupted_prefix_tokens: int
    action_position: int
    action_target: int
    action_supervised: bool
    continuation_position: int
    continuation_target: int
    continuation_supervised: bool
    support_bucket: int
    chunk_end_ms: int
    soft_deadline_ms: int
    hard_deadline_ms: int
    deadline_forced: bool
    deadline_loss_enabled: bool
    previous_committed_length: int
    stable_target_length: int
    playback_buffer_ms: int
    translation_ids: tuple[int, ...]
    schema_version: str = RECOVERY_SCHEMA

    def __post_init__(self) -> None:
        lengths = {
            len(self.tokens),
            len(self.labels),
            len(self.loss_mask),
            len(self.token_roles),
            len(self.position_ids),
        }
        if len(lengths) != 1 or not self.tokens:
            raise ValueError("recovery tensors have inconsistent lengths")
        if len(self.frontend_positions) != len(self.frontend_ids):
            raise ValueError("recovery frontend position/code lengths differ")
        for position in (
            self.recovery_position,
            self.action_position,
            self.continuation_position,
        ):
            if not 0 <= position < len(self.tokens):
                raise ValueError("recovery position is outside the sequence")
        if self.labels[self.action_position] not in {
            c.TOKEN_WAIT_READ,
            c.TOKEN_WRITE_GENERATE,
        }:
            raise ValueError("recovery action label is not WAIT/WRITE")
        if self.labels[self.continuation_position] not in {
            c.TOKEN_START_GLM,
            c.TOKEN_EOS,
        }:
            raise ValueError("recovery continuation label is not START_GLM/EOS")
        if not self.loss_mask[self.recovery_position]:
            raise ValueError("recovery target position is not supervised")
        if any(self.loss_mask[: self.recovery_position]):
            raise ValueError("generated prefix must not receive oracle loss")


def parse_write_outcome(values: Sequence[int]) -> ParsedWrite:
    tokens = tuple(int(value) for value in values)
    if len(tokens) < 8 or tokens[0] != c.TOKEN_WRITE_GENERATE:
        raise ValueError("malformed WRITE outcome")
    try:
        content_start = tokens.index(c.TOKEN_START_CONTENT) + 1
        content_end = tokens.index(c.TOKEN_END_CONTENT, content_start)
        semantic_start = tokens.index(c.TOKEN_START_SEMANTIC, content_end + 1) + 1
        semantic_end = tokens.index(c.TOKEN_END_SEMANTIC, semantic_start)
    except ValueError as exc:
        raise ValueError("WRITE outcome has incomplete content/semantic grammar") from exc
    if semantic_end != len(tokens) - 1:
        raise ValueError("WRITE outcome has trailing tokens after END_SEMANTIC")
    semantic = tuple(
        int(value) - c.BICODEC_SEMANTIC_OFFSET
        for value in tokens[semantic_start:semantic_end]
    )
    if not semantic:
        raise ValueError("WRITE outcome has no semantic content")
    if any(not 0 <= value < c.BICODEC_SEMANTIC_SIZE for value in semantic):
        raise ValueError("WRITE semantic token escaped the BiCodec interval")
    return ParsedWrite(tokens[content_start:content_end], semantic)


def build_write_outcome(
    target_lang: str,
    text_ids: Sequence[int],
    semantic_codes: Sequence[int],
    *,
    speed: float = 1.0,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    text = tuple(int(value) for value in text_ids)
    semantic = tuple(int(value) for value in semantic_codes)
    if not semantic:
        raise ValueError("WRITE outcome requires semantic codes")
    tokens = (
        c.TOKEN_WRITE_GENERATE,
        c.language_token_id(target_lang),
        c.speed_token_id(speed),
        c.TOKEN_START_CONTENT,
        *text,
        c.TOKEN_END_CONTENT,
        c.TOKEN_START_SEMANTIC,
        *c.encode_bicodec_semantic(semantic),
        c.TOKEN_END_SEMANTIC,
    )
    roles = (
        ROLE_ACTION,
        ROLE_BOUNDARY,
        ROLE_BOUNDARY,
        ROLE_BOUNDARY,
        *([ROLE_TEXT] * len(text)),
        ROLE_BOUNDARY,
        ROLE_BOUNDARY,
        *([ROLE_SEMANTIC] * len(semantic)),
        ROLE_BOUNDARY,
    )
    return tokens, roles


def _decode_header(header: Sequence[int]) -> tuple[str, tuple[int, ...]]:
    values = tuple(int(value) for value in header)
    if len(values) != 39:
        raise ValueError(f"unexpected streaming header length: {len(values)}")
    if values[:3] != (
        c.TOKEN_TASK_STREAMING_S2ST,
        c.TOKEN_STREAMING_MODE,
        c.TOKEN_DYNAMIC_MODE,
    ):
        raise ValueError("packed session has an unexpected streaming header")
    language = (
        "eng"
        if values[3] == c.TOKEN_ENG
        else "cmn"
        if values[3] == c.TOKEN_CMN
        else None
    )
    if language is None:
        raise ValueError("packed session target language token is invalid")
    if values[5] != c.TOKEN_START_GLOBAL or values[-1] != c.TOKEN_END_GLOBAL:
        raise ValueError("packed session has an invalid global-speaker envelope")
    speaker = tuple(value - c.BICODEC_GLOBAL_OFFSET for value in values[6:-1])
    if len(speaker) != 32 or any(
        not 0 <= value < c.BICODEC_GLOBAL_SIZE for value in speaker
    ):
        raise ValueError("packed session speaker codes are invalid")
    return language, speaker


def oracle_sessions_from_pack(value: Mapping[str, object]) -> tuple[OracleSession, ...]:
    """Parse complete sessions without merging or relabeling any WRITE."""

    if value.get("schema_version") != PACK_SCHEMA:
        raise ValueError("unexpected dense pack schema")
    packed_tokens = [int(item) for item in value["tokens"]]  # type: ignore[index]
    packed_labels = [int(item) for item in value["labels"]]  # type: ignore[index]
    raw_sessions = value.get("sessions")
    if not isinstance(raw_sessions, list) or not raw_sessions:
        raise ValueError("dense pack contains no sessions")
    parsed_sessions: list[OracleSession] = []
    for raw_session in raw_sessions:
        session = dict(raw_session)
        start, end = (int(item) for item in session["boundary"])
        conceptual = tuple(packed_tokens[start:end] + [packed_labels[end - 1]])
        annotations = sorted(
            (dict(item) for item in session["annotations"]),
            key=lambda item: int(item["event_index"]),
        )
        tick_starts: list[int] = []
        action_positions: list[int] = []
        for annotation in annotations:
            action = int(annotation["action_position"]) - start
            source_ids = tuple(int(item) for item in annotation["frontend_ids"])
            tick_start = action - len(source_ids) - 1
            expected_source = (
                c.TOKEN_START_GLM,
                *c.encode_glm_semantic(source_ids),
                c.TOKEN_END_GLM,
            )
            if tick_start < 0 or conceptual[tick_start : action + 1] != expected_source:
                raise ValueError("dense source block differs from its inline codes")
            tick_starts.append(tick_start)
            action_positions.append(action)
        header = conceptual[: tick_starts[0]]
        target_lang, speaker = _decode_header(header)
        events: list[OracleEvent] = []
        for index, (annotation, action_position) in enumerate(
            zip(annotations, action_positions)
        ):
            stop = (
                tick_starts[index + 1]
                if index + 1 < len(tick_starts)
                else len(conceptual) - 1
            )
            outcome = tuple(conceptual[action_position + 1 : stop])
            if not outcome:
                raise ValueError("dense event has an empty outcome")
            if outcome[0] == c.TOKEN_WAIT_READ:
                if outcome != (c.TOKEN_WAIT_READ,):
                    raise ValueError("WAIT event carries unexpected payload")
                action = "WAIT"
                roles = (ROLE_ACTION,)
            elif outcome[0] == c.TOKEN_WRITE_GENERATE:
                parsed = parse_write_outcome(outcome)
                rebuilt, roles = build_write_outcome(
                    target_lang, parsed.text_ids, parsed.semantic_codes
                )
                if rebuilt != outcome:
                    raise ValueError("WRITE outcome is not canonical")
                action = "WRITE"
            else:
                raise ValueError("dense event outcome is neither WAIT nor WRITE")
            events.append(
                OracleEvent(
                    event_index=index,
                    source_codes=tuple(
                        int(item) for item in annotation["frontend_ids"]
                    ),
                    source_block=conceptual[
                        tick_starts[index] : action_position + 1
                    ],
                    action=action,
                    outcome_tokens=outcome,
                    outcome_roles=tuple(roles),
                    continuation_token=(
                        c.TOKEN_EOS
                        if index + 1 == len(annotations)
                        else c.TOKEN_START_GLM
                    ),
                    source_finished=bool(annotation["source_finished"]),
                    previous_committed_length=int(
                        annotation["previous_committed_length"]
                    ),
                    stable_target_length=int(annotation["stable_target_length"]),
                    support_bucket=int(annotation["support_bucket"]),
                    chunk_end_ms=int(annotation["chunk_end_ms"]),
                    soft_deadline_ms=int(annotation["soft_deadline_ms"]),
                    hard_deadline_ms=int(annotation["hard_deadline_ms"]),
                    deadline_forced=bool(annotation["deadline_forced"]),
                    deadline_loss_enabled=bool(
                        annotation["deadline_loss_enabled"]
                    ),
                    playback_buffer_ms=int(annotation["playback_buffer_ms"]),
                )
            )
        parsed_sessions.append(
            OracleSession(
                sample_id=str(session["sample_id"]),
                target_lang=target_lang,
                speaker_global=speaker,
                header=tuple(header),
                events=tuple(events),
                full_translation_ids=tuple(
                    int(item) for item in session["translation_ids"]
                ),
            )
        )
    return tuple(parsed_sessions)


def generated_outcome_tokens(
    session: OracleSession, generated: GeneratedTick
) -> tuple[int, ...]:
    """Return the tokens actually appended to the persistent runtime KV.

    Safety-ceiling closures are included because they entered the actual KV,
    while ``natural_*_end`` records whether the model selected those closures.
    """

    if generated.action == "WAIT":
        return (c.TOKEN_WAIT_READ,)
    return (
        c.TOKEN_WRITE_GENERATE,
        c.language_token_id(session.target_lang),
        c.speed_token_id(1.0),
        c.TOKEN_START_CONTENT,
        *generated.text_ids,
        c.TOKEN_END_CONTENT,
        c.TOKEN_START_SEMANTIC,
        *c.encode_bicodec_semantic(generated.semantic_codes),
        c.TOKEN_END_SEMANTIC,
    )


def _common_prefix(left: Sequence[int], right: Sequence[int]) -> int:
    count = 0
    for first, second in zip(left, right):
        if int(first) != int(second):
            break
        count += 1
    return count


def first_tick_divergence(
    expected: OracleEvent, generated: GeneratedTick
) -> dict[str, object] | None:
    """Locate the first decision/content divergence inside one event."""

    if generated.action != expected.action:
        return {"kind": "action"}
    if generated.action == "WAIT":
        expected_eos = expected.continuation_token == c.TOKEN_EOS
        if bool(generated.choose_eos) != expected_eos:
            return {
                "kind": "event_continuation",
                "include_complete_generated_outcome": True,
            }
        return None

    oracle = parse_write_outcome(expected.outcome_tokens)
    text_common = _common_prefix(generated.text_ids, oracle.text_ids)
    if text_common < min(len(generated.text_ids), len(oracle.text_ids)):
        return {"kind": "text_token", "generated_text_prefix": text_common}
    if len(generated.text_ids) != len(oracle.text_ids):
        # Keep any extra generated text in the model history before asking it
        # to close or resume the oracle suffix.
        return {
            "kind": "text_end",
            "generated_text_prefix": len(generated.text_ids),
            "contains_corruption": len(generated.text_ids) > text_common,
        }
    if not generated.natural_text_end:
        return {
            "kind": "text_end",
            "generated_text_prefix": len(generated.text_ids),
        }

    semantic_common = _common_prefix(generated.semantic_codes, oracle.semantic_codes)
    if semantic_common < min(
        len(generated.semantic_codes), len(oracle.semantic_codes)
    ):
        return {
            "kind": "semantic_token",
            "generated_text_prefix": len(generated.text_ids),
            "generated_semantic_prefix": semantic_common,
        }
    if len(generated.semantic_codes) != len(oracle.semantic_codes):
        return {
            "kind": "semantic_end",
            "generated_text_prefix": len(generated.text_ids),
            "generated_semantic_prefix": len(generated.semantic_codes),
            "contains_corruption": len(generated.semantic_codes) > semantic_common,
        }
    if not generated.natural_semantic_end:
        return {
            "kind": "semantic_end",
            "generated_text_prefix": len(generated.text_ids),
            "generated_semantic_prefix": len(generated.semantic_codes),
        }
    expected_eos = expected.continuation_token == c.TOKEN_EOS
    if bool(generated.choose_eos) != expected_eos:
        return {
            "kind": "event_continuation",
            "include_complete_generated_outcome": True,
        }
    return None


def generated_tick_matches_oracle(
    expected: OracleEvent, generated: GeneratedTick
) -> bool:
    return first_tick_divergence(expected, generated) is None


def build_rollout_trace(
    session: OracleSession, generated_ticks: Sequence[GeneratedTick]
) -> RolloutTrace:
    """Apply generated variable events to the append-only runtime grammar."""

    transcript = list(session.header)
    action_prefixes: list[tuple[int, ...]] = []
    source_positions: list[tuple[int, int]] = []
    committed: list[GeneratedTick] = []
    stopped_early = False
    for event, generated in zip(session.events, generated_ticks):
        source_start = len(transcript) + 1
        transcript.extend(event.source_block)
        for offset, code in enumerate(event.source_codes):
            position = source_start + offset
            if transcript[position] != c.GLM_SEMANTIC_OFFSET + int(code):
                raise ValueError("rollout source position/code parity failed")
            source_positions.append((position, int(code)))
        action_prefixes.append(tuple(transcript))
        transcript.extend(generated_outcome_tokens(session, generated))
        committed.append(generated)
        if generated.choose_eos:
            transcript.append(c.TOKEN_EOS)
            stopped_early = event.event_index + 1 < len(session.events)
            break
    return RolloutTrace(
        sample_id=session.sample_id,
        transcript=tuple(transcript),
        action_prefixes=tuple(action_prefixes),
        source_positions=tuple(source_positions),
        generated_ticks=tuple(committed),
        stopped_early=stopped_early,
    )


def recovery_points(
    session: OracleSession, trace: RolloutTrace
) -> tuple[RecoveryPoint, ...]:
    """Return correction and corrupted-state recovery points.

    The first point corrects the exact divergent choice.  For a generated
    WRITE that already contains wrong text or semantic units, an additional
    terminal point keeps those wrong units in history and supervises the next
    oracle-aligned token or natural END decision.
    """

    first = trace.first_divergence(session)
    if first is None:
        if not trace.generated_ticks:
            raise ValueError("rollout reached no event")
        last = len(trace.generated_ticks) - 1
        return (
            RecoveryPoint(
                last,
                "event_continuation",
                include_complete_generated_outcome=True,
            ),
        )
    points = [first]
    if first.event_index < len(trace.generated_ticks):
        generated = trace.generated_ticks[first.event_index]
        if generated.action == "WRITE" and first.kind in {
            "text_token",
            "text_end",
            "semantic_token",
            "semantic_end",
        }:
            terminal = RecoveryPoint(
                first.event_index,
                "semantic_end",
                generated_text_prefix=len(generated.text_ids),
                generated_semantic_prefix=len(generated.semantic_codes),
                contains_corruption=True,
            )
            if terminal != first:
                points.append(terminal)
    return tuple(points)


def choose_recovery_point(
    session: OracleSession,
    trace: RolloutTrace,
    *,
    prefer_corrupted_state: bool = True,
) -> RecoveryPoint:
    points = recovery_points(session, trace)
    if prefer_corrupted_state:
        for point in reversed(points):
            if point.contains_corruption:
                return point
    return points[0]


def _write_prefix_and_suffix(
    session: OracleSession,
    event: OracleEvent,
    generated: GeneratedTick,
    point: RecoveryPoint,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    oracle = parse_write_outcome(event.outcome_tokens)
    fixed = (
        c.TOKEN_WRITE_GENERATE,
        c.language_token_id(session.target_lang),
        c.speed_token_id(1.0),
        c.TOKEN_START_CONTENT,
    )
    if point.kind in {"text_token", "text_end"}:
        count = min(point.generated_text_prefix, len(generated.text_ids))
        prefix = (*fixed, *generated.text_ids[:count])
        oracle_index = min(count, len(oracle.text_ids))
        suffix = (
            *oracle.text_ids[oracle_index:],
            c.TOKEN_END_CONTENT,
            c.TOKEN_START_SEMANTIC,
            *c.encode_bicodec_semantic(oracle.semantic_codes),
            c.TOKEN_END_SEMANTIC,
            event.continuation_token,
        )
        roles = (
            *([ROLE_TEXT] * (len(oracle.text_ids) - oracle_index)),
            ROLE_BOUNDARY,
            ROLE_BOUNDARY,
            *([ROLE_SEMANTIC] * len(oracle.semantic_codes)),
            ROLE_BOUNDARY,
            ROLE_BOUNDARY,
        )
        return tuple(prefix), tuple(suffix), tuple(roles)
    if point.kind in {"semantic_token", "semantic_end"}:
        text_count = min(point.generated_text_prefix, len(generated.text_ids))
        semantic_count = min(
            point.generated_semantic_prefix, len(generated.semantic_codes)
        )
        prefix = (
            *fixed,
            *generated.text_ids[:text_count],
            c.TOKEN_END_CONTENT,
            c.TOKEN_START_SEMANTIC,
            *c.encode_bicodec_semantic(
                generated.semantic_codes[:semantic_count]
            ),
        )
        oracle_index = min(semantic_count, len(oracle.semantic_codes))
        suffix = (
            *c.encode_bicodec_semantic(oracle.semantic_codes[oracle_index:]),
            c.TOKEN_END_SEMANTIC,
            event.continuation_token,
        )
        roles = (
            *([ROLE_SEMANTIC] * (len(oracle.semantic_codes) - oracle_index)),
            ROLE_BOUNDARY,
            ROLE_BOUNDARY,
        )
        return tuple(prefix), tuple(suffix), tuple(roles)
    if point.kind == "event_continuation":
        prefix = generated_outcome_tokens(session, generated)
        return tuple(prefix), (event.continuation_token,), (ROLE_BOUNDARY,)
    raise ValueError(f"WRITE recovery does not support {point.kind}")


def build_recovery_example(
    session: OracleSession,
    trace: RolloutTrace,
    point: RecoveryPoint | None = None,
) -> RecoveryExample:
    """Build oracle supervision from an exact generated sub-event state."""

    if trace.sample_id != session.sample_id:
        raise ValueError("trace/session IDs differ")
    point = point or choose_recovery_point(session, trace)
    if not 0 <= point.event_index < len(trace.action_prefixes):
        raise IndexError("recovery event was not reached by the rollout")
    event = session.events[point.event_index]
    action_prefix = trace.action_prefixes[point.event_index]
    generated = trace.generated_ticks[point.event_index]

    if point.kind == "action":
        generated_suffix: tuple[int, ...] = ()
        oracle_suffix = (*event.outcome_tokens, event.continuation_token)
        oracle_roles = (*event.outcome_roles, ROLE_BOUNDARY)
    elif event.action == "WAIT":
        if point.kind != "event_continuation":
            raise ValueError("WAIT supports only action/continuation recovery")
        generated_suffix = (c.TOKEN_WAIT_READ,)
        oracle_suffix = (event.continuation_token,)
        oracle_roles = (ROLE_BOUNDARY,)
    else:
        generated_suffix, oracle_suffix, oracle_roles = _write_prefix_and_suffix(
            session, event, generated, point
        )

    prefix = (*action_prefix, *generated_suffix)
    if not oracle_suffix:
        raise ValueError("recovery has no oracle target")
    conceptual = (*prefix, *oracle_suffix)
    conceptual_roles = (
        *([ROLE_OBSERVED] * len(prefix)),
        *oracle_roles,
    )
    tokens = tuple(conceptual[:-1])
    labels = tuple(conceptual[1:])
    shifted_roles = tuple(conceptual_roles[1:])
    recovery_position = len(prefix) - 1
    action_position = len(action_prefix) - 1
    continuation_position = len(tokens) - 1
    loss_mask = tuple(
        1 if index >= recovery_position else 0 for index in range(len(tokens))
    )
    source = tuple(
        (position, code)
        for position, code in trace.source_positions
        if position < len(action_prefix)
    )
    return RecoveryExample(
        sample_id=session.sample_id,
        event_index=point.event_index,
        divergence_kind=point.kind,
        tokens=tokens,
        labels=labels,
        loss_mask=loss_mask,
        token_roles=shifted_roles,
        position_ids=tuple(range(len(tokens))),
        frontend_positions=tuple(position for position, _ in source),
        frontend_ids=tuple(code for _, code in source),
        recovery_position=recovery_position,
        generated_prefix_length=len(prefix),
        corrupted_prefix_tokens=(
            point.generated_text_prefix + point.generated_semantic_prefix
            if point.contains_corruption
            else 0
        ),
        action_position=action_position,
        action_target=1 if event.action == "WRITE" else 0,
        action_supervised=point.kind == "action",
        continuation_position=continuation_position,
        continuation_target=1 if event.continuation_token == c.TOKEN_EOS else 0,
        continuation_supervised=point.kind == "event_continuation",
        support_bucket=event.support_bucket,
        chunk_end_ms=event.chunk_end_ms,
        soft_deadline_ms=event.soft_deadline_ms,
        hard_deadline_ms=event.hard_deadline_ms,
        deadline_forced=event.deadline_forced,
        deadline_loss_enabled=event.deadline_loss_enabled,
        previous_committed_length=event.previous_committed_length,
        stable_target_length=event.stable_target_length,
        playback_buffer_ms=event.playback_buffer_ms,
        translation_ids=session.full_translation_ids,
    )


__all__ = [
    "DIVERGENCE_KINDS",
    "GeneratedTick",
    "OracleEvent",
    "OracleSession",
    "ParsedWrite",
    "RECOVERY_SCHEMA",
    "RecoveryExample",
    "RecoveryPoint",
    "RolloutTrace",
    "build_recovery_example",
    "build_rollout_trace",
    "build_write_outcome",
    "choose_recovery_point",
    "first_tick_divergence",
    "generated_outcome_tokens",
    "generated_tick_matches_oracle",
    "oracle_sessions_from_pack",
    "parse_write_outcome",
    "recovery_points",
]
