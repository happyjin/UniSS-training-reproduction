"""Exact runtime-event transcripts and variable-length recovery examples.

The functions here are deliberately model/backend agnostic.  A native
Megatron KV backend and a small fake backend both produce :class:`GeneratedTick`
objects.  The same transcript builder then reconstructs the actual history and
creates a differentiable oracle-correction example at a model-induced state.
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


RECOVERY_SCHEMA = "uniss_event_rollout_recovery_v1"


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
class GeneratedTick:
    action: str
    text_ids: tuple[int, ...] = ()
    semantic_codes: tuple[int, ...] = ()
    natural_semantic_end: bool = True
    choose_eos: bool = False

    def __post_init__(self) -> None:
        if self.action not in {"WAIT", "WRITE"}:
            raise ValueError("generated action must be WAIT or WRITE")
        if self.action == "WAIT" and (self.text_ids or self.semantic_codes):
            raise ValueError("WAIT cannot carry a generated payload")
        if self.action == "WRITE" and not self.semantic_codes:
            raise ValueError("WRITE requires non-empty semantic content")
        for token in self.text_ids:
            if not 0 <= int(token) <= c.QWEN_BASE_VOCAB_END:
                raise ValueError("generated text escaped the base Qwen vocabulary")
        for code in self.semantic_codes:
            c.validate_range(int(code), c.BICODEC_SEMANTIC_SIZE, "target_bicodec")


@dataclass(frozen=True)
class RolloutTrace:
    sample_id: str
    transcript: tuple[int, ...]
    action_prefixes: tuple[tuple[int, ...], ...]
    source_positions: tuple[tuple[int, int], ...]
    generated_ticks: tuple[GeneratedTick, ...]
    stopped_early: bool

    def first_divergence(self, oracle: OracleSession) -> int | None:
        for index, generated in enumerate(self.generated_ticks):
            expected = oracle.events[index]
            if generated.action != expected.action:
                return index
            if generated.action == "WRITE":
                parsed = parse_write_outcome(expected.outcome_tokens)
                if (
                    tuple(generated.text_ids) != parsed.text_ids
                    or tuple(generated.semantic_codes) != parsed.semantic_codes
                    or not generated.natural_semantic_end
                ):
                    return index
            expected_eos = expected.continuation_token == c.TOKEN_EOS
            if bool(generated.choose_eos) != expected_eos:
                return index
        if len(self.generated_ticks) < len(oracle.events):
            return len(self.generated_ticks)
        return None


@dataclass(frozen=True)
class ParsedWrite:
    text_ids: tuple[int, ...]
    semantic_codes: tuple[int, ...]


@dataclass(frozen=True)
class RecoveryExample:
    sample_id: str
    event_index: int
    tokens: tuple[int, ...]
    labels: tuple[int, ...]
    loss_mask: tuple[int, ...]
    token_roles: tuple[int, ...]
    position_ids: tuple[int, ...]
    frontend_positions: tuple[int, ...]
    frontend_ids: tuple[int, ...]
    action_position: int
    action_target: int
    continuation_position: int
    continuation_target: int
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
        if not 0 <= self.action_position < len(self.tokens):
            raise ValueError("recovery action position is outside the sequence")
        if not 0 <= self.continuation_position < len(self.tokens):
            raise ValueError("recovery continuation position is outside the sequence")
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


def parse_write_outcome(values: Sequence[int]) -> ParsedWrite:
    tokens = tuple(int(value) for value in values)
    if len(tokens) < 8 or tokens[0] != c.TOKEN_WRITE_GENERATE:
        raise ValueError("malformed WRITE outcome")
    try:
        content_start = tokens.index(c.TOKEN_START_CONTENT) + 1
        content_end = tokens.index(c.TOKEN_END_CONTENT, content_start)
        semantic_start_marker = tokens.index(c.TOKEN_START_SEMANTIC, content_end + 1)
        semantic_end = tokens.index(c.TOKEN_END_SEMANTIC, semantic_start_marker + 1)
    except ValueError as exc:
        raise ValueError("WRITE outcome has incomplete content/semantic grammar") from exc
    if semantic_end != len(tokens) - 1:
        raise ValueError("WRITE outcome has trailing tokens after END_SEMANTIC")
    text = tokens[content_start:content_end]
    semantic_tokens = tokens[semantic_start_marker + 1 : semantic_end]
    semantic = tuple(int(value) - c.BICODEC_SEMANTIC_OFFSET for value in semantic_tokens)
    if not semantic:
        raise ValueError("WRITE outcome has no semantic content")
    if any(not 0 <= value < c.BICODEC_SEMANTIC_SIZE for value in semantic):
        raise ValueError("WRITE semantic token escaped the BiCodec interval")
    return ParsedWrite(text, semantic)


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
    language = "eng" if values[3] == c.TOKEN_ENG else "cmn" if values[3] == c.TOKEN_CMN else None
    if language is None:
        raise ValueError("packed session target language token is invalid")
    if values[5] != c.TOKEN_START_GLOBAL or values[-1] != c.TOKEN_END_GLOBAL:
        raise ValueError("packed session has an invalid global-speaker envelope")
    speaker = tuple(value - c.BICODEC_GLOBAL_OFFSET for value in values[6:-1])
    if len(speaker) != 32 or any(not 0 <= value < c.BICODEC_GLOBAL_SIZE for value in speaker):
        raise ValueError("packed session speaker codes are invalid")
    return language, speaker


def oracle_sessions_from_pack(value: Mapping[str, object]) -> tuple[OracleSession, ...]:
    """Parse packed complete sessions and collapse semantic-only top-level WRITEs.

    The deployed four-unit microblock loop emits all continuation blocks inside
    one top-level WRITE.  Dense data historically stored some of those blocks
    as later WRITE ticks with no new text.  Their semantic spans are merged into
    the preceding lexical WRITE, and the later top-level targets become WAIT.
    """

    if value.get("schema_version") != PACK_SCHEMA:
        raise ValueError("unexpected dense pack schema")
    packed_tokens = [int(item) for item in value["tokens"]]  # type: ignore[index]
    packed_labels = [int(item) for item in value["labels"]]  # type: ignore[index]
    sessions = value.get("sessions")
    if not isinstance(sessions, list) or not sessions:
        raise ValueError("dense pack contains no sessions")
    parsed_sessions: list[OracleSession] = []
    for raw_session in sessions:
        session = dict(raw_session)
        start, end = (int(item) for item in session["boundary"])
        conceptual = tuple(packed_tokens[start:end] + [packed_labels[end - 1]])
        annotations = sorted(
            (dict(item) for item in session["annotations"]),
            key=lambda item: int(item["event_index"]),
        )
        tick_starts: list[int] = []
        action_positions: list[int] = []
        raw_outcomes: list[tuple[int, ...]] = []
        raw_roles: list[tuple[int, ...]] = []
        for annotation in annotations:
            action = int(annotation["action_position"]) - start
            ids = tuple(int(item) for item in annotation["frontend_ids"])
            tick_start = action - len(ids) - 1
            if tick_start < 0 or conceptual[tick_start] != c.TOKEN_START_GLM:
                raise ValueError("could not locate dense event START_GLM")
            if conceptual[action] != c.TOKEN_END_GLM:
                raise ValueError("dense action context is not END_GLM")
            expected_source = (
                c.TOKEN_START_GLM,
                *c.encode_glm_semantic(ids),
                c.TOKEN_END_GLM,
            )
            if conceptual[tick_start : action + 1] != expected_source:
                raise ValueError("dense source block differs from its inline codes")
            tick_starts.append(tick_start)
            action_positions.append(action)
        header = conceptual[: tick_starts[0]]
        target_lang, speaker = _decode_header(header)
        for index, action in enumerate(action_positions):
            stop = tick_starts[index + 1] if index + 1 < len(tick_starts) else len(conceptual) - 1
            outcome = conceptual[action + 1 : stop]
            if not outcome:
                raise ValueError("dense event has an empty outcome")
            if outcome[0] == c.TOKEN_WAIT_READ:
                roles = (ROLE_ACTION,)
            elif outcome[0] == c.TOKEN_WRITE_GENERATE:
                parsed = parse_write_outcome(outcome)
                _, roles = build_write_outcome(target_lang, parsed.text_ids, parsed.semantic_codes)
            else:
                raise ValueError("dense event outcome is neither WAIT nor WRITE")
            raw_outcomes.append(tuple(outcome))
            raw_roles.append(tuple(roles))

        # Tokenizing an accumulated text prefix can retokenize its suffix, so
        # ``stable_target_length`` occasionally remains unchanged even though
        # the exact packed WRITE carries a non-empty new lexical delta.  The
        # runtime distinction is structural: only a WRITE with an empty text
        # span is a semantic-only continuation block.
        lexical_indices = [
            index
            for index, outcome in enumerate(raw_outcomes)
            if outcome[0] == c.TOKEN_WRITE_GENERATE
            and bool(parse_write_outcome(outcome).text_ids)
        ]
        semantic_groups: dict[int, list[int]] = {index: [] for index in lexical_indices}
        active_lexical: int | None = None
        for index, outcome in enumerate(raw_outcomes):
            if index in semantic_groups:
                active_lexical = index
            if outcome[0] == c.TOKEN_WRITE_GENERATE:
                if active_lexical is None:
                    raise ValueError("semantic-only WRITE precedes every lexical WRITE")
                semantic_groups[active_lexical].extend(parse_write_outcome(outcome).semantic_codes)

        events: list[OracleEvent] = []
        for index, (annotation, raw_outcome) in enumerate(zip(annotations, raw_outcomes)):
            lexical = index in semantic_groups
            if lexical:
                parsed = parse_write_outcome(raw_outcome)
                outcome, roles = build_write_outcome(
                    target_lang, parsed.text_ids, semantic_groups[index]
                )
                action = "WRITE"
            else:
                outcome = (c.TOKEN_WAIT_READ,)
                roles = (ROLE_ACTION,)
                action = "WAIT"
            events.append(
                OracleEvent(
                    event_index=index,
                    source_codes=tuple(int(item) for item in annotation["frontend_ids"]),
                    source_block=conceptual[tick_starts[index] : action_positions[index] + 1],
                    action=action,
                    outcome_tokens=outcome,
                    outcome_roles=roles,
                    continuation_token=(
                        c.TOKEN_EOS if index + 1 == len(annotations) else c.TOKEN_START_GLM
                    ),
                    source_finished=bool(annotation["source_finished"]),
                    previous_committed_length=int(annotation["previous_committed_length"]),
                    stable_target_length=int(annotation["stable_target_length"]),
                    support_bucket=int(annotation["support_bucket"]),
                    chunk_end_ms=int(annotation["chunk_end_ms"]),
                    soft_deadline_ms=int(annotation["soft_deadline_ms"]),
                    hard_deadline_ms=int(annotation["hard_deadline_ms"]),
                    deadline_forced=bool(annotation["deadline_forced"]),
                    deadline_loss_enabled=bool(annotation["deadline_loss_enabled"]),
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
                full_translation_ids=tuple(int(item) for item in session["translation_ids"]),
            )
        )
    return tuple(parsed_sessions)


def generated_outcome_tokens(
    session: OracleSession, generated: GeneratedTick
) -> tuple[int, ...]:
    if generated.action == "WAIT":
        return (c.TOKEN_WAIT_READ,)
    values, _ = build_write_outcome(
        session.target_lang, generated.text_ids, generated.semantic_codes
    )
    return values


def build_rollout_trace(
    session: OracleSession, generated_ticks: Sequence[GeneratedTick]
) -> RolloutTrace:
    """Apply generated variable events to the exact append-only runtime grammar."""

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


def build_recovery_example(
    session: OracleSession,
    trace: RolloutTrace,
    event_index: int,
) -> RecoveryExample:
    """Build one oracle correction at an actual model-induced event state."""

    if trace.sample_id != session.sample_id:
        raise ValueError("trace/session IDs differ")
    if not 0 <= event_index < len(trace.action_prefixes):
        raise IndexError("recovery event was not reached by the rollout")
    event = session.events[event_index]
    prefix = trace.action_prefixes[event_index]
    conceptual = (
        *prefix,
        *event.outcome_tokens,
        event.continuation_token,
    )
    conceptual_roles = (
        *([ROLE_OBSERVED] * len(prefix)),
        *event.outcome_roles,
        ROLE_BOUNDARY,
    )
    tokens = tuple(conceptual[:-1])
    labels = tuple(conceptual[1:])
    shifted_roles = tuple(conceptual_roles[1:])
    action_position = len(prefix) - 1
    continuation_position = len(tokens) - 1
    loss_mask = tuple(
        1 if index >= action_position else 0 for index in range(len(tokens))
    )
    source = tuple(
        (position, code)
        for position, code in trace.source_positions
        if position < len(prefix)
    )
    return RecoveryExample(
        sample_id=session.sample_id,
        event_index=event_index,
        tokens=tokens,
        labels=labels,
        loss_mask=loss_mask,
        token_roles=shifted_roles,
        position_ids=tuple(range(len(tokens))),
        frontend_positions=tuple(position for position, _ in source),
        frontend_ids=tuple(code for _, code in source),
        action_position=action_position,
        action_target=1 if event.action == "WRITE" else 0,
        continuation_position=continuation_position,
        continuation_target=1 if event.continuation_token == c.TOKEN_EOS else 0,
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


def choose_recovery_event(session: OracleSession, trace: RolloutTrace) -> int:
    """Prefer the first runtime divergence; otherwise supervise the final state."""

    divergence = trace.first_divergence(session)
    if divergence is not None and divergence < len(trace.action_prefixes):
        return divergence
    if not trace.action_prefixes:
        raise ValueError("rollout reached no action state")
    return len(trace.action_prefixes) - 1


__all__ = [
    "GeneratedTick",
    "OracleEvent",
    "OracleSession",
    "RECOVERY_SCHEMA",
    "RecoveryExample",
    "RolloutTrace",
    "build_recovery_example",
    "build_rollout_trace",
    "build_write_outcome",
    "choose_recovery_event",
    "generated_outcome_tokens",
    "oracle_sessions_from_pack",
    "parse_write_outcome",
]
