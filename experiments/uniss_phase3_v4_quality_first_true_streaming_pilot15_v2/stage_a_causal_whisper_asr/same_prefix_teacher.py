"""Construct future-safe Phase3 ASR teacher requests from Stage A packs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v1.stage_a_causal_whisper_asr.packing import (
    LOSS_CAUSAL_FULL_ASR,
    LOSS_STREAMING_ASR,
)
from training import constants_uniss as c


@dataclass(frozen=True)
class TeacherRequest:
    """One standalone-ASR teacher view and its Stage A student positions."""

    prompt_ids: tuple[int, ...]
    target_ids: tuple[int, ...]
    selected_target_indices: tuple[int, ...]
    student_positions: tuple[int, ...]
    reference_labels: tuple[int, ...]
    visible_glm_tokens: int
    event_index: int

    def __post_init__(self) -> None:
        if not self.prompt_ids or not self.target_ids:
            raise ValueError("teacher request prompt/target cannot be empty")
        lengths = {
            len(self.selected_target_indices),
            len(self.student_positions),
            len(self.reference_labels),
        }
        if lengths != {len(self.selected_target_indices)} or not self.student_positions:
            raise ValueError("teacher request selected arrays differ or are empty")
        if any(not 0 <= index < len(self.target_ids) for index in self.selected_target_indices):
            raise ValueError("teacher request target selection is out of range")
        if tuple(self.target_ids[index] for index in self.selected_target_indices) != (
            self.reference_labels
        ):
            raise ValueError("teacher request labels differ from selected target tokens")
        if any(left >= right for left, right in zip(self.student_positions, self.student_positions[1:])):
            raise ValueError("teacher student positions must be strictly increasing")
        if self.visible_glm_tokens <= 0:
            raise ValueError("teacher request must expose a non-empty GLM prefix")


def _quality_asr_prompt(
    source_glm: Sequence[int],
    *,
    src_lang: str,
    fixed_speaker: Sequence[int],
) -> tuple[int, ...]:
    if not source_glm or len(fixed_speaker) != 32:
        raise ValueError("same-prefix teacher prompt geometry is invalid")
    target_language = "cmn" if c.normalize_language(src_lang) == "eng" else "eng"
    return tuple(
        [
            c.TOKEN_TASK_S2S_TRANSLATION,
            c.TOKEN_SLOW_MODE,
            c.language_token_id(target_language),
            *c.wrap_global_tokens([int(value) for value in fixed_speaker]),
            *c.encode_glm_semantic([int(value) for value in source_glm]),
            c.TOKEN_WRITE_GENERATE,
            c.TOKEN_TASK_ASR,
            c.language_token_id(src_lang),
            c.speed_token_id(1.0),
            c.TOKEN_START_CONTENT,
        ]
    )


def _longest_common_prefix(left: Sequence[int], right: Sequence[int]) -> int:
    count = 0
    for first, second in zip(left, right):
        if int(first) != int(second):
            break
        count += 1
    return count


def _lcs_pairs(left: Sequence[int], right: Sequence[int]) -> list[tuple[int, int]]:
    """Return an ordered exact-token alignment for two short BPE suffixes."""

    rows = len(left) + 1
    columns = len(right) + 1
    lengths = [[0] * columns for _ in range(rows)]
    for row in range(len(left) - 1, -1, -1):
        for column in range(len(right) - 1, -1, -1):
            if int(left[row]) == int(right[column]):
                lengths[row][column] = lengths[row + 1][column + 1] + 1
            else:
                lengths[row][column] = max(
                    lengths[row + 1][column], lengths[row][column + 1]
                )
    pairs: list[tuple[int, int]] = []
    row = column = 0
    while row < len(left) and column < len(right):
        if int(left[row]) == int(right[column]):
            pairs.append((row, column))
            row += 1
            column += 1
        elif lengths[row + 1][column] >= lengths[row][column + 1]:
            row += 1
        else:
            column += 1
    return pairs


def _sample_view(
    pack: Mapping[str, object], acoustic: Mapping[str, object]
) -> tuple[int, int, list[int], list[int], list[int], list[int]]:
    boundary_index = int(acoustic["batch_boundary_index"])
    boundaries = pack["sample_boundaries"]
    if not isinstance(boundaries, list) or not 0 <= boundary_index < len(boundaries):
        raise ValueError("teacher acoustic boundary index is malformed")
    start, end = (int(value) for value in boundaries[boundary_index])
    tokens = [int(value) for value in pack["tokens"]]  # type: ignore[index]
    labels = [int(value) for value in pack["labels"]]  # type: ignore[index]
    loss_mask = [int(value) for value in pack["loss_mask"]]  # type: ignore[index]
    loss_kinds = [int(value) for value in pack["loss_kinds"]]  # type: ignore[index]
    if not 0 <= start < end <= len(tokens):
        raise ValueError("teacher sample boundary is outside its pack")
    conceptual = [*tokens[start:end], labels[end - 1]]
    return (
        start,
        end,
        conceptual,
        labels[start:end],
        loss_mask[start:end],
        loss_kinds[start:end],
    )


def fixed_speaker_from_pack(
    pack: Mapping[str, object], acoustic: Mapping[str, object]
) -> tuple[int, ...]:
    """Recover the immutable neutral speaker directly from the student prompt.

    Stage A packs already contain the exact 32-token speaker prompt used when
    they were built.  Reading it from the selected sample avoids a second,
    mutable source-snapshot dependency and guarantees teacher/student parity.
    """

    _, _, conceptual, _, _, _ = _sample_view(pack, acoustic)
    try:
        start = conceptual.index(c.TOKEN_START_GLOBAL)
        stop = conceptual.index(c.TOKEN_END_GLOBAL, start + 1)
    except ValueError as exc:
        raise ValueError("teacher sample has no complete speaker prompt") from exc
    encoded = conceptual[start + 1 : stop]
    if len(encoded) != 32:
        raise ValueError("teacher sample speaker prompt must contain 32 tokens")
    speaker = tuple(c.BICODEC_GLOBAL_SPAN.value_for(int(value)) for value in encoded)
    if conceptual.count(c.TOKEN_START_GLOBAL) != 1 or conceptual.count(
        c.TOKEN_END_GLOBAL
    ) != 1:
        raise ValueError("teacher sample contains multiple speaker prompts")
    return speaker


def _causal_full_request(
    pack: Mapping[str, object],
    acoustic: Mapping[str, object],
    fixed_speaker: Sequence[int],
) -> list[TeacherRequest]:
    start, _, _, labels, loss_mask, loss_kinds = _sample_view(pack, acoustic)
    active = [
        index
        for index, (mask, kind) in enumerate(zip(loss_mask, loss_kinds))
        if mask and kind == LOSS_CAUSAL_FULL_ASR
    ]
    if not active or active != list(range(active[0], active[-1] + 1)):
        raise ValueError("causal-full teacher targets must be one contiguous region")
    target = [labels[index] for index in active]
    source_glm = [int(value) for value in acoustic["source_glm"]]  # type: ignore[index]
    return [
        TeacherRequest(
            prompt_ids=_quality_asr_prompt(
                source_glm,
                src_lang=str(acoustic["src_lang"]),
                fixed_speaker=fixed_speaker,
            ),
            target_ids=tuple(target),
            selected_target_indices=tuple(range(len(target))),
            student_positions=tuple(start + index for index in active),
            reference_labels=tuple(target),
            visible_glm_tokens=len(source_glm),
            event_index=0,
        )
    ]


def _streaming_requests(
    pack: Mapping[str, object],
    acoustic: Mapping[str, object],
    fixed_speaker: Sequence[int],
    encode_text: Callable[[str], Sequence[int]],
    decode_text: Callable[[Sequence[int]], str],
) -> list[TeacherRequest]:
    start, _, conceptual, labels, loss_mask, loss_kinds = _sample_view(pack, acoustic)
    language = str(acoustic["src_lang"])
    visible_glm: list[int] = []
    committed_text = ""
    cumulative_ids: list[int] = []
    requests: list[TeacherRequest] = []
    cursor = 0
    event_index = 0
    while cursor < len(conceptual):
        if conceptual[cursor] != c.TOKEN_START_GLM:
            cursor += 1
            continue
        try:
            close_glm = conceptual.index(c.TOKEN_END_GLM, cursor + 1)
        except ValueError as exc:
            raise ValueError("streaming teacher found an unclosed GLM region") from exc
        encoded_glm = conceptual[cursor + 1 : close_glm]
        if not encoded_glm or any(
            not c.GLM_SEMANTIC_OFFSET <= value <= c.GLM_SEMANTIC_SPAN.last_id
            for value in encoded_glm
        ):
            raise ValueError("streaming teacher GLM region contains invalid tokens")
        visible_glm.extend(value - c.GLM_SEMANTIC_OFFSET for value in encoded_glm)
        cursor = close_glm + 1
        if cursor >= len(conceptual) or conceptual[cursor] != c.TOKEN_WRITE_GENERATE:
            continue
        if (
            cursor + 3 >= len(conceptual)
            or conceptual[cursor + 1] != c.language_token_id(language)
            or conceptual[cursor + 2] != c.TOKEN_START_CONTENT
        ):
            raise ValueError("streaming teacher event header is malformed")
        try:
            close_content = conceptual.index(c.TOKEN_END_CONTENT, cursor + 3)
        except ValueError as exc:
            raise ValueError("streaming teacher found an unclosed content region") from exc
        delta_start = cursor + 3
        delta = conceptual[delta_start:close_content]
        if not delta:
            raise ValueError("streaming teacher event has an empty text delta")
        delta_text = decode_text(delta).strip()
        if not delta_text:
            raise ValueError("streaming teacher decoded an empty text delta")
        if c.normalize_language(language) == "eng":
            updated_text = f"{committed_text} {delta_text}".strip()
        else:
            updated_text = f"{committed_text}{delta_text}".replace(" ", "")
        updated_ids = [int(value) for value in encode_text(updated_text)]
        if not updated_ids:
            raise ValueError("streaming teacher cumulative text encoded empty")
        stable_prefix = _longest_common_prefix(cumulative_ids, updated_ids)
        teacher_suffix = updated_ids[stable_prefix:]
        pairs = _lcs_pairs(teacher_suffix, delta)
        selected = [stable_prefix + teacher_index for teacher_index, _ in pairs]
        conceptual_positions = [
            delta_start + student_index for _, student_index in pairs
        ]
        target = [*updated_ids, c.TOKEN_END_CONTENT, c.TOKEN_EOS]
        selected.append(len(updated_ids))
        conceptual_positions.append(close_content)
        student_positions = [start + position - 1 for position in conceptual_positions]
        reference = [int(labels[position - 1]) for position in conceptual_positions]
        if reference != [target[index] for index in selected]:
            raise ValueError("streaming teacher/student target mapping differs")
        for local_position in student_positions:
            relative = local_position - start
            if not loss_mask[relative] or loss_kinds[relative] != LOSS_STREAMING_ASR:
                raise ValueError("streaming teacher selected an inactive student position")
        requests.append(
            TeacherRequest(
                prompt_ids=_quality_asr_prompt(
                    visible_glm,
                    src_lang=language,
                    fixed_speaker=fixed_speaker,
                ),
                target_ids=tuple(target),
                selected_target_indices=tuple(selected),
                student_positions=tuple(student_positions),
                reference_labels=tuple(reference),
                visible_glm_tokens=len(visible_glm),
                event_index=event_index,
            )
        )
        committed_text = updated_text
        cumulative_ids = updated_ids
        event_index += 1
        cursor = close_content + 1
    source_glm = [int(value) for value in acoustic["source_glm"]]  # type: ignore[index]
    if not requests or requests[-1].visible_glm_tokens > len(source_glm):
        raise ValueError("streaming teacher did not construct a valid text event")
    if visible_glm != source_glm:
        raise ValueError("streaming teacher events do not cover the source GLM sequence")
    canonical = " ".join(str(acoustic["canonical_transcript"]).split())
    reconstructed = " ".join(committed_text.split())
    if c.normalize_language(language) == "cmn":
        canonical = canonical.replace(" ", "")
        reconstructed = reconstructed.replace(" ", "")
    if canonical != reconstructed:
        raise ValueError(
            f"streaming teacher text reconstruction differs: {reconstructed!r} vs {canonical!r}"
        )
    flattened_positions = [position for request in requests for position in request.student_positions]
    if len(flattened_positions) != len(set(flattened_positions)):
        raise ValueError("streaming teacher selected duplicate student positions")
    return requests


def requests_for_acoustic(
    pack: Mapping[str, object],
    acoustic: Mapping[str, object],
    *,
    fixed_speaker: Sequence[int],
    encode_text: Callable[[str], Sequence[int]],
    decode_text: Callable[[Sequence[int]], str],
) -> list[TeacherRequest]:
    """Return only comparable, prefix-supported teacher/student positions."""

    task = str(acoustic["task"])
    if task == "causal_full_asr":
        return _causal_full_request(pack, acoustic, fixed_speaker)
    if task == "streaming_asr":
        return _streaming_requests(
            pack, acoustic, fixed_speaker, encode_text, decode_text
        )
    raise ValueError(f"same-prefix teacher does not support acoustic task {task!r}")


__all__ = [
    "TeacherRequest",
    "fixed_speaker_from_pack",
    "requests_for_acoustic",
]
