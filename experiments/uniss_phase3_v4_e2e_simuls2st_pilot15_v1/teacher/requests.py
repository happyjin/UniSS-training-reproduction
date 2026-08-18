"""Construct future-safe Phase3 MT and semantic teacher requests per event."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Sequence

from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.data.schema import (
    E2ETrajectory,
)
from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.rollout.schema import (
    V1Rollout,
)
from training import constants_uniss as c
from training.sample_builders import build_mt_sample, build_tts_sample


TeacherFamily = Literal["phase3_mt", "phase3_semantic"]
HistoryKind = Literal["gold_source", "v1_source", "gold_target"]


@dataclass(frozen=True)
class Phase3TeacherRequest:
    sample_id: str
    split: str
    event_index: int
    family: TeacherFamily
    history_kind: HistoryKind
    prompt_ids: tuple[int, ...]
    target_ids: tuple[int, ...]
    selected_target_indices: tuple[int, ...]
    reference_labels: tuple[int, ...]
    content_candidate_tokens: int
    content_selected_tokens: int
    visible_source_prefix: str
    visible_target_prefix: str
    visible_semantic_tokens: int

    def __post_init__(self) -> None:
        if not self.sample_id or not self.prompt_ids or not self.target_ids:
            raise ValueError("Phase3 teacher request is incomplete")
        if len(self.selected_target_indices) != len(self.reference_labels):
            raise ValueError("Phase3 teacher request selection geometry differs")
        if not self.selected_target_indices:
            raise ValueError("Phase3 teacher request denominator is zero")
        if any(
            not 0 <= index < len(self.target_ids)
            for index in self.selected_target_indices
        ):
            raise ValueError("Phase3 teacher selection is outside its target")
        if tuple(self.target_ids[index] for index in self.selected_target_indices) != (
            self.reference_labels
        ):
            raise ValueError("Phase3 teacher references differ from selected targets")
        if any(
            left >= right
            for left, right in zip(
                self.selected_target_indices, self.selected_target_indices[1:]
            )
        ):
            raise ValueError("Phase3 teacher selected positions are not strictly increasing")
        if not 0 <= self.content_selected_tokens <= self.content_candidate_tokens:
            raise ValueError("Phase3 teacher content mapping counts are invalid")
        if self.visible_semantic_tokens < 0:
            raise ValueError("Phase3 teacher visible semantic count is negative")


def _longest_common_prefix(left: Sequence[int], right: Sequence[int]) -> int:
    count = 0
    for first, second in zip(left, right):
        if int(first) != int(second):
            break
        count += 1
    return count


def _lcs_pairs(left: Sequence[int], right: Sequence[int]) -> list[tuple[int, int]]:
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


def _mt_request(
    trajectory: E2ETrajectory,
    *,
    event_index: int,
    source_prefix: str,
    history_kind: Literal["gold_source", "v1_source"],
    previous_target_prefix: str,
    encode_text: Callable[[str], Sequence[int]],
) -> Phase3TeacherRequest | None:
    event = trajectory.events[event_index]
    if not event.target_text_delta or not source_prefix.strip():
        return None
    sample = build_mt_sample(
        src_lang=trajectory.src_lang,
        tgt_lang=trajectory.tgt_lang,
        source_text=source_prefix,
        target_text=event.target_text_prefix,
        text_encoder=lambda text: [int(value) for value in encode_text(text)],
        source_id=trajectory.sample_id,
    )
    previous = [int(value) for value in encode_text(previous_target_prefix)] if previous_target_prefix else []
    current = [int(value) for value in encode_text(event.target_text_prefix)]
    delta = [int(value) for value in encode_text(event.target_text_delta)]
    stable = _longest_common_prefix(previous, current)
    pairs = _lcs_pairs(current[stable:], delta)
    selected = [stable + teacher_index for teacher_index, _ in pairs]
    # Every incremental MT WRITE closes a content fragment. The boundary is a
    # valid same-prefix teacher target even when BPE retokenization maps only a
    # subset of independently encoded delta tokens.
    selected.append(len(current))
    return Phase3TeacherRequest(
        sample_id=trajectory.sample_id,
        split=trajectory.split,
        event_index=event_index,
        family="phase3_mt",
        history_kind=history_kind,
        prompt_ids=tuple(sample.prompt_ids),
        target_ids=tuple(sample.target_ids),
        selected_target_indices=tuple(selected),
        reference_labels=tuple(sample.target_ids[index] for index in selected),
        content_candidate_tokens=len(delta),
        content_selected_tokens=len(pairs),
        visible_source_prefix=source_prefix,
        visible_target_prefix=event.target_text_prefix,
        visible_semantic_tokens=event.target_semantic_start,
    )


def _semantic_request(
    trajectory: E2ETrajectory,
    *,
    event_index: int,
    semantic_prefix: Sequence[int],
    encode_text: Callable[[str], Sequence[int]],
    semantic_stride: int,
) -> Phase3TeacherRequest | None:
    event = trajectory.events[event_index]
    if not event.target_semantic_delta:
        return None
    if not event.target_text_prefix:
        raise ValueError("semantic teacher event has no visible target text")
    if len(semantic_prefix) != event.target_semantic_end:
        raise ValueError("semantic teacher prefix differs from event semantic boundary")
    sample = build_tts_sample(
        bicodec_global=trajectory.speaker_global,
        src_lang=trajectory.tgt_lang,
        transcription=event.target_text_prefix,
        source_bicodec=semantic_prefix,
        text_encoder=lambda text: [int(value) for value in encode_text(text)],
        source_id=trajectory.sample_id,
    )
    if semantic_stride <= 0:
        raise ValueError("semantic teacher stride must be positive")
    selected_content = list(
        range(event.target_semantic_start, event.target_semantic_end, semantic_stride)
    )
    if selected_content[-1] != event.target_semantic_end - 1:
        selected_content.append(event.target_semantic_end - 1)
    selected = list(selected_content)
    selected.append(len(semantic_prefix))
    if event.target_final:
        selected.append(len(semantic_prefix) + 1)
    return Phase3TeacherRequest(
        sample_id=trajectory.sample_id,
        split=trajectory.split,
        event_index=event_index,
        family="phase3_semantic",
        history_kind="gold_target",
        prompt_ids=tuple(sample.prompt_ids),
        target_ids=tuple(sample.target_ids),
        selected_target_indices=tuple(selected),
        reference_labels=tuple(sample.target_ids[index] for index in selected),
        content_candidate_tokens=len(event.target_semantic_delta),
        content_selected_tokens=len(selected_content),
        visible_source_prefix=event.gold_source_prefix,
        visible_target_prefix=event.target_text_prefix,
        visible_semantic_tokens=event.target_semantic_start,
    )


def build_phase3_requests(
    trajectory: E2ETrajectory,
    rollout: V1Rollout,
    *,
    encode_text: Callable[[str], Sequence[int]],
    semantic_stride: int = 8,
) -> list[Phase3TeacherRequest]:
    if trajectory.sample_id != rollout.sample_id or len(trajectory.events) != len(rollout.events):
        raise ValueError("gold/rollout geometry differs for Phase3 teacher requests")
    requests: list[Phase3TeacherRequest] = []
    previous_target = ""
    semantic_prefix: list[int] = []
    for event_index, (event, rollout_event) in enumerate(
        zip(trajectory.events, rollout.events)
    ):
        gold_request = _mt_request(
            trajectory,
            event_index=event_index,
            source_prefix=event.gold_source_prefix,
            history_kind="gold_source",
            previous_target_prefix=previous_target,
            encode_text=encode_text,
        )
        if gold_request is not None:
            requests.append(gold_request)
        v1_request = _mt_request(
            trajectory,
            event_index=event_index,
            source_prefix=rollout_event.v1_source_prefix,
            history_kind="v1_source",
            previous_target_prefix=previous_target,
            encode_text=encode_text,
        )
        if v1_request is not None:
            requests.append(v1_request)
        semantic_prefix.extend(int(value) for value in event.target_semantic_delta)
        semantic_request = _semantic_request(
            trajectory,
            event_index=event_index,
            semantic_prefix=semantic_prefix,
            encode_text=encode_text,
            semantic_stride=semantic_stride,
        )
        if semantic_request is not None:
            requests.append(semantic_request)
        previous_target = event.target_text_prefix
    if semantic_prefix != [
        int(value)
        for event in trajectory.events
        for value in event.target_semantic_delta
    ]:
        raise AssertionError("semantic teacher request cursor did not close")
    return requests


__all__ = ["Phase3TeacherRequest", "build_phase3_requests"]
