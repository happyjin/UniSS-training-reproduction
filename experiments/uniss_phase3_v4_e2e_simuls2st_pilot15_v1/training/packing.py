"""Pack one homogeneous E2E task family with loss and cache sidecars."""

from __future__ import annotations

from typing import Iterable, Iterator

from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.training.task_samples import (
    E2ETaskSample,
    LOSS_NONE,
)
from training import constants_uniss as c


PACKED_TASK_SCHEMA = "uniss_phase3_v4_e2e_task_pack_v2"


def _pad(values: list, length: int, fill):
    return [*values, *([fill] * (length - len(values)))]


def pack_task_samples(
    samples: Iterable[E2ETaskSample],
    *,
    seq_length: int,
    drop_overlong: bool = False,
) -> Iterator[dict[str, object]]:
    if seq_length <= 0:
        raise ValueError("E2E pack sequence length must be positive")
    current: list[E2ETaskSample] = []
    current_length = 0
    current_family: str | None = None

    def emit() -> dict[str, object] | None:
        if not current:
            return None
        tokens: list[int] = []
        labels: list[int] = []
        loss_kinds: list[int] = []
        position_ids: list[int] = []
        boundaries: list[list[int]] = []
        source_ids: list[str] = []
        sequence_ids: list[str] = []
        source_records: list[int] = []
        acoustic_rows: list[dict[str, object]] = []
        teacher_bindings: list[dict[str, object]] = []
        commit_consistency: list[dict[str, object]] = []
        sample_bases: list[int] = []
        for sample_ordinal, sample in enumerate(current):
            base = len(tokens)
            sample_bases.append(base)
            shifted_length = sample.shifted_length
            tokens.extend(sample.token_ids[:-1])
            labels.extend(sample.token_ids[1:])
            loss_kinds.extend(sample.loss_kinds[1:])
            position_ids.extend(range(shifted_length))
            boundaries.append([base, base + shifted_length])
            source_ids.append(sample.sample_id)
            sequence_ids.append(sample.sequence_id)
            source_records.append(sample.source_manifest_record)
            speech_positions = [
                base + position
                for position, value in enumerate(sample.speech_indices[:-1])
                if value is not None
            ]
            speech_sources = [
                int(value)
                for value in sample.speech_indices[:-1]
                if value is not None
            ]
            if speech_positions:
                if len(speech_positions) != sample.source_glm_length:
                    raise ValueError("E2E packed acoustic coverage differs")
                acoustic_rows.append(
                    {
                        "sample_ordinal": sample_ordinal,
                        "sample_id": sample.sample_id,
                        "source_manifest_record": sample.source_manifest_record,
                        "source_audio": sample.source_audio,
                        "source_glm_length": sample.source_glm_length,
                        "source_glm": list(sample.source_glm_ids),
                        "packed_positions": speech_positions,
                        "source_indices": speech_sources,
                    }
                )
            for binding in sample.teacher_bindings:
                packed_start = base + binding.target_start - 1
                packed_stop = base + binding.target_stop - 1
                if not base <= packed_start < packed_stop <= base + shifted_length:
                    raise ValueError("E2E packed teacher binding exceeds sample")
                teacher_bindings.append(
                    {
                        "sample_ordinal": sample_ordinal,
                        "sample_id": sample.sample_id,
                        "source_manifest_record": sample.source_manifest_record,
                        "cache_kind": binding.cache_kind,
                        "request_id": binding.request_id,
                        "cache_position_start": binding.cache_position_start,
                        "cache_position_stop": binding.cache_position_stop,
                        "packed_start": packed_start,
                        "packed_stop": packed_stop,
                    }
                )
        previous_by_key: dict[str, tuple[int, E2ETaskSample]] = {}
        for sample_ordinal, sample in enumerate(current):
            if sample.commit_key is None:
                continue
            previous = previous_by_key.get(sample.commit_key)
            if previous is not None:
                previous_ordinal, previous_sample = previous
                previous_tokens = [
                    previous_sample.token_ids[position]
                    for position in previous_sample.commit_positions
                ]
                current_tokens = [
                    sample.token_ids[position]
                    for position in sample.commit_positions
                ]
                stable = 0
                for left, right in zip(previous_tokens, current_tokens):
                    if left != right:
                        break
                    stable += 1
                if stable:
                    previous_start = (
                        sample_bases[previous_ordinal]
                        + previous_sample.commit_positions[0]
                        - 1
                    )
                    current_start = (
                        sample_bases[sample_ordinal] + sample.commit_positions[0] - 1
                    )
                    commit_consistency.append(
                        {
                            "commit_key": sample.commit_key,
                            "previous_sample_ordinal": previous_ordinal,
                            "current_sample_ordinal": sample_ordinal,
                            "previous_packed_start": previous_start,
                            "previous_packed_stop": previous_start + stable,
                            "current_packed_start": current_start,
                            "current_packed_stop": current_start + stable,
                            "positions": stable,
                        }
                    )
            previous_by_key[sample.commit_key] = (sample_ordinal, sample)
        if len(tokens) > seq_length:
            raise AssertionError("E2E pack exceeded configured sequence length")
        return {
            "schema_version": PACKED_TASK_SCHEMA,
            "family": current[0].family,
            "tokens": _pad(tokens, seq_length, c.TOKEN_PAD),
            "labels": _pad(labels, seq_length, c.TOKEN_PAD),
            "loss_kinds": _pad(loss_kinds, seq_length, LOSS_NONE),
            "loss_mask": _pad(
                [float(value != LOSS_NONE) for value in loss_kinds],
                seq_length,
                0.0,
            ),
            "position_ids": _pad(position_ids, seq_length, 0),
            "sample_boundaries": boundaries,
            "source_ids": source_ids,
            "sequence_ids": sequence_ids,
            "source_manifest_records": source_records,
            "acoustic_rows": acoustic_rows,
            "teacher_bindings": teacher_bindings,
            "commit_consistency": commit_consistency,
            "used_tokens": len(tokens),
            "supervised_tokens": sum(value != LOSS_NONE for value in loss_kinds),
        }

    for sample in samples:
        length = sample.shifted_length
        if length > seq_length:
            if drop_overlong:
                continue
            raise ValueError(
                f"E2E sample length {length} exceeds sequence length {seq_length}"
            )
        if current_family is None:
            current_family = sample.family
        if sample.family != current_family:
            raise ValueError("E2E pack input mixes task families")
        if current and current_length + length > seq_length:
            packed = emit()
            if packed is not None:
                yield packed
            current = []
            current_length = 0
        current.append(sample)
        current_length += length
    packed = emit()
    if packed is not None:
        yield packed


def validate_packed_task(value: dict[str, object], *, seq_length: int) -> None:
    if value.get("schema_version") != PACKED_TASK_SCHEMA:
        raise ValueError("unexpected E2E packed task schema")
    for key in ("tokens", "labels", "loss_kinds", "loss_mask", "position_ids"):
        rows = value.get(key)
        if not isinstance(rows, list) or len(rows) != seq_length:
            raise ValueError(f"E2E packed {key} length differs")
    boundaries = value.get("sample_boundaries")
    if not isinstance(boundaries, list) or not boundaries:
        raise ValueError("E2E packed sample boundaries are missing")
    cursor = 0
    for start, stop in boundaries:
        if int(start) != cursor or not cursor < int(stop) <= seq_length:
            raise ValueError("E2E packed sample boundaries contain a gap")
        cursor = int(stop)
    if cursor != int(value["used_tokens"]):
        raise ValueError("E2E packed used-token count differs")
    loss_kinds = value["loss_kinds"]
    loss_mask = value["loss_mask"]
    if any(
        float(mask) != float(int(kind) != LOSS_NONE)
        for kind, mask in zip(loss_kinds, loss_mask)
    ):
        raise ValueError("E2E packed loss mask differs from loss kinds")
    supervised = sum(int(kind) != LOSS_NONE for kind in loss_kinds)
    if supervised != int(value["supervised_tokens"]) or supervised <= 0:
        raise ValueError("E2E packed supervised-token count differs")
    for binding in value.get("teacher_bindings", []):
        start = int(binding["packed_start"])
        stop = int(binding["packed_stop"])
        cache_start = int(binding["cache_position_start"])
        cache_stop = int(binding["cache_position_stop"])
        if not 0 <= start < stop <= cursor or stop - start != cache_stop - cache_start:
            raise ValueError("E2E packed teacher binding geometry differs")
        if any(int(loss_kinds[index]) == LOSS_NONE for index in range(start, stop)):
            raise ValueError("E2E packed teacher binding covers masked labels")
    labels = value["labels"]
    for binding in value.get("commit_consistency", []):
        previous_start = int(binding["previous_packed_start"])
        previous_stop = int(binding["previous_packed_stop"])
        current_start = int(binding["current_packed_start"])
        current_stop = int(binding["current_packed_stop"])
        positions = int(binding["positions"])
        if (
            not 0 <= previous_start < previous_stop <= cursor
            or not 0 <= current_start < current_stop <= cursor
            or previous_stop - previous_start != positions
            or current_stop - current_start != positions
            or list(labels[previous_start:previous_stop])
            != list(labels[current_start:current_stop])
        ):
            raise ValueError("E2E packed commit-consistency geometry differs")


__all__ = ["PACKED_TASK_SCHEMA", "pack_task_samples", "validate_packed_task"]
