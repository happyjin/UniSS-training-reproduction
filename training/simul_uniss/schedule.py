"""Build deterministic pseudo-streaming schedules from tokenized UniST rows.

These schedules are for bootstrap integration only.  They preserve exact token
coverage but use proportional source/target timing because public UniST parquet
rows do not contain word timestamps.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

from training.simul_uniss import SCHEDULE_SCHEMA_VERSION
from training.simul_uniss.schema import chunk_spans, proportional_boundary, tokens_per_chunk


TextEncoder = Callable[[str], list[int]]
_PHRASE_BOUNDARY = re.compile(r"(?<=[,.;:!?，。；：！？])\s*")


@dataclass(frozen=True)
class Phrase:
    text: str
    text_ids: list[int]
    text_start: int
    text_end: int
    semantic_start: int
    semantic_end: int

    def to_json(self) -> dict[str, object]:
        return {
            "text": self.text,
            "text_ids": self.text_ids,
            "text_start": self.text_start,
            "text_end": self.text_end,
            "semantic_start": self.semantic_start,
            "semantic_end": self.semantic_end,
        }


def _split_long_ids(ids: Sequence[int], max_phrase_tokens: int) -> list[list[int]]:
    return [list(ids[start : start + max_phrase_tokens]) for start in range(0, len(ids), max_phrase_tokens)]


def build_phrases(
    text: str,
    text_encoder: TextEncoder,
    target_semantic_length: int,
    max_phrase_tokens: int = 16,
) -> list[Phrase]:
    if not text.strip():
        raise ValueError("translation must be non-empty")
    if target_semantic_length <= 0:
        raise ValueError("target_semantic_length must be positive")
    if max_phrase_tokens <= 0:
        raise ValueError("max_phrase_tokens must be positive")

    raw_parts = [part for part in _PHRASE_BOUNDARY.split(text) if part]
    if not raw_parts:
        raw_parts = [text]

    encoded_parts: list[tuple[str, list[int]]] = []
    for raw_part in raw_parts:
        ids = text_encoder(raw_part)
        if not ids:
            continue
        split_ids = _split_long_ids(ids, max_phrase_tokens)
        for index, part_ids in enumerate(split_ids):
            display = raw_part if len(split_ids) == 1 else f"{raw_part}#{index}"
            encoded_parts.append((display, part_ids))
    if not encoded_parts:
        raise ValueError("translation encoded to no tokens")

    total_text_tokens = sum(len(ids) for _, ids in encoded_parts)
    phrases: list[Phrase] = []
    text_cursor = 0
    semantic_cursor = 0
    for index, (part_text, part_ids) in enumerate(encoded_parts):
        text_end = text_cursor + len(part_ids)
        if index == len(encoded_parts) - 1:
            semantic_end = target_semantic_length
        else:
            semantic_end = proportional_boundary(target_semantic_length, text_end, total_text_tokens)
            semantic_end = max(semantic_cursor, semantic_end)
        phrases.append(
            Phrase(
                text=part_text,
                text_ids=part_ids,
                text_start=text_cursor,
                text_end=text_end,
                semantic_start=semantic_cursor,
                semantic_end=semantic_end,
            )
        )
        text_cursor = text_end
        semantic_cursor = semantic_end
    return phrases


def _supported_target_tokens(
    source_seen: int,
    source_length: int,
    target_text_length: int,
    wait_glm_tokens: int,
    is_final: bool,
) -> int:
    if is_final:
        return target_text_length
    effective_seen = max(0, source_seen - wait_glm_tokens)
    effective_total = max(1, source_length - wait_glm_tokens)
    return min(target_text_length, int(math.floor(target_text_length * effective_seen / effective_total)))


def build_pseudo_schedule(
    record: Mapping[str, object],
    text_encoder: TextEncoder,
    *,
    chunk_ms: int = 640,
    wait_k_chunks: int = 2,
    max_phrase_tokens: int = 16,
) -> dict[str, object]:
    source_glm = list(record["source_glm"])  # type: ignore[arg-type]
    source_bicodec = list(record["source_bicodec"])  # type: ignore[arg-type]
    target_bicodec = list(record["target_bicodec"])  # type: ignore[arg-type]
    if not source_glm or not source_bicodec or not target_bicodec:
        raise ValueError("source_glm, source_bicodec, and target_bicodec must be non-empty")

    glm_chunk_size = tokens_per_chunk(chunk_ms)
    wait_glm_tokens = max(0, wait_k_chunks) * glm_chunk_size
    spans = chunk_spans(len(source_glm), glm_chunk_size)
    phrases = build_phrases(
        str(record["translation"]),
        text_encoder,
        len(target_bicodec),
        max_phrase_tokens=max_phrase_tokens,
    )
    target_text_length = phrases[-1].text_end

    phrase_cursor = 0
    events: list[dict[str, object]] = []
    for chunk_index, span in enumerate(spans):
        source_seen = span.end
        is_final = chunk_index == len(spans) - 1
        supported = _supported_target_tokens(
            source_seen,
            len(source_glm),
            target_text_length,
            wait_glm_tokens,
            is_final,
        )

        write_start = phrase_cursor
        while phrase_cursor < len(phrases) and phrases[phrase_cursor].text_end <= supported:
            phrase_cursor += 1
        if is_final:
            phrase_cursor = len(phrases)

        source_bicodec_start = proportional_boundary(len(source_bicodec), span.start, len(source_glm))
        source_bicodec_end = proportional_boundary(len(source_bicodec), span.end, len(source_glm))
        event: dict[str, object] = {
            "chunk_index": chunk_index,
            "source_start_ms": chunk_index * chunk_ms,
            "source_end_ms": (chunk_index + 1) * chunk_ms,
            "source_glm_start": span.start,
            "source_glm_end": span.end,
            "source_glm": source_glm[span.start : span.end],
            "source_bicodec_start": source_bicodec_start,
            "source_bicodec_end": source_bicodec_end,
            "source_bicodec": source_bicodec[source_bicodec_start:source_bicodec_end],
            "source_ctc_count_proxy": source_seen,
            "target_ctc_count_proxy": supported,
            "source_is_final": is_final,
        }
        if phrase_cursor == write_start:
            if is_final:
                raise AssertionError("final event must flush at least one phrase")
            event["action"] = "wait"
        else:
            emitted = phrases[write_start:phrase_cursor]
            event.update(
                {
                    "action": "write",
                    "target_phrases": [phrase.to_json() for phrase in emitted],
                    "target_text_ids": [token for phrase in emitted for token in phrase.text_ids],
                    "target_semantic_start": emitted[0].semantic_start,
                    "target_semantic_end": emitted[-1].semantic_end,
                    "target_semantic": target_bicodec[
                        emitted[0].semantic_start : emitted[-1].semantic_end
                    ],
                }
            )
        events.append(event)

    if phrase_cursor != len(phrases):
        raise AssertionError("schedule did not emit all target phrases")
    emitted_semantic = [
        token
        for event in events
        if event["action"] == "write"
        for token in event["target_semantic"]  # type: ignore[index]
    ]
    if emitted_semantic != target_bicodec:
        raise AssertionError("schedule semantic spans do not exactly cover target_bicodec")

    return {
        "schema_version": SCHEDULE_SCHEMA_VERSION,
        "alignment_kind": "pseudo_proportional_token_alignment",
        "id": str(record["id"]),
        "dataset_name": str(record.get("dataset_name", "unknown")),
        "split": str(record.get("split", "unknown")),
        "src_lang": str(record["src_lang"]),
        "tgt_lang": str(record["tgt_lang"]),
        "transcription": str(record["transcription"]),
        "translation": str(record["translation"]),
        "chunk_ms": chunk_ms,
        "wait_k_chunks": wait_k_chunks,
        "glm_tokens_per_chunk": glm_chunk_size,
        "source_glm_length": len(source_glm),
        "source_bicodec_length": len(source_bicodec),
        "target_text_length": target_text_length,
        "target_bicodec_length": len(target_bicodec),
        "speaker_tokens": list(record["bicodec_global"]),  # type: ignore[arg-type]
        "events": events,
    }
