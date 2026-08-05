"""Compact CTC vocabulary maps derived from the Phase3 Qwen tokenizer."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class CompactCTCMap:
    language: str
    qwen_to_compact: dict[int, int]
    compact_to_qwen: tuple[int, ...]

    @property
    def blank_id(self) -> int:
        return len(self.compact_to_qwen)

    @property
    def output_size(self) -> int:
        return self.blank_id + 1

    def encode(self, qwen_ids: Iterable[int]) -> list[int]:
        return [self.qwen_to_compact[int(value)] for value in qwen_ids]

    def decode(self, compact_ids: Iterable[int]) -> list[int]:
        values = []
        for value in compact_ids:
            value = int(value)
            if value == self.blank_id:
                continue
            values.append(self.compact_to_qwen[value])
        return values

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "uniss_qwen_compact_ctc_map_v1",
            "language": self.language,
            "compact_to_qwen": list(self.compact_to_qwen),
            "blank_id": self.blank_id,
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "CompactCTCMap":
        if value.get("schema_version") != "uniss_qwen_compact_ctc_map_v1":
            raise ValueError("unsupported compact CTC map schema")
        ids = tuple(int(item) for item in value["compact_to_qwen"])  # type: ignore[index]
        if len(ids) != len(set(ids)):
            raise ValueError("compact_to_qwen contains duplicate Qwen IDs")
        expected_blank = len(ids)
        if int(value["blank_id"]) != expected_blank:
            raise ValueError("blank_id must follow the compact vocabulary")
        return cls(
            language=str(value["language"]),
            qwen_to_compact={token: index for index, token in enumerate(ids)},
            compact_to_qwen=ids,
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "CompactCTCMap":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def build_compact_map(language: str, token_sequences: Iterable[Iterable[int]]) -> CompactCTCMap:
    """Build a deterministic ID map; corpus order cannot change the result."""

    qwen_ids = sorted({int(token) for sequence in token_sequences for token in sequence})
    if not qwen_ids:
        raise ValueError("cannot build an empty CTC vocabulary")
    return CompactCTCMap(
        language=language,
        qwen_to_compact={token: index for index, token in enumerate(qwen_ids)},
        compact_to_qwen=tuple(qwen_ids),
    )
