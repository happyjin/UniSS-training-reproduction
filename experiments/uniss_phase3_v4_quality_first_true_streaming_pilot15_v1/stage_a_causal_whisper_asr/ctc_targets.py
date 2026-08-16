"""Stage A source-CTC target encodings with explicit provenance schemas."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from training.phase3_whisper_streamspeech_joint.tokenizer_maps import CompactCTCMap


@dataclass(frozen=True)
class UTF8ByteCTCMap:
    language: str

    @property
    def compact_to_byte(self) -> tuple[int, ...]:
        return tuple(range(256))

    @property
    def blank_id(self) -> int:
        return 256

    @property
    def output_size(self) -> int:
        return 257

    def encode_text(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, compact_ids: list[int]) -> str:
        values = bytes(int(value) for value in compact_ids if int(value) != self.blank_id)
        return values.decode("utf-8")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "uniss_utf8_byte_ctc_map_v1",
            "language": self.language,
            "compact_to_byte": list(self.compact_to_byte),
            "blank_id": self.blank_id,
            "provenance": "label-independent fixed UTF-8 byte inventory",
        }

    def save(self, path: str | Path) -> None:
        output = Path(path)
        if output.exists():
            raise FileExistsError(f"refusing to overwrite UTF-8 byte CTC map: {output}")
        output.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "UTF8ByteCTCMap":
        if value.get("schema_version") != "uniss_utf8_byte_ctc_map_v1":
            raise ValueError("unsupported UTF-8 byte CTC map schema")
        inventory = tuple(int(item) for item in value["compact_to_byte"])  # type: ignore[index]
        if inventory != tuple(range(256)):
            raise ValueError("UTF-8 byte CTC inventory must contain exactly 0..255")
        if int(value["blank_id"]) != 256:
            raise ValueError("UTF-8 byte CTC blank_id must be 256")
        return cls(language=str(value["language"]))

    @classmethod
    def load(cls, path: str | Path) -> "UTF8ByteCTCMap":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


CTCMap = CompactCTCMap | UTF8ByteCTCMap


def load_ctc_map(path: str | Path) -> CTCMap:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("schema_version") == "uniss_utf8_byte_ctc_map_v1":
        return UTF8ByteCTCMap.from_dict(value)
    return CompactCTCMap.from_dict(value)


def encode_ctc_text(mapping: CTCMap, text: str, tokenizer: Any | None = None) -> list[int]:
    if isinstance(mapping, UTF8ByteCTCMap):
        return mapping.encode_text(text)
    if tokenizer is None:
        raise ValueError("Qwen compact CTC encoding requires a tokenizer")
    qwen_ids = tokenizer.encode(text, add_special_tokens=False)
    return mapping.encode(qwen_ids)


def minimum_ctc_steps(target_ids: list[int]) -> int:
    """Return target length plus mandatory blanks between repeated labels."""

    return len(target_ids) + sum(
        int(left == right) for left, right in zip(target_ids, target_ids[1:])
    )


__all__ = [
    "CTCMap",
    "UTF8ByteCTCMap",
    "encode_ctc_text",
    "load_ctc_map",
    "minimum_ctc_steps",
]
