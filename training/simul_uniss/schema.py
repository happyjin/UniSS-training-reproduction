"""Schemas and validation helpers for isolated Simul-UniSS data."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


DEFAULT_CHUNK_MS = 640
GLM_TOKENS_PER_SECOND = 12.5
BICODEC_TOKENS_PER_SECOND = 50.0


def coerce_token_list(value: object, field_name: str) -> list[int]:
    if not isinstance(value, list):
        raise TypeError(f"{field_name} must be a list, got {type(value).__name__}")
    tokens: list[int] = []
    for item in value:
        if not isinstance(item, int):
            raise TypeError(f"{field_name} items must be int, got {type(item).__name__}")
        tokens.append(item)
    return tokens


def normalize_record(raw: Mapping[str, object]) -> dict[str, object]:
    required = {
        "id",
        "transcription",
        "translation",
        "source_glm",
        "source_bicodec",
        "target_bicodec",
        "bicodec_global",
        "src_lang",
        "tgt_lang",
    }
    missing = sorted(required - set(raw))
    if missing:
        raise KeyError(f"UniST row is missing required fields: {missing}")

    record: dict[str, object] = {
        "id": str(raw["id"]),
        "transcription": str(raw["transcription"]),
        "translation": str(raw["translation"]),
        "source_glm": coerce_token_list(raw["source_glm"], "source_glm"),
        "source_bicodec": coerce_token_list(raw["source_bicodec"], "source_bicodec"),
        "target_bicodec": coerce_token_list(raw["target_bicodec"], "target_bicodec"),
        "bicodec_global": coerce_token_list(raw["bicodec_global"], "bicodec_global"),
        "src_lang": str(raw["src_lang"]),
        "tgt_lang": str(raw["tgt_lang"]),
        "dataset_name": str(raw.get("dataset_name", "unknown")),
        "split": str(raw.get("split", "unknown")),
    }
    if len(record["bicodec_global"]) != 32:  # type: ignore[arg-type]
        raise ValueError("bicodec_global must contain exactly 32 tokens")
    if raw.get("duration_ratio") is not None:
        record["duration_ratio"] = float(raw["duration_ratio"])
    return record


def tokens_per_chunk(chunk_ms: int, rate: float = GLM_TOKENS_PER_SECOND) -> int:
    if chunk_ms <= 0:
        raise ValueError("chunk_ms must be positive")
    return max(1, int(round(chunk_ms * rate / 1000.0)))


def proportional_boundary(total: int, completed: int, denominator: int) -> int:
    if total < 0 or completed < 0 or denominator <= 0:
        raise ValueError("invalid proportional boundary arguments")
    return min(total, max(0, int(round(total * completed / denominator))))


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def select_train_shards(root: Path, count: int = 15, start_index: int = 0) -> list[Path]:
    if count <= 0:
        raise ValueError("count must be positive")
    if start_index < 0:
        raise ValueError("start_index must be non-negative")
    paths = [root / f"train-{index:05d}.parquet" for index in range(start_index, start_index + count)]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing requested UniST shards: {missing}")
    return paths


@dataclass(frozen=True)
class TokenSpan:
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError(f"invalid token span: {(self.start, self.end)}")

    @property
    def length(self) -> int:
        return self.end - self.start


def chunk_spans(length: int, chunk_size: int) -> list[TokenSpan]:
    if length <= 0:
        raise ValueError("length must be positive")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    return [TokenSpan(start, min(start + chunk_size, length)) for start in range(0, length, chunk_size)]


def flatten(parts: Iterable[Sequence[int]]) -> list[int]:
    return [token for part in parts for token in part]
