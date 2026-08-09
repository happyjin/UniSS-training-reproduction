"""Locality-aware full198 parquet datasets for Megatron data parallelism."""

from __future__ import annotations

import bisect
import json
import random
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset


REQUIRED_COLUMNS = (
    "id",
    "transcription",
    "translation",
    "source_glm",
    "target_bicodec",
    "bicodec_global",
    "src_lang",
    "tgt_lang",
)


def nonempty_record_mask(table) -> np.ndarray:
    """Return rows that satisfy every field consumed by the joint objective."""

    mask = None
    for name in ("source_glm", "target_bicodec", "bicodec_global"):
        present = pc.greater(pc.fill_null(pc.list_value_length(table[name]), 0), 0)
        mask = present if mask is None else pc.and_(mask, present)
    for name in ("transcription", "translation"):
        text = pc.fill_null(table[name], "")
        present = pc.greater(pc.utf8_length(pc.utf8_trim_whitespace(text)), 0)
        mask = pc.and_(mask, present)
    return pc.fill_null(mask, False).to_numpy(zero_copy_only=False)


@dataclass(frozen=True)
class DirectionBlock:
    shard: int
    direction: str
    start: int
    length: int


class _TokenizerMixin:
    tokenizer_path: Path
    _tokenizer: Any

    def _encode(self, text: str) -> list[int]:
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_path, local_files_only=True
            )
        values = self._tokenizer.encode(text, add_special_tokens=False)
        if not values:
            raise ValueError("text encoded to an empty sequence")
        return [int(value) for value in values]

    def _record(self, value: dict[str, object], sample_index: int) -> dict[str, object]:
        record = {
            "id": str(value["id"]),
            "src_lang": str(value["src_lang"]),
            "tgt_lang": str(value["tgt_lang"]),
            "source_glm": [int(item) for item in value["source_glm"]],
            "target_bicodec": [int(item) for item in value["target_bicodec"]],
            "bicodec_global": [int(item) for item in value["bicodec_global"]],
            "transcription_ids": self._encode(str(value["transcription"])),
            "translation_ids": self._encode(str(value["translation"])),
            "sample_index": int(sample_index),
        }
        if not record["source_glm"] or not record["target_bicodec"]:
            raise ValueError(f"record {record['id']} has an empty speech token sequence")
        return record


class Full198CurriculumDataset(Dataset, _TokenizerMixin):
    """Pre-shuffled direction-balanced blocks with per-worker parquet locality."""

    def __init__(
        self,
        index_json: str | Path,
        tokenizer_path: str | Path,
        *,
        block_size: int = 4096,
        seed: int = 20260809,
        cache_shards: int = 2,
    ) -> None:
        if block_size <= 0 or cache_shards <= 0:
            raise ValueError("block_size and cache_shards must be positive")
        metadata = json.loads(Path(index_json).read_text(encoding="utf-8"))
        if metadata.get("schema_version") != "uniss_phase3_prefix_streaming_direction_index_v2":
            raise ValueError("unsupported full198 direction index")
        self.shards = list(metadata["shards"])
        if len(self.shards) != 198:
            raise ValueError(f"expected 198 indexed shards, found {len(self.shards)}")
        self.tokenizer_path = Path(tokenizer_path)
        self._tokenizer = None
        self.cache_shards = int(cache_shards)
        self._tables: OrderedDict[int, Any] = OrderedDict()
        self._indices: dict[tuple[int, str], np.ndarray] = {}
        by_direction: dict[str, list[DirectionBlock]] = {"eng": [], "cmn": []}
        for shard_index, shard in enumerate(self.shards):
            for direction in ("eng", "cmn"):
                count = int(shard[direction])
                for start in range(0, count, block_size):
                    by_direction[direction].append(
                        DirectionBlock(
                            shard_index,
                            direction,
                            start,
                            min(block_size, count - start),
                        )
                    )
        random.Random(seed).shuffle(by_direction["eng"])
        random.Random(seed + 1).shuffle(by_direction["cmn"])
        blocks: list[DirectionBlock] = []
        maximum = max(len(by_direction["eng"]), len(by_direction["cmn"]))
        for index in range(maximum):
            if index < len(by_direction["eng"]):
                blocks.append(by_direction["eng"][index])
            if index < len(by_direction["cmn"]):
                blocks.append(by_direction["cmn"][index])
        self.blocks = blocks
        self.ends: list[int] = []
        total = 0
        for block in blocks:
            total += block.length
            self.ends.append(total)
        self.length = total
        expected = int(metadata["eng"]) + int(metadata["cmn"])
        if self.length != expected:
            raise AssertionError(f"block schedule covers {self.length}, expected {expected}")

    def __len__(self) -> int:
        return self.length

    def _table(self, shard_index: int):
        table = self._tables.pop(shard_index, None)
        if table is None:
            table = pq.read_table(
                self.shards[shard_index]["file"], columns=list(REQUIRED_COLUMNS)
            )
        self._tables[shard_index] = table
        while len(self._tables) > self.cache_shards:
            self._tables.popitem(last=False)
        return table

    def _direction_index(self, shard_index: int, direction: str) -> np.ndarray:
        key = (shard_index, direction)
        values = self._indices.get(key)
        if values is None:
            values = np.load(
                self.shards[shard_index][f"{direction}_index"],
                mmap_mode="r",
                allow_pickle=False,
            )
            self._indices[key] = values
        return values

    def __getitem__(self, index: int) -> dict[str, object]:
        if not 0 <= index < self.length:
            raise IndexError(index)
        block_index = bisect.bisect_right(self.ends, index)
        previous = self.ends[block_index - 1] if block_index else 0
        block = self.blocks[block_index]
        local = block.start + index - previous
        row_index = int(self._direction_index(block.shard, block.direction)[local])
        value = self._table(block.shard).slice(row_index, 1).to_pylist()[0]
        record = self._record(value, index)
        expected_src = block.direction
        if record["src_lang"] != expected_src:
            raise AssertionError(
                f"direction index mismatch: expected {expected_src}, got {record['src_lang']}"
            )
        return {
            "record_json": json.dumps(record, ensure_ascii=False, separators=(",", ":")),
            "direction_id": 0 if expected_src == "eng" else 1,
            "sample_index": int(index),
        }


class UniSTDevDataset(Dataset, _TokenizerMixin):
    def __init__(
        self,
        parquet: str | Path,
        tokenizer_path: str | Path,
        limit: int | None = None,
        *,
        balance_directions: bool = True,
    ) -> None:
        self.path = Path(parquet)
        self.tokenizer_path = Path(tokenizer_path)
        self._tokenizer = None
        self.table = pq.read_table(self.path, columns=list(REQUIRED_COLUMNS))
        valid_rows = np.flatnonzero(nonempty_record_mask(self.table))
        maximum = len(valid_rows) if limit is None else min(len(valid_rows), int(limit))
        if limit is None or not balance_directions:
            self.row_indices = valid_rows[:maximum].astype(np.int64, copy=False)
        else:
            languages = self.table.column("src_lang").to_numpy(zero_copy_only=False)
            eng = valid_rows[languages[valid_rows] == "eng"]
            cmn = valid_rows[languages[valid_rows] == "cmn"]
            paired = min(len(eng), len(cmn), maximum // 2)
            selected = np.empty(paired * 2, dtype=np.int64)
            selected[0::2] = eng[:paired]
            selected[1::2] = cmn[:paired]
            if len(selected) < maximum:
                used = np.zeros(self.table.num_rows, dtype=np.bool_)
                used[selected] = True
                remaining = valid_rows[~used[valid_rows]][: maximum - len(selected)]
                selected = np.concatenate((selected, remaining))
            self.row_indices = selected
        self.length = int(len(self.row_indices))
        if self.length <= 0:
            raise ValueError("validation dataset is empty")

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, object]:
        if not 0 <= index < self.length:
            raise IndexError(index)
        row_index = int(self.row_indices[index])
        value = self.table.slice(row_index, 1).to_pylist()[0]
        record = self._record(value, index)
        return {
            "record_json": json.dumps(record, ensure_ascii=False, separators=(",", ":")),
            "direction_id": 0 if record["src_lang"] == "eng" else 1,
            "sample_index": int(index),
        }
