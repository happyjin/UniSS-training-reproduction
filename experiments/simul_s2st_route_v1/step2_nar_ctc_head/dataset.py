"""Indexed joint-manifest dataset for teacher-forced NAR CTC training.

Reads only the fields the head needs (no waveform I/O). Degenerate pairs — those
whose required CTC frames per text token exceed ``degenerate_ratio_limit`` — are
filtered out of the train split so they cannot dictate the frame budget.

Filtered index lists are cached next to the manifest so eight Megatron ranks do
not each rescan a multi-gigabyte JSONL on every launch.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from experiments.simul_s2st_route_v1.step2_nar_ctc_head.duration_anchored_nar_ctc import (
    adjacent_repeats,
    required_ctc_frames,
)
from training.simul_uniss.jsonl_index import load_index


class NarCtcJointDataset(Dataset[dict[str, object]]):
    def __init__(
        self,
        manifest: str | Path,
        *,
        max_audio_seconds: float = 12.0,
        min_audio_seconds: float = 0.4,
        max_unit_tokens: int = 1200,
        degenerate_ratio_limit: float = 100.0,
        filter_degenerate: bool = True,
        max_samples: int | None = None,
        cache_dir: str | Path | None = None,
    ) -> None:
        self.path = Path(manifest)
        offsets = load_index(self.path)
        if offsets is None:
            raise ValueError(f"missing offset index for {self.path}")
        self.offsets = offsets
        self.max_audio_seconds = float(max_audio_seconds)
        self.min_audio_seconds = float(min_audio_seconds)
        self.max_unit_tokens = int(max_unit_tokens)
        self.degenerate_ratio_limit = float(degenerate_ratio_limit)
        self.filter_degenerate = bool(filter_degenerate)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else self.path.parent / ".nar_ctc_index_cache"
        self.indices = self._select_indices(max_samples)

    def _filter_signature(self, max_samples: int | None) -> str:
        payload = {
            "manifest": str(self.path.resolve()),
            "data_size_bytes": self.path.stat().st_size,
            "data_mtime_ns": self.path.stat().st_mtime_ns,
            "records": len(self.offsets),
            "max_audio_seconds": self.max_audio_seconds,
            "min_audio_seconds": self.min_audio_seconds,
            "max_unit_tokens": self.max_unit_tokens,
            "degenerate_ratio_limit": self.degenerate_ratio_limit,
            "filter_degenerate": self.filter_degenerate,
            "max_samples": max_samples,
        }
        return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]

    def _cache_paths(self, max_samples: int | None) -> tuple[Path, Path]:
        stem = f"{self.path.name}.{self._filter_signature(max_samples)}"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        return self.cache_dir / f"{stem}.idx.npy", self.cache_dir / f"{stem}.json"

    def _select_indices(self, max_samples: int | None) -> list[int]:
        cache_bin, cache_meta = self._cache_paths(max_samples)
        if cache_bin.is_file() and cache_meta.is_file():
            return [int(value) for value in np.load(cache_bin)]
        selected: list[int] = []
        with self.path.open("rb") as handle:
            for index, offset in enumerate(self.offsets):
                handle.seek(int(offset))
                record = json.loads(handle.readline())
                if not self._keep(record):
                    continue
                selected.append(index)
                if max_samples is not None and len(selected) >= max_samples:
                    break
        if not selected:
            raise RuntimeError(f"no usable rows in {self.path}")
        array = np.asarray(selected, dtype=np.int64)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{cache_bin.name}.", dir=self.cache_dir, suffix=".npy"
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            with temporary.open("wb") as handle:
                np.save(handle, array)
            os.replace(temporary, cache_bin)
        finally:
            temporary.unlink(missing_ok=True)
        cache_meta.write_text(
            json.dumps({"records": len(selected), "manifest": str(self.path)}, indent=2) + "\n",
            encoding="utf-8",
        )
        return selected

    def _keep(self, record: dict[str, object]) -> bool:
        duration_ms = float(record.get("source_duration_ms", 0))
        seconds = duration_ms / 1000.0
        if not self.min_audio_seconds <= seconds <= self.max_audio_seconds:
            return False
        units = record.get("target_bicodec") or []
        text = record.get("target_qwen_ids") or []
        if not units or not text:
            return False
        if len(units) > self.max_unit_tokens:
            return False
        if self.filter_degenerate:
            repeats = sum(
                1 for left, right in zip(units, units[1:]) if int(left) == int(right)
            )
            required = required_ctc_frames(len(units), repeats)
            if required > self.degenerate_ratio_limit * len(text):
                return False
        return True

    def __len__(self) -> int:
        return len(self.indices)

    def _read(self, physical_index: int) -> dict[str, object]:
        with self.path.open("rb") as handle:
            handle.seek(int(self.offsets[physical_index]))
            return json.loads(handle.readline())

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self._read(self.indices[index])
        units = torch.tensor([int(value) for value in record["target_bicodec"]], dtype=torch.long)
        # Strings go through one JSON blob so Megatron's default collator (which
        # rejects bare str fields) only has to stack tensors.
        meta = {
            "id": str(record["id"]),
            "src_lang": str(record["src_lang"]),
            "tgt_lang": str(record["tgt_lang"]),
            "translation": str(record["translation"]),
        }
        return {
            "record_json": json.dumps(meta, ensure_ascii=False),
            "source_duration_ms": torch.tensor(int(record["source_duration_ms"]), dtype=torch.long),
            "source_glm": torch.tensor([int(value) for value in record["source_glm"]], dtype=torch.long),
            "target_bicodec": units,
            "target_bicodec_length": torch.tensor(len(units), dtype=torch.long),
            "unit_repeats": adjacent_repeats(units),
            "bicodec_global": torch.tensor(
                [int(value) for value in record["bicodec_global"]], dtype=torch.long
            ),
        }
