#!/usr/bin/env python3
"""Empty CoVoST parquet cells must not kill a DataLoader worker."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from simuls2st_omni.train.datasets.audio_refs import AudioRefError
from simuls2st_omni.train.datasets.manifest import ManifestDataset


def _write_s2tt(path: Path, rows: list[dict]) -> None:
    path.mkdir(parents=True)
    with (path / "part-000.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _row(i: int, audio: str) -> dict:
    return {
        "id": f"utt-{i}",
        "task_type": "s2tt_translate",
        "audio": audio,
        "corpus": "covost",
        "src_lang": "en",
        "tgt_lang": "zh",
        "source_text": f"hello {i}",
        "target_text": f"你好 {i}",
        "duration": 1.0,
    }


def test_getitem_skips_empty_audio_cell(tmp_path: Path) -> None:
    manifest = tmp_path / "s2tt"
    _write_s2tt(
        manifest,
        [
            _row(0, "parquet:bad.parquet:0:0"),
            _row(1, "parquet:good.parquet:0:1"),
        ],
    )
    dataset = ManifestDataset([manifest], corpus_roots={"covost": tmp_path})
    waves = {
        "parquet:bad.parquet:0:0": AudioRefError("parquet:bad.parquet:0:0: empty audio cell"),
        "parquet:good.parquet:0:1": np.zeros(16000, dtype=np.float32),
    }

    def fake_resolve(record: dict):
        item = waves[record["audio"]]
        if isinstance(item, Exception):
            raise item
        return item

    dataset._resolve_audio = fake_resolve  # type: ignore[method-assign]
    bad = next(i for i, rec in enumerate(dataset.records) if rec["id"] == "utt-0")
    sample = dataset[bad]
    assert sample["id"] == "utt-1"
    assert sample["audio"].shape == (16000,)


def test_getitem_raises_if_all_audio_empty(tmp_path: Path) -> None:
    manifest = tmp_path / "s2tt"
    _write_s2tt(manifest, [_row(0, "parquet:bad.parquet:0:0")])
    dataset = ManifestDataset([manifest], corpus_roots={"covost": tmp_path})

    def boom(record: dict):
        raise AudioRefError(f"{record['audio']}: empty audio cell")

    dataset._resolve_audio = boom  # type: ignore[method-assign]
    with pytest.raises(AudioRefError, match="empty audio cell"):
        dataset[0]
