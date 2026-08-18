from __future__ import annotations

from pathlib import Path

from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.data.fingerprint import (
    fingerprint_checkpoint,
)


def test_tree_fingerprint_covers_file_content(tmp_path: Path) -> None:
    checkpoint = tmp_path / "iter_1"
    checkpoint.mkdir()
    (checkpoint / "metadata.json").write_text("{}\n", encoding="utf-8")
    shard = checkpoint / "__0_0.distcp"
    shard.write_bytes(b"weights-v1")
    first = fingerprint_checkpoint(checkpoint, workers=2)
    assert first["files"] == 2
    assert first["bytes"] == 3 + len(b"weights-v1")
    shard.write_bytes(b"weights-v2")
    second = fingerprint_checkpoint(checkpoint, workers=2)
    assert first["sha256"] != second["sha256"]
