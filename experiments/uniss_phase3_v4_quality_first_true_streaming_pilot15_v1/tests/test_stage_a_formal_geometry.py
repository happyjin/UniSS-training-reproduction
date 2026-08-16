from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v1.stage_a_causal_whisper_asr.formal_geometry import (
    formal_geometry,
    write_formal_manifest,
)


def _report(path: Path, packs: int) -> dict[str, object]:
    path.write_bytes(b"{}\n" * packs)
    return {
        "schema_version": "uniss_quality_first_stage_a_pack_build_v1",
        "status": "complete",
        "seq_length": 18_000,
        "counts": {"packs": packs},
        "index": {
            "records": packs,
            "data_path": str(path),
            "data_size_bytes": path.stat().st_size,
        },
    }


def test_formal_geometry_matches_full_pack_counts(tmp_path: Path) -> None:
    geometry = formal_geometry(
        _report(tmp_path / "train.jsonl", 16_195),
        _report(tmp_path / "valid.jsonl", 167),
    )
    assert geometry["steps_per_epoch"] == 127
    assert geometry["epoch_samples"] == 16_256
    assert geometry["train_iters"] == 381
    assert geometry["train_samples"] == 48_768
    assert geometry["eval_iters"] == 21
    assert geometry["warmup_iters"] == 12


def test_formal_manifest_refuses_overwrite_and_failed_gate(tmp_path: Path) -> None:
    train_build = tmp_path / "train.build.json"
    valid_build = tmp_path / "valid.build.json"
    train_build.write_text(
        json.dumps(_report(tmp_path / "train.jsonl", 128)), encoding="utf-8"
    )
    valid_build.write_text(
        json.dumps(_report(tmp_path / "valid.jsonl", 8)), encoding="utf-8"
    )
    gate = tmp_path / "gate.json"
    gate.write_text(
        json.dumps(
            {"schema_version": "uniss_stage_a_training_gate_v1", "passed": True}
        ),
        encoding="utf-8",
    )
    output = tmp_path / "manifest.json"
    manifest = write_formal_manifest(
        train_build=train_build,
        valid_build=valid_build,
        training_gate=gate,
        output=output,
        run_id="formal-test",
        git_head="deadbeef",
    )
    assert manifest["geometry"]["train_iters"] == 3
    with pytest.raises(FileExistsError):
        write_formal_manifest(
            train_build=train_build,
            valid_build=valid_build,
            training_gate=gate,
            output=output,
            run_id="formal-test",
            git_head="deadbeef",
        )
    gate.write_text(
        json.dumps(
            {"schema_version": "uniss_stage_a_training_gate_v1", "passed": False}
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        write_formal_manifest(
            train_build=train_build,
            valid_build=valid_build,
            training_gate=gate,
            output=tmp_path / "blocked.json",
            run_id="formal-test",
            git_head="deadbeef",
        )

