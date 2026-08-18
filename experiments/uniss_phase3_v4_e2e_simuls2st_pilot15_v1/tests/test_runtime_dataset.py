from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.tests.test_teacher_requests import (
    _encode,
    _rollout,
    _trajectory,
)
from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.training.build_task_pools import (
    BUILD_SCHEMA,
)
from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.training.packing import (
    PACKED_TASK_SCHEMA,
    pack_task_samples,
)
from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.training.runtime_dataset import (
    E2EPackedFamilyDataset,
    collate_e2e_family,
)
from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.training.task_samples import (
    FAMILY_STREAMING_ASR,
    build_streaming_asr_task,
)
from training.simul_uniss.jsonl_index import write_index


def _fixture(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    sample = build_streaming_asr_task(
        _trajectory(), _rollout(), encode_text=_encode
    )
    value = next(pack_task_samples([sample], seq_length=18_000))
    packed = tmp_path / "valid_streaming_asr_event.jsonl"
    encoded = (json.dumps(value, separators=(",", ":")) + "\n").encode()
    packed.write_bytes(encoded)
    index = write_index(packed, [0])
    metadata = {
        "family": FAMILY_STREAMING_ASR,
        "schema_version": PACKED_TASK_SCHEMA,
        "path": str(packed.resolve()),
        "records": 1,
        "bytes": len(encoded),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "counts": {"supervised_tokens": value["supervised_tokens"]},
        "index": index,
    }
    report = {
        "schema_version": BUILD_SCHEMA,
        "status": "passed",
        "seq_length": 18_000,
        "families": {FAMILY_STREAMING_ASR: metadata},
    }
    report_path = tmp_path / "BUILD_COMPLETE.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    return report_path, value


def test_runtime_dataset_loads_audio_and_preserves_training_sidecars(
    tmp_path: Path,
) -> None:
    report, packed = _fixture(tmp_path)

    def load_audio(path: Path) -> tuple[torch.Tensor, int]:
        assert str(path) == _trajectory().source_audio
        return torch.arange(16_000, dtype=torch.float32).unsqueeze(0), 16_000

    dataset = E2EPackedFamilyDataset.from_build_report(
        report,
        FAMILY_STREAMING_ASR,
        verify_sha256=True,
        audio_loader=load_audio,
    )
    item = dataset[0]
    assert item["family"] == FAMILY_STREAMING_ASR
    assert item["tokens"].shape == (18_000,)
    assert item["supervised_tokens"] == packed["supervised_tokens"]
    assert len(item["acoustic_rows"]) == 1
    acoustic = item["acoustic_rows"][0]
    assert acoustic["waveform"].shape == (16_000,)
    assert acoustic["source_indices"].tolist() == list(
        range(_trajectory().source_glm_length)
    )
    assert len(item["teacher_bindings"]) == len(packed["teacher_bindings"])
    assert item["commit_consistency"] == packed["commit_consistency"]


def test_runtime_collate_stacks_fixed_tensors_and_tags_sidecars(tmp_path: Path) -> None:
    report, _ = _fixture(tmp_path)
    dataset = E2EPackedFamilyDataset.from_build_report(
        report, FAMILY_STREAMING_ASR, load_audio=False
    )
    first = dataset[0]
    second = dict(dataset[0])
    second["source_manifest_records"] = torch.tensor([0, 1])
    batch = collate_e2e_family([first, second])
    assert batch["tokens"].shape == (2, 18_000)
    assert batch["loss_kinds"].shape == (2, 18_000)
    assert batch["cu_seqlens"].shape == (2, 18_001)
    assert [len(value) for value in batch["source_manifest_records"]] == [1, 2]
    assert {row["batch_index"] for row in batch["acoustic_rows"]} == {0, 1}
    assert {row["batch_index"] for row in batch["teacher_bindings"]} == {0, 1}
    assert batch["commit_consistency"] == []


def test_runtime_dataset_rejects_changed_pack_and_mixed_family_batch(
    tmp_path: Path,
) -> None:
    report, _ = _fixture(tmp_path)
    metadata = json.loads(report.read_text())["families"][FAMILY_STREAMING_ASR]
    packed = Path(metadata["path"])
    contents = packed.read_bytes()
    packed.write_bytes(b"[" + contents[1:])
    with pytest.raises(ValueError, match="SHA256 changed"):
        E2EPackedFamilyDataset.from_build_report(
            report,
            FAMILY_STREAMING_ASR,
            verify_sha256=True,
            load_audio=False,
        )

    report, _ = _fixture(tmp_path / "second")
    dataset = E2EPackedFamilyDataset.from_build_report(
        report, FAMILY_STREAMING_ASR, load_audio=False
    )
    first = dataset[0]
    second = dict(first)
    second["family"] = "interleaved_e2e_s2st"
    with pytest.raises(ValueError, match="cannot mix"):
        collate_e2e_family([first, second])
