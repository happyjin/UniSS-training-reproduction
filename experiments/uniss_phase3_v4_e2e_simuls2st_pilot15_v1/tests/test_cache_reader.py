from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.tests.test_teacher_cache_merge import (
    _fixture as phase3_fixture,
)
from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.tests.test_teacher_cache_merge import (
    _merge as merge_phase3,
)
from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.tests.test_teacher_requests import (
    _encode,
    _rollout,
    _trajectory,
)
from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.tests.test_v1_cache_merge import (
    _fixture as v1_fixture,
)
from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.tests.test_v1_cache_merge import (
    _merge as merge_v1,
)
from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.training.cache_reader import (
    TopKTeacherCacheReader,
    resolve_teacher_bindings,
)
from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.training.packing import (
    pack_task_samples,
)
from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.training.task_samples import (
    build_incremental_mt_tasks,
    build_streaming_asr_task,
)


def _packed_binding(sample) -> tuple[dict[str, object], dict[str, object]]:
    packed = next(pack_task_samples([sample], seq_length=512))
    return packed, packed["teacher_bindings"][0]


def _record_zero():
    return (
        replace(_trajectory(), sample_id="sample-0", source_manifest_record=0),
        replace(_rollout(), sample_id="sample-0", source_manifest_record=0),
    )


def test_phase3_reader_returns_the_exact_incremental_mt_target_slice(
    tmp_path: Path,
) -> None:
    root = tmp_path / "phase3"
    root.mkdir()
    gold, parts = phase3_fixture(root)
    merge_phase3(root, gold, parts)
    reader = TopKTeacherCacheReader(
        root / "AUDIT.json",
        cache_kind="phase3",
        verify_manifest_sha256=True,
        verify_bundle_sha256=True,
    )
    trajectory, rollout = _record_zero()
    sample = build_incremental_mt_tasks(
        trajectory, rollout, encode_text=_encode
    )[0]
    packed, binding = _packed_binding(sample)
    posterior = reader.read_binding(binding)
    assert posterior.positions == binding["packed_stop"] - binding["packed_start"]
    assert posterior.indices.shape[1] == 2
    assert torch.equal(
        posterior.reference_labels,
        torch.tensor(
            packed["labels"][binding["packed_start"] : binding["packed_stop"]]
        ),
    )
    assert torch.allclose(posterior.probabilities.sum(dim=1), torch.ones(posterior.positions))


def test_v1_reader_and_binding_resolver_preserve_packed_coordinates(
    tmp_path: Path,
) -> None:
    root = tmp_path / "v1"
    root.mkdir()
    gold, parts = v1_fixture(root)
    merge_v1(root, gold, parts)
    reader = TopKTeacherCacheReader(
        root / "AUDIT.json", cache_kind="v1_asr", verify_bundle_sha256=True
    )
    trajectory, rollout = _record_zero()
    sample = build_streaming_asr_task(
        trajectory, rollout, encode_text=_encode
    )
    packed = next(pack_task_samples([sample], seq_length=512))
    values = resolve_teacher_bindings(
        packed["teacher_bindings"],
        {"v1_asr": reader},
        packed_labels=torch.tensor(packed["labels"]),
    )
    assert len(values) == len(packed["teacher_bindings"])
    for value in values:
        posterior = value["posterior"]
        assert posterior.positions == value["packed_stop"] - value["packed_start"]
        assert torch.equal(
            posterior.reference_labels,
            torch.tensor(
                packed["labels"][value["packed_start"] : value["packed_stop"]]
            ),
        )
    changed = torch.tensor(packed["labels"])
    changed[packed["teacher_bindings"][0]["packed_start"]] += 1
    with pytest.raises(ValueError, match="differ from packed labels"):
        resolve_teacher_bindings(
            packed["teacher_bindings"],
            {"v1_asr": reader},
            packed_labels=changed,
        )


def test_reader_rejects_identity_and_request_range_mismatches(tmp_path: Path) -> None:
    root = tmp_path / "phase3"
    root.mkdir()
    gold, parts = phase3_fixture(root)
    merge_phase3(root, gold, parts)
    reader = TopKTeacherCacheReader(root / "AUDIT.json", cache_kind="phase3")
    trajectory, rollout = _record_zero()
    sample = build_incremental_mt_tasks(
        trajectory, rollout, encode_text=_encode
    )[0]
    _, binding = _packed_binding(sample)
    wrong_identity = dict(binding)
    wrong_identity["sample_id"] = "not-the-sample"
    with pytest.raises(ValueError, match="identity differs"):
        reader.read_binding(wrong_identity)
    wrong_range = dict(binding)
    wrong_range["cache_position_stop"] = 1_000_000
    with pytest.raises(ValueError, match="exceeds its cache request"):
        reader.read_binding(wrong_range)
