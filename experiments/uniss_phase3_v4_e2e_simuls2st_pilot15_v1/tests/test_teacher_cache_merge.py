from __future__ import annotations

import json
from array import array
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.rollout.io import (
    file_sha256,
)
from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.teacher.build_phase3_cache import (
    PART_SCHEMA,
)
from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.teacher.cache import (
    CACHE_SCHEMA,
    combine_sample,
    save_bundle,
)
from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.teacher.merge_phase3_cache import (
    merge_phase3_cache,
)
from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.teacher.requests import (
    build_phase3_requests,
)
from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.tests.test_teacher_requests import (
    DIGEST,
    _encode,
    _rollout,
    _trajectory,
)
from training import constants_uniss as c
from training.simul_uniss.jsonl_index import load_index, write_index


def _write_jsonl(path: Path, rows: list[str]) -> None:
    offsets = array("Q")
    byte_offset = 0
    with path.open("wb") as handle:
        for row in rows:
            encoded = (row + "\n").encode("utf-8")
            offsets.append(byte_offset)
            handle.write(encoded)
            byte_offset += len(encoded)
    write_index(path, offsets)


def _summaries(trajectory, rollout):
    requests = build_phase3_requests(trajectory, rollout, encode_text=_encode)
    summaries = []
    for request in requests:
        labels = np.asarray(request.reference_labels, dtype=np.int32)
        summaries.append(
            {
                "indices": np.stack((labels, (labels + 1) % c.VOCAB_SIZE), axis=1),
                "probabilities": np.tile(
                    np.asarray([[0.75, 0.25]], dtype=np.float16),
                    (len(labels), 1),
                ),
                "top1": labels.copy(),
                "confidence": np.full(len(labels), 0.75, dtype=np.float16),
            }
        )
    return requests, summaries


def _rewrite_manifest(part_root: Path, mutate) -> None:
    manifest = part_root / "teacher_cache.jsonl"
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    mutate(rows)
    manifest.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    marker_path = part_root / "PART_COMPLETE.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["manifest_bytes"] = manifest.stat().st_size
    marker["manifest_sha256"] = file_sha256(manifest)
    marker_path.write_text(json.dumps(marker), encoding="utf-8")


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    gold = tmp_path / "gold.jsonl"
    trajectories = []
    rollouts = []
    for index in range(2):
        trajectories.append(
            replace(
                _trajectory(),
                sample_id=f"sample-{index}",
                source_manifest_record=index,
            )
        )
        rollouts.append(
            replace(
                _rollout(),
                sample_id=f"sample-{index}",
                source_manifest_record=index,
            )
        )
    _write_jsonl(gold, [value.to_json() for value in trajectories])

    parts_root = tmp_path / "parts"
    for rank, (trajectory, rollout) in enumerate(zip(trajectories, rollouts)):
        part_root = parts_root / f"part_{rank:03d}"
        requests, summaries = _summaries(trajectory, rollout)
        arrays, descriptors = combine_sample(requests, summaries)
        bundle = part_root / "bundles" / "bundle-000000.npz"
        rows = save_bundle(
            bundle,
            [
                {
                    "sample_id": trajectory.sample_id,
                    "split": trajectory.split,
                    "source_manifest_record": rank,
                    "arrays": arrays,
                    "requests": descriptors,
                }
            ],
        )
        rows[0]["bundle_sha256"] = file_sha256(bundle)
        manifest = part_root / "teacher_cache.jsonl"
        _write_jsonl(
            manifest,
            [json.dumps(rows[0], ensure_ascii=False, separators=(",", ":"))],
        )
        counts: Counter[str] = Counter(
            {
                "records": 1,
                "requests": len(descriptors),
                "teacher_positions": len(arrays["reference_label"]),
                "teacher_top1_correct": int(
                    np.count_nonzero(arrays["top1"] == arrays["reference_label"])
                ),
                "reference_in_topk": int(
                    np.count_nonzero(
                        (
                            arrays["indices"]
                            == arrays["reference_label"][:, None]
                        ).any(axis=1)
                    )
                ),
            }
        )
        for descriptor in descriptors:
            counts[f"family:{descriptor['family']}"] += 1
            counts[f"history:{descriptor['history_kind']}"] += 1
            counts["content_candidate_tokens"] += int(
                descriptor["content_candidate_tokens"]
            )
            counts["content_selected_tokens"] += int(
                descriptor["content_selected_tokens"]
            )
        marker = {
            "schema_version": PART_SCHEMA,
            "cache_schema": CACHE_SCHEMA,
            "status": "complete",
            "rank": rank,
            "world_size": 2,
            "selection_start": 0,
            "selection_stop": 2,
            "assigned_start": rank,
            "assigned_stop": rank + 1,
            "gold": str(gold.resolve()),
            "rollouts": str((tmp_path / "rollouts.jsonl").resolve()),
            "model": str((tmp_path / "phase3_hf").resolve()),
            "phase3_hf_sha256": DIGEST,
            "runtime_sha256": DIGEST,
            "topk": 2,
            "temperature": 1.5,
            "semantic_stride": 8,
            "max_padded_tokens": 1024,
            "max_batch_size": 8,
            "max_selected_positions": 64,
            "counts": dict(counts),
            "manifest": str(manifest.resolve()),
            "manifest_bytes": manifest.stat().st_size,
            "manifest_sha256": file_sha256(manifest),
        }
        (part_root / "PART_COMPLETE.json").write_text(
            json.dumps(marker), encoding="utf-8"
        )
    return gold, parts_root


def _merge(tmp_path: Path, gold: Path, parts_root: Path):
    return merge_phase3_cache(
        gold_path=gold,
        parts_root=parts_root,
        world_size=2,
        output_path=tmp_path / "merged.jsonl",
        audit_path=tmp_path / "AUDIT.json",
    )


def test_phase3_teacher_cache_merge_audits_all_rows(tmp_path) -> None:
    gold, parts_root = _fixture(tmp_path)
    audit = _merge(tmp_path, gold, parts_root)
    assert audit["status"] == "passed"
    assert audit["counts"]["records"] == 2
    assert audit["teacher_top1_accuracy"] == 1.0
    assert audit["reference_in_topk_rate"] == 1.0
    assert len(load_index(tmp_path / "merged.jsonl") or []) == 2


def test_phase3_teacher_cache_merge_rejects_part_range_gap(tmp_path) -> None:
    gold, parts_root = _fixture(tmp_path)
    marker_path = parts_root / "part_001" / "PART_COMPLETE.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["assigned_start"] = 0
    marker_path.write_text(json.dumps(marker), encoding="utf-8")
    with pytest.raises(ValueError, match="range gap or overlap"):
        _merge(tmp_path, gold, parts_root)


def test_phase3_teacher_cache_merge_rejects_bundle_digest_change(tmp_path) -> None:
    gold, parts_root = _fixture(tmp_path)
    _rewrite_manifest(
        parts_root / "part_000",
        lambda rows: rows[0].__setitem__("bundle_sha256", "0" * 64),
    )
    with pytest.raises(ValueError, match="bundle digest changed"):
        _merge(tmp_path, gold, parts_root)


def test_phase3_teacher_cache_merge_rejects_gold_identity_change(tmp_path) -> None:
    gold, parts_root = _fixture(tmp_path)
    _rewrite_manifest(
        parts_root / "part_000",
        lambda rows: rows[0].__setitem__("sample_id", "wrong-sample"),
    )
    with pytest.raises(ValueError, match="differs from gold sample identity"):
        _merge(tmp_path, gold, parts_root)


def test_phase3_teacher_cache_merge_rejects_descriptor_gap(tmp_path) -> None:
    gold, parts_root = _fixture(tmp_path)

    def mutate(rows) -> None:
        rows[0]["requests"][0]["position_stop"] -= 1

    _rewrite_manifest(parts_root / "part_000", mutate)
    with pytest.raises(ValueError, match="descriptor length differs"):
        _merge(tmp_path, gold, parts_root)


@pytest.mark.parametrize("invalid", ["sum", "nan"])
def test_phase3_teacher_cache_merge_rejects_invalid_probabilities(
    tmp_path, invalid
) -> None:
    gold, parts_root = _fixture(tmp_path)
    part_root = parts_root / "part_000"
    bundle_path = part_root / "bundles" / "bundle-000000.npz"
    with np.load(bundle_path, allow_pickle=False) as source:
        values = {name: source[name].copy() for name in source.files}
    probabilities = values["row_0_probabilities"]
    if invalid == "sum":
        probabilities[0] = np.asarray([0.75, 0.75], dtype=np.float16)
    else:
        probabilities[0, 0] = np.nan
    temporary = bundle_path.with_name("replacement.npz")
    np.savez(temporary, **values)
    temporary.replace(bundle_path)
    digest = file_sha256(bundle_path)
    _rewrite_manifest(
        part_root,
        lambda rows: rows[0].__setitem__("bundle_sha256", digest),
    )
    message = "NaN/Inf" if invalid == "nan" else "do not sum to one"
    with pytest.raises(ValueError, match=message):
        _merge(tmp_path, gold, parts_root)
