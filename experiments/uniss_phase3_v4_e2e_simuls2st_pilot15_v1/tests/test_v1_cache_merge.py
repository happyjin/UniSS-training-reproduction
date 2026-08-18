from __future__ import annotations

import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.rollout.io import (
    file_sha256,
)
from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.teacher.build_v1_cache import (
    V1_PART_SCHEMA,
)
from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.teacher.merge_v1_cache import (
    merge_v1_cache,
)
from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.teacher.v1_cache import (
    V1_CACHE_SCHEMA,
    combine_v1_sample,
    save_v1_bundle,
)
from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.teacher.v1_requests import (
    build_v1_teacher_sequences,
)
from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.tests.test_teacher_cache_merge import (
    _write_jsonl,
)
from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.tests.test_teacher_requests import (
    DIGEST,
    _encode,
    _rollout,
    _trajectory,
)
from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.tests.test_v1_teacher_cache import (
    _summaries,
)
from training.simul_uniss.jsonl_index import load_index


def _rewrite_manifest(part_root: Path, mutate) -> None:
    manifest = part_root / "v1_teacher_cache.jsonl"
    rows = [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
    ]
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
        sequences = build_v1_teacher_sequences(
            trajectory, rollout, encode_text=_encode
        )
        arrays, descriptors = combine_v1_sample(
            sequences, _summaries(sequences)
        )
        bundle = part_root / "bundles" / "bundle-000000.npz"
        rows = save_v1_bundle(
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
        manifest = part_root / "v1_teacher_cache.jsonl"
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
            counts[f"history:{descriptor['history_kind']}"] += 1
            counts["final_requests"] += int(bool(descriptor["final"]))
        marker = {
            "schema_version": V1_PART_SCHEMA,
            "cache_schema": V1_CACHE_SCHEMA,
            "status": "complete",
            "rank": rank,
            "world_size": 2,
            "selection_start": 0,
            "selection_stop": 2,
            "assigned_start": rank,
            "assigned_stop": rank + 1,
            "gold": str(gold.resolve()),
            "rollouts": str((tmp_path / "rollouts.jsonl").resolve()),
            "checkpoint": str((tmp_path / "v1_native").resolve()),
            "hf_model": str((tmp_path / "v1_hf").resolve()),
            "whispervq_model": str((tmp_path / "whispervq").resolve()),
            "v1_hf_sha256": DIGEST,
            "runtime_sha256": DIGEST,
            "topk": 2,
            "temperature": 1.5,
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
    return merge_v1_cache(
        gold_path=gold,
        parts_root=parts_root,
        world_size=2,
        output_path=tmp_path / "merged.jsonl",
        audit_path=tmp_path / "AUDIT.json",
    )


def test_v1_teacher_cache_merge_audits_both_histories(tmp_path) -> None:
    gold, parts_root = _fixture(tmp_path)
    audit = _merge(tmp_path, gold, parts_root)
    assert audit["status"] == "passed"
    assert audit["counts"]["records"] == 2
    assert audit["counts"]["history:gold_asr"] > 0
    assert audit["counts"]["history:v1_asr"] > 0
    assert audit["teacher_top1_accuracy"] == 1.0
    assert len(load_index(tmp_path / "merged.jsonl") or []) == 2


def test_v1_teacher_cache_merge_rejects_target_digest_change(tmp_path) -> None:
    gold, parts_root = _fixture(tmp_path)

    def mutate(rows) -> None:
        rows[0]["requests"][0]["target_sha256"] = "0" * 64

    _rewrite_manifest(parts_root / "part_000", mutate)
    with pytest.raises(ValueError, match="target digest differs"):
        _merge(tmp_path, gold, parts_root)


def test_v1_teacher_cache_merge_rejects_history_descriptor_change(tmp_path) -> None:
    gold, parts_root = _fixture(tmp_path)

    def mutate(rows) -> None:
        rows[0]["requests"][0]["history_kind"] = "v1_asr"

    _rewrite_manifest(parts_root / "part_000", mutate)
    with pytest.raises(ValueError, match="history_id differs"):
        _merge(tmp_path, gold, parts_root)


def test_v1_teacher_cache_merge_rejects_probability_sum(tmp_path) -> None:
    gold, parts_root = _fixture(tmp_path)
    part_root = parts_root / "part_000"
    bundle_path = part_root / "bundles" / "bundle-000000.npz"
    with np.load(bundle_path, allow_pickle=False) as source:
        values = {name: source[name].copy() for name in source.files}
    values["row_0_probabilities"][0] = np.asarray(
        [0.75, 0.75], dtype=np.float16
    )
    temporary = bundle_path.with_name("replacement.npz")
    np.savez(temporary, **values)
    temporary.replace(bundle_path)
    digest = file_sha256(bundle_path)
    _rewrite_manifest(
        part_root,
        lambda rows: rows[0].__setitem__("bundle_sha256", digest),
    )
    with pytest.raises(ValueError, match="do not sum to one"):
        _merge(tmp_path, gold, parts_root)
