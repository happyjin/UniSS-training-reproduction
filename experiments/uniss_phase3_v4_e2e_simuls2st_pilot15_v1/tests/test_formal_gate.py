from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.data.finalize_gold_gate import (
    finalize,
)
from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.data.freeze_split import (
    freeze_split,
)
from training.simul_uniss.jsonl_index import write_index


def _write_json(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")
    return path


def test_freeze_split_rechecks_disjoint_ids(tmp_path: Path) -> None:
    manifests = {}
    id_paths = {}
    split_audit = {}
    for split, ids in (("train", [1, 3]), ("valid", [2])):
        manifest = tmp_path / f"{split}.jsonl"
        lines = [json.dumps({"id": f"{split}-{index}"}) + "\n" for index in range(len(ids))]
        manifest.write_text("".join(lines), encoding="utf-8")
        offsets = []
        cursor = 0
        for line in lines:
            offsets.append(cursor)
            cursor += len(line.encode("utf-8"))
        write_index(manifest, offsets)
        ids_path = tmp_path / f"{split}_ids.npy"
        np.save(ids_path, np.asarray(ids, dtype=np.uint64))
        manifests[split] = {"path": str(manifest), "records": len(ids)}
        id_paths[split] = ids_path
        split_audit[split] = {
            "counters": {
                "records": len(ids),
                "duration_ms": 1000 * len(ids),
                "direction:eng->cmn": len(ids),
            },
            "sorted_id_hashes": str(ids_path),
        }
    snapshot = _write_json(
        tmp_path / "snapshot.json",
        {
            "schema_version": "uniss_quality_first_stage_a_source_snapshot_v2",
            **manifests,
        },
    )
    audit = _write_json(
        tmp_path / "audit.json",
        {
            "passed": True,
            "train_valid_id_overlap": 0,
            **split_audit,
        },
    )
    fingerprints = _write_json(
        tmp_path / "fingerprints.json",
        {
            "status": "complete",
            "checkpoints": {
                "v1": {"sha256": hashlib.sha256(b"v1").hexdigest()},
                "phase3": {"sha256": hashlib.sha256(b"phase3").hexdigest()},
            },
        },
    )
    frozen = freeze_split(snapshot, audit, fingerprints)
    assert frozen["status"] == "frozen"
    assert frozen["train_valid_id_overlap"] == 0
    assert frozen["splits"]["train"]["records"] == 2


def test_finalize_keeps_gpu_training_blocked_until_rollout(tmp_path: Path) -> None:
    frozen = _write_json(
        tmp_path / "frozen.json",
        {
            "status": "frozen",
            "train_valid_id_overlap": 0,
            "splits": {"train": {"records": 2}, "valid": {"records": 1}},
        },
    )
    paths = {}
    for split, records in (("train", 2), ("valid", 1)):
        paths[f"{split}_build"] = _write_json(
            tmp_path / f"{split}_build.json",
            {
                "status": "complete",
                "hash_audio": True,
                "audit_audio": True,
                "counts": {"records": records},
            },
        )
        paths[f"{split}_audit"] = _write_json(
            tmp_path / f"{split}_audit.json",
            {
                "status": "passed",
                "require_audio_hash": True,
                "require_audio_audit": True,
                "counts": {"records": records, "events": records * 3, "prefinal_target_writes": records},
            },
        )
    gate = finalize(
        frozen,
        paths["train_build"],
        paths["valid_build"],
        paths["train_audit"],
        paths["valid_audit"],
    )
    assert gate["status"] == "passed"
    assert gate["formal_training_authorized"] is False
    assert gate["v1_rollout_status"] == "pending"
