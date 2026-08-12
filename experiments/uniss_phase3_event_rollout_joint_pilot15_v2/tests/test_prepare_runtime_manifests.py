from __future__ import annotations

import json
from pathlib import Path

from experiments.uniss_phase3_event_rollout_joint_pilot15_v2.evaluation.prepare_runtime_manifests import (
    prepare,
)
from training.simul_uniss.jsonl_index import load_index, write_index


def _source(path: Path) -> None:
    offsets = []
    with path.open("wb") as handle:
        for index in range(20):
            offsets.append(handle.tell())
            direction = ("cmn", "eng") if index % 2 == 0 else ("eng", "cmn")
            handle.write(
                (
                    json.dumps(
                        {
                            "id": f"sample-{index}",
                            "src_lang": direction[0],
                            "tgt_lang": direction[1],
                            "source_audio": f"/audio/{index}.wav",
                            "translation": f"translation {index}",
                        }
                    )
                    + "\n"
                ).encode()
            )
    write_index(path, offsets)


def test_full_selection_is_complete_and_disjoint(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    _source(source)
    summary = prepare(
        source,
        tmp_path / "full",
        split="valid",
        num_shards=3,
        samples_per_direction=None,
        seed=7,
    )
    assert summary["selected_records"] == 20
    assert summary["selected_unique_sample_ids"] == 20
    assert sum(part["records"] for part in summary["parts"]) == 20
    assert summary["directions"] == {"cmn->eng": 10, "eng->cmn": 10}
    for part in summary["parts"]:
        assert len(load_index(Path(part["manifest"]))) == part["records"]


def test_balanced_selection_is_deterministic(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    _source(source)
    first = prepare(
        source,
        tmp_path / "first",
        split="train",
        num_shards=2,
        samples_per_direction=3,
        seed=19,
    )
    second = prepare(
        source,
        tmp_path / "second",
        split="train",
        num_shards=2,
        samples_per_direction=3,
        seed=19,
    )
    assert first["directions"] == {"cmn->eng": 3, "eng->cmn": 3}
    first_ids = [
        json.loads(Path(part["sample_ids"]).read_text()) for part in first["parts"]
    ]
    second_ids = [
        json.loads(Path(part["sample_ids"]).read_text()) for part in second["parts"]
    ]
    assert first_ids == second_ids
