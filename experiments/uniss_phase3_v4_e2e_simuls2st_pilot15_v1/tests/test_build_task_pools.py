from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.tests.test_teacher_requests import (
    _encode,
    _rollout,
    _trajectory,
)
from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.rollout.stratify_rollouts import (
    STRATA_SCHEMA,
    STRATUM_CLEAN,
)
from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.training import (
    build_task_pools,
)
from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.training.packing import (
    validate_packed_task,
)
from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.training.task_samples import (
    FAMILY_STREAMING_ASR,
    LOSS_KIND_NAMES,
    TASK_FAMILIES,
    build_streaming_asr_task,
)
from training.simul_uniss.jsonl_index import load_index, write_index


class _Tokenizer:
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return _encode(text)


def _write_jsonl(path: Path, rows: list[str]) -> None:
    offsets = []
    offset = 0
    with path.open("wb") as handle:
        for row in rows:
            encoded = (row + "\n").encode("utf-8")
            offsets.append(offset)
            handle.write(encoded)
            offset += len(encoded)
    write_index(path, offsets)


def _stratum_row() -> str:
    return json.dumps(
        {
            "schema_version": STRATA_SCHEMA,
            "sample_id": "sample-1",
            "split": "valid",
            "src_lang": "eng",
            "source_manifest_record": 0,
            "rollout_ordinal": 0,
            "stratum": STRATUM_CLEAN,
            "reasons": [],
            "structural": {
                "malformed_write_events": 0,
                "early_eos_events": 0,
                "final_reached_eos": True,
            },
            "content": {
                "metric": "wer",
                "errors": 0,
                "reference_units": 2,
                "error_rate": 0.0,
                "empty_events": 0,
                "events": 2,
            },
        },
        separators=(",", ":"),
    )


def test_worker_ranges_are_contiguous_and_cover_every_record_once() -> None:
    assert build_task_pools._ranges(10, 3) == [(0, 3), (3, 6), (6, 10)]
    assert build_task_pools._ranges(3, 8) == [(0, 1), (1, 2), (2, 3)]


def test_worker_builds_all_five_readable_families_with_exact_loss_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gold = tmp_path / "gold.jsonl"
    rollouts = tmp_path / "rollouts.jsonl"
    strata = tmp_path / "strata.jsonl"
    _write_jsonl(gold, [_trajectory().to_json()])
    _write_jsonl(rollouts, [_rollout().to_json()])
    _write_jsonl(strata, [_stratum_row()])
    monkeypatch.setattr(
        build_task_pools.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: _Tokenizer(),
    )
    report = build_task_pools._worker(
        (
            0,
            gold,
            rollouts,
            strata,
            0,
            1,
            0,
            1,
            tmp_path / "tokenizer",
            tmp_path / "parts",
            18_000,
        )
    )
    assert report["records"] == 1
    assert set(report["families"]) == set(TASK_FAMILIES)
    for family in TASK_FAMILIES:
        family_report = report["families"][family]
        path = Path(family_report["path"])
        offsets = load_index(path)
        assert offsets is not None and len(offsets) == family_report["records"]
        packed_loss_counts = {name: 0 for name in LOSS_KIND_NAMES.values()}
        with path.open("rb") as handle:
            for offset in offsets:
                handle.seek(int(offset))
                value = json.loads(handle.readline())
                validate_packed_task(value, seq_length=18_000)
                assert value["family"] == family
                for kind in value["loss_kinds"][: value["used_tokens"]]:
                    packed_loss_counts[LOSS_KIND_NAMES[int(kind)]] += 1
        assert family_report["counts"]["supervised_tokens"] > 0
        if family == "interleaved_e2e_s2st":
            assert any(
                json.loads(path.read_text(encoding="utf-8").splitlines()[0])[
                    "teacher_bindings"
                ]
            )
        for name, count in packed_loss_counts.items():
            assert family_report["counts"][f"loss:{name}"] == count


def test_merge_rejects_a_part_changed_after_worker_completion(tmp_path: Path) -> None:
    sample = build_streaming_asr_task(
        _trajectory(), _rollout(), encode_text=_encode
    )
    writer = build_task_pools._PackedWriter(
        tmp_path / "part.jsonl", FAMILY_STREAMING_ASR, 512
    )
    writer.add(sample)
    report = writer.close()
    path = Path(report["path"])
    contents = path.read_bytes()
    path.write_bytes(b"[" + contents[1:])
    with pytest.raises(ValueError, match="digest changed"):
        build_task_pools._merge_family(
            FAMILY_STREAMING_ASR,
            [{"families": {FAMILY_STREAMING_ASR: report}}],
            tmp_path / "merged.jsonl",
        )


def test_writer_rejects_an_overlong_sample_without_truncation(tmp_path: Path) -> None:
    sample = build_streaming_asr_task(
        _trajectory(), _rollout(), encode_text=_encode
    )
    writer = build_task_pools._PackedWriter(
        tmp_path / "overlong.jsonl",
        FAMILY_STREAMING_ASR,
        sample.shifted_length - 1,
    )
    try:
        with pytest.raises(ValueError, match="exceeds 18k pack"):
            writer.add(sample)
    finally:
        writer.abort()
