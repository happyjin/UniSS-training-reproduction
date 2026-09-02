"""The pool has to satisfy the readers it was not written for.

Two of them matter.  ``validate_packed_task`` is the base experiment's own
checker, and ``decode_packed_e2e_task`` in ``runtime_dataset`` re-validates the
acoustic rows at load time -- it raises unless ``source_indices`` is exactly
``range(source_glm_length)`` and the three parallel arrays agree.  Both are
exercised here against rows produced from real trajectories, because a pool
that only satisfies its own writer is not evidence of anything.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.data.schema import (
    E2ETrajectory,
)
from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.training.packing import (
    validate_packed_task,
)
from experiments.uniss_streaming_p2st_pure_ce_v1.training.build_p2st_pools import (
    _merge_family,
    _ranges,
    build_trajectory_samples,
    pack_p2st_samples,
)
from experiments.uniss_streaming_p2st_pure_ce_v1.training.task_samples_p2st import (
    REPLAY_FAMILIES,
    FAMILY_P2ST_ASR,
    causal_glm_token_count,
)
from training.simul_uniss.jsonl_index import load_index

SEQ_LENGTH = 18_000
REPO_ROOT = Path(__file__).resolve().parents[3]
GOLD = (
    REPO_ROOT
    / "data/processed/uniss_phase3_v4_e2e_simuls2st_pilot15_v1"
    / "formal_gold_20260818T090515Z/source_events/valid_gold_trajectories.jsonl"
)


def _encode(text: str) -> list[int]:
    return [ord(char) % 1000 + 1 for char in text]


@pytest.fixture(scope="module")
def trajectories() -> list[E2ETrajectory]:
    if not GOLD.exists():
        pytest.skip(f"gold trajectories not present at {GOLD}")
    records = []
    with GOLD.open() as handle:
        for index, line in enumerate(handle):
            records.append(E2ETrajectory.from_mapping(json.loads(line)))
            if index >= 39:
                break
    return records


@pytest.fixture(scope="module")
def rows_by_family(trajectories):
    grouped: dict[str, list] = {}
    for trajectory in trajectories:
        for family, samples in build_trajectory_samples(
            trajectory, encode_text=_encode
        ).items():
            grouped.setdefault(family, []).extend(samples)
    return {
        family: list(pack_p2st_samples(samples, seq_length=SEQ_LENGTH))
        for family, samples in grouped.items()
        if samples
    }


def test_ranges_partition_exactly():
    for total, workers in ((13469, 32), (200, 4), (7, 16), (1, 8), (0, 4)):
        bounds = _ranges(total, workers)
        assert sum(stop - start for start, stop in bounds) == total
        assert all(start < stop for start, stop in bounds)
        for (_, left), (right, _) in zip(bounds, bounds[1:]):
            assert left == right
        if total:
            assert bounds[0][0] == 0 and bounds[-1][1] == total


def test_rows_pass_the_established_validator(rows_by_family):
    assert rows_by_family
    for family, rows in rows_by_family.items():
        assert rows, family
        for row in rows:
            validate_packed_task(row, seq_length=SEQ_LENGTH)


def test_only_the_family_whitelist_blocks_the_established_reader(rows_by_family):
    """Pin down exactly why C needs its own dataset.

    ``packed_task_to_runtime_item`` re-validates a packed row at load time and
    then rejects it on one thing only -- ``family not in TASK_FAMILIES``.  If
    that is the sole complaint, C's dataset can be a thin wrapper over the
    established reader rather than a fork of it, and any *other* error here
    would mean the pool is genuinely malformed.  Renaming a row's family to an
    accepted one and watching it load proves which of the two it is.
    """
    from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.training.runtime_dataset import (  # noqa: E501
        packed_task_to_runtime_item,
    )

    def loader(path):  # pragma: no cover - audio is not loaded here
        raise AssertionError("audio must not be read when load_audio is False")

    checked = 0
    for family, rows in rows_by_family.items():
        if family in REPLAY_FAMILIES:
            # These two *are* the base experiment's families, built by its own
            # builder, so the established reader accepts them unrenamed.  That
            # is the point of reusing them: replay needs no fork.
            for row in rows:
                item = packed_task_to_runtime_item(
                    row,
                    seq_length=SEQ_LENGTH,
                    load_audio=False,
                    audio_loader=loader,
                )
                assert item["family"] == family
                assert not item["acoustic_rows"]
                checked += 1
            continue
        for row in rows:
            with pytest.raises(ValueError, match="unknown family"):
                packed_task_to_runtime_item(
                    row,
                    seq_length=SEQ_LENGTH,
                    load_audio=False,
                    audio_loader=loader,
                )
            accepted = dict(row)
            accepted["family"] = (
                "streaming_asr_event"
                if family == FAMILY_P2ST_ASR
                else "incremental_mt_event"
            )
            item = packed_task_to_runtime_item(
                accepted,
                seq_length=SEQ_LENGTH,
                load_audio=False,
                audio_loader=loader,
            )
            assert int(item["tokens"].shape[-1]) == SEQ_LENGTH
            checked += 1
    assert checked > 0


def test_every_acoustic_row_carries_a_consistent_audio_cut(rows_by_family):
    seen = 0
    for row in rows_by_family[FAMILY_P2ST_ASR]:
        for acoustic in row["acoustic_rows"]:
            seen += 1
            cut = int(acoustic["source_pcm_end"])
            length = int(acoustic["source_glm_length"])
            assert cut > 0
            assert length == causal_glm_token_count(cut)
            assert len(acoustic["source_glm"]) == length
            assert len(acoustic["packed_positions"]) == length
            assert acoustic["source_indices"] == list(range(length))
    assert seen > 0


def test_text_families_emit_no_acoustic_rows(rows_by_family):
    for family, rows in rows_by_family.items():
        if family == FAMILY_P2ST_ASR:
            continue
        for row in rows:
            assert row["acoustic_rows"] == []


def test_no_row_carries_a_teacher_binding(rows_by_family):
    """Pure CE: a binding here would mean a teacher cache lookup at train time."""
    for rows in rows_by_family.values():
        for row in rows:
            assert row["teacher_bindings"] == []
            assert row.get("commit_consistency", []) == []


def test_packed_rows_hold_one_family_each(rows_by_family):
    for family, rows in rows_by_family.items():
        for row in rows:
            assert row["family"] == family


def test_merge_writes_an_offset_for_every_line(tmp_path):
    parts = []
    expected = []
    for part_index in range(3):
        path = tmp_path / f"part{part_index}.jsonl"
        lines = [
            json.dumps({"part": part_index, "row": row, "pad": "x" * (row * 97)})
            for row in range(4)
        ]
        path.write_text("\n".join(lines) + "\n")
        expected.extend(lines)
        parts.append(path)
    output = tmp_path / "merged.jsonl"
    summary = _merge_family(parts, output)
    assert summary["rows"] == len(expected)
    offsets = load_index(output)
    assert offsets is not None and len(offsets) == len(expected)
    raw = output.read_bytes()
    for offset, line in zip(offsets, expected):
        end = raw.index(b"\n", offset)
        assert raw[offset:end].decode("utf-8") == line


def test_merge_rejects_a_part_without_a_trailing_newline(tmp_path):
    """A truncated part would silently drop or fuse rows."""
    part = tmp_path / "part0.jsonl"
    part.write_text('{"a": 1}\n{"b": 2}')
    with pytest.raises(ValueError, match="does not end on a newline"):
        _merge_family([part], tmp_path / "merged.jsonl")
