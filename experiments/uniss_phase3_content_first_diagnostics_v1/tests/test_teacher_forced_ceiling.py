"""CPU unit tests for the 0-C code-source ablation."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch

from experiments.uniss_phase3_content_first_diagnostics_v1.diagnostics import (
    teacher_forced_ceiling as tfc,
)


class _Objective:
    """Minimal stand-in exposing only what the substitution touches."""

    def __init__(self) -> None:
        self.calls = 0
        # ``model_loader._nearest_codes`` reads ``objective.codebook.weight``,
        # which is the shortfall fallback the substitution must delegate to.
        self.codebook = SimpleNamespace(weight=torch.eye(4))

    def causal(self, hidden: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return torch.full((hidden.shape[0],), -1, dtype=torch.long)


def _fresh() -> tuple[_Objective, object]:
    """An objective whose ``_nearest_codes`` is an instance attribute.

    ``attach_cascade_compatibility`` installs it exactly this way, so the
    restore path has to put the instance attribute back byte-for-byte.
    """

    objective = _Objective()
    objective._nearest_codes = objective.causal
    return objective, objective._nearest_codes


def test_gold_codes_are_served_in_order_across_blocks() -> None:
    objective, original = _fresh()
    gold = [10, 11, 12, 13, 14, 15]
    with tfc.gold_code_source(objective, gold) as server:
        first = objective._nearest_codes(torch.zeros(2, 4))
        second = objective._nearest_codes(torch.zeros(4, 4))
        assert first.tolist() == [10, 11]
        assert second.tolist() == [12, 13, 14, 15]
        stats = server.stats()
        assert stats["blocks"] == 2
        assert stats["consumed"] == 6
        assert stats["exhausted"] == 0
    assert objective._nearest_codes is original
    assert objective.calls == 0


def test_the_second_consumer_of_a_block_does_not_advance_the_cursor() -> None:
    """The residual adapter re-quantizes the same block the embedding used."""

    objective, _ = _fresh()
    block = torch.zeros(2, 4)
    with tfc.gold_code_source(objective, [1, 2, 3, 4]) as server:
        embedding_codes = objective._nearest_codes(block)
        residual_codes = tfc.content_first_loader._nearest_codes(objective, block)
        next_block = objective._nearest_codes(torch.ones(2, 4))
        assert embedding_codes.tolist() == [1, 2]
        assert residual_codes.tolist() == [1, 2]
        assert next_block.tolist() == [3, 4]
        stats = server.stats()
        assert stats["blocks"] == 2
        assert stats["memoized_repeats"] == 1


def test_original_callables_are_restored_after_an_exception() -> None:
    objective, original = _fresh()
    module_original = tfc.content_first_loader._nearest_codes
    try:
        with tfc.gold_code_source(objective, [1, 2]):
            raise RuntimeError("session failed")
    except RuntimeError:
        pass
    assert objective._nearest_codes is original
    assert tfc.content_first_loader._nearest_codes is module_original


def test_restore_removes_an_attribute_that_did_not_exist_before() -> None:
    objective = _Objective()
    assert "_nearest_codes" not in vars(objective)
    with tfc.gold_code_source(objective, [1]):
        assert "_nearest_codes" in vars(objective)
    assert "_nearest_codes" not in vars(objective)


def test_exhaustion_falls_back_to_the_causal_codes_and_is_reported() -> None:
    """A short gold stream must never be padded with a fabricated code."""

    objective, _ = _fresh()
    with tfc.gold_code_source(objective, [7]) as server:
        served = objective._nearest_codes(torch.zeros(3, 4))
        assert served.tolist()[0] == 7
        assert len(served) == 3
        assert server.stats()["exhausted"] == 2


def test_served_tensor_keeps_the_requested_length_and_dtype() -> None:
    objective, _ = _fresh()
    with tfc.gold_code_source(objective, list(range(100))):
        served = objective._nearest_codes(torch.zeros(5, 3))
    assert served.shape == (5,)
    assert served.dtype == torch.long


def test_component_row_carries_the_teacher_references() -> None:
    component = {
        "sample_id": "s0",
        "src_lang": "cmn",
        "tgt_lang": "eng",
        "source_audio": "/tmp/s0.wav",
        "transcription": "你好",
        "translation": "hello",
    }
    row = tfc.component_row(component, [1, 2], [3, 4], "causal")
    assert row["id"] == "s0_causal"
    assert row["teacher_transcription"] == "你好"
    assert row["teacher_translation"] == "hello"
    assert row["bicodec_global"] == [3, 4]
    assert row["_stage_a_fixed_speaker_global"] == [1, 2]


def test_component_directions_reads_the_first_occurrence(tmp_path: Path) -> None:
    episodes = tmp_path / "episodes.jsonl"
    with episodes.open("w", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "episode_id": "e0",
                    "src_lang": "cmn",
                    "tgt_lang": "eng",
                    "components": [
                        {
                            "sample_id": "s0",
                            "transcription": "甲",
                            "translation": "A",
                            "speaker_global": [1],
                        }
                    ],
                }
            )
            + "\n"
        )
        handle.write(
            json.dumps(
                {
                    "episode_id": "e1",
                    "src_lang": "eng",
                    "tgt_lang": "cmn",
                    "components": [
                        {
                            "sample_id": "s0",
                            "transcription": "later",
                            "translation": "later",
                            "speaker_global": [9],
                        }
                    ],
                }
            )
            + "\n"
        )
    directions = tfc._component_directions(episodes)
    assert directions["s0"]["src_lang"] == "cmn"
    assert directions["s0"]["translation"] == "A"
    assert directions["s0"]["speaker_global"] == [1]


def test_arms_are_exactly_the_two_code_sources() -> None:
    assert tfc.ARMS == ("causal", "gold_offline")
