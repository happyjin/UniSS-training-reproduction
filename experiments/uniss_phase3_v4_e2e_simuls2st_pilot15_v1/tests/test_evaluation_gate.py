from __future__ import annotations

from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.evaluation.gate import (
    generated_runs,
    incremental_text_metrics,
    lcs_length,
    text_units,
)


def test_text_units_use_characters_for_chinese_and_words_for_english() -> None:
    assert text_units("你 好", "cmn") == ["你", "好"]
    assert text_units("Hello  WORLD", "eng") == ["hello", "world"]


def test_incremental_metrics_detect_rollback_and_final_coverage() -> None:
    value = incremental_text_metrics(
        ["one two", "one three", "one three four"],
        "one three four",
        "eng",
    )
    assert value["rollback_events"] == 1
    assert value["coverage"] == 1.0


def test_lcs_and_generated_runs() -> None:
    assert lcs_length([1, 2, 3], [1, 9, 3]) == 2
    assert generated_runs([0, 1, 1, 0, 2, 0]) == [(1, 3), (4, 5)]

