from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v1.stage_a_causal_whisper_asr.evaluate_checkpoint import (
    collapse_ctc,
    content_ids,
    edit_distance,
    generated_runs,
)
from training import constants_uniss as c


def test_generated_runs_preserve_disjoint_write_segments() -> None:
    assert generated_runs([False, True, True, False, True, False]) == [(1, 3), (4, 5)]


def test_ctc_collapse_removes_repeats_and_blank() -> None:
    assert collapse_ctc([256, 65, 65, 256, 65, 66, 66]) == [65, 65, 66]


def test_content_ids_collect_multiple_write_events() -> None:
    values = [
        c.TOKEN_WRITE_GENERATE,
        c.TOKEN_ENG,
        c.TOKEN_START_CONTENT,
        11,
        c.TOKEN_END_CONTENT,
        c.TOKEN_START_CONTENT,
        12,
        13,
        c.TOKEN_END_CONTENT,
    ]
    assert content_ids(values) == [11, 12, 13]


def test_edit_distance_handles_insert_delete_replace() -> None:
    assert edit_distance([1, 2, 3], [1, 4, 3]) == 1
    assert edit_distance([1, 2, 3], [1, 3]) == 1
    assert edit_distance([], [1, 2]) == 2


def test_causal_full_target_has_content_start_in_prompt() -> None:
    conceptual = [c.TOKEN_START_CONTENT, 11, 12, c.TOKEN_END_CONTENT, c.TOKEN_EOS]
    start, end = generated_runs([False, True, True, True, True])[0]
    assert c.TOKEN_START_CONTENT in conceptual[:start]
    assert c.TOKEN_END_CONTENT in conceptual[start:end]
