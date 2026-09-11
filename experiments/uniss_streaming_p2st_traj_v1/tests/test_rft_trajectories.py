"""Rebuilding a trajectory around the winner's fragments."""
from experiments.uniss_streaming_p2st_traj_v1.data.rft_trajectories import (
    merge_source,
    rebuild,
)


def _event(index, end_ms, glm, word, text):
    return {
        "event_index": index, "source_start_ms": end_ms - 640, "source_end_ms": end_ms,
        "source_pcm_start": (end_ms - 640) * 16, "source_pcm_end": end_ms * 16,
        "source_glm_start": (end_ms - 640) // 80, "source_glm_end": end_ms // 80,
        "source_glm_delta": glm, "gold_source_word_start": word,
        "gold_source_word_end": word + 1, "gold_source_delta": text,
        "gold_source_prefix": "", "v1_source_delta": text, "v1_source_prefix": "",
        "target_text_delta": "x", "target_text_prefix": "",
        "target_semantic_start": 0, "target_semantic_end": 1,
        "target_semantic_delta": [1], "target_support_end_ms": end_ms,
        "source_final": False, "target_final": False,
        "alignment_confidence": 0.9, "noise_severity": "none",
    }


def _gold():
    return {
        "sample_id": "s1", "split": "train", "src_lang": "eng", "tgt_lang": "cmn",
        "source_audio": "/tmp/a.flac", "source_duration_ms": 1920,
        "speaker_global": list(range(32)),
        "full_transcription": "a b c", "full_translation": "甲乙丙",
        "target_semantic_sha256": "deadbeef", "phase3_teacher_sha256": "cafe",
        "events": [_event(0, 640, [1, 2], 0, "a"),
                   _event(1, 1280, [3, 4], 1, "b"),
                   _event(2, 1920, [5, 6], 2, "c")],
    }


def test_merge_source_concatenates_the_window():
    e = _gold()["events"]
    m = merge_source(e, -1.0, 1280.0)
    assert m["source_glm_delta"] == [1, 2, 3, 4]
    assert m["gold_source_delta"] == "ab"
    assert m["source_glm_start"] == 0 and m["source_glm_end"] == 16


def test_an_empty_window_returns_nothing():
    assert merge_source(_gold()["events"], 1920.0, 2560.0) == {}


def test_rebuild_uses_the_winner_text_and_codes():
    r = rebuild(_gold(), [640.0, 1280.0], ["你", "你好"], [[10, 11], [12]])
    assert r is not None
    assert [e["target_text_delta"] for e in r["events"]] == ["你", "好"]
    assert [list(e["target_semantic_delta"]) for e in r["events"]] == [[10, 11], [12]]
    assert r["full_translation"] == "你好"
    assert r["target_semantic_length"] == 3


def test_the_semantic_ranges_are_cumulative_and_contiguous():
    r = rebuild(_gold(), [640.0, 1280.0], ["你", "你好"], [[10, 11], [12]])
    spans = [(e["target_semantic_start"], e["target_semantic_end"]) for e in r["events"]]
    assert spans == [(0, 2), (2, 3)]


def test_the_last_window_absorbs_the_remaining_source():
    """The final fragment is emitted once the source is exhausted."""
    r = rebuild(_gold(), [640.0, 1280.0], ["你", "你好"], [[10], [11]])
    assert r["events"][-1]["source_end_ms"] == 1920, "event 2 must not be dropped"
    assert r["events"][-1]["source_glm_delta"] == [3, 4, 5, 6]


def test_only_the_last_event_is_marked_final():
    r = rebuild(_gold(), [640.0, 1280.0], ["你", "你好"], [[10], [11]])
    assert [e["source_final"] for e in r["events"]] == [False, True]
    assert [e["target_final"] for e in r["events"]] == [False, True]


def test_the_source_side_is_carried_through_unchanged():
    g = _gold()
    r = rebuild(g, [640.0], ["你"], [[10]])
    assert r["source_audio"] == g["source_audio"]
    assert r["speaker_global"] == g["speaker_global"]
    assert r["src_lang"] == g["src_lang"] and r["tgt_lang"] == g["tgt_lang"]


def test_the_semantic_hash_matches_the_new_codes():
    """The schema checks this hash against the codes, so it must be recomputed.

    Dropping it -- which the first version did, on the reasoning that the
    target is no longer the teacher's -- made the pool builder reject every
    record with a missing-argument error.
    """
    from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.data.schema import (
        hash_int_sequence,
    )

    r = rebuild(_gold(), [640.0, 1280.0], ["你", "你好"], [[10, 11], [12]])
    assert r["target_semantic_sha256"] == hash_int_sequence([10, 11, 12])
    assert r["target_semantic_sha256"] != _gold()["target_semantic_sha256"]


def test_the_teacher_hash_is_carried_through():
    """It identifies the teacher cache the source side came from, which these
    records still share with their gold originals; only its hex format is
    validated."""
    r = rebuild(_gold(), [640.0], ["你"], [[10]])
    assert r["phase3_teacher_sha256"] == _gold()["phase3_teacher_sha256"]


def test_rebuild_tolerates_more_fragments_than_gold_events():
    """The winner can speak in more fragments than gold had events."""
    r = rebuild(_gold(), [640.0, 1280.0, 1920.0], ["你", "你好", "你好啊"],
                [[10], [11], [12]])
    assert r is not None and len(r["events"]) == 3
    assert r["full_translation"] == "你好啊"


def test_a_fragment_in_an_empty_source_window_is_skipped_not_crashed():
    """Two fragments inside one gold event's window: the second has no source."""
    gold = _gold()
    gold["events"] = [_event(0, 640, [1, 2], 0, "a")]
    gold["source_duration_ms"] = 640
    r = rebuild(gold, [640.0, 1280.0], ["你", "你好"], [[10], [11]])
    assert r is not None
    assert len(r["events"]) == 1, "the window with no source events is dropped"
