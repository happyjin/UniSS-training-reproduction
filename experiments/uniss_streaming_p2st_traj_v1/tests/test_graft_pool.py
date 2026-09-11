"""Grafting one family across pools."""
import pytest

from experiments.uniss_streaming_p2st_traj_v1.training.graft_pool import graft


def _pool(tag):
    return {"gold": f"/{tag}.jsonl", "chunk_ms": 640,
            "families": {f: {"samples": 10, "data": f"/{tag}/{f}"} for f in
                         ("p2st_streaming_asr", "p2st_incremental_mt",
                          "p2st_streaming_tts")}}


def test_only_the_named_family_comes_from_the_donor():
    m = graft(_pool("gold"), _pool("win"), ["p2st_streaming_tts"])
    assert m["families"]["p2st_streaming_tts"]["data"] == "/win/p2st_streaming_tts"
    assert m["families"]["p2st_incremental_mt"]["data"] == "/gold/p2st_incremental_mt"
    assert m["families"]["p2st_streaming_asr"]["data"] == "/gold/p2st_streaming_asr"


def test_the_base_is_not_mutated():
    base = _pool("gold")
    graft(base, _pool("win"), ["p2st_streaming_tts"])
    assert base["families"]["p2st_streaming_tts"]["data"] == "/gold/p2st_streaming_tts"


def test_the_graft_is_recorded_in_the_manifest():
    m = graft(_pool("gold"), _pool("win"), ["p2st_streaming_tts"])
    assert m["grafted_families"] == ["p2st_streaming_tts"]
    assert m["grafted_from"] == "/win.jsonl"


def test_a_missing_family_is_refused_rather_than_silently_skipped():
    with pytest.raises(SystemExit):
        graft(_pool("gold"), _pool("win"), ["nope"])
