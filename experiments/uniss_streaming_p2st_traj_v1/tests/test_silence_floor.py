"""What the floor report collects, and from where."""

from __future__ import annotations

import json

from experiments.uniss_streaming_p2st_traj_v1.evaluation.silence_floor import collect


def _arm(root, name, ids):
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    rows = [{"sample_id": i,
             "translation_concat": str(d / f"{i}_c.wav"),
             "translation_placed": str(d / f"{i}_p.wav")} for i in ids]
    (d / "MANIFEST.json").write_text(json.dumps({"samples": rows}))


def test_both_variants_are_collected_for_each_arm(tmp_path):
    """Concatenated is the speech itself; placed adds the schedule's gaps.

    The difference between them is the part a policy can still attack, so the
    report is useless if it carries only one.
    """
    _arm(tmp_path, "a", ["s0", "s1"])
    got = collect(tmp_path, ["a"], None)
    assert {k for k, _, _ in got} == {"a concat", "a placed"}
    assert len(got) == 4


def test_the_source_recordings_come_from_the_selection(tmp_path):
    sel = tmp_path / "sel.json"
    sel.write_text(json.dumps({"samples": [
        {"sample_id": "s0", "audio_path": "/x/s0.wav"},
        {"sample_id": "s1", "audio_path": "/x/s1.wav"}]}))
    got = collect(tmp_path, [], sel)
    assert [k for k, _, _ in got] == ["source", "source"]
    assert [p for _, _, p in got] == ["/x/s0.wav", "/x/s1.wav"]


def test_a_missing_arm_is_skipped_rather_than_raising(tmp_path):
    _arm(tmp_path, "a", ["s0"])
    got = collect(tmp_path, ["a", "not_here"], None)
    assert {k for k, _, _ in got} == {"a concat", "a placed"}
