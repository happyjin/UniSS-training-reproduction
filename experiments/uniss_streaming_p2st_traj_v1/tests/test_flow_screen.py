"""The paired difference and its interval."""

from __future__ import annotations

from experiments.uniss_streaming_p2st_traj_v1.evaluation.flow_screen import (
    onset_ms,
    paired,
)


def test_the_difference_is_taken_per_sample_before_averaging():
    """Pairing is the whole point: it cancels per-sample variation.

    These two arms differ by exactly -1 on every sample while the samples
    themselves range over 100, so an unpaired comparison would be swamped.
    """
    a = {f"s{i}": float(i) for i in range(100)}
    b = {f"s{i}": float(i) + 1.0 for i in range(100)}
    ids = sorted(a)
    mean, half = paired(a, b, ids)
    assert mean == -1.0
    assert half == 0.0, "no spread in the differences, so no interval"


def test_the_interval_widens_as_the_differences_scatter():
    ids = [f"s{i}" for i in range(100)]
    tight = {i: 0.0 for i in ids}
    loose = {i: (1.0 if n % 2 else -1.0) for n, i in enumerate(ids)}
    _, narrow = paired(tight, {i: 0.5 for i in ids}, ids)
    _, wide = paired(loose, {i: 0.0 for i in ids}, ids)
    assert narrow == 0.0 and wide > 0.1


def test_a_single_sample_has_no_interval():
    mean, half = paired({"s": 2.0}, {"s": 1.0}, ["s"])
    assert mean == 1.0 and half == 0.0


def test_onset_reads_the_first_delay_and_survives_an_empty_one():
    assert onset_ms({"delays": [320.0, 900.0]}) == 320.0
    assert onset_ms({"starts_ms": [640.0]}) == 640.0
    assert onset_ms({"delays": []}) == 0.0
    assert onset_ms({}) == 0.0


def test_an_arm_that_does_not_cover_the_baseline_is_dropped_not_intersected(tmp_path):
    """A rollout root accumulates arms; one short arm must not erase the rest.

    Intersecting every arm reduced a real comparison to zero samples, because
    two arms left over from older experiments held 65 samples and none.
    """
    import json
    import subprocess
    import sys

    import numpy as np
    import soundfile as sf

    def write(name, ids):
        d = tmp_path / name
        d.mkdir(parents=True, exist_ok=True)
        rows = []
        for i in ids:
            wav = d / f"{i}.wav"
            sf.write(str(wav), np.zeros(1600, dtype="float32"), 16_000)
            rows.append({
                "sample_id": i, "translation_placed": str(wav),
                "fragments": 3, "target_hypothesis": "abc", "delays": [100.0],
            })
        (d / "MANIFEST.json").write_text(json.dumps({"samples": rows}))

    write("base", [f"s{i}" for i in range(10)])
    write("good", [f"s{i}" for i in range(10)])
    write("stub", ["s0"])

    out = subprocess.run(
        [sys.executable, "-m",
         "experiments.uniss_streaming_p2st_traj_v1.evaluation.flow_screen",
         "--rollout-root", str(tmp_path), "--baseline", "base", "--workers", "2"],
        capture_output=True, text=True,
    )
    assert "skipped 1 arm" in out.stdout, out.stdout + out.stderr
    assert "10 paired samples" in out.stdout, out.stdout + out.stderr
