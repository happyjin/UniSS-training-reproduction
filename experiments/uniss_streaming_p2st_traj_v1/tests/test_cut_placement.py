"""Synthetic signals with a known answer, because the metric has no ground truth.

The 74% mid-voice figure was measured once by hand.  Turning it into a gate
means the classifier has to be pinned against cases where the right answer is
obvious by construction: a cut through a steady tone is mid-voice, a cut at
the edge of a tone is an edge, and a cut through silence is silence.
"""

from __future__ import annotations

import numpy as np
import pytest

from experiments.uniss_streaming_p2st_traj_v1.evaluation.cut_placement import (
    SAMPLE_RATE,
    classify_boundaries,
    summarise,
)


def _tone(ms: int, amplitude: float = 0.3) -> np.ndarray:
    n = SAMPLE_RATE * ms // 1000
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    return (amplitude * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)


def _silence(ms: int) -> np.ndarray:
    return np.zeros(SAMPLE_RATE * ms // 1000, dtype=np.float32)


def test_cut_through_steady_speech_is_mid_voice():
    audio = np.concatenate([_tone(500), _tone(500)])
    records = classify_boundaries(audio, [500.0, 500.0])
    assert [r["kind"] for r in records] == ["mid_voice"]
    assert summarise(records)["mid_voice_fraction"] == 1.0


def test_cut_at_the_end_of_speech_is_an_edge():
    audio = np.concatenate([_tone(500), _silence(500)])
    records = classify_boundaries(audio, [500.0, 500.0])
    assert [r["kind"] for r in records] == ["offset"]
    assert summarise(records)["mid_voice_fraction"] == 0.0
    assert summarise(records)["edge_fraction"] == 1.0


def test_cut_before_speech_starts_is_an_edge():
    audio = np.concatenate([_silence(500), _tone(500)])
    records = classify_boundaries(audio, [500.0, 500.0])
    assert [r["kind"] for r in records] == ["onset"]


def test_cut_inside_silence_is_silent():
    audio = np.concatenate([_tone(300), _silence(400), _silence(300)])
    records = classify_boundaries(audio, [300.0, 400.0, 300.0])
    assert records[1]["kind"] == "silent"


def test_a_loudness_step_is_not_called_mid_voice():
    """Both sides voiced but very different -- a seam, not a truncation."""
    audio = np.concatenate([_tone(500, 0.02), _tone(500, 0.4)])
    records = classify_boundaries(audio, [500.0, 500.0])
    assert records[0]["kind"] == "step"
    assert summarise(records)["mid_voice_fraction"] == 0.0


def test_only_interior_boundaries_are_counted():
    audio = np.concatenate([_tone(300), _tone(300), _tone(300)])
    records = classify_boundaries(audio, [300.0, 300.0, 300.0])
    assert len(records) == 2, "the final fragment's end is not a boundary"


def test_a_single_fragment_has_no_boundary():
    assert classify_boundaries(_tone(500), [500.0]) == []


def test_empty_input_is_handled():
    assert classify_boundaries(np.zeros(0, dtype=np.float32), [100.0, 100.0]) == []
    assert summarise([])["mid_voice_fraction"] is None


def test_summary_counts_partition_the_boundaries():
    audio = np.concatenate([_tone(300), _tone(300), _silence(300), _tone(300)])
    records = classify_boundaries(audio, [300.0, 300.0, 300.0, 300.0])
    summary = summarise(records)
    assert sum(summary["counts"].values()) == summary["boundaries"] == len(records)


def test_quiet_recording_is_not_called_silent_throughout():
    """The floor is relative, so a quiet file still has voiced frames."""
    audio = np.concatenate([_tone(500, 0.01), _tone(500, 0.01)])
    records = classify_boundaries(audio, [500.0, 500.0])
    assert records[0]["kind"] == "mid_voice"
