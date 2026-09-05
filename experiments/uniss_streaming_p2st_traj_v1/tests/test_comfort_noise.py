"""Comfort noise must be inaudible as a change to what was said.

The whole claim for this change is that its downside is zero, so that is what
is tested: voiced samples are untouched, the fill sits at the level asked for,
and a source with no room tone gets no fill rather than invented hiss.
"""

from __future__ import annotations

import numpy as np

from experiments.uniss_streaming_p2st_traj_v1.evaluation.comfort_noise import (
    FADE,
    MAX_LEVEL,
    MIN_LEVEL,
    SAMPLE_RATE,
    _shaped_noise,
    fill,
    noise_floor,
)


def _tone(ms: int, amplitude: float = 0.3) -> np.ndarray:
    n = SAMPLE_RATE * ms // 1000
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    return (amplitude * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)


def _silence(ms: int) -> np.ndarray:
    return np.zeros(SAMPLE_RATE * ms // 1000, dtype=np.float32)


def test_voiced_samples_are_never_touched():
    speech = _tone(400)
    audio = np.concatenate([speech, _silence(600), speech])
    filled, _ = fill(audio, 1.0e-3)
    assert np.array_equal(filled[: len(speech)], speech)
    assert np.array_equal(filled[-len(speech):], speech)


def test_the_gap_stops_being_digital_zero():
    audio = np.concatenate([_tone(300), _silence(700), _tone(300)])
    before = audio[len(_tone(300)) + FADE : len(_tone(300)) + 5000]
    filled, stats = fill(audio, 1.0e-3)
    after = filled[len(_tone(300)) + FADE : len(_tone(300)) + 5000]
    assert np.all(before == 0.0)
    assert not np.all(after == 0.0)
    assert stats["runs"] >= 1


def test_the_fill_lands_near_the_requested_level():
    audio = np.concatenate([_tone(200), _silence(1000), _tone(200)])
    level = 8.0e-4
    filled, _ = fill(audio, level)
    core = filled[
        len(_tone(200)) + 2 * FADE : len(_tone(200)) + SAMPLE_RATE - 2 * FADE
    ]
    measured = float(np.sqrt(np.mean(core.astype(np.float64) ** 2)))
    assert 0.5 * level <= measured <= 1.6 * level


def test_a_clean_source_gets_no_fill():
    """A synthetic source has no room tone; inventing some would be worse."""
    audio = np.concatenate([_tone(300), _silence(700), _tone(300)])
    filled, stats = fill(audio, noise_floor(np.concatenate([_tone(300), _silence(300)])))
    assert stats["runs"] == 0
    assert np.array_equal(filled, audio)


def test_a_very_noisy_source_is_capped():
    audio = np.concatenate([_tone(300), _silence(700), _tone(300)])
    _, stats = fill(audio, 1.0)
    assert stats["level"] == MAX_LEVEL


def test_level_below_the_floor_is_a_no_op():
    audio = np.concatenate([_tone(300), _silence(700), _tone(300)])
    filled, stats = fill(audio, MIN_LEVEL / 2)
    assert stats["runs"] == 0
    assert np.array_equal(filled, audio)


def test_edges_are_faded_so_the_join_is_not_a_step():
    audio = np.concatenate([_tone(300), _silence(900), _tone(300)])
    filled, _ = fill(audio, 2.0e-3)
    start = len(_tone(300))
    head = np.abs(filled[start : start + 8])
    body = np.abs(filled[start + FADE : start + FADE + 400])
    assert head.max() < body.max()


def test_noise_floor_reads_the_quiet_part():
    quiet = 5.0e-4
    rng = np.random.default_rng(0)
    room = (rng.standard_normal(SAMPLE_RATE) * quiet).astype(np.float32)
    audio = np.concatenate([_tone(300), room, _tone(300)])
    measured = noise_floor(audio)
    assert 0.3 * quiet <= measured <= 3.0 * quiet


def test_shaped_noise_is_reproducible():
    a = _shaped_noise(1000, 1e-3, seed=7)
    b = _shaped_noise(1000, 1e-3, seed=7)
    assert np.array_equal(a, b)


def test_output_length_is_unchanged():
    audio = np.concatenate([_tone(300), _silence(700), _tone(300)])
    filled, _ = fill(audio, 1.0e-3)
    assert len(filled) == len(audio)
