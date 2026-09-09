"""The three stages compose, and in the right order."""
import numpy as np

from experiments.uniss_streaming_p2st_traj_v1.evaluation.deploy_render import deploy
from experiments.uniss_streaming_p2st_traj_v1.evaluation.playout_buffer import SAMPLE_RATE


def _packed(durations):
    """Full-amplitude speech for every fragment, so edges are unambiguous."""
    n = int(round(sum(durations) * SAMPLE_RATE / 1000))
    return np.ones(n, dtype=np.float32) * 0.5


def test_no_buffer_no_tone_leaves_a_gap_at_exact_zero():
    du = [200.0, 200.0]; dl = [0.0, 1000.0]
    audio, _ = deploy(_packed(du), dl, du, buffer_ms=0.0, tone=False)
    mid = int(0.5 * SAMPLE_RATE)
    assert abs(float(audio[mid])) == 0.0, "an unfilled gap is digital silence"


def test_room_tone_removes_the_exact_zero():
    du = [200.0, 200.0]; dl = [0.0, 1000.0]
    audio, stats = deploy(_packed(du), dl, du, buffer_ms=0.0, tone=True)
    mid = int(0.5 * SAMPLE_RATE)
    assert float(audio[mid]) != 0.0, "the gap should carry room tone"
    assert stats["fill_level"] > 0.0


def test_the_buffer_closes_the_gap_and_then_nothing_is_tapered():
    """A gap the buffer removes must not be tapered -- the phone is continuous."""
    du = [200.0, 200.0]; dl = [0.0, 1000.0]
    _, wide = deploy(_packed(du), dl, du, buffer_ms=1000.0, tone=False)
    assert wide["edges_tapered"] == 0.0
    _, none = deploy(_packed(du), dl, du, buffer_ms=0.0, tone=False)
    assert none["edges_tapered"] == 2.0


def test_the_buffer_only_delays_the_onset():
    du = [200.0]; dl = [500.0]
    _, a = deploy(_packed(du), dl, du, buffer_ms=0.0, tone=False)
    _, b = deploy(_packed(du), dl, du, buffer_ms=750.0, tone=False)
    assert a["onset_ms"] == 500.0
    assert b["onset_ms"] == 1250.0


def test_speech_energy_is_preserved_apart_from_the_tapers():
    du = [400.0, 400.0]; dl = [0.0, 400.0]      # contiguous: no taper, no gap
    packed = _packed(du)
    audio, stats = deploy(packed, dl, du, buffer_ms=0.0, tone=False)
    assert stats["edges_tapered"] == 0.0
    assert abs(float((audio ** 2).sum() - (packed ** 2).sum())) < 1e-3
