"""Preference pairs: the second quintile, and margins that keep signal in."""
from experiments.uniss_streaming_p2st_traj_v1.data.silver_pairs import pair_for_group


def _rows(spec):
    """spec: list of (silence, bleu); chrf is set high so the gate passes."""
    return [{"arm": f"k{i:02d}", "silence_ratio": s, "bleu": b, "chrf": 90.0,
             "text": f"t{i}", "direction": "en2zh", "sample_id": "s", "reference": "r"}
            for i, (s, b) in enumerate(spec)]


def test_the_quietest_is_never_chosen():
    rows = _rows([(0.05, 30.0), (0.10, 31.0), (0.20, 32.0), (0.30, 33.0),
                  (0.40, 34.0), (0.50, 35.0), (0.60, 36.0), (0.70, 37.0)])
    chosen, rejected, _ = pair_for_group(
        rows, bleu_margin=5.0, silence_margin=0.15, min_chrf=0.0)
    assert chosen is not None
    assert chosen["silence_ratio"] > 0.05, "the extreme is what the rule avoids"


def test_the_rejected_is_noisier_than_the_chosen():
    rows = _rows([(0.05, 30.0), (0.10, 31.0), (0.20, 32.0), (0.30, 33.0),
                  (0.40, 34.0), (0.50, 35.0), (0.60, 36.0), (0.70, 37.0)])
    chosen, rejected, _ = pair_for_group(
        rows, bleu_margin=5.0, silence_margin=0.15, min_chrf=0.0)
    assert rejected["silence_ratio"] > chosen["silence_ratio"]


def test_a_group_with_no_silence_spread_yields_no_pair():
    rows = _rows([(0.20, 30.0)] * 8)
    chosen, rejected, why = pair_for_group(
        rows, bleu_margin=5.0, silence_margin=0.15, min_chrf=0.0)
    assert rejected is None and "spread" in why


def test_the_bleu_margin_keeps_the_pair_quality_matched():
    """The pair must differ in flow and agree in quality.

    The candidate at BLEU 1.0 is noisier than the chosen one, so a
    silence-only rule would reject it -- but then the pair differs in quality
    too and DPO would learn to translate better rather than to speak
    continuously.
    """
    rows = _rows([(0.20, 40.0), (0.25, 39.0), (0.30, 38.0), (0.35, 37.0),
                  (0.40, 36.0), (0.50, 35.0), (0.60, 34.0), (0.90, 1.0)])
    chosen, rejected, _ = pair_for_group(
        rows, bleu_margin=5.0, silence_margin=0.15, min_chrf=0.0)
    assert rejected is not None
    assert rejected["bleu"] != 1.0, "the outlier is beyond the BLEU margin"
    assert abs(chosen["bleu"] - rejected["bleu"]) <= 5.0


def test_the_chrf_gate_drops_candidates_that_are_not_paraphrases():
    rows = _rows([(0.10, 30.0), (0.20, 31.0), (0.30, 32.0), (0.40, 33.0)])
    for r in rows[:3]:
        r["chrf"] = 10.0
    chosen, rejected, why = pair_for_group(
        rows, bleu_margin=5.0, silence_margin=0.15, min_chrf=50.0)
    assert chosen is None or chosen["chrf"] >= 50.0
