"""Preference pairs: the second quintile, and margins that keep signal in."""
from experiments.uniss_streaming_p2st_traj_v1.data.silver_pairs import pair_for_group


def _rows(spec, onsets=None):
    """spec: list of (silence, bleu); chrf is set high so the gate passes.

    Onsets default to equal, which is the case the guard lets through, so the
    existing tests keep testing the margins and nothing else.
    """
    return [{"arm": f"k{i:02d}", "silence_ratio": s, "bleu": b, "chrf": 90.0,
             "text": f"t{i}", "direction": "en2zh", "sample_id": "s", "reference": "r",
             "onset_ms": 1000.0 if onsets is None else float(onsets[i])}
            for i, (s, b) in enumerate(spec)]


def test_the_quietest_is_never_chosen():
    # BLEU is held constant so the group turns only on silence.  Under the
    # paper's margin a group whose quieter candidates translate *worse* yields
    # no pair at all, which is the rule working, not a failure.
    rows = _rows([(0.05, 30.0), (0.10, 30.0), (0.20, 30.0), (0.30, 30.0),
                  (0.40, 30.0), (0.50, 30.0), (0.60, 30.0), (0.70, 30.0)])
    chosen, rejected, _ = pair_for_group(
        rows, bleu_margin=0.0, silence_margin=0.15, min_chrf=0.0)
    assert chosen is not None
    assert chosen["silence_ratio"] > 0.05, "the extreme is what the rule avoids"


def test_the_rejected_is_noisier_than_the_chosen():
    # BLEU is held constant so the group turns only on silence.  Under the
    # paper's margin a group whose quieter candidates translate *worse* yields
    # no pair at all, which is the rule working, not a failure.
    rows = _rows([(0.05, 30.0), (0.10, 30.0), (0.20, 30.0), (0.30, 30.0),
                  (0.40, 30.0), (0.50, 30.0), (0.60, 30.0), (0.70, 30.0)])
    chosen, rejected, _ = pair_for_group(
        rows, bleu_margin=0.0, silence_margin=0.15, min_chrf=0.0)
    assert rejected["silence_ratio"] > chosen["silence_ratio"]


def test_a_group_with_no_silence_spread_yields_no_pair():
    rows = _rows([(0.20, 30.0)] * 8)
    chosen, rejected, why = pair_for_group(
        rows, bleu_margin=5.0, silence_margin=0.15, min_chrf=0.0)
    assert rejected is None and "spread" in why


def test_the_match_mode_keeps_the_pair_quality_matched():
    """The pair must differ in flow and agree in quality.

    The candidate at BLEU 1.0 is noisier than the chosen one, so a
    silence-only rule would reject it -- but then the pair differs in quality
    too and DPO would learn to translate better rather than to speak
    continuously.
    """
    rows = _rows([(0.20, 40.0), (0.25, 39.0), (0.30, 38.0), (0.35, 37.0),
                  (0.40, 36.0), (0.50, 35.0), (0.60, 34.0), (0.90, 1.0)])
    chosen, rejected, _ = pair_for_group(
        rows, bleu_margin=5.0, silence_margin=0.15, min_chrf=0.0, bleu_margin_mode="match")
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


def test_a_quieter_candidate_that_only_starts_later_is_refused():
    """The metric's artefact, and the reason for the guard.

    k03 is much quieter, but only because it opens its mouth 2 seconds later,
    which removes leading gap from the placed timeline without improving flow.
    Training on that pair would teach the model to delay speaking.
    """
    rows = _rows([(0.30, 30.0), (0.45, 30.0), (0.12, 30.0)],
                 onsets=[1000, 1000, 3000])
    chosen, rejected, why = pair_for_group(
        rows, bleu_margin=0.0, silence_margin=0.15, min_chrf=0.0)
    assert chosen is not None
    assert float(chosen["onset_ms"]) <= float(rejected["onset_ms"])


def test_the_guard_blocks_a_group_where_every_pair_is_the_artefact():
    """Three candidates, so the second quintile is a genuine middle one."""
    rows = _rows([(0.10, 30.0), (0.30, 30.0), (0.50, 30.0)],
                 onsets=[1000, 4000, 1000])
    chosen, rejected, why = pair_for_group(
        rows, bleu_margin=0.0, silence_margin=0.15, min_chrf=0.0)
    assert rejected is None and "starts later" in why


def test_a_chosen_candidate_that_starts_earlier_is_kept():
    """Earlier is better on both axes, so the guard is one-sided."""
    rows = _rows([(0.10, 30.0), (0.30, 30.0), (0.50, 30.0)],
                 onsets=[1000, 500, 2000])
    chosen, rejected, why = pair_for_group(
        rows, bleu_margin=0.0, silence_margin=0.15, min_chrf=0.0)
    assert rejected is not None
    assert float(chosen["onset_ms"]) < float(rejected["onset_ms"])


def test_the_paper_margin_requires_the_chosen_to_be_better():
    """NaturalFlow: the chosen candidate must *outperform* by the margin.

    k02 is noisier and 10 BLEU worse, so it clears a separation margin of 5.
    k01 is noisier but only 1 BLEU worse, so it does not.
    """
    rows = _rows([(0.10, 40.0), (0.30, 40.0), (0.50, 39.0), (0.55, 30.0)])
    chosen, rejected, why = pair_for_group(
        rows, bleu_margin=5.0, silence_margin=0.15, min_chrf=0.0,
        bleu_margin_mode="separation")
    assert rejected is not None
    assert float(chosen["bleu"]) - float(rejected["bleu"]) >= 5.0


def test_the_two_margin_modes_select_different_rejects():
    """The direction is not a detail: it picks a different candidate."""
    rows = _rows([(0.10, 40.0), (0.30, 40.0), (0.50, 39.0), (0.55, 30.0)])
    _, sep, _ = pair_for_group(rows, bleu_margin=5.0, silence_margin=0.15,
                               min_chrf=0.0, bleu_margin_mode="separation")
    _, mat, _ = pair_for_group(rows, bleu_margin=5.0, silence_margin=0.15,
                               min_chrf=0.0, bleu_margin_mode="match")
    assert sep is not None and mat is not None
    assert sep["arm"] != mat["arm"]
