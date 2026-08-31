"""CPU unit tests for the simultaneous-S2ST latency and quality metrics."""

from __future__ import annotations

import pytest

from experiments.uniss_phase3_e2e_commit_policy_v1.evaluation import (
    streaming_metrics as sm,
)


def test_percentile_endpoints_and_interpolation() -> None:
    values = [0.0, 10.0, 20.0, 30.0]
    assert sm.percentile(values, 0.0) == 0.0
    assert sm.percentile(values, 1.0) == 30.0
    assert sm.percentile(values, 0.5) == 15.0
    assert sm.percentile([], 0.5) == 0.0
    assert sm.percentile([7.0], 0.9) == 7.0


def test_summarize_reports_zero_n_for_empty() -> None:
    assert sm.summarize([]) == {"n": 0}
    value = sm.summarize([1.0, 2.0, 3.0])
    assert value["n"] == 3 and value["mean"] == 2.0 and value["max"] == 3.0


def test_emission_times_expands_text_units_and_speech_counts() -> None:
    events = [
        {"source_end_ms": 320, "mt_deltas": ["Such a"], "semantic_tokens": 0},
        {"source_end_ms": 640, "mt_deltas": [], "semantic_tokens": 3},
        {"source_end_ms": 960, "mt_deltas": ["person"], "semantic_tokens": 2},
    ]
    text, speech = sm.emission_times(events, "eng")
    assert text == [320.0, 320.0, 960.0]
    assert speech == [640.0, 640.0, 640.0, 960.0, 960.0]


def test_emission_times_uses_characters_for_chinese() -> None:
    events = [{"source_end_ms": 100, "mt_deltas": ["他是"], "semantic_tokens": 0}]
    text, _ = sm.emission_times(events, "cmn")
    assert text == [100.0, 100.0]


def test_ideal_oracle_policy_has_zero_al_and_laal() -> None:
    """A system emitting unit i exactly at its proportional source time lags 0."""

    source = 1000.0
    reference = 5
    times = [index * source / reference for index in range(reference)]
    value = sm.latency_family(
        times, source_duration_ms=source, reference_units=reference
    )
    assert value["al_ms"] == pytest.approx(0.0)
    assert value["laal_ms"] == pytest.approx(0.0)
    assert value["length_ratio"] == 1.0


def test_offline_policy_lags_by_the_whole_utterance() -> None:
    """Waiting for all input then dumping the target is the worst case."""

    source = 1000.0
    reference = 4
    value = sm.latency_family(
        [source] * reference, source_duration_ms=source, reference_units=reference
    )
    assert value["first_emission_ms"] == source
    assert value["al_ms"] == pytest.approx(source)
    assert value["average_proportion"] == pytest.approx(1.0)


def test_laal_is_stricter_than_al_under_over_generation() -> None:
    """The load-bearing distinction: AL and LAAL are not the same metric.

    ``training/simul_uniss/latency_metrics.py`` aliases them, which is only
    valid when the hypothesis is no longer than the reference.  This model
    over-generates speech by five to twelve times, and there LAAL is the
    stricter number: replacing the reference length with
    ``max(hypothesis, reference)`` shrinks the per-unit budget, so the ideal
    emission time for unit i moves earlier and the measured lag grows.
    """

    source = 1000.0
    times = [0.0, 250.0, 500.0, 750.0, 1000.0, 1000.0, 1000.0, 1000.0]
    value = sm.latency_family(times, source_duration_ms=source, reference_units=4)
    assert value["length_ratio"] == 2.0
    assert value["al_ms"] == pytest.approx(0.0)
    assert value["laal_ms"] == pytest.approx(250.0)
    assert value["laal_ms"] > value["al_ms"]


def test_laal_equals_al_when_the_hypothesis_is_not_longer() -> None:
    """``max`` picks the reference, so the two metrics coincide there."""

    source = 1000.0
    times = [0.0, 250.0, 500.0, 750.0]
    value = sm.latency_family(times, source_duration_ms=source, reference_units=8)
    assert value["length_ratio"] == 0.5
    assert value["laal_ms"] == pytest.approx(value["al_ms"])


def test_dal_never_undercuts_al_for_a_bursty_policy() -> None:
    source = 1000.0
    times = [0.0, 0.0, 0.0, 0.0]
    value = sm.latency_family(times, source_duration_ms=source, reference_units=4)
    assert value["dal_ms"] >= value["al_ms"]


def test_no_emission_is_reported_rather_than_scored() -> None:
    value = sm.latency_family([], source_duration_ms=1000.0, reference_units=4)
    assert value["emitted"] is False
    assert "al_ms" not in value


def test_zero_duration_source_is_not_scored() -> None:
    value = sm.latency_family([0.0], source_duration_ms=0.0, reference_units=4)
    assert value["emitted"] is False


def test_maximum_gap_is_the_largest_advance_between_emissions() -> None:
    value = sm.latency_family(
        [0.0, 100.0, 100.0, 5000.0], source_duration_ms=6000.0, reference_units=4
    )
    assert value["maximum_gap_ms"] == 4900.0


def test_action_counts_tallies_every_continuation() -> None:
    events = [
        {"chosen_continuations": ["WRITE_ASR", "WAIT"]},
        {"chosen_continuations": ["WRITE_MT", "WRITE_SEMANTIC", "WAIT"]},
    ]
    assert sm.action_counts(events) == {
        "WRITE_ASR": 1,
        "WAIT": 2,
        "WRITE_MT": 1,
        "WRITE_SEMANTIC": 1,
    }


def _row() -> dict[str, object]:
    return {
        "sample_id": "s0",
        "src_lang": "cmn",
        "tgt_lang": "eng",
        "source_duration_ms": 1000,
        "e_asr": {
            "metric": "CER",
            "error_rate": 0.1,
            "errors": 2,
            "reference_units": 20,
            "empty_events": 1,
            "early_eos_events": 0,
            "malformed_write_events": 0,
            "source_rollbacks": 0,
            "final_reached_eos": True,
        },
        "e_mt_gold": {
            "coverage": 0.5,
            "hypothesis_units": 4,
            "reference_units": 8,
            "commit_conflicts": 1,
            "rollback_events": 0,
            "events": 4,
            "nonempty_events": 4,
            "final_hypothesis": "a b c d",
        },
        "e_s2s_free": {
            "semantic_tokens": 400,
            "semantic_reference_tokens": 100,
            "semantic_coverage": 1.0,
            "malformed_segments": 2,
            "invalid_semantic_tokens": 0,
            "natural_eos": False,
            "target_text_before_source_eos": True,
            "target_semantic_before_source_eos": True,
            "audio": {"duration_seconds": 8.0, "non_silent": True, "rms": 0.07, "peak": 0.8},
            "events": [
                {
                    "source_end_ms": 500,
                    "mt_deltas": ["a b"],
                    "semantic_tokens": 200,
                    "chosen_continuations": ["WRITE_MT", "WRITE_SEMANTIC"],
                },
                {
                    "source_end_ms": 1000,
                    "mt_deltas": ["c d"],
                    "semantic_tokens": 200,
                    "chosen_continuations": ["WRITE_MT", "WRITE_SEMANTIC", "EOS"],
                },
            ],
        },
    }


def test_sample_metrics_exposes_unclamped_over_generation() -> None:
    value = sm.sample_metrics(_row())
    assert value["s2s"]["semantic_coverage_clamped"] == 1.0
    # The gate metric saturates and hides that this is four times too long.
    assert value["s2s"]["semantic_length_ratio"] == 4.0
    assert value["s2s"]["audio_over_source_ratio"] == 8.0
    assert value["s2s"]["natural_eos"] is False


def test_sample_metrics_scores_both_latency_streams() -> None:
    value = sm.sample_metrics(_row())
    assert value["s2s"]["latency_text"]["hypothesis_units"] == 4
    assert value["s2s"]["latency_speech"]["hypothesis_units"] == 400
    assert value["s2s"]["latency_text"]["first_emission_ms"] == 500.0
    assert value["mt_gold"]["length_ratio"] == 0.5


def test_aggregate_splits_by_direction() -> None:
    left = sm.sample_metrics(_row())
    other = _row()
    other["src_lang"], other["tgt_lang"] = "eng", "cmn"
    right = sm.sample_metrics(other)
    value = sm.aggregate([left, right])
    assert value["all"]["samples"] == 2
    assert set(value["by_direction"]) == {"cmn->eng", "eng->cmn"}
    assert value["by_direction"]["cmn->eng"]["samples"] == 1
    assert value["all"]["s2s.semantic_length_ratio"]["mean"] == 4.0


def test_aggregate_counts_booleans_as_rates() -> None:
    value = sm.aggregate([sm.sample_metrics(_row())])
    assert value["all"]["s2s.non_silent"]["mean"] == 1.0
    assert value["all"]["s2s.natural_eos"]["mean"] == 0.0
