"""The prior must suppress a premature END and leave a legitimate one alone."""
from __future__ import annotations

import math

import pytest

from experiments.uniss_streaming_p2st_pure_ce_v1.runtime.length_prior import (
    LengthPrior,
    terminator_bias,
)


@pytest.fixture(scope="module")
def prior() -> LengthPrior:
    return LengthPrior.load()


def test_both_target_languages_are_fitted(prior):
    assert prior.available("cmn")
    assert prior.available("eng")


def test_a_premature_stop_is_suppressed_hard(prior):
    """Chinese runs 12-13 codes per character, so one code for ten is absurd."""
    bias = prior.log_completion(1, text_length=10, language="cmn")
    assert bias < -5.0, bias


def test_the_plausible_range_is_untouched(prior):
    """At the prior's own median the bias must not fight the model.

    Ten Chinese characters have a median of 128 codes, so by then most of the
    mass is behind and log P(N <= n) is close to zero.
    """
    bias = prior.log_completion(128, text_length=10, language="cmn")
    assert -1.2 < bias <= 0.0, bias


def test_the_bias_is_monotone_in_generated_length(prior):
    values = [
        prior.log_completion(n, text_length=8, language="cmn")
        for n in (1, 10, 30, 60, 100, 140, 200)
    ]
    assert values == sorted(values), values
    assert values[0] < values[-1]


def test_the_bias_never_forces_a_stop(prior):
    """log of a probability is at most zero, so END is only ever suppressed.

    Over-generation already has the pace budget and the repetition penalty,
    both measured to work; this mechanism exists for the other direction.
    """
    for n in (0, 1, 50, 500, 5000):
        assert prior.log_completion(n, text_length=6, language="cmn") <= 0.0


def test_longer_text_tolerates_more_codes_before_stopping(prior):
    """The whole point: the bias scales with how much text is being spoken."""
    short = prior.log_completion(40, text_length=2, language="cmn")
    long = prior.log_completion(40, text_length=16, language="cmn")
    assert long < short, (short, long)


def test_an_unfitted_language_is_a_no_op(prior):
    assert prior.log_completion(3, text_length=5, language="deu") == 0.0


def test_callback_is_none_when_it_cannot_apply(prior):
    assert terminator_bias(prior, text_length=0, language="cmn") is None
    assert terminator_bias(prior, text_length=5, language="cmn", scale=0.0) is None
    assert terminator_bias(None, text_length=5, language="cmn") is None
    assert terminator_bias(prior, text_length=5, language="deu") is None


def test_scale_multiplies_the_bias(prior):
    one = terminator_bias(prior, text_length=9, language="cmn", scale=1.0)
    half = terminator_bias(prior, text_length=9, language="cmn", scale=0.5)
    assert math.isclose(half(3), 0.5 * one(3), rel_tol=1e-9)
