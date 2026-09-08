"""The hard per-fragment budget: off by default, exact when on."""
import math

from experiments.uniss_streaming_p2st_pure_ce_v1.runtime.p2st_cascade import (
    GOLD_CODES_PER_CHAR,
    budget_from_text,
    floor_bias,
)


def test_budget_is_characters_times_gold_density():
    # 10 Chinese characters at gold's median 11.80 codes each
    assert budget_from_text("一二三四五六七八九十", "cmn", 1.0) == 118


def test_budget_scales_and_ignores_surrounding_space():
    assert budget_from_text("  hello  ", "eng", 1.0) == round(5 * 3.76)
    assert budget_from_text("hello", "eng", 0.5) == round(5 * 3.76 * 0.5)


def test_budget_is_off_at_scale_zero_and_for_empty_text():
    assert budget_from_text("hello", "eng", 0.0) == 0
    assert budget_from_text("   ", "cmn", 1.0) == 0


def test_unknown_target_language_disables_the_budget():
    assert budget_from_text("hello", "deu", 1.0) == 0


def test_the_two_directions_differ_by_the_measured_factor():
    # en2zh spends about 3.1x the codes per character that zh2en does, so a
    # single pooled density would mis-size one direction badly.
    assert GOLD_CODES_PER_CHAR["cmn"] / GOLD_CODES_PER_CHAR["eng"] > 3.0


def test_floor_bias_forbids_the_terminator_until_the_floor():
    bias = floor_bias(16)
    assert bias(0) == -math.inf
    assert bias(15) == -math.inf
    assert bias(16) == 0.0
    assert bias(40) == 0.0
