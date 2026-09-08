"""Which families semantic roll-in applies to.

The default must stay the historical single family, because every published run
was trained with it; opting a p2st family in is what this adds.
"""
import pytest

from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.training import (
    pretrain_e2e_megatron as base,
)
from experiments.uniss_streaming_p2st_pure_ce_v1.training import (
    pretrain_p2st_megatron as p2st,
)

ENV = base.SEMANTIC_ROLLIN_FAMILIES_ENV


@pytest.fixture
def p2st_registered():
    """``install_p2st_overrides`` raises if run twice, so guard on its effect."""
    if "p2st_streaming_tts" not in base.TASK_FAMILIES:
        p2st.install_p2st_overrides()
    return True


def test_default_is_the_historical_family_only(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    assert base.semantic_rollin_families() == frozenset({base.FAMILY_INTERLEAVED})


def test_blank_value_is_also_the_default(monkeypatch):
    monkeypatch.setenv(ENV, "   ")
    assert base.semantic_rollin_families() == frozenset({base.FAMILY_INTERLEAVED})


def test_named_families_replace_the_default(monkeypatch, p2st_registered):
    monkeypatch.setenv(ENV, "p2st_streaming_tts")
    assert base.semantic_rollin_families() == frozenset({"p2st_streaming_tts"})
    # naming a family opts out of interleaved unless it is listed too
    assert base.FAMILY_INTERLEAVED not in base.semantic_rollin_families()


def test_several_families_and_surrounding_space(monkeypatch, p2st_registered):
    monkeypatch.setenv(ENV, " p2st_streaming_tts , interleaved_e2e_s2st ")
    assert base.semantic_rollin_families() == frozenset(
        {"p2st_streaming_tts", base.FAMILY_INTERLEAVED}
    )


def test_an_unknown_family_is_rejected_rather_than_ignored(monkeypatch):
    monkeypatch.setenv(ENV, "p2st_streaming_typo")
    with pytest.raises(ValueError, match="unknown semantic roll-in families"):
        base.semantic_rollin_families()


def test_rollin_is_disabled_for_a_family_outside_the_set(monkeypatch):
    """The guard inside the roll-in function, not just the call site."""
    import torch

    monkeypatch.delenv(ENV, raising=False)
    ids = torch.zeros((1, 4), dtype=torch.long)
    result = base.apply_symmetric_model_generated_semantic_rollin(
        ids, ids.clone(), ids.clone(),
        sample_boundaries=[[(0, 4)]],
        family="p2st_streaming_tts",     # not in the default set
        training=True, rate=1.0, ramp_updates=0, continue_ratio=0.5, update=10,
    )
    assert result.selected_tokens == 0
    assert result.eligible_tokens == 0
