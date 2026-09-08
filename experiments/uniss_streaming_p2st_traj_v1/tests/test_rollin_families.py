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


def test_rollin_runs_for_an_opted_in_family(monkeypatch, p2st_registered):
    """The positive direction, which the deny test above cannot catch.

    The first attempt at this change patched the family gate in
    ``corrupt_interleaved_semantic_prefixes`` instead of the roll-in function,
    and every deny-direction test still passed because the default set excludes
    the p2st families either way.  A whole training launch was spent finding
    that out, so assert that an opted-in family actually reaches the candidate
    accounting.
    """
    import torch

    import training.constants_uniss as c

    monkeypatch.setenv(ENV, "p2st_streaming_tts")
    ids = torch.zeros((1, 6), dtype=torch.long)
    # two model-proposed codes inside the BiCodec semantic span, which is what
    # the function counts as eligible
    end = torch.full((1, 6), -1, dtype=torch.long)
    end[0, 2] = c.BICODEC_SEMANTIC_OFFSET + 5
    cont = torch.full((1, 6), -1, dtype=torch.long)
    cont[0, 4] = c.BICODEC_SEMANTIC_OFFSET + 9
    result = base.apply_symmetric_model_generated_semantic_rollin(
        ids, end, cont,
        sample_boundaries=[[(0, 6)]],
        family="p2st_streaming_tts",
        training=True, rate=1.0, ramp_updates=0, continue_ratio=0.5, update=0,
    )
    assert result.eligible_tokens == 2, "the opted-in family must be counted"
    assert result.effective_rate > 0.0


def test_prefix_corruption_keeps_its_own_family_gate(monkeypatch, p2st_registered):
    """The roll-in variable must not silently widen prefix corruption too."""
    import torch

    monkeypatch.setenv(ENV, "p2st_streaming_tts")
    ids = torch.zeros((1, 6), dtype=torch.long)
    effective, corrupted, eligible, rate = base.corrupt_interleaved_semantic_prefixes(
        ids, ids.clone(),
        family="p2st_streaming_tts",
        training=True, rate=1.0, tail=4, ramp_updates=0, update=0,
    )
    assert corrupted == 0 and eligible == 0 and rate == 0.0
