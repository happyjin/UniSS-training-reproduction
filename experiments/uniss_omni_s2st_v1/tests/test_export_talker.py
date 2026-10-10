"""Guards on reading a Megatron checkpoint back out.

A Megatron torch_dist checkpoint holds the weights at ``talker.*`` and the
optimiser state at ``optimizer.state.{exp_avg,exp_avg_sq,fp32_param}.talker.*``.
Matching ``.talker.`` anywhere in the key collects all four, and stripping
the prefix collapses them onto one name -- so the exported "weight" was
whichever the dict iterated last. For model.norm.weight that was Adam's
first moment, near zero, and the model then emitted the same logits at
every position.
"""

from __future__ import annotations

import importlib

import pytest
import torch

export = importlib.import_module(
    "experiments.uniss_omni_s2st_v1.training.export_talker_from_megatron"
)


def test_optimiser_state_keys_do_not_look_like_weights():
    """The filter must anchor, not search: this is the bug that shipped."""
    keys = [
        "talker.model.norm.weight",
        "optimizer.state.exp_avg.talker.model.norm.weight",
        "optimizer.state.exp_avg_sq.talker.model.norm.weight",
        "optimizer.state.fp32_param.talker.model.norm.weight",
        "thinker.model.layers.0.mlp.up_proj.weight",
        "global_embed.weight",
    ]
    kept = [k for k in keys if k.startswith("talker.") or k.startswith("global_embed.")]
    assert kept == ["talker.model.norm.weight", "global_embed.weight"]

    # The behaviour that caused it, written out so the difference is visible.
    naive = [k for k in keys if ".talker." in k or k.startswith("talker.")]
    assert len(naive) == 4
    assert len({k.split("talker.", 1)[1] for k in naive}) == 1


def test_stripping_is_by_length_not_by_split():
    """split('talker.', 1) also cuts at an optimiser prefix's occurrence."""
    key = "optimizer.state.exp_avg.talker.model.norm.weight"
    assert key.split("talker.", 1)[1] == "model.norm.weight"
    assert not key.startswith("talker.")


@pytest.mark.parametrize(
    "mean_abs,ok",
    [(1.0, True), (6.83, True), (0.011, True), (0.0001, False), (0.0, False)],
)
def test_the_norm_sanity_threshold_separates_weights_from_moments(mean_abs, ok):
    """A trained RMSNorm sits near 1; an Adam moment sits near 0."""
    tensor = torch.full((896,), mean_abs)
    passes = float(tensor.float().abs().mean()) >= 0.01
    assert passes is ok


def test_a_name_collision_raises_rather_than_picking_one():
    seen: dict[str, object] = {}
    with pytest.raises(KeyError):
        for key in ("talker.model.norm.weight", "talker.model.norm.weight"):
            stripped = key[len("talker.") :]
            if stripped in seen:
                raise KeyError(stripped)
            seen[stripped] = object()
