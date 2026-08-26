import torch

from experiments.uniss_phasea_stateful_longepisode_rl_v1.training.pretrain_megatron import (
    METRIC_NAMES,
)


def test_metric_contract_is_unique_and_includes_quality_anchors():
    assert len(METRIC_NAMES) == len(set(METRIC_NAMES))
    assert "loss/policy" in METRIC_NAMES
    assert "loss/phase3_replay" in METRIC_NAMES
    assert "loss/reference_kl" in METRIC_NAMES
    assert torch.isfinite(torch.tensor(0.0))

