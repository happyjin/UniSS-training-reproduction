from __future__ import annotations

import torch
from torch import nn

from experiments.uniss_stagea_quality_first_joint_grpo_v1.evaluation.hf_routed_lora import (
    RoutedHFLoRA,
    split_megatron_fc1_b,
    split_megatron_qkv_b,
)


def test_qkv_and_fc1_splits_match_megatron_layout():
    rows = torch.arange(18.0).reshape(18, 1).repeat_interleave(2, dim=0)
    q, k, v = split_megatron_qkv_b(
        rows, num_attention_heads=14, num_query_groups=2, head_dim=2
    )
    assert q[:, 0].reshape(14, 2)[:, 0].tolist() == [
        0, 1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 15
    ]
    assert k[:, 0].reshape(2, 2)[:, 0].tolist() == [7, 16]
    assert v[:, 0].reshape(2, 2)[:, 0].tolist() == [8, 17]
    gate, up = split_megatron_fc1_b(torch.arange(12).reshape(6, 2))
    assert gate.tolist() == [[0, 1], [2, 3], [4, 5]]
    assert up.tolist() == [[6, 7], [8, 9], [10, 11]]


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(2, 2, bias=False)
        nn.init.zeros_(self.proj.weight)

    def forward(self, value):
        return self.proj(value)


def test_route_mask_and_disable_are_exact():
    model = Tiny()
    a = torch.eye(2)
    b = torch.eye(2)
    route = RoutedHFLoRA(model, {"proj": (a, b)}, scale=2.0)
    value = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    assert torch.equal(model(value), torch.zeros_like(value))
    route.set_route(True, torch.tensor([[True, False]]))
    assert torch.equal(model(value), torch.tensor([[[2.0, 4.0], [0.0, 0.0]]]))
    with route.route(False):
        assert torch.equal(model(value), torch.zeros_like(value))
    assert torch.equal(model(value), torch.tensor([[[2.0, 4.0], [0.0, 0.0]]]))

