from __future__ import annotations

import unittest

import torch
from torch import nn

from experiments.uniss_phase3_prefix_streaming_full198_v1.lora import (
    LoRALinear,
    inject_lora,
    lora_enabled,
)


class Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(4, 4, bias=False)
        self.v_proj = nn.Linear(4, 4, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.q_proj(value) + self.v_proj(value)


class LoRATest(unittest.TestCase):
    def test_disabled_path_is_exact_base_teacher(self) -> None:
        torch.manual_seed(7)
        model = Tiny()
        value = torch.randn(2, 4)
        expected = model(value).detach()
        result = inject_lora(model, rank=2, alpha=4, dropout=0.0)
        self.assertEqual(result.trainable_parameters, 32)
        modules = [module for module in model.modules() if isinstance(module, LoRALinear)]
        with torch.no_grad():
            for module in modules:
                module.lora_B.weight.normal_(std=0.1)
        with lora_enabled(model, False):
            observed = model(value)
        self.assertTrue(torch.equal(expected, observed))
        self.assertFalse(torch.equal(expected, model(value)))


if __name__ == "__main__":
    unittest.main()

