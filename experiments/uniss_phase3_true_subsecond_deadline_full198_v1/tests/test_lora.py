from __future__ import annotations

import unittest

import torch
from torch import nn

from experiments.uniss_phase3_true_subsecond_deadline_full198_v1.model.lora import (
    LoRALinear,
    inject_phase3_qwen_lora,
)


class Attention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(8, 8, bias=False)
        self.k_proj = nn.Linear(8, 4, bias=False)
        self.v_proj = nn.Linear(8, 4, bias=False)
        self.o_proj = nn.Linear(8, 8, bias=False)


class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(8, 16, bias=False)
        self.up_proj = nn.Linear(8, 16, bias=False)
        self.down_proj = nn.Linear(16, 8, bias=False)


class Layer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = Attention()
        self.mlp = MLP()


class ToyQwen(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList((Layer(), Layer(), Layer()))


class LoRATest(unittest.TestCase):
    def test_zero_initialized_lora_preserves_linear_output(self) -> None:
        torch.manual_seed(3)
        base = nn.Linear(5, 7, bias=False)
        value = torch.randn(2, 5)
        expected = base(value).detach()
        lora = LoRALinear(base, rank=2, alpha=4, dropout=0.0).eval()
        torch.testing.assert_close(lora(value), expected)
        self.assertFalse(lora.base.weight.requires_grad)
        self.assertTrue(lora.lora_a.requires_grad)
        self.assertTrue(lora.lora_b.requires_grad)

    def test_qwen_injection_targets_all_attention_and_last_mlp(self) -> None:
        model = ToyQwen()
        summary = inject_phase3_qwen_lora(
            model, rank=2, alpha=4, dropout=0.0, mlp_last_layers=1
        )
        self.assertEqual(summary.attention_modules, 12)
        self.assertEqual(summary.mlp_modules, 3)
        self.assertIsInstance(model.model.layers[0].self_attn.q_proj, LoRALinear)
        self.assertIsInstance(model.model.layers[-1].mlp.down_proj, LoRALinear)
        self.assertIsInstance(model.model.layers[0].mlp.down_proj, nn.Linear)
        self.assertGreater(summary.trainable_parameters, 0)


if __name__ == "__main__":
    unittest.main()
