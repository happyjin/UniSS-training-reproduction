from __future__ import annotations

import unittest

import torch
from torch import nn

from experiments.uniss_phase3_true_subsecond_deadline_full198_v1.model.megatron_lora import (
    inject_native_megatron_lora,
)


class ToyAttention(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.linear_qkv = nn.Linear(hidden, 3 * hidden, bias=False)
        self.linear_proj = nn.Linear(hidden, hidden, bias=False)


class ToyMLP(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.linear_fc1 = nn.Linear(hidden, 2 * hidden, bias=False)
        self.linear_fc2 = nn.Linear(2 * hidden, hidden, bias=False)


class ToyLayer(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.self_attention = ToyAttention(hidden)
        self.mlp = ToyMLP(hidden)


class ToyMegatron(nn.Module):
    def __init__(self, hidden: int = 8, layers: int = 4) -> None:
        super().__init__()
        self.decoder = nn.Module()
        self.decoder.layers = nn.ModuleList(ToyLayer(hidden) for _ in range(layers))


class MegatronLoRATest(unittest.TestCase):
    def test_zero_initialized_hooks_preserve_base_and_target_expected_layers(self) -> None:
        torch.manual_seed(4)
        model = ToyMegatron()
        value = torch.randn(3, 2, 8)
        before = model.decoder.layers[0].self_attention.linear_qkv(value)
        summary = inject_native_megatron_lora(
            model, rank=2, alpha=4, dropout=0.0, mlp_last_layers=2
        )
        after = model.decoder.layers[0].self_attention.linear_qkv(value)
        torch.testing.assert_close(before, after)
        self.assertEqual(summary.attention_modules, 8)
        self.assertEqual(summary.mlp_modules, 4)
        self.assertGreater(summary.trainable_parameters, 0)
        base_trainable = [
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and "true_subsecond_lora" not in name
        ]
        self.assertEqual(base_trainable, [])

    def test_hook_adds_gradient_to_lora_b(self) -> None:
        model = ToyMegatron(layers=1)
        inject_native_megatron_lora(
            model, rank=2, alpha=4, dropout=0.0, mlp_last_layers=1
        )
        output = model.decoder.layers[0].self_attention.linear_qkv(
            torch.randn(2, 8)
        )
        output.square().mean().backward()
        gradients = [
            parameter.grad
            for name, parameter in model.named_parameters()
            if "lora_b" in name
        ]
        self.assertTrue(
            any(value is not None and bool(value.abs().sum()) for value in gradients)
        )


if __name__ == "__main__":
    unittest.main()
