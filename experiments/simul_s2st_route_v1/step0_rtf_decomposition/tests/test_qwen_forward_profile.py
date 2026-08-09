"""Correctness tests for the two speed-ups the Step 0b profile proposes.

Both are candidate production changes, so equivalence with the current Stage08/Stage10
behaviour has to hold before the timing numbers mean anything.
"""

import sys
import unittest
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[4]
TREE = ROOT / "experiments/uniss_streamspeech_ctc_v1"
for _path in (
    ROOT,
    TREE / "stage02_ctc_probe",
    TREE / "stage03_multitask_encoder",
    TREE / "stage03_multitask_encoder/ar_s2tt_v1",
    TREE / "stage04_b2_discrete_bridge",
):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from experiments.simul_s2st_route_v1.step0_rtf_decomposition.qwen_forward_profile import (  # noqa: E402
    merge_lora,
    vectorized_repetition_penalty,
)
from experiments.uniss_streamspeech_ctc_v1.stage08_emformer_phase3_joint.step2_qwen_lora_replay_v1.lora import (  # noqa: E402
    LoRALinear,
    inject_lora,
)
from experiments.uniss_streamspeech_ctc_v1.stage10_cached_micro_write.adapter import (  # noqa: E402
    apply_repetition_penalty,
)


class TinyAttention(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.q_proj = nn.Linear(size, size)
        self.v_proj = nn.Linear(size, size)
        self.o_proj = nn.Linear(size, size)

    def forward(self, value):
        return self.o_proj(self.q_proj(value) + self.v_proj(value))


class TinyModel(nn.Module):
    def __init__(self, size=16, layers=3):
        super().__init__()
        self.layers = nn.ModuleList(TinyAttention(size) for _ in range(layers))

    def forward(self, value):
        for layer in self.layers:
            value = layer(value)
        return value


class MergeLoraTest(unittest.TestCase):
    def build(self):
        torch.manual_seed(0)
        model = TinyModel()
        injection = inject_lora(model, target_modules=("q_proj", "v_proj"), rank=4, dropout=0.0)
        # A freshly injected adapter has lora_B at zero, which would make the merge trivially
        # correct; give it real content first.
        for module in model.modules():
            if isinstance(module, LoRALinear):
                nn.init.normal_(module.lora_B.weight, std=0.1)
        model.eval()
        return model, injection

    def test_merge_preserves_outputs_and_removes_every_adapter(self):
        model, injection = self.build()
        inputs = torch.randn(2, 5, 16)
        expected = model(inputs)

        merged = merge_lora(model)

        self.assertEqual(merged, len(injection.module_names))
        self.assertEqual(merged, 6)
        self.assertFalse(any(isinstance(m, LoRALinear) for m in model.modules()))
        self.assertTrue(torch.allclose(model(inputs), expected, atol=1e-5))

    def test_merge_keeps_bias_and_shapes(self):
        model, _ = self.build()
        original_bias = model.layers[0].q_proj.base.bias.detach().clone()

        merge_lora(model)

        layer = model.layers[0].q_proj
        self.assertIsInstance(layer, nn.Linear)
        self.assertEqual(layer.weight.shape, (16, 16))
        self.assertTrue(torch.allclose(layer.bias, original_bias))
        self.assertFalse(layer.weight.requires_grad)

    def test_merge_on_a_model_without_adapters_is_a_no_op(self):
        model = TinyModel()
        inputs = torch.randn(1, 3, 16)
        expected = model(inputs)

        self.assertEqual(merge_lora(model), 0)
        self.assertTrue(torch.allclose(model(inputs), expected))


class VectorizedRepetitionPenaltyTest(unittest.TestCase):
    def assert_matches(self, logits, tokens, penalty):
        expected = apply_repetition_penalty(logits, tokens, penalty)
        observed = vectorized_repetition_penalty(logits, tokens, penalty)
        self.assertTrue(
            torch.equal(observed, expected),
            f"mismatch for tokens={tokens} penalty={penalty}",
        )

    def test_matches_the_loop_on_mixed_sign_logits(self):
        torch.manual_seed(1)
        logits = torch.randn(1, 64)
        self.assert_matches(logits, [0, 5, 5, 63], 1.1)
        self.assert_matches(logits, list(range(64)), 1.5)

    def test_matches_the_loop_for_edge_cases(self):
        logits = torch.tensor([[1.0, -1.0, 0.0]])
        self.assert_matches(logits, [], 1.1)
        self.assert_matches(logits, [0, 1, 2], 1.0)
        self.assert_matches(logits, [-1, 3, 99], 1.2)
        self.assert_matches(logits, [1], 2.0)

    def test_zero_logits_are_untouched(self):
        logits = torch.tensor([[0.0, 4.0]])
        observed = vectorized_repetition_penalty(logits, [0], 1.1)
        self.assertEqual(float(observed[0, 0]), 0.0)

    def test_input_is_not_modified_in_place(self):
        logits = torch.tensor([[1.0, -1.0]])
        snapshot = logits.clone()
        vectorized_repetition_penalty(logits, [0, 1], 1.3)
        self.assertTrue(torch.equal(logits, snapshot))


if __name__ == "__main__":
    unittest.main()
