from __future__ import annotations

import unittest

import torch
from torch import nn

from web_demo.true_subsecond_pilot15_streaming_v1.checkpoint_export import (
    interleave_hf_gqa_outputs,
    map_native_lora_to_hf,
    split_megatron_gqa_lora_b,
    verify_fused_mapping,
)
from web_demo.true_subsecond_pilot15_streaming_v1.model_loader import (
    _CapturedInputLoRALinear,
    _PreNormInputCapture,
)


def _pair(state, layer, module, in_features, out_features):
    prefix = (
        "true_subsecond_lora.branches."
        f"decoder__layers__{layer}__{module.replace('.', '__')}"
    )
    state[f"{prefix}.lora_a"] = torch.randn(32, in_features, dtype=torch.bfloat16)
    state[f"{prefix}.lora_b"] = torch.randn(out_features, 32, dtype=torch.bfloat16)


class CheckpointExportTest(unittest.TestCase):
    def test_hf_lora_uses_megatron_fused_prenorm_input(self) -> None:
        torch.manual_seed(9)
        norm = nn.RMSNorm(8)
        base = nn.Linear(8, 6, bias=False)
        capture = _PreNormInputCapture(norm)
        projection = _CapturedInputLoRALinear(
            base, rank=3, alpha=6.0, capture=capture
        )
        with torch.no_grad():
            projection.lora_A.weight.normal_()
            projection.lora_B.weight.normal_()
        raw = torch.randn(2, 4, 8) * 3.0
        normalized = norm(raw)
        actual = projection(normalized)
        expected = base(normalized) + (
            projection.lora_B(projection.lora_A(raw)) * projection.scaling
        ).to(base.weight.dtype)
        wrong_post_norm = base(normalized) + (
            projection.lora_B(projection.lora_A(normalized)) * projection.scaling
        ).to(base.weight.dtype)
        torch.testing.assert_close(actual, expected)
        self.assertFalse(torch.allclose(actual, wrong_post_norm))

    def test_gqa_rows_are_gathered_from_each_query_group(self) -> None:
        fused = torch.arange(1152, dtype=torch.float32).view(1152, 1).repeat(1, 32)
        query, key, value = split_megatron_gqa_lora_b(fused)
        expected_query = torch.cat((fused[:448], fused[576:1024]))
        expected_key = torch.cat((fused[448:512], fused[1024:1088]))
        expected_value = torch.cat((fused[512:576], fused[1088:1152]))
        torch.testing.assert_close(query, expected_query)
        torch.testing.assert_close(key, expected_key)
        torch.testing.assert_close(value, expected_value)

    def test_hf_outputs_rebuild_megatron_interleaved_layout(self) -> None:
        query = torch.arange(2 * 896, dtype=torch.float32).reshape(2, 896)
        key = torch.arange(2 * 128, dtype=torch.float32).reshape(2, 128) + 10_000
        value = torch.arange(2 * 128, dtype=torch.float32).reshape(2, 128) + 20_000
        rebuilt = interleave_hf_gqa_outputs(query, key, value)
        expected = torch.cat(
            (
                query.reshape(2, 2, 448),
                key.reshape(2, 2, 64),
                value.reshape(2, 2, 64),
            ),
            dim=-1,
        ).reshape(2, 1152)
        torch.testing.assert_close(rebuilt, expected)

    def test_fused_qwen_mapping_is_exact(self) -> None:
        torch.manual_seed(4)
        state = {}
        for layer in range(24):
            _pair(state, layer, "self_attention.linear_qkv", 896, 1152)
            _pair(state, layer, "self_attention.linear_proj", 896, 896)
        for layer in range(12, 24):
            _pair(state, layer, "mlp.linear_fc1", 896, 9728)
            _pair(state, layer, "mlp.linear_fc2", 4864, 896)
        mapped = map_native_lora_to_hf(state)
        self.assertEqual(len(mapped), 264)
        verify_fused_mapping(state, mapped)


if __name__ == "__main__":
    unittest.main()
