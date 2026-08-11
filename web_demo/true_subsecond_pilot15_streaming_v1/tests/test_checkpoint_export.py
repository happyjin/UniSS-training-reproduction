from __future__ import annotations

import unittest

import torch

from web_demo.true_subsecond_pilot15_streaming_v1.checkpoint_export import (
    map_native_lora_to_hf,
    verify_fused_mapping,
)


def _pair(state, layer, module, in_features, out_features):
    prefix = (
        "true_subsecond_lora.branches."
        f"decoder__layers__{layer}__{module.replace('.', '__')}"
    )
    state[f"{prefix}.lora_a"] = torch.randn(32, in_features, dtype=torch.bfloat16)
    state[f"{prefix}.lora_b"] = torch.randn(out_features, 32, dtype=torch.bfloat16)


class CheckpointExportTest(unittest.TestCase):
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
