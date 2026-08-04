import os
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from torch import nn


STEP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STEP))

from checkpoint_io import load_step2_lora_into_qwen
from lora import LoRALinear, inject_lora, lora_tensor_names


class FakeAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(4, 4)
        self.k_proj = nn.Linear(4, 2)
        self.v_proj = nn.Linear(4, 2)


class FakeQwen(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = FakeAttention()

    def forward(self, value):
        return self.self_attn.q_proj(value), self.self_attn.v_proj(value)


class LoRATest(unittest.TestCase):
    def test_zero_initialized_adapter_preserves_output(self) -> None:
        model = FakeQwen()
        value = torch.randn(2, 4)
        before = model(value)
        injection = inject_lora(model, rank=2, alpha=4, dropout=0.0)
        after = model(value)
        self.assertEqual(injection.module_names, ("self_attn.q_proj", "self_attn.v_proj"))
        self.assertIsInstance(model.self_attn.q_proj, LoRALinear)
        torch.testing.assert_close(before[0], after[0])
        torch.testing.assert_close(before[1], after[1])
        trainable = [name for name, value in model.named_parameters() if value.requires_grad]
        self.assertTrue(trainable)
        self.assertTrue(all("lora_" in name for name in trainable))

    def test_selective_checkpoint_load(self) -> None:
        model = FakeQwen()
        inject_lora(model, rank=2, alpha=4, dropout=0.0)
        names = lora_tensor_names(model)
        expected = {
            f"qwen.{name}": torch.full_like(model.state_dict()[name], index + 1)
            for index, name in enumerate(names)
        }
        tmp_root = Path(os.environ.get("TMPDIR", "/opt/dlami/nvme/jasonleeeli/tmp"))
        tmp_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=tmp_root) as directory:
            checkpoint = Path(directory) / "iter_0000002"
            dcp.save(expected, checkpoint_id=checkpoint)
            provenance = load_step2_lora_into_qwen(model, checkpoint)
        for name in names:
            torch.testing.assert_close(model.state_dict()[name], expected[f"qwen.{name}"])
        self.assertEqual(provenance["iteration"], 2)


if __name__ == "__main__":
    unittest.main()
