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

from checkpoint_io import (
    inference_tensor_names,
    load_step1_inference_into_model,
    load_step1_trainable_into_model,
    trainable_tensor_names,
)


class FakeJoint(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.endpoint = nn.Module()
        self.endpoint.base = nn.Module()
        self.endpoint.base.encoder = nn.Module()
        self.endpoint.base.encoder.emformer_layers = nn.ModuleList(
            [nn.Linear(3, 3) for _ in range(4)]
        )
        self.endpoint.base.output_norm = nn.LayerNorm(3)
        self.residual = nn.Linear(3, 2)
        self.register_buffer("residual_scale", torch.tensor(0.05))


class Step1CheckpointIOTest(unittest.TestCase):
    def test_loads_only_inference_path_tensors(self) -> None:
        tmp_root = Path(os.environ.get("TMPDIR", "/opt/dlami/nvme/jasonleeeli/tmp"))
        tmp_root.mkdir(parents=True, exist_ok=True)
        model = FakeJoint()
        names = inference_tensor_names(model, unfreeze_encoder_layers=2)
        self.assertFalse(any("emformer_layers.1." in name for name in names))
        expected = {
            f"joint.{name}": torch.full_like(model.state_dict()[name], index + 1)
            for index, name in enumerate(names)
        }
        with tempfile.TemporaryDirectory(dir=tmp_root) as directory:
            checkpoint = Path(directory) / "iter_0000042"
            dcp.save({**expected, "qwen.unused": torch.ones(2)}, checkpoint_id=checkpoint)
            provenance = load_step1_inference_into_model(
                model, checkpoint, unfreeze_encoder_layers=2
            )
        for name in names:
            torch.testing.assert_close(model.state_dict()[name], expected[f"joint.{name}"])
        self.assertEqual(provenance["iteration"], 42)
        self.assertEqual(provenance["loaded_tensors"], len(names))

    def test_rejects_invalid_unfreeze_depth(self) -> None:
        with self.assertRaises(ValueError):
            inference_tensor_names(FakeJoint(), unfreeze_encoder_layers=0)

    def test_loads_every_and_only_trainable_parameter(self) -> None:
        tmp_root = Path(os.environ.get("TMPDIR", "/opt/dlami/nvme/jasonleeeli/tmp"))
        tmp_root.mkdir(parents=True, exist_ok=True)
        model = FakeJoint()
        model.endpoint.base.encoder.emformer_layers[0].requires_grad_(False)
        model.endpoint.base.encoder.emformer_layers[1].requires_grad_(False)
        names = trainable_tensor_names(model)
        self.assertFalse(any("emformer_layers.0." in name for name in names))
        expected = {
            f"joint.{name}": torch.full_like(model.state_dict()[name], index + 2)
            for index, name in enumerate(names)
        }
        frozen_before = model.endpoint.base.encoder.emformer_layers[0].weight.detach().clone()
        with tempfile.TemporaryDirectory(dir=tmp_root) as directory:
            checkpoint = Path(directory) / "iter_0000800"
            dcp.save(expected, checkpoint_id=checkpoint)
            provenance = load_step1_trainable_into_model(model, checkpoint)
        for name in names:
            torch.testing.assert_close(model.state_dict()[name], expected[f"joint.{name}"])
        torch.testing.assert_close(
            model.endpoint.base.encoder.emformer_layers[0].weight, frozen_before
        )
        self.assertEqual(provenance["iteration"], 800)


if __name__ == "__main__":
    unittest.main()
