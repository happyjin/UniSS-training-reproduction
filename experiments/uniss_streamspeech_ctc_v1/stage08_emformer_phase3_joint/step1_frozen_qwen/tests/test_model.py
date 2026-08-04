import sys
import unittest
from pathlib import Path

import torch
from torch import nn


STEP = Path(__file__).resolve().parents[1]
TREE = STEP.parents[1]
for path in (
    TREE / "stage03_multitask_encoder",
    TREE / "stage03_multitask_encoder" / "ar_s2tt_v1",
    TREE / "stage04_b2_discrete_bridge",
    TREE / "stage07_end_to_end_eval",
    STEP,
):
    sys.path.insert(0, str(path))

from model import JointEmformerB1


class FakeEndpoint(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base = nn.Module()
        self.base.encoder = nn.Module()
        self.base.encoder.emformer_layers = nn.ModuleList([nn.Linear(4, 4) for _ in range(6)])
        self.base.output_norm = nn.LayerNorm(4)
        self.base.heads = nn.ModuleDict({"a": nn.Linear(4, 3)})
        self.target_embeddings = nn.ModuleDict({"eng": nn.Embedding(3, 4)})
        self.target_positions = nn.Embedding(8, 4)
        self.decoder = nn.Linear(4, 4)
        self.target_outputs = nn.ModuleDict({"eng": nn.Linear(4, 3)})


class FakeBridge(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(4, 4)


class JointModelTest(unittest.TestCase):
    def test_only_last_encoder_layers_and_heads_are_trainable(self) -> None:
        model = JointEmformerB1(
            FakeEndpoint(), FakeBridge(), nn.Linear(4, 5), residual_scale=0.05
        )
        model.configure_trainable(2)
        layers = model.endpoint.base.encoder.emformer_layers
        self.assertFalse(any(parameter.requires_grad for parameter in layers[3].parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in layers[4].parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in layers[5].parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in model.residual.parameters()))
        self.assertFalse(any(parameter.requires_grad for parameter in model.bridge.parameters()))

    def test_rejects_invalid_unfreeze_depth(self) -> None:
        model = JointEmformerB1(
            FakeEndpoint(), FakeBridge(), nn.Linear(4, 5), residual_scale=0.05
        )
        with self.assertRaises(ValueError):
            model.configure_trainable(0)


if __name__ == "__main__":
    unittest.main()
