import os
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from torch import nn


STAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STAGE))

from checkpoint_io import checkpoint_iteration, load_residual_into_model


class FakeResidualModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.residual = nn.Linear(3, 2)
        self.register_buffer("residual_scale", torch.tensor(0.05))


class CheckpointIOTest(unittest.TestCase):
    def test_loads_only_residual_tensors(self) -> None:
        tmp_root = Path(
            os.environ.get("TMPDIR", "/opt/dlami/nvme/jasonleeeli/tmp")
        )
        tmp_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=tmp_root) as directory:
            checkpoint = Path(directory) / "iter_0000042"
            expected_weight = torch.arange(6, dtype=torch.float32).reshape(2, 3)
            expected_bias = torch.tensor([0.25, -0.5])
            dcp.save(
                {
                    "bridge.residual.weight": expected_weight,
                    "bridge.residual.bias": expected_bias,
                    "bridge.residual_scale": torch.tensor(0.125),
                    "unrelated.large.tensor": torch.ones(4),
                },
                checkpoint_id=checkpoint,
            )
            model = FakeResidualModel()
            provenance = load_residual_into_model(model, checkpoint)
            torch.testing.assert_close(model.residual.weight, expected_weight)
            torch.testing.assert_close(model.residual.bias, expected_bias)
            self.assertAlmostEqual(float(model.residual_scale), 0.125)
            self.assertEqual(provenance["iteration"], 42)

    def test_rejects_non_iteration_directory_name(self) -> None:
        with self.assertRaises(ValueError):
            checkpoint_iteration("checkpoint_600")


if __name__ == "__main__":
    unittest.main()
