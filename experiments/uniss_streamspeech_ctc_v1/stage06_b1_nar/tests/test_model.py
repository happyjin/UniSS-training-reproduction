import sys
import unittest
from pathlib import Path

import torch
from torch import nn


STAGE = Path(__file__).resolve().parents[1]
TREE = STAGE.parents[0]
STAGE04 = TREE / "stage04_b2_discrete_bridge"
STAGE03 = TREE / "stage03_multitask_encoder"
ROOT = STAGE.parents[2]
for path in (ROOT, STAGE04, STAGE03, STAGE):
    sys.path.insert(0, str(path))

from bridge import StraightThroughCodebookBridge
from model import FrozenB2ResidualBridge


class FakeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = type("Config", (), {"hidden_size": 4})()

    def encode(self, waveform, lengths):
        hidden = waveform.unsqueeze(-1).repeat(1, 1, 4)
        return hidden, lengths


class FakeBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = FakeEncoder()
        self.bridge = StraightThroughCodebookBridge(
            encoder_dim=4,
            codebook=torch.randn(8, 6),
            qwen_glm_embeddings=torch.randn(8, 5),
            top_k=2,
        )


class B1ResidualTest(unittest.TestCase):
    def test_zero_residual_is_exact_b2_initialization(self) -> None:
        base = FakeBase().eval()
        model = FrozenB2ResidualBridge(base).eval()
        waveform = torch.randn(2, 6)
        lengths = torch.tensor([6, 5])
        with torch.no_grad():
            hidden, hidden_lengths = base.encoder.encode(waveform, lengths)
            expected = base.bridge(hidden, hidden_lengths)
            actual = model(waveform, lengths)
        torch.testing.assert_close(
            actual.qwen_speech_embeddings, expected.qwen_speech_embeddings
        )
        self.assertEqual(float(actual.residual_rms), 0.0)


if __name__ == "__main__":
    unittest.main()
