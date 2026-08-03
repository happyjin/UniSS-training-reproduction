import sys
import unittest
from pathlib import Path

import torch


HERE = Path(__file__).resolve()
AR = HERE.parents[1]
STAGE03 = AR.parent
ROOT = HERE.parents[5]
for path in (ROOT, STAGE03, AR):
    sys.path.insert(0, str(path))

from endpoint_model import EndpointCTCStudent
from model import EndpointCTCARStudent
from training.simul_uniss.subsecond_v2.stage_b_latent_model import LatentStageBModelConfig


class ARModelTest(unittest.TestCase):
    def test_direction_conditioned_ar_outputs(self) -> None:
        config = LatentStageBModelConfig(
            policy_vocab_size=10,
            hidden_size=16,
            num_layers=2,
            num_heads=2,
            ffn_dim=32,
            n_mels=8,
            stack_factor=4,
            segment_frames=4,
            right_context_frames=2,
            left_context_frames=8,
        )
        base = EndpointCTCStudent(config, eng_vocab_size=11, cmn_vocab_size=13)
        model = EndpointCTCARStudent(base, eng_vocab_size=11, cmn_vocab_size=13, decoder_layers=2)
        targets = torch.tensor([[1, 2, 3, -1], [4, 5, -1, -1]])
        output = model(
            torch.randn(2, 16000),
            torch.tensor([16000, 14000]),
            targets,
            torch.tensor([3, 2]),
            torch.tensor([0, 1]),
        )
        self.assertEqual(set(output["ar_logits"]), {"eng", "cmn"})
        self.assertEqual(output["ar_logits"]["cmn"][0].shape[-1], 13)
        self.assertEqual(output["ar_logits"]["eng"][0].shape[-1], 11)


if __name__ == "__main__":
    unittest.main()
