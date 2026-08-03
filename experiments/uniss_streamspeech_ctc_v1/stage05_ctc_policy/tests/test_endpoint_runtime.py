import sys
import unittest
from pathlib import Path

import torch


STAGE = Path(__file__).resolve().parents[1]
STAGE03 = STAGE.parents[0] / "stage03_multitask_encoder"
ROOT = STAGE.parents[2]
for path in (ROOT, STAGE, STAGE03):
    sys.path.insert(0, str(path))

from endpoint_model import EndpointCTCStudent
from endpoint_runtime import streaming_ctc_paths
from training.simul_uniss.subsecond_v2.stage_b_latent_model import (
    LatentStageBModelConfig,
)


class EndpointRuntimeTest(unittest.TestCase):
    def test_streaming_paths_grow_monotonically_by_real_frames(self) -> None:
        config = LatentStageBModelConfig(
            policy_vocab_size=2,
            hidden_size=16,
            num_layers=2,
            num_heads=2,
            ffn_dim=32,
            n_mels=8,
            stack_factor=2,
            segment_frames=2,
            right_context_frames=1,
            left_context_frames=4,
        )
        model = EndpointCTCStudent(config, eng_vocab_size=7, cmn_vocab_size=9).eval()
        waveform = torch.randn(1, 6400)
        lengths = torch.tensor([waveform.shape[1]])
        observations = list(
            streaming_ctc_paths(
                model,
                waveform,
                lengths,
                source_head="asr_eng",
                target_head="nar_s2tt_cmn",
            )
        )
        self.assertGreater(len(observations), 1)
        prior = 0
        for source, target, consumed, _ in observations:
            self.assertEqual(len(source), len(target))
            self.assertGreater(len(source), prior)
            self.assertGreaterEqual(consumed, len(source))
            prior = len(source)
        self.assertTrue(observations[-1][-1])


if __name__ == "__main__":
    unittest.main()

